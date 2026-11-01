"""torchrun entry point using LLaMAFactory loaders, template, collator and trainer."""

import argparse
from pathlib import Path

import torch
import yaml
from transformers import set_seed

from llamafactory.data import SFTDataCollatorWith4DAttentionMask, get_template_and_fix_tokenizer
from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
from llamafactory.hparams import get_train_args
from llamafactory.model import load_model, load_tokenizer
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer

from agentworld.backbone import GroundingContext
from agentworld.config import AgentConfig
from agentworld.data import DrivingCollator, DrivingDataset
from agentworld.modules import AgentWorld, grounding_loss
from agentworld.tokens import gather_slots, perception_positions, register_tokens


class AgentTrainer(CustomSeq2SeqTrainer):
    def __init__(self, *, stage, token_ids, context, **kwargs):
        super().__init__(**kwargs)
        self.stage, self.token_ids, self.context = stage, token_ids, context
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        geometry, geometry_mask, targets = (inputs.pop(k) for k in ("geometry", "geometry_mask", "targets"))
        tokenizer = self.processing_class
        assistant_ids = inputs["input_ids"].masked_fill(inputs["labels"].eq(-100), -1)
        positions, _ = perception_positions(
            assistant_ids,
            self.token_ids,
            tokenizer.convert_tokens_to_ids("<perception>"),
            tokenizer.convert_tokens_to_ids("</perception>"),
        )
        progress = self.state.global_step / max(self.state.max_steps, 1)
        # Unspecified paper schedule: first/middle/final thirds = full/linear dropout/none.
        probability = 1.0 if self.stage == "grounding" else min(1.0, max(0.0, 2.0 - 3.0 * progress))
        keep = torch.rand(inputs["input_ids"].size(0), device=geometry.device) < probability
        if probability > 0:
            self.context.batch = (geometry, geometry_mask, positions, keep, inputs["input_ids"])
        if self.stage == "grounding":
            inputs.pop("labels")
        try:
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
        finally:
            self.context.clear()
        states = gather_slots(outputs.hidden_states[-1], positions)
        module = self.context.model.agentworld
        if self.stage == "grounding":
            loss, _ = grounding_loss(module.head(states.to(next(module.head.parameters()).dtype)), targets)
        else:
            loss = outputs.loss
            eta = 0.1 * probability  # optional grounding regularizer; exactly zero in final phase
            if eta:
                physical_loss, _ = grounding_loss(
                    module.head(states.to(next(module.head.parameters()).dtype)), targets
                )
                loss = loss + eta * physical_loss
        return (loss, outputs) if return_outputs else loss

    def _save(self, output_dir=None, state_dict=None):
        # Keep HF checkpoint/vLLM weights standard; store training scaffold separately.
        directory = output_dir or self.args.output_dir
        state_dict = self.model.state_dict() if state_dict is None else state_dict
        policy = {key: value for key, value in state_dict.items() if not key.startswith("agentworld.")}
        super()._save(directory, state_dict=policy)
        self.context.model.agentworld.save(directory)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        super()._load_from_checkpoint(resume_from_checkpoint, model=model)
        saved = AgentWorld.load(resume_from_checkpoint)
        self.context.model.agentworld.load_state_dict(saved.state_dict())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--stage", choices=["grounding", "sft"], required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--agent-config", default=None)
    args = parser.parse_args()
    options = yaml.safe_load(Path(args.config).read_text())
    options.setdefault("dataset", "agentworld")  # parsed by LF; prepared records are loaded below
    options["remove_unused_columns"] = False
    model_args, data_args, training_args, finetuning_args, _ = get_train_args(options)
    if (
        finetuning_args.finetuning_type != "full"
        or data_args.packing
        or training_args.predict_with_generate
        or training_args.deepspeed
        or training_args.fsdp
        or model_args.enable_liger_kernel
    ):
        raise ValueError("AgentWorld entry supports full-tuning, unpacked DDP, standard loss and no generation eval.")
    model_args.agentworld = True
    set_seed(training_args.seed)
    checkpoint = Path(model_args.model_name_or_path)
    module = AgentWorld.load(checkpoint) if (checkpoint / "agentworld_config.json").exists() else None
    if module is None and args.stage == "sft":
        raise ValueError("Stage 2 requires a Stage-1 checkpoint with its grounding sidecar.")
    config = module.config if module else AgentConfig.load(args.agent_config) if args.agent_config else AgentConfig()
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    token_ids = register_tokens(tokenizer, config)
    model_args.resize_vocab = True
    model = load_model(tokenizer, model_args, finetuning_args, is_trainable=True)
    if model.config.model_type not in ("qwen2_vl", "qwen2_5_vl"):
        raise ValueError(
            f"Use a Qwen2-VL or Qwen2.5-VL backbone; got {type(model).__name__}, {model.config.model_type}."
        )
    config.hidden_size = model.get_input_embeddings().weight.size(1)
    config.image_token_id = model.config.image_token_id
    model.agentworld = module or AgentWorld(config)
    model.agentworld.to(dtype=model.get_input_embeddings().weight.dtype)
    if args.stage == "sft":
        model.agentworld.requires_grad_(False)
    elif model.get_output_embeddings().weight is not model.get_input_embeddings().weight:
        # Grounding uses hidden states only; an untied LM output head has no loss path.
        model.get_output_embeddings().requires_grad_(False)
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    context = GroundingContext(model, token_ids)
    encoder = SupervisedDatasetProcessor(template, tokenizer, tokenizer_module["processor"], data_args)
    dataset = DrivingDataset(args.data, encoder, config, args.stage)
    collator = SFTDataCollatorWith4DAttentionMask(
        template=template, model=model, pad_to_multiple_of=8, **tokenizer_module
    )
    trainer = AgentTrainer(
        stage=args.stage,
        token_ids=token_ids,
        context=context,
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        train_dataset=dataset,
        data_collator=DrivingCollator(collator),
        **tokenizer_module,
    )
    try:
        result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.save_state()
        trainer.save_metrics("train", result.metrics)
    finally:
        context.remove()


if __name__ == "__main__":
    main()
