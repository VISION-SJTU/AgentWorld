# AgentWorld

**Learning Agent-centric Vision-Language-Action World Models for Autonomous Driving (NeurIPS 2026)**

AgentWorld introduces agent-centric representations into a multimodal language model for driving perception, reasoning, and planning. Training follows three stages: physical grounding, reasoning SFT, and GRPO. Inference uses only the MLLM backbone.

This implementation supports **Qwen2-VL / Qwen2.5-VL**, with grounding and SFT built on **LLaMAFactory**, and reinforcement learning built on **EasyR1**.

## 🛠️ Installation

We recommend separate Python environments for supervised training and reinforcement learning. Run the following commands from the project root.

```bash
# Grounding / SFT environment
pip install -e .
pip install -e ./LLaMAFactory

# RL environment
pip install -e .
pip install -e ./EasyR1
pip install -r ./EasyR1/requirements.txt
```

Install compatible versions of PyTorch, CUDA, vLLM, and FlashAttention for your hardware. Transformers 4.57.1 is supported.

Download a pretrained backbone such as [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct). Install [VGGT](https://github.com/facebookresearch/vggt) for offline geometry feature extraction.

## 📂 Data Preparation

Prepare JSONL records containing camera images, prompts, geometry feature paths, agent annotations, and ego future trajectories. Coordinates use the current ego frame: X points right, Y points forward, and Z points upward.

For nuScenes:

```bash
python scripts/prepare_nuscenes.py \
    --dataroot /path/to/nuscenes --split train \
    --output data/train.jsonl --geometry-dir data/geometry

python scripts/extract_geometry.py data/train.jsonl \
    --model facebook/VGGT-1B
```

Prepare validation data separately using `--split val`. The nuScenes converter requires `nuscenes-devkit`.

## ⚙️ Training

Set model paths, data paths, batch sizes, and GPU counts in the corresponding YAML files before training. Commands below run from the project root unless otherwise specified.

### Stage 1: Physical Grounding

Train agent representations with geometry/visual routing and a shared perception head.

```bash
torchrun --nproc_per_node=4 LLaMAFactory/examples/agentworld/train.py \
    LLaMAFactory/examples/agentworld/grounding.yaml \
    --stage grounding --data data/train.jsonl \
    --agent-config configs/agentworld.json
```

### Stage 2: Reasoning SFT

Build reasoning supervision using the Stage-1 slot assignments, then train with the routing module and perception head frozen. The grounding scaffold is progressively removed.

```bash
python LLaMAFactory/examples/agentworld/build_sft_data.py \
    --model outputs/grounding --data data/train.jsonl \
    --output data/sft_train.jsonl

torchrun --nproc_per_node=4 LLaMAFactory/examples/agentworld/train.py \
    LLaMAFactory/examples/agentworld/sft.yaml \
    --stage sft --data data/sft_train.jsonl
```

The data builder provides template-based reasoning labels; external reasoning annotations can be supplied while preserving slot assignments. Prepare `data/sft_val.jsonl` from the independent validation split in the same way.

### Stage 3: GRPO

Export the policy and frozen reward scaffold, then prepare the RL datasets.

```bash
python scripts/export_policy.py \
    outputs/sft outputs/policy outputs/reward_scaffold

python scripts/build_rl_data.py data/sft_train.jsonl data/rl_train.jsonl
python scripts/build_rl_data.py data/sft_val.jsonl data/rl_val.jsonl

cd EasyR1
python -m verl.trainer.main config=examples/agentworld/grpo.yaml
```

GRPO optimizes the MLLM policy. A frozen perception head decodes current-policy hidden states for agent interaction rewards; geometry extraction and routing are not used during this stage.

## 🔑 Inference

Run planning with the exported policy:

```bash
python scripts/plan.py \
    --model outputs/policy --data data/sft_val.jsonl --index 0
```

Inference requires only images, the prompt, and an optional system message.

## 🙏 Acknowledgements

This implementation builds on [LLaMAFactory](https://github.com/hiyouga/LLaMA-Factory), [EasyR1](https://github.com/hiyouga/EasyR1), [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL), and [VGGT](https://github.com/facebookresearch/vggt).
