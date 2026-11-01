from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass
class AgentConfig:
    num_agents: int = 20
    num_classes: int = 10
    hidden_size: int = 1536  # overwritten from the actual backbone
    geometry_dim: int = 2048
    routing_dim: int = 256
    routing_heads: int = 8
    routing_layers: int = 3
    box_dim: int = 8  # x,y,z,w,l,h,sin(yaw),cos(yaw), in ego frame
    horizon: int = 6
    image_token_id: int = 151655

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    @property
    def tokens(self):
        return [f"<agent_{i + 1}>" for i in range(self.num_agents)]

    @property
    def perception(self):
        return "<perception>" + "".join(self.tokens) + "</perception>"


def normalize_qwen_config(config):
    """Prevent Transformers 4.57 Qwen-VL legacy proxies from hiding the outer model type.

    Nested configs serialized by newer Transformers can carry model_type as an
    instance attribute. The legacy VL __getattribute__ then incorrectly exposes
    the text-only type on the multimodal config. The text config's class still
    supplies its own model_type after removing the redundant instance attribute.
    """
    if type(config).model_type in ("qwen2_vl", "qwen2_5_vl"):
        text = vars(config).get("text_config")
        if text is not None:
            vars(text).pop("model_type", None)
    return config
