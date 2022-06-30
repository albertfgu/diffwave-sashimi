from .sashimi import Sashimi
from .wavenet import WaveNet

def construct_model(model_cfg):
    model_cls = {
        "wavenet": WaveNet,
        "sashimi": Sashimi,
    }[model_cfg.pop("backbone")]
    return model_cls(**model_cfg)
    # if model_cfg.backbone == "wavenet":
    #     return WaveNet(**model_cfg)
    # elif model_cfg.backbone == ["sashimi"]:
    #     return Sashimi(**model_cfg)
