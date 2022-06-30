from .sashimi import Sashimi
from .wavenet import WaveNet

def construct_model(model_cfg):
    name = model_cfg.pop("backbone")
    model_cls = {
        "wavenet": WaveNet,
        "sashimi": Sashimi,
    }[name]
    model = model_cls(**model_cfg)
    model_cfg["backbone"] = name # restore
    return model
    # if model_cfg.backbone == "wavenet":
    #     return WaveNet(**model_cfg)
    # elif model_cfg.backbone == ["sashimi"]:
    #     return Sashimi(**model_cfg)
