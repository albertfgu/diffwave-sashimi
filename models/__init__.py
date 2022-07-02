from .sashimi import Sashimi
from .wavenet import WaveNet

def construct_model(model_cfg):
    name = model_cfg.pop("_name_")
    model_cls = {
        "wavenet": WaveNet,
        "sashimi": Sashimi,
    }[name]
    model = model_cls(**model_cfg)
    model_cfg["_name_"] = name # restore
    return model
    # if model_cfg.backbone == "wavenet":
    #     return WaveNet(**model_cfg)
    # elif model_cfg.backbone == ["sashimi"]:
    #     return Sashimi(**model_cfg)

def model_identifier(model_cfg):
    model_cls = {
        "wavenet": WaveNet,
        "sashimi": Sashimi,
    }[model_cfg._name_]
    return model_cls.name(model_cfg)
