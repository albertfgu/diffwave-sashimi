from sashimi import Sashimi
from wavenet import WaveNet

def construct_model(wavenet_config):
    if wavenet_config["sashimi"]:
        return Sashimi(**wavenet_config)
    else:
        return WaveNet(**wavenet_config)
