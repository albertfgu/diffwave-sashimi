## Modified based on https://github.com/pytorch/audio/blob/master/torchaudio/datasets/speechcommands.py

import os
from pathlib import Path
import numpy as np
import torch
from torchvision import datasets, models, transforms

from torch.utils.data.distributed import DistributedSampler
from scipy.io.wavfile import read as wavread

from typing import Tuple

import torchaudio
from torch.utils.data import Dataset
from torch import Tensor
from torchaudio.datasets.utils import (
    download_url,
    extract_archive,
)

HASH_DIVIDER = "_nohash_"
EXCEPT_FOLDER = "_background_noise_"

def fix_length(tensor, length):
    assert len(tensor.shape) == 2 and tensor.shape[0] == 1
    if tensor.shape[1] > length:
        return tensor[:,:length]
    elif tensor.shape[1] < length:
        return torch.cat([tensor, torch.zeros(1, length-tensor.shape[1])], dim=1)
    else:
        return tensor

def load_speechcommands_item(filepath: str, path: str):
    relpath = os.path.relpath(filepath, path)
    label, filename = os.path.split(relpath)
    speaker, _ = os.path.splitext(filename)

    speaker_id, utterance_number = speaker.split(HASH_DIVIDER)
    utterance_number = int(utterance_number)

    # Load audio
    waveform, sample_rate = torchaudio.load(filepath)
    return (fix_length(waveform, length=16000), sample_rate, label)

class SpeechCommands(Dataset):
    """
    Create a Dataset for Speech Commands. Each item is a tuple of the form:
    waveform, sample_rate, label
    """

    def __init__(self, path: str):
        self._path = path # os.path.join(root, folder_in_archive)
        # walker = walk_files(self._path, suffix=".wav", prefix=True)
        walker = sorted(str(p) for p in Path(self._path).glob('**/*.wav'))
        walker = filter(lambda w: HASH_DIVIDER in w and EXCEPT_FOLDER not in w, walker)
        self._walker = list(walker)

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str, int]:
        fileid = self._walker[n]
        return load_speechcommands_item(fileid, self._path)

    def __len__(self) -> int:
        return len(self._walker)
