import torch
from torch.utils.data.distributed import DistributedSampler
from .sc import SpeechCommands
from .mel2samp import Mel2Samp

def dataloader(dataset_cfg, batch_size, num_gpus, unconditional=True):
    # TODO would be nice if unconditional was decoupled from dataset

    dataset_name = dataset_cfg.pop("_name_")
    if dataset_name == "sc09":
        assert unconditional
        dataset = SpeechCommands(dataset_cfg.data_path)
    elif dataset_name == "ljspeech":
        assert not unconditional
        dataset = Mel2Samp(**dataset_cfg)
    dataset_cfg["_name_"] = dataset_name # Restore

    # distributed sampler
    train_sampler = DistributedSampler(dataset) if num_gpus > 1 else None

    trainloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=False,
        drop_last=True,
    )
    return trainloader
