import torch
from torch.utils.data.distributed import DistributedSampler
from mel2samp import Mel2Samp

def load_LJSpeech(dataset_config, batch_size=4, num_gpus=1):
    LJSpeech_dataset = Mel2Samp(**dataset_config)

    # distributed sampler
    train_sampler = DistributedSampler(LJSpeech_dataset) if num_gpus > 1 else None

    trainloader = torch.utils.data.DataLoader(LJSpeech_dataset,
                                              batch_size=batch_size,
                                              sampler=train_sampler,
                                              num_workers=4,
                                              pin_memory=False,
                                              drop_last=True)
    return trainloader
