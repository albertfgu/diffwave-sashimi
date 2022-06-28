import os
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter # If tensorboard is preferred over wandb

from scipy.io.wavfile import write as wavwrite
from scipy.io.wavfile import read as wavread

from util import rescale, find_max_epoch, print_size, sampling, calc_diffusion_hyperparams
from util import local_directory
from util import smooth_ckpt
# from WaveNet import WaveNet_Speech_Commands as WaveNet
from model import construct_model

@torch.no_grad()
def generate(
        # output_directory, # tensorboard_directory,
        num_samples,
        # ckpt_path,
        ckpt_iter,
        diffusion_config,
        model_config,
        dataset_config,
        batch_size=0,
        ckpt_smooth=-1,
        rank=0,
        mel_path="mel_spectrogram", mel_name="LJ001-0001",
    ):
    """
    Generate audio based on ground truth mel spectrogram

    Parameters:
    output_directory (str):         checkpoint path
    tensorboard_directory (str):    save tensorboard events to this path
    num_samples (int):              number of samples to generate, default is 4
    ckpt_path (str):                checkpoint path
    ckpt_iter (int or 'max'):       the pretrained checkpoint to be loaded;
                                    automitically selects the maximum iteration if 'max' is selected
    """

    if rank is not None:
        print(f"rank {rank} {torch.cuda.device_count()} GPUs")
        torch.cuda.set_device(rank % torch.cuda.device_count())

    local_path, output_directory = local_directory(model_config, diffusion_config, dataset_config, 'waveforms')

    # map diffusion hyperparameters to gpu
    diffusion_hyperparams   = calc_diffusion_hyperparams(**diffusion_config, fast=True)  # dictionary of all diffusion hyperparameters
    # for key in diffusion_hyperparams:
    #     if key != "T":
    #         diffusion_hyperparams[key] = diffusion_hyperparams[key].cuda()

    # predefine model
    # net = WaveNet(**model_config).cuda()
    net = construct_model(model_config).cuda()
    print_size(net)
    net.eval()

    # load checkpoint
    print('ckpt_iter', ckpt_iter)
    ckpt_path = os.path.join('exp', local_path, 'checkpoint')
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(ckpt_path)
    ckpt_iter = int(ckpt_iter)

    if ckpt_smooth < 0: # TODO not a good default, should be None
        try:
            model_path = os.path.join(ckpt_path, '{}.pkl'.format(ckpt_iter))
            checkpoint = torch.load(model_path, map_location='cpu')
            net.load_state_dict(checkpoint['model_state_dict'])
            print('Successfully loaded model at iteration {}'.format(ckpt_iter))
        except:
            raise Exception('No valid model found')
    else:
        state_dict = smooth_ckpt(ckpt_path, ckpt_smooth, ckpt_iter, alpha=None)
        net.load_state_dict(state_dict)

    # Add checkpoint number to output directory
    output_directory = os.path.join(output_directory, str(ckpt_iter))
    # if rank is None: rank = 0
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        print("saving to output directory", output_directory)

    # if rank is not None:
    #     output_directory = os.path.join(output_directory, str(rank))
    #     if not os.path.isdir(output_directory):
    #         os.makedirs(output_directory)
    #         os.chmod(output_directory, 0o775)

    if batch_size <= 0:
        batch_size = num_samples
    assert num_samples % batch_size == 0

    if mel_path is not None and mel_name is not None:
        # use ground truth mel spec
        try:
            ground_truth_mel_name = os.path.join(mel_path, '{}.wav.pt'.format(mel_name))
            ground_truth_mel_spectrogram = torch.load(ground_truth_mel_name).unsqueeze(0).cuda()
        except:
            raise Exception('No ground truth mel spectrogram found')
        audio_length = ground_truth_mel_spectrogram.shape[-1] * dataset_config["hop_length"]
    else:
        # predefine audio shape
        audio_length = dataset_config["segment_length"]  # 16000
        ground_truth_mel_spectrogram = None
    print(f'begin generating audio of length {audio_length} | {num_samples} samples with batch size {batch_size}')

    # inference
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    generated_audio = []

    for _ in range(num_samples // batch_size):
        _audio = sampling(
            net,
            (batch_size,1,audio_length),
            diffusion_hyperparams,
            condition=ground_truth_mel_spectrogram,
        )
        generated_audio.append(_audio)
    generated_audio = torch.cat(generated_audio, dim=0)

    end.record()
    torch.cuda.synchronize()
    print('generated {} samples shape {} at iteration {} in {} seconds'.format(num_samples,
        generated_audio.shape,
        ckpt_iter,
        int(start.elapsed_time(end)/1000)))

    # save audio to .wav
    for i in range(num_samples):
        outfile = '{}k_{}.wav'.format(
        # outfile = '{}_{}_{}k_{}.wav'.format(
            # model_config["res_channels"],
            # diffusion_config["T"],
            ckpt_iter // 1000,
            num_samples*rank + i,
        )
        wavwrite(os.path.join(output_directory, outfile),
                    dataset_config["sampling_rate"],
                    generated_audio[i].squeeze().cpu().numpy())

        # save audio to tensorboard
        # tb = SummaryWriter(os.path.join('exp', local_path, tensorboard_directory))
        # tb.add_audio(tag=outfile, snd_tensor=generated_audio[i], sample_rate=dataset_config["sampling_rate"])
        # tb.close()

    print('saved generated samples at iteration %s' % ckpt_iter)
    return generated_audio


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config.json',
                        help='JSON file for configuration')
    parser.add_argument('-ckpt_iter', '--ckpt_iter', default='max',
                        help='Which checkpoint to use; assign a number or "max"')
    parser.add_argument('-s', '--ckpt_smooth', default=-1, type=int,
                        help='Which checkpoint to start averaging from')
    parser.add_argument('-n', '--num_samples', type=int, default=4,
                        help='Number of utterances to be generated')
    parser.add_argument('-b', '--batch_size', type=int, default=0,
                        help='Number of samples to generate at once per GPU')
    parser.add_argument('-r', '--rank', type=int, default=0,
                        help='rank of process for distributed')
    args = parser.parse_args()

    # Parse configs. Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    # gen_config              = config["gen_config"]
    # global model_config
    model_config          = config["model_config"]      # to define wavenet
    # global diffusion_config
    diffusion_config        = config["diffusion_config"]    # basic hyperparameters
    # global dataset_config
    dataset_config         = config["dataset_config"]     # to read trainset configurations
    # global diffusion_hyperparams
    # global train_config
    # train_config            = config["train_config"]


    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    generate(
        # **gen_config,
        # output_directory=train_config["output_directory"],
        ckpt_iter=args.ckpt_iter,
        ckpt_smooth=args.ckpt_smooth,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        diffusion_config=diffusion_config,
        model_config=model_config,
        dataset_config=dataset_config,
        rank=args.rank,
    )
