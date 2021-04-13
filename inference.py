import os
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from scipy.io.wavfile import write as wavwrite
from scipy.io.wavfile import read as wavread

from util import rescale, find_max_epoch, print_size, sampling, calc_diffusion_hyperparams
from WaveNet import WaveNet_vocoder as WaveNet

def generate(output_directory, tensorboard_directory,
             mel_path, condition_name,
             ckpt_path, ckpt_iter):
    """
    Generate audio based on ground truth mel spectrogram

    Parameters:
    output_directory (str):         save generated speeches to this path
    tensorboard_directory (str):    save tensorboard events to this path
    mel_path (str):                 ground truth mel spectrogram path
    condition_name (str):           name of ground truth mel spectrogram to be conditioned on
                                    e.g. LJ001-0001
    ckpt_path (str):                checkpoint path
    ckpt_iter (int or 'max'):       the pretrained checkpoint to be loaded; 
                                    automitically selects the maximum iteration if 'max' is selected
    """

    # generate experiment (local) path
    local_path = "ch{}_T{}_betaT{}".format(wavenet_config["res_channels"], 
                                           diffusion_config["T"], 
                                           diffusion_config["beta_T"])

    # Get shared output_directory ready
    output_directory = os.path.join('exp', local_path, output_directory)
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    print("output directory", output_directory, flush=True)

    # map diffusion hyperparameters to gpu
    for key in diffusion_hyperparams:
        if key is not "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].cuda()

    # predefine model
    net = WaveNet(**wavenet_config).cuda()
    print_size(net)

    # load checkpoint
    ckpt_path = os.path.join('exp', local_path, ckpt_path)
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(ckpt_path)
    model_path = os.path.join(ckpt_path, '{}.pkl'.format(ckpt_iter))
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        net.load_state_dict(checkpoint['model_state_dict'])
        print('Successfully loaded model at iteration {}'.format(ckpt_iter))
    except:
        raise Exception('No valid model found')
    
    # use ground truth mel spec
    try:
        ground_truth_mel_name = os.path.join(mel_path, '{}.wav.pt'.format(condition_name))
        ground_truth_mel_spectrogram = torch.load(ground_truth_mel_name).unsqueeze(0).cuda()
    except:
        raise Exception('No ground truth mel spectrogram found')
    audio_length = ground_truth_mel_spectrogram.shape[-1] * trainset_config["hop_length"]
    print('begin generating audio of length %s' % audio_length)

    # inference
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    generated_audio = sampling(net, (1,1,audio_length), 
                            diffusion_hyperparams, 
                            condition=ground_truth_mel_spectrogram)

    end.record()
    torch.cuda.synchronize()
    print('generated {} at iteration {} in {} seconds'.format(condition_name, 
                                                              ckpt_iter, 
                                                              int(start.elapsed_time(end)/1000)))

    # save audio to .wav
    outfile = '{}_{}_{}k_{}.wav'.format(wavenet_config["res_channels"], 
                                        diffusion_config["T"], 
                                        ckpt_iter // 1000, 
                                        condition_name)
    wavwrite(os.path.join(output_directory, outfile), 
                trainset_config["sampling_rate"],
                generated_audio.squeeze().cpu().numpy())

    # save audio to tensorboard
    tb = SummaryWriter(os.path.join('exp', local_path, tensorboard_directory))
    tb.add_audio(tag=outfile, snd_tensor=generated_audio.squeeze(0), sample_rate=trainset_config["sampling_rate"])
    tb.close()

    print('saved generated samples at iteration %s' % ckpt_iter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        help='JSON file for configuration')
    parser.add_argument('-ckpt_iter', '--ckpt_iter', default='max',
                        help='Which checkpoint to use; assign a number or "max"')
    parser.add_argument('-cond', '--condition_name', type=str,
                        help='Name of the ground truth mel spectrogram to be conditioned on')                  
    args = parser.parse_args()

    # Parse configs. Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    gen_config              = config["gen_config"]
    global wavenet_config
    wavenet_config          = config["wavenet_config"]      # to define wavenet
    global diffusion_config 
    diffusion_config        = config["diffusion_config"]    # basic hyperparameters
    global trainset_config
    trainset_config         = config["trainset_config"]     # to read trainset configurations
    global diffusion_hyperparams
    diffusion_hyperparams   = calc_diffusion_hyperparams(**diffusion_config)  # dictionary of all diffusion hyperparameters

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    generate(**gen_config,
             ckpt_iter=args.ckpt_iter,
             condition_name=args.condition_name)
    