import os
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter
import wandb
from tqdm import tqdm

from dataset_sc import load_Speech_commands
from dataset_ljspeech import load_LJSpeech
from util import rescale, find_max_epoch, print_size
from util import training_loss, calc_diffusion_hyperparams
from util import local_directory

from distributed_util import init_distributed, apply_gradient_allreduce, reduce_tensor
from inference import generate

# from WaveNet import WaveNet_Speech_Commands as WaveNet
from model import construct_model


def train(num_gpus, rank, group_name, wandb_id,
        diffusion_config,
        model_config,
        dataset_config,
        # checkpoint_directory,
        ckpt_iter, n_iters, iters_per_ckpt, iters_per_logging,
        learning_rate, batch_size_per_gpu,
        wandb_mode,
        n_samples,
        mel_path=None,
    ):
    """
    Train the WaveNet model on the Speech Commands dataset

    Parameters:
    num_gpus, rank, group_name:     parameters for distributed training
    checkpoint_directory (str):         save model checkpoints to this path
    ckpt_iter (int or 'max'):       the pretrained checkpoint to be loaded;
                                    automitically selects the maximum iteration if 'max' is selected
    n_iters (int):                  number of iterations to train, default is 1M
    iters_per_ckpt (int):           number of iterations to save checkpoint,
                                    default is 10k, for models with residual_channel=64 this number can be larger
    iters_per_logging (int):        number of iterations to save training log and compute validation loss, default is 100
    learning_rate (float):          learning rate
    batch_size_per_gpu (int):       batchsize per gpu, default is 2 so total batchsize is 16 with 8 gpus
    """

    # # generate experiment (local) path
    # local_path = "ch{}_T{}_betaT{}".format(model_config["res_channels"],
    #                                        diffusion_config["T"],
    #                                        diffusion_config["beta_T"])
    # Create tensorboard logger.
    if rank == 0:
        # tb = SummaryWriter(os.path.join('exp', local_path, tensorboard_directory))
        cfg = {
            'model': model_config,
            'train': train_config,
            'diffusion': diffusion_config,
        }
        wandb_id = None if len(wandb_id) == 0 else wandb_id
        wandb.init(
            project="hippo", job_type='training', mode=wandb_mode, id=wandb_id,
            config=cfg,
        )

    # distributed running initialization
    if num_gpus > 1:
        init_distributed(rank, num_gpus, group_name, **dist_config)

    # # Get shared checkpoint_directory ready
    # checkpoint_directory = os.path.join('exp', local_path, checkpoint_directory)
    # if rank == 0:
    #     if not os.path.isdir(checkpoint_directory):
    #         os.makedirs(checkpoint_directory)
    #         os.chmod(checkpoint_directory, 0o775)
    #     print("output directory", checkpoint_directory, flush=True)

    local_path, checkpoint_directory = local_directory(model_config, diffusion_config, dataset_config, 'checkpoint')

    # map diffusion hyperparameters to gpu
    diffusion_hyperparams   = calc_diffusion_hyperparams(**diffusion_config, fast=False)  # dictionary of all diffusion hyperparameters

    # load training data
    if model_config['unconditional']:
        trainloader = load_Speech_commands(path=dataset_config["data_path"],
                                           batch_size=batch_size_per_gpu,
                                           num_gpus=num_gpus)
    else:
        trainloader = load_LJSpeech(dataset_config=dataset_config,
                                    batch_size=batch_size_per_gpu,
                                    num_gpus=num_gpus)
    print('Data loaded')

    # predefine model
    # net = WaveNet(**model_config).cuda()
    net = construct_model(model_config).cuda()
    print_size(net, verbose=False)

    # apply gradient all reduce
    if num_gpus > 1:
        net = apply_gradient_allreduce(net)

    # define optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # load checkpoint
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(checkpoint_directory)
    if ckpt_iter >= 0:
        try:
            # load checkpoint file
            model_path = os.path.join(checkpoint_directory, '{}.pkl'.format(ckpt_iter))
            checkpoint = torch.load(model_path, map_location='cpu')

            # feed model dict and optimizer state
            net.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                # HACK to reset learning rate
                optimizer.param_groups[0]['lr'] = learning_rate

            print('Successfully loaded model at iteration {}'.format(ckpt_iter))
        except:
            ckpt_iter = -1
            print('No valid checkpoint model found, start training from initialization.')
    else:
        ckpt_iter = -1
        print('No valid checkpoint model found, start training from initialization.')

    # training
    n_iter = ckpt_iter + 1
    while n_iter < n_iters + 1:
        # for audio, _, _ in trainloader:
        epoch_loss = 0.
        for data in tqdm(trainloader, desc=f'Epoch {n_iter // len(trainloader)}'):
            if model_config["unconditional"]:
                audio, _, _ = data
                # load audio
                audio = audio.cuda()
                mel_spectrogram = None
            else:
                mel_spectrogram, audio = data
                mel_spectrogram = mel_spectrogram.cuda()
                audio = audio.cuda()

            # back-propagation
            optimizer.zero_grad()
            loss = training_loss(net, nn.MSELoss(), audio, diffusion_hyperparams, mel_spec=mel_spectrogram)
            if num_gpus > 1:
                reduced_loss = reduce_tensor(loss.data, num_gpus).item()
            else:
                reduced_loss = loss.item()
            loss.backward()
            optimizer.step()

            epoch_loss += reduced_loss

            # output to log
            # note, only do this on the first gpu
            if n_iter % iters_per_logging == 0 and rank == 0:
                # save training loss to tensorboard
                # print("iteration: {} \treduced loss: {} \tloss: {}".format(n_iter, reduced_loss, loss.item()))
                # tb.add_scalar("Log-Train-Loss", torch.log(loss).item(), n_iter)
                # tb.add_scalar("Log-Train-Reduced-Loss", np.log(reduced_loss), n_iter)
                wandb.log({
                    # 'train/loss': loss.item(),
                    # 'train/reduced_loss': reduced_loss,
                    'train/loss': reduced_loss,
                    # 'train/log_loss': torch.log(loss).item(),
                    # 'train/log_reduced_loss': np.log(reduced_loss),
                    'train/log_loss': np.log(reduced_loss),
                    # 'feature/audio': wandb.Audio(features['audio'][0].cpu(), sample_rate=self.params.sample_rate)
                }, step=n_iter)

            # save checkpoint
            # if n_iter > 0 and n_iter % iters_per_ckpt == 0 and rank == 0:
            if n_iter % iters_per_ckpt == 0 and rank == 0:
                checkpoint_name = '{}.pkl'.format(n_iter)
                torch.save({'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                           os.path.join(checkpoint_directory, checkpoint_name))
                print('model at iteration %s is saved' % n_iter)

                # Generate samples
                if model_config["unconditional"]:
                    mel_path = None
                    mel_name = None
                else:
                    # mel_path="mel_spectrogram"
                    mel_name="LJ001-0001"
                samples = generate(
                    n_samples, n_iter,
                    diffusion_config, model_config, dataset_config,
                    mel_path=mel_path,
                    mel_name=mel_name,
                )
                samples = [wandb.Audio(sample.squeeze().cpu(), sample_rate=dataset_config['sampling_rate']) for sample in samples]
                wandb.log(
                    {'inference/audio': samples},
                    step=n_iter,
                    # commit=False,
                )

            n_iter += 1
        if rank == 0:
            epoch_loss /= len(trainloader)
            wandb.log({'train/loss_epoch': epoch_loss, 'train/log_loss_epoch': np.log(epoch_loss)}, step=n_iter)

    # Close TensorBoard.
    if rank == 0:
        # tb.close()
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config.json',
                        help='JSON file for configuration')
    parser.add_argument('-r', '--rank', type=int, default=0,
                        help='rank of process for distributed')
    parser.add_argument('-g', '--group_name', type=str, default='',
                        help='name of group for distributed')
    parser.add_argument('-w', '--wandb_id', type=str, default='')
    parser.add_argument('-m', '--mel_path', type=str, default='')
    args = parser.parse_args()

    # Parse configs. Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    train_config            = config["train_config"]        # training parameters
    # global dist_config
    dist_config             = config["dist_config"]         # to initialize distributed training
    # global model_config
    model_config          = config["model_config"]      # to define wavenet
    # global diffusion_config
    diffusion_config        = config["diffusion_config"]    # basic hyperparameters
    # global dataset_config
    dataset_config         = config["dataset_config"]     # to load trainset
    # global diffusion_hyperparams
    # diffusion_hyperparams   = calc_diffusion_hyperparams(**diffusion_config)  # dictionary of all diffusion hyperparameters

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        if args.group_name == '':
            print("WARNING: Multiple GPUs detected but no distributed group set")
            print("Only running 1 GPU. Use distributed.py for multiple GPUs")
            num_gpus = 1

    if num_gpus == 1 and args.rank != 0:
        raise Exception("Doing single GPU training on rank > 0")

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    train(num_gpus, args.rank, args.group_name, args.wandb_id, diffusion_config, model_config, dataset_config, mel_path=args.mel_path, **train_config)
