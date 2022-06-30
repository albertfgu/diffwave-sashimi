# *****************************************************************************
# Adapted from https://github.com/NVIDIA/waveglow/blob/master/distributed.py
# *****************************************************************************

# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************\

import os
import sys
import time
import subprocess
import argparse
import json
import warnings
from functools import partial
import multiprocessing as mp
warnings.filterwarnings("ignore")

import torch
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

from train import train

# from distributed_util import *


def main(config, stdout_dir, args_str, name, wandb_id, mel_path, diffusion_config, model_config, dataset_config, dist_config, train_config):
    args_list = ['train.py']
    args_list += args_str.split(' ') if len(args_str) > 0 else []

    args_list.append('--config={}'.format(config))

    num_gpus = torch.cuda.device_count()
    args_list.append('--num_gpus={}'.format(num_gpus))
    args_list.append("--group_name=group_{}".format(time.strftime("%Y_%m_%d-%H%M%S")))
    args_list.append(f'--name={name}')
    args_list.append(f'--wandb_id={wandb_id}')
    args_list.append(f'--mel_path={mel_path}')

    if not os.path.isdir(stdout_dir):
        os.makedirs(stdout_dir)
        os.chmod(stdout_dir, 0o775)

    # workers = []

    # for i in range(num_gpus):
    #     args_list[2] = '--rank={}'.format(i) # Overwrite num_gpus
    #     stdout = None if i == 0 else open(
    #         os.path.join(stdout_dir, "GPU_{}.log".format(i)), "w")
    #     print(args_list)
    #     p = subprocess.Popen([str(sys.executable)]+args_list, stdout=stdout)
    #     workers.append(p)

    train_fn = partial(
        train,
        num_gpus=num_gpus,
        group_name=time.strftime("%Y%m%d-%H%M%S"),
        wandb_id=wandb_id,
        diffusion_config=diffusion_config,
        model_config=model_config,
        dataset_config=dataset_config,
        dist_config=dist_config,
        train_config=train_config,
        name=name,
        mel_path=mel_path,
        **train_config,
    )
    mp.set_start_method("spawn")
    processes = []
    for i in range(num_gpus):
        p = mp.Process(target=train_fn, args=(i,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    # with Pool(num_gpus) as pool:
    #     pool.map(train_fn, list(range(num_gpus)))
        # train(num_gpus, i, time.strftime("%Y%m%d-%H%M%S"), wandb_id, diffusion_config, model_config, dataset_config, dist_config, train_config, name=name, mel_path=mel_path, **train_config)

    # for p in workers:
    #     p.wait()


@hydra.main(version_base=None, config_path="", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)  # Allow writing keys

# if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-c', '--config', type=str,
    #                     help='JSON file for configuration')
    # parser.add_argument('-s', '--stdout_dir', type=str, default="exp/",
    #                     help='directory to save stdout logs')
    # parser.add_argument('-a', '--args_str', type=str, default='',
    #                     help='double quoted string with space separated key value pairs')
    # parser.add_argument('-w', '--wandb_id', type=str, default='')
    # parser.add_argument('-m', '--mel_path', type=str, default='')
    # parser.add_argument('-n', '--name', type=str, default='')

    # args = parser.parse_args()

    # Parse configs
    # with open(args.config) as f:
    #     data = f.read()
    # config = json.loads(data)
    # train_config            = config["train_config"]        # training parameters
    # dist_config             = config["dist_config"]         # to initialize distributed training
    # model_config          = config["model_config"]      # to define wavenet
    # diffusion_config        = config["diffusion_config"]    # basic hyperparameters
    # dataset_config         = config["dataset_config"]     # to load trainset
    # diffusion_hyperparams   = calc_diffusion_hyperparams(**diffusion_config)  # dictionary of all diffusion hyperparameters

    # main(args.config, args.stdout_dir, args.args_str, args.name, args.wandb_id, args.mel_path, diffusion_config, model_config, dataset_config, dist_config, train_config)

    if cfg.wandb is not None:
        wandb_cfg = cfg.pop("wandb")
        wandb.init(
            **wandb_cfg, config=OmegaConf.to_container(cfg, resolve=True)
        )

    if not os.path.isdir("exp/"):
        os.makedirs("exp/")
        os.chmod("exp/", 0o775)

    num_gpus = torch.cuda.device_count()
    train_fn = partial(
        train,
        num_gpus=num_gpus,
        group_name=time.strftime("%Y%m%d-%H%M%S"),
        # wandb_id=wandb_id,
        diffusion_config=cfg.diffusion_config,
        model_config=cfg.model_config,
        dataset_config=cfg.dataset_config,
        dist_config=cfg.dist_config,
        # train_config=train_config,
        # name=name,
        # mel_path=mel_path,
        **cfg.train_config,
    )

    if num_gpus <= 1:
        generate_fn(0)
    else:
        mp.set_start_method("spawn")
        processes = []
        for i in range(num_gpus):
            p = mp.Process(target=train_fn, args=(i,))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

if __name__ == "__main__":
    main()
