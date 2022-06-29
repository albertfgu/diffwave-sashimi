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
import warnings
import json
from functools import partial
import multiprocessing as mp
warnings.filterwarnings("ignore")

import torch
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

from inference import generate
from distributed_util import *


def main(config, ckpt_iter, ckpt_smooth, num_samples, batch_size, name, model_config, diffusion_config, dataset_config):
    num_gpus = torch.cuda.device_count()
    # args_list = ["CUDA_VISIBLE_DEVICES", str(sys.executable), 'inference.py']
    # args_list = [str(sys.executable), 'inference.py']

    # args_list.append('--config={}'.format(config))
    # args_list.append('--ckpt_iter={}'.format(ckpt_iter))
    # args_list.append('--ckpt_smooth={}'.format(ckpt_smooth))
    # assert num_samples % num_gpus == 0
    # args_list.append('--num_samples={}'.format(num_samples // num_gpus))
    # args_list.append('--batch_size={}'.format(batch_size))

    # args_list.append('--num_gpus={}'.format(num_gpus))
    # # args_list.append("--group_name=group_{}".format(time.strftime("%Y_%m_%d-%H%M%S")))
    # # args_list.append(f'--wandb_id={wandb_id}')

    # # if not os.path.isdir(stdout_dir):
    # #     os.makedirs(stdout_dir)
    # #     os.chmod(stdout_dir, 0o775)

    # workers = []

    # for i in range(num_gpus):
    #     # args_list[0] = f"CUDA_VISIBLE_DEVICES={i%num_gpus}"
    #     args_list[-1] = '--rank={}'.format(i) # Overwrite num_gpus
    #     # stdout = None if i == 0 else open(
    #     #     os.path.join(stdout_dir, "GPU_{}.log".format(i)), "w")
    #     stdout = None
    #     print(args_list)
    #     p = subprocess.Popen(args_list, stdout=stdout)
    #     # p = subprocess.Popen([str(sys.executable)]+args_list)
    #     workers.append(p)

    # for p in workers:
    #     p.wait()



@hydra.main(version_base=None, config_path="", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)  # Allow writing keys

    # # Track with wandb
    # if wandb is not None:
    #     wandb_cfg = cfg.pop("wandb")
    #     wandb.init(
    #         **wandb_cfg, config=OmegaConf.to_container(cfg, resolve=True)
    #     )



    # parser = argparse.ArgumentParser()
    # parser.add_argument('-c', '--config', type=str,
    #                     help='JSON file for configuration')
    # parser.add_argument('-ckpt_iter', '--ckpt_iter', default='max',
    #                     help='Which checkpoint to use; assign a number or "max"')
    # parser.add_argument('-s', '--ckpt_smooth', default=-1, type=int,
    #                     help='Which checkpoint to start averaging from')
    # parser.add_argument('-n', '--n_samples', type=int, default=4,
    #                     help='Number of utterances to be generated')        
    # parser.add_argument('-b', '--batch_size', type=int, default=0,
    #                     help='Number of samples to generate at once per GPU')        
    # parser.add_argument('--name', type=str, default='',
    #                     help='Name of experiment (prefix of experiment directory)')
    # args = parser.parse_args()

    # Parse configs
    # with open(args.config) as f:
    #     data = f.read()
    # config = json.loads(data)
    # model_config          = config["model_config"]      # to define wavenet
    # diffusion_config        = config["diffusion_config"]    # basic hyperparameters
    # dataset_config         = config["dataset_config"]     # to read trainset configurations

    # main(args.config, args.ckpt_iter, args.ckpt_smooth, args.num_samples, args.batch_size, args.name, model_config, diffusion_config, dataset_config)

    # model_config          = config["model_config"]      # to define wavenet
    # diffusion_config        = config["diffusion_config"]    # basic hyperparameters
    # dataset_config         = config["dataset_config"]     # to read trainset configurations

    num_gpus = torch.cuda.device_count()
    generate_fn = partial(
        generate,
        # num_samples=args.n_samples//num_gpus, # Samples per GPU
        # n_samples=n_samples,
        # batch_size=args.batch_size,
        # ckpt_iter=args.ckpt_iter,
        # ckpt_smooth=args.ckpt_smooth,
        # name=args.name,
        diffusion_config=cfg.diffusion_config,
        model_config=cfg.model_config,
        dataset_config=cfg.dataset_config,
        **cfg.generate_config,
    )

    if num_gpus <= 1:
        generate_fn(0)
    else:
        mp.set_start_method("spawn")
        processes = []
        for i in range(num_gpus):
            p = mp.Process(target=generate_fn, args=(i,))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


if __name__ == "__main__":
    main()
