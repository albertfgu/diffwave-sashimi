This repository is an implementation of the waveform synthesizer in [DIFFWAVE: A VERSATILE DIFFUSION MODEL FOR AUDIO SYNTHESIS](https://arxiv.org/pdf/2009.09761.pdf).
It also has code to reproduce the SaShiMi+DiffWave experiments from [It’s Raw! Audio Generation with State-Space Models](https://arxiv.org/abs/2202.09729) (Goel et al. 2022).

This is a fork/combination of the implementations [philsyn/DiffWave-unconditional](https://github.com/philsyn/DiffWave-unconditional) and [philsyn/DiffWave-Vocoder](https://github.com/philsyn/DiffWave-Vocoder).
This repo uses Git LFS to store model checkpoints, which unfortunately [does not work with public forks](https://github.com/git-lfs/git-lfs/issues/1939).
For this reason it is not an official GitHub fork.

## Overview

This repository aims to provide a clean implementation of the DiffWave audio diffusion model.
The `checkpoints` branch of this repository has the original code used for reproducing experiments from the SaShiMi paper ([instructions](#pretrained-models)).
The `master` branch of this repository has the latest versions of the S4/SaShiMi model and can be used to train new models from scratch.


Compared to the parent fork for DiffWave, this repository has:
- Both unconditional (SC09) and vocoding (LJSpeech) waveform synthesis. It's also designed in a way to be easy to add new datasets
- Significantly improved infrastructure and documentation
- Configuration system with Hydra for modular configs and flexible command-line API
- Logging with WandB instead of Tensorboard, including automatically generating and uploading samples during training
- Vocoding does not require a separate pre-processing step to generate spectrograms, making it easier and less error-prone to use
- Option to replace WaveNet with the SaShiMi backbone (based on the [S4 layer](https://github.com/HazyResearch/state-spaces))
- Pretrained checkpoints and samples for both DiffWave (+Wavenet) and DiffWave+SaShiMi

These are some features that would be nice to add.
PRs are very welcome!
- Use the pip S4 package once it's released, instead of manually updating the standalone files
- Mixed-precision training
- Fast inference procedure from later versions of the DiffWave paper
- Can add an option to allow original Tensorboard logging instead of WandB (code is still there, just commented out)
- The different backbones (WaveNet and SaShiMi) can be consolidated more cleanly with the diffusion logic factored out

## ToC

- [Usage](#usage)
- [Data](#data)
- [Training](#training)
- [Vocoding](#vocoding)
- [Pretrained Models](#pretrained-models)
  - [DiffWave+SaShiMi](#sashimi-1)
  - [DiffWave](#wavenet-1)

## Usage

A basic experiment can be run with `python train.py`.
This default config is for SC09 unconditional generation with the SaShiMi backbone.

### Hydra

Configuration is managed by [Hydra](https://hydra.cc).
Config files are under `configs/`.
Examples of different configs and running experiments via command line are provided throughout this README.
Hydra has a steeper learning curve than standard `argparse`-based workflows, but offers much more flexibility and better experiment management. Feel free to file issues for help with configs.

### Multi-GPU training
By default, all available GPUs are used (according to [`torch.cuda.device_count()`](https://pytorch.org/docs/stable/cuda.html#torch.cuda.device_count)).
You can specify which GPUs to use by setting the [`CUDA_DEVICES_AVAILABLE`](https://developer.nvidia.com/blog/cuda-pro-tip-control-gpu-visibility-cuda_visible_devices/) environment variable before running the training module, or e.g. `CUDA_VISIBLE_DEVICES=0,1 python train.py`.


## Data

Unconditional generation uses the SC09 dataset by default, while vocoding uses [LJSpeech](https://keithito.com/LJ-Speech-Dataset/).
The entry `dataset_config.data_path` of the config should point to the desired folder, e.g. `data/sc09` or `data/LJSpeech-1.1/wavs`

### SC09
For SC09, extract the Speech Commands dataset and move the digits subclasses into a separate folder, e.g. `./data/sc09/{zero,one,two,three,four,five,six,seven,eight,nine}`

### LJSpeech

Download the LJSpeech dataset into a folder. For example (as of 2022/06/28):
```
mkdir data && cd data
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar xvf LJSpeech-1.1.tar.bz2
```
The waveforms should be organized in the folder `./data/LJSpeech-1.1/wavs`


## Models

All models use the DiffWave diffusion model with either a WaveNet or SaShiMi backbone.
The model configuration is controlled by the dictionary `model` inside the config file.

### Diffusion

The flags `in_channels` and `out_channels` control the data dimension.
The three flags `diffusion_step_embed_dim_{in,mid,out}` control DiffWave parameters for the diffusion step embedding.
The flag `unconditional` determines whether to do unconditional or conditional generation.
These flags are common to both backbones.

### WaveNet

The WaveNet backbone is used by setting `model._name_=wavenet`.
The parameters `res_channels`, `skip_channels`, `num_res_layers`, and `dilation_cycle` control the WaveNet backbone.

### SaShiMi

The SaShiMi backbone is used by setting `model._name_=sashimi`.
The parameters are:
```yaml
unet:     # If true, use S4 layers in both the downsample and upsample parts of the backbone. If false, use S4 layers in only the upsample part.
d_model:  # Starting model dimension of outermost stage
n_layers: # Number of layers per stage
pool:     # List of pooling factors per stage (e.g. [4, 4] means three total stages, pooling by a factor of 4 in between each)
expand:   # Multiplicative factor to increase model dimension between stages
ff:       # Feedforward expansion factor: MLP block has dimensions d_model -> d_model*ff -> d_model)
```

## Training

A basic experiment can be run with `python train.py`, which defaults to `python train.py experiment=sc09` (SC09 unconditional waveform synthesis).

Experiments are saved under `exp/<run>` with an automatically generated run name identifying the experiment (model and setting).
Checkpoints are saved to `exp/<run>/checkpoint/` and generated audio samples to `exp/<run>/waveforms/`.

### Logging
Set `wandb.mode=online` to turn on WandB logging, or `wandb.mode=disabled` to turn it off.
Standard wandb arguments such as entity and project are configurable.

### Resuming

To resume from the checkpoint `exp/<run>/checkpoint/1000.pkl`, simply re-run the same training command with the additional flag `train.ckpt_iter=1000`.
Passing in `train_config.ckpt_iter=max` resumes from the last checkpoint, and `train_config.ckpt_iter=-1` trains from scratch.

Use `wandb.id=<id>` to resume logging to a previous run.

### Generating

After training with `python train.py <flags>`, `python generate.py <flags>` generates samples according to the `generate` dictionary of the config.

For example,
```
python generate.py <flags> generate.ckpt_iter=500000 generate.n_samples=256 generate.batch_size=128
```
generates 256 samples per GPU, at a batch size of 128 per GPU, from the model specified in the config at checkpoint iteration 500000.

Generated samples will be stored in `exp/<run>/waveforms/<generate.ckpt_iter>/`

## Vocoding

- After downloading the data, make the config's `dataset.data_path` point to the `.wav` files
- Toggle `model.unconditional=false`
- Pass in the name of a `.wav` file for generation, e.g. `generate.mel_name=LJ001-0001`. Every checkpoint, vocoded samples for this audio file will be logged to wandb

Currently, vocoding is only set up for the LJSpeech dataset. See the config `configs/experiment/ljspeech.yaml` for details.
The following is an example command for LJSpeech vocoding with a smaller SaShiMi model. A checkpoint for this model at 200k steps is also provided.
```
python train.py experiment=ljspeech model=sashimi model.d_model=32 wandb.mode=online
```

Another example with a smaller WaveNet backbone, similar to the results from the DiffWave paper:
```
python train.py experiment=ljspeech model=wavenet model.res_channels=64 model.skip_channels=64 wandb.mode=online
```

Generation can be done in the usual way, conditioning on any spectrogram, e.g.
```
python generate.py experiment=ljspeech model=sashimi model.d_model=32 generate.mel_name=LJ001-0002
```

### Pre-processed Spectrograms

Other DiffWave vocoder implementations such as https://github.com/philsyn/DiffWave-Vocoder and https://github.com/lmnt-com/diffwave require first generating spectrograms in a separate pre-processing step.
This implementation does not require this step, which we find more convenient.
However, pre-processing and saving the spectrograms is still possible.

To generate a folder of spectrograms according to the `dataset` config, run the `mel2samp` script and specify an output directory (e.g. here 256 refers to the hop size):
```
python -m dataloaders.mel2samp experiment=ljspeech +output_dir=mel256
```

Then during training or generation, add in the additional flag `generate.mel_path=mel256` to use the pre-processed spectrograms, e.g.
```
python generate.py experiment=ljspeech model=sashimi model.d_model=32 generate.mel_name=LJ001-0002 generate.mel_path=mel256
```


# Pretrained Models

The remainder of this README pertains only to pre-trained models from the SaShiMi paper.

The branch `git checkout checkpoints` provides checkpoints for these models.

**This branch is meant only for reproducing generated samples from the SaShiMi paper from ICML 2022 - please do not attempt train-from-scratch results from this code.**
The older models in this branch have issues that are explained below.

Training from scratch is covered in the previous part of this README and should be done from the `master` branch.

### Checkpoints

Install [Git LFS](https://git-lfs.github.com/) and `git lfs pull` to download the checkpoints.

### Samples
For each of the provided checkpoints, 16 audio samples are provided.

More pre-generated samples for all models from the SaShiMi paper can be downloaded from: https://huggingface.co/krandiash/sashimi-release/tree/main/samples/sc09

The below four models correspond to "sashimi-diffwave", "sashimi-diffwave-small", "diffwave", and "diffwave-small" respectively.
Command lines are also provided to reproduce these samples (up to random seed).


## SaShiMi

The version of S4 used in the experiments in the SaShiMi paper is an outdated version of S4 from January 2022 that predates V2 (February 2022) of the [S4 repository](https://github.com/HazyResearch/state-spaces). S4 is currently on V3 as of July 2022.

### DiffWave+SaShiMi

- Experiment folder: `exp/unet_d128_n6_pool_2_expand2_ff2_T200_betaT0.02_uncond/`
- Train from scratch: `python train.py model=sashimi train.ckpt_iter=-1`
- Resume training: `python train.py model=sashimi train.ckpt_iter=max train.learning_rate=1e-4`
(as described in the paper, this model used a manual learning rate decay after 500k steps)

Generation examples:
- `python generate.py experiment=sc09 model=sashimi` (Latest model at 800k steps)
- `python generate.py experiment=sc09 model=sashimi generate.ckpt_iter=500000` (Earlier model at 500k steps)
- `python generate.py generate.n_samples=256 generate.batch_size=128` (Generate 256 samples per GPU with the largest batch that fits on an A100. The paper uses this command to generate 2048 samples on an 8xA100 machine for evaluation metrics.)

### DiffWave+SaShiMi small

Experiment folder: `exp/unet_d64_n6_pool_2_expand2_ff2_T200_betaT0.02_uncond/`

Train (since the model is smaller, you can increase the batch size and logged samples per checkpoint):
```
python train.py experiment=sc09 model=sashimi model.d_model=64 train.batch_size_per_gpu=4 generate.n_samples=32
```

Generate:
```
python generate.py experiment=sc09 model=sashimi model.d_model=64 generate.n_samples=256 generate.batch_size=256
```

## WaveNet

The WaveNet backbone provided in the parent fork had a [small](https://github.com/albertfgu/diffwave-sashimi/blob/checkpoints/models/wavenet.py#L92) [bug](https://github.com/albertfgu/diffwave-sashimi/blob/checkpoints/models/wavenet.py#L163) where it used `x += y` instead of `x = x + y`.
This can cause a difficult-to-trace error in some PyTorch + environment combinations (but sometimes it works; I never figured out when it's ok).
These two lines are fixed in the master branch of this repo.

However, for some reason when models are *trained using the wrong code* and *loaded using the correct code*,
the model runs fine but produces inconsistent outputs, even in inference mode (i.e. generation produces static noise).
So this branch for reproducing the checkpoints uses the incorrect version of these two lines.
This allows generating from the pre-trained models, but may not train in some environments.
**If anyone knows why this happens, I would love to know! Shoot me an email or file an issue!**


### DiffWave(+WaveNet)
- Experiment folder: `exp/wnet_h256_d36_T200_betaT0.02_uncond/`
- Usage: `python <train|generate>.py model=wavenet`

More notes:
- The fully trained model (1000000 steps) is the original checkpoint from the original repo philsyn/DiffWave-unconditional
- The checkpoint at 500000 steps is our version trained from scratch
- These should both be compatible with this codebase (e.g. generation works with both), but for some reason the original `checkpoint/1000000.pkl` file is much smaller than our `checkpoint/500000.pkl`
- I don't remember if I changed anything in the code to cause this; perhaps it could also be differences in PyTorch or versions or environments?

### DiffWave(+WaveNet) small
Experiment folder: `exp/wnet_h128_d30_T200_betaT0.02_uncond/`

Train:
```
python train.py model=wavenet model.res_channels=128 model.num_res_layers=30 model.dilation_cycle=10 train.batch_size_per_gpu=4 generate.n_samples=32
```
A shorthand model config is also defined
```
python train.py model=wavenet_small train.batch_size_per_gpu=4 generate.n_samples=32
```


## Vocoders
The parent fork has a few pretrained LJSpeech vocoder models. Because of the WaveNet bug, we recommend not using these and simply training from scratch from the `master` branch; these vocoder models are small and faster to train than the unconditional SC09 models.
Feel free to file an issue for help with configs.
<!--
- [channel=64 model](https://github.com/philsyn/DiffWave-Vocoder/tree/master/exp/ch64_T50_betaT0.05/logs/checkpoint)
- [channel=64 samples](https://github.com/philsyn/DiffWave-Vocoder/tree/master/exp/ch64_T50_betaT0.05/speeches)
- [channel=128 model](https://github.com/philsyn/DiffWave-Vocoder/tree/master/exp/ch128_T200_betaT0.02/logs/checkpoint)
- [channel=128 samples](https://github.com/philsyn/DiffWave-Vocoder/tree/master/exp/ch128_T200_betaT0.02/speeches)
-->

