This is a fork of philsyn/diffwave, a reimplementation of the unconditional waveform synthesizer in [DIFFWAVE: A VERSATILE DIFFUSION MODEL FOR AUDIO SYNTHESIS](https://arxiv.org/pdf/2009.09761.pdf).
This repository contains code to reproduce the SaShiMi+DiffWave experiments from [It’s Raw! Audio Generation with State-Space Models](https://arxiv.org/abs/2202.09729) (Goel et al. 2022).

## Overview

This repository is a supplement to the main [S4 repository]() which contains the official and up-to-date code for S4 and SaShiMi.
The code here is *research code* - it was not originally planned to be released because it was forked from an external codebase instead of being incorporated into the main S4 codebase.
However, we have decided to release the implementation used to produce results in the SaShiMi paper to improve reproducibility for downstream researchers.
Instructions for generating samples using checkpoints are in [#pretrained-models].

Working examples of how to train models from scratch with the latest version of S4/SaShiMi are also provided, but currently untested.

Compared to the parent fork for DiffWave, this repository has:
- Both unconditional (SC09) and vocoding (LJSpeech) waveform synthesis. It's also designed in a way to be easy to add new datasets
- Significantly improved infrastructure and documentation
- Configuration system with Hydra for modular configs and flexible command-line API
- Logging with WandB instead of Tensorboard, including automatically generating and uploading samples during training
- Option to replace WaveNet with the [SaShiMi backbone]() (based on the [S4 layer]())
- Pretrained checkpoints and samples for both DiffWave (+Wavenet) and DiffWave+SaShiMi

These are some features that would be nice to add:
- Incorporate latest S4/SaShiMi standalone file; currently this reimplements the architecture using a model predating V2 of the S4 standalone [#sashimi]. Would be even better to use the pip S4 package once it's released
- Mixed-precision training
- Fast inference procedure from later versions of the DiffWave paper
- Generate spectrograms on the fly based on the config instead of requiring a [separate preprocessing step](#vocoding)
- Can add an option to allow original Tensorboard logging instead of WandB (code is still there, just commented out)
- The different backbones (WaveNet and SaShiMi) can be consolidated more cleanly with the diffusion portions factored out

PRs are very welcome!

## Usage

A basic experiment can be run with `python train.py`.
This default config is for SC09 unconditional generation.

Configuration is managed by [Hydra](https://hydra.cc).
Config files are under `configs/`.
Examples of different configs and configuring via command line are provided throughout this README.

### Multi-GPU training
By default, all available GPUs are used (according to [`torch.cuda.device_count()`](https://pytorch.org/docs/stable/cuda.html#torch.cuda.device_count)).
You can specify which GPUs to use by setting the [`CUDA_DEVICES_AVAILABLE`](https://developer.nvidia.com/blog/cuda-pro-tip-control-gpu-visibility-cuda_visible_devices/) environment variable before running the training module, or `CUDA_VISIBLE_DEVICES=0,1 python train.py`.


## Data

Unconditional generation uses the SC09 dataset by default, while vocoding uses [LJSpeech](https://keithito.com/LJ-Speech-Dataset/).
The entry `dataset_config.data_path` of the config should point to the desired folder, e.g. `data/sc09` or `data/LJSpeech-1.1/wavs`.

### SC09
For SC09, extract the Speech Commands dataset and move the digits subclasses into a separate folder, e.g. `./data/sc09/{zero,one,two,three,four,five,six,seven,eight,nine}`.

### LJSpeech

Download the LJSpeech dataset into a folder. For example (as of 2022/06/28):
```
mkdir data && cd data
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar xvf LJSpeech-1.1.tar.bz2
```
The waveforms should be organized in the folder `./data/LJSpeech-1.1/wavs`.

Follow instructions in [#vocoding] to generate spectrograms before training.


## Models

All models use the DiffWave diffusion model with either a WaveNet or SaShiMi backbone.
The model configuration is controlled by the dictionary `model` inside the config file.

### Diffusion

The flags `in_channels` and `out_channels` control the data dimension.
The three flags `diffusion_step_embed_dim_{in,mid,out}` control DiffWave parameters for the diffusion step embedding.
The flag `unconditional` determines whether to do unconditional or conditional generation (using dataset SC09 and LibreSpeech respectively).
These flags are common to both backbones.

### WaveNet

The WaveNet backbone is used by setting `model.backbone=wavenet`.
The parameters `res_channels`, `skip_channels`, `num_res_layers`, and `dilation_cycle` control the WaveNet backbone.

### SaShiMi

The SaShiMi backbone is used by setting `model.backbone=sashimi`.
The parameters are:
```
unet: If true, use S4 layers in both the downsample and upsample parts of the backbone. If false, use S4 layers in only the upsample part.
d_model: Starting model dimension of outermost stage
n_layers: Number of layers per stage
pool: List of pooling factors per stage (e.g. [4, 4] means three total stages, pooling by a factor of 4 in between each)
expand: Multiplicative factor to increase model dimension between stages
ff: Width of inner layer of MLP (i.e. MLP block has dimensions d_model -> d_model*ff -> d_model)
```

## Training

Experiments are saved under `exp/<run>` with an automatically generated run name identifying the experiment (model and setting).
Checkpoints are inside `exp/<run>/checkpoint` and generated audio samples in `exp/<name>/waveforms`.

## Logging
Set `wandb.mode=online` to turn on WandB logging, or `wandb.mode=disabled` to turn it off.
Standard wandb arguments such as entity and project are configurable.

## Resuming

To resume from the checkpoint `exp/<run>/checkpoint/1000.pkl`, simply re-run the same training command with `train.ckpt_iter=1000` .
`train_config.ckpt_iter=max` resumes from the last checkpoint, and `train_config.ckpt_iter=-1` trains from scratch.

Use `wandb.id=<id>` to resume logging to a previous run.

## Generating

After training,
```
python generate.py generate.ckpt_iter=500000 generate.n_samples=2048 generate.batch_size=128
```
generates 2048 total samples at a batch size of 128 per GPU from the model specified in the config at checkpoint iteration 500000.
Generated samples will be stored in `exp/<run>/waveforms/<ckpt_iter>/`

## Vocoding

First create spectrograms to condition on: `python mel2samp.py -f ../data/ljspeech/LJSpeech-1.1/wavs -c config_vocoder.json -o mel256` to put which creates spectrograms based on the arguments in the `dataset_config`. 
Some notes:
- The hop size affects how much upsampling is needed in the diffusion upsampling. The parameter `mel_upsample` controls this; the product of the upsample sizes should equal the hop size.
- Note that these spectrograms are only used for inference/generation. With a small tweak it should be possible to generate them on the fly to avoid this hassle of creating spectrograms separately

Run the same training script, with the additional command argument pointing to the spectrogramfolder, e.g. `python distributed_train.py -c config_vocoder_baseline.json --mel_path mel256`

### Pre-processed Spectrograms

To pre-generate a folder of spectrograms corresponding to an experiment, run the same config and specify an output directory:
```python -m dataloaders.mel2samp experiment=ljspeech +output_dir=mel256```
(Here 256 refers to the hop size, but you can use any folder name.)

Then during training, you can pass in this directory and a specific file name to only log samples for that spectrogram:
```python -m train experiment=ljspeech generate.mel_path=mel256 generate.mel_name=LJ001-0001```

### Conditional generation
- To generate audio, run ```python inference.py -c ${config}.json -cond ${conditioner_name}```. For example, if the name of the mel spectrogram is ```LJ001-0001.wav.pt```, then ```${conditioner_name}``` is ```LJ001-0001```. Provided mel spectrograms include ```LJ001-0001``` through ```LJ001-0186```.


# Pretrained Models

The branch `git checkout checkpoints` is provided for the code used in the checkpoints.
**This branch is meant only for reproducing generated samples from the ICML 2022 paper - please do not attempt train-from-scratch results from this code.**
Reasons are explained below.

### Checkpoints

Install [Git LFS](https://git-lfs.github.com/) and `git lfs pull` to download the checkpoints.

### Samples
Pre-generated samples for all models from the SaShiMi paper can be downloaded from: https://huggingface.co/krandiash/sashimi-release/tree/main/samples/sc09

The below four models correspond to "sashimi-diffwave", "sashimi-diffwave-small", "diffwave", and "diffwave-small" respectively.

Command lines are also provided to reproduce these samples (up to random seed).


## SaShiMi

The version of S4 used in these experiments is an outdated version of S4 that predates V2 (February 2022) of the [S4 repository](https://github.com/HazyResearch/state-spaces) (currently on V3 as of July 2022).

### SaShiMi+DiffWave

Experiment folder: `exp/unet_d128_n6_pool_2_expand2_ff2_T200_betaT0.02_uncond/`
Train from scratch: `python train.py model=sashimi train.ckpt_iter=-1`
Resume training: `python train.py model=sashimi train.ckpt_iter=max train.learning_rate=1e-4`
(as described in the paper, this model used a manual learning rate decay after 500k steps)

Generation examples:
`python generate.py model=sashimi` (Best model)
`python generate.py model=sashimi generate.ckpt_iter=500000` (Earlier model)

`python generate.py generate.n_samples=256 generate.batch_size=128` (Generate 2048 total samples with largest batch that fits on an A100; used for evaluation metrics in the paper)

### SaShiMi+DiffWave small
Experiment folder: `exp/unet_d64_n6_pool_2_expand2_ff2_T200_betaT0.02_uncond/`
Train: `python train.py model=sashimi model.d_model=64 train.batch_size_per_gpu=4 train.n_samples=32` (since generation is faster, you can increase the logged samples per epoch)
Generate: `python generate.py model=sashimi model.d_model=64 generate.n_samples=256 generate.batch_size=256`

## WaveNet

The WaveNet backbone provided in the parent fork had a [small]() [bug]() where it used `x += y` instead of `x = x + y`.
This can cause a difficult-to-trace error in some PyTorch + environment combinations (but sometimes it works; I never figured out when it's ok).
These two lines are fixed in the main branch of this repo.

However, for some reason when models are *trained using the wrong version* and *loaded using the correct version*,
the model runs fine but produces inconsistent outputs, even in inference mode (i.e. generation produces static noise).
So this branch for reproducing the checkpoints uses the incorrect version of these two lines.
This allows generating from the model, but may not train in some environments.
**If anyone knows why this happens, I would love to know! Shoot me an email or file an Issue!**


### (WaveNet)+DiffWave
Experiment folder: `exp/wnet_h256_d36_T200_betaT0.02_uncond/`
Usage: `python <train|generate>.py model=wavenet`

More Notes:
The fully trained model (1000000 steps) is the original checkpoint from the original repo philsyn/DiffWave-unconditional
The checkpoint at 500000 steps is our version trained from scratch.
These should both be compatible with this codebase (e.g. generation works with both), but for some reason the original `checkpoint/1000000.pkl` file is much smaller than our `checkpoint/500000.pkl`.
I don't remember if I changed anything in the code to cause this; perhaps it could also be differences in PyTorch or versions or environments?

### (WaveNet)+DiffWave small
Experiment folder: `exp/wnet_h128_d30_T200_betaT0.02_uncond/`
Train: `python train.py model=wavenet model.res_channels=128 model.num_res_layers=30 model.dilation_cycle=10 train.batch_size_per_gpu=4 train.n_samples=32`
A shorthand model config is also defined:
`python train.py model=wavenet_small train.batch_size_per_gpu=4 train.n_samples=32`


### Conditional
- [channel=64 model](https://github.com/philsyn/DiffWave-Vocoder/tree/master/exp/ch64_T50_betaT0.05/logs/checkpoint)
- [channel=64 samples](https://github.com/philsyn/DiffWave-Vocoder/tree/master/exp/ch64_T50_betaT0.05/speeches)
- [channel=128 model](https://github.com/philsyn/DiffWave-Vocoder/tree/master/exp/ch128_T200_betaT0.02/logs/checkpoint)
- [channel=128 samples](https://github.com/philsyn/DiffWave-Vocoder/tree/master/exp/ch128_T200_betaT0.02/speeches)


# Notes

