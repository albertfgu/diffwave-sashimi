This is a fork of philsyn/diffwave, a reimplementation of the unconditional waveform synthesizer in [DIFFWAVE: A VERSATILE DIFFUSION MODEL FOR AUDIO SYNTHESIS](https://arxiv.org/pdf/2009.09761.pdf).
This repository contains code to reproduce the SaShiMi+DiffWave experiments from []().

## Overview

This repository is a supplement to the main [S4 repository]() which contains the official and up-to-date code for S4 and SaShiMi.
The code here is *research code* - it was not originally planned to be released because it was forked from an external codebase instead of being incorporated into the main S4 codebase.
However, we have decided to release the existing implementation to improve reproducibility for downstream researchers.

Compared to the parent fork, this repository has
- Significantly improved infrastructure and documentation
- Incorporation of the standalone [S4 module]() to create the SaShiMi DiffWave model (as well as the original DiffWave model which uses a WaveNet backbone)
- Checkpoints for the main SaShiMi DiffWave models described in the paper

## Usage

A basic experiment can be run with by passing in a config in the form of a JSON file.

## Data

Unconditional generation uses the SC09 dataset by default, while vocoding uses [LJSpeech](https://keithito.com/LJ-Speech-Dataset/).
For SC09, extract the Speech Commands dataset and move the digits subclasses into a separate folder, e.g. `./data/sc09/{zero,one,two,three,four,five,six,seven,eight,nine}`.
For LJSpeech, after unzipping, the waveforms should be organized in the folder `./data/ljspeech/LJSpeech-1.1/wavs`.

The entry `dataset_config.data_path` of the config should point to the desired folder.

## Models

All models use the DiffWave diffusion model with either a WaveNet or SaShiMi backbone.
The model configuration is controlled by the dictionary `model_config` inside the config file.

### Diffusion

The flags `in_channels` and `out_channels` control the data dimension.
The three flags `diffusion_step_embed_dim_{in,mid,out}` control DiffWave parameters for the diffusion step embedding.
The flag `unconditional` determines whether to do unconditional or conditional generation (using dataset SC09 and LibreSpeech respectively).
These flags are common to both embeddings.

### WaveNet

The WaveNet backbone is toggled by setting `sashimi=false`.
The parameters `res_channels`, `skip_channels`, `num_res_layers`, and `dilation_cycle` control the WaveNet backbone.
The below SaShiMi parameters will then be ignored.

### SaShiMi

The SaShiMi backbone is toggled by setting `sashimi=true`.
The parameters are:
```
unet: If true, use S4 layers in both the downsample and upsample parts of the backbone. If false, use S4 layers in only the upsample part.
d_model: Starting model dimension of outermost stage
n_layers: Number of layers per stage
pool: List of pooling factors per stage (e.g. [4, 4] means three total stages, pooling by a factor of 4 in between each)
expand: Multiplicative factor to increase model dimension between stages
ff: Width of inner layer of MLP (i.e. MLP block has dimensions d_model -> d_model*ff -> d_model)
```
The above WaveNet parameters will be ignored.

## Training

Experiments are saved under `exp/<run>` with an automatically generated run name identifying the experiment (model and setting), e.g. `exp/unet_d64_n6_pool_2_expand2_ff2_T200_betaT0.02_uncond`.
Checkpoints are inside `exp/<run>/checkpoint` and generated audio samples in `exp/<name>/waveforms`.

## Logging
Set `train_config.wandb_mode=online` to turn on WandB logging, or `train_config.wandb_mode=disabled` to turn it off.
You may want to change the project name on line []

## Resuming

To resume from the checkpoint `exp/<run>/checkpoint/1000.pkl`, simply set `train_config.ckpt_iter=1000` in the config and re-run the same command, `python distributed_train.py -c config.json`.
`train_config.ckpt_iter=max` resumes from the last checkpoint, and `train_config.ckpt_iter=-1` trains from scratch.

Use `-w <id>` to resume logging to a wandb run ID

## Generating

After training,
```
python distributed_inference.py -c generate.json -n 2048 -b 128 --ckpt_iter 500000
```
to generate 2048 samples (total) at a batch size of 128 per GPU from the model specified in the config, which will be at `exp/{model}/waveforms/500000`.
Generated samples will be stored in `exp/<run>/waveforms/<ckpt_iter>/`

## Vocoding

## Usage:

- To continue training the model, run ```python distributed_train.py -c config_vocoder.json```

- To retrain the model, change the parameter ```ckpt_iter``` in the corresponding ```json``` file to ```-1``` and use the above command.

- Note, you may need to carefully adjust some parameters in the ```json``` file, such as ```data_path``` and ```batch_size_per_gpu```.


### Unconditional generation
- To generate audio, run ```python inference.py -c config.json -n 16``` to generate 16 utterances.

### Conditional generation
- To generate audio, run ```python inference.py -c config_${channel}.json -cond ${conditioner_name}```. For example, if the name of the mel spectrogram is ```LJ001-0001.wav.pt```, then ```${conditioner_name}``` is ```LJ001-0001```. Provided mel spectrograms include ```LJ001-0001``` through ```LJ001-0186```.


## Pretrained models and generated samples:

### Unconditional
- [model](https://github.com/philsyn/DiffWave-unconditional/tree/master/exp/ch256_T200_betaT0.02/logs/checkpoint)
- [samples](https://github.com/philsyn/DiffWave-unconditional/tree/master/exp/ch256_T200_betaT0.02/speeches)

config_sashimi_large.json
Note that the learning rate in this model is 1e-4 because it was decayed by 0.5 at step 500k (see note in paper)

### Conditional
- [channel=64 model](https://github.com/philsyn/DiffWave-Vocoder/tree/master/exp/ch64_T50_betaT0.05/logs/checkpoint)
- [channel=64 samples](https://github.com/philsyn/DiffWave-Vocoder/tree/master/exp/ch64_T50_betaT0.05/speeches)
- [channel=128 model](https://github.com/philsyn/DiffWave-Vocoder/tree/master/exp/ch128_T200_betaT0.02/logs/checkpoint)
- [channel=128 samples](https://github.com/philsyn/DiffWave-Vocoder/tree/master/exp/ch128_T200_betaT0.02/speeches)


# Notes

- create spectrograms: `python mel2samp.py -f ../data/ljspeech/LJSpeech-1.1/wavs -c config_vocoder.json -o mel1024` to put which creates them based on the dataset_config in the folder mel1024 (my convention is to store the hop length or other relevant information in the folder name)

(Note that these spectrograms are only used for inference/generation; with a small tweak it should be possible to generate them on the fly to avoid this hassle)

