This is a reimplementaion of the neural vocoder in [DIFFWAVE: A VERSATILE DIFFUSION MODEL FOR AUDIO SYNTHESIS](https://arxiv.org/pdf/2009.09761.pdf).

## Usage: 

- To continue training the model, run ```python distributed_train.py -c config_${channel}.json```, where ```${channel}``` can be either ```64``` or ```128```. 

- To retrain the model, change the parameter ```ckpt_iter``` in the corresponding ```json``` file to ```-1``` and use the above command.

- To generate audio, run ```python inference.py -c config_${channel}.json -cond ${conditioner_name}```. For example, if the name of the mel spectrogram is ```LJ001-0001.wav.pt```, then ```${conditioner_name}``` is ```LJ001-0001```. Provided mel spectrograms include ```LJ001-0001``` through ```LJ001-0186```.


- Note, you may need to carefully adjust some parameters in the ```json``` file, such as ```data_path``` and ```batch_size_per_gpu```.
