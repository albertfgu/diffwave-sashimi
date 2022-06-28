import os
import numpy as np
import torch
from tqdm import tqdm

def flatten(v):
    """
    Flatten a list of lists/tuples
    """

    return [x for y in v for x in y]


def rescale(x):
    """
    Rescale a tensor to 0-1
    """

    return (x - x.min()) / (x.max() - x.min())


def find_max_epoch(path):
    """
    Find maximum epoch/iteration in path, formatted ${n_iter}.pkl
    E.g. 100000.pkl

    Parameters:
    path (str): checkpoint path

    Returns:
    maximum iteration, -1 if there is no (valid) checkpoint
    """

    files = os.listdir(path)
    epoch = -1
    for f in files:
        if len(f) <= 4:
            continue
        if f[-4:]  == '.pkl':
            try:
                epoch = max(epoch, int(f[:-4]))
            except:
                continue
    return epoch

def smooth_ckpt(path, min_ckpt, max_ckpt, alpha=None):
    print(f"finding checkpoints in ({min_ckpt}, {max_ckpt}] in {path}")
    files = os.listdir(path)
    ckpts = []
    for f in files:
        if len(f) <= 4:
            continue
        if f[-4:]  == '.pkl':
            print(f)
            try:
                it = int(f[:-4])
                if min_ckpt < it and it <= max_ckpt:
                    ckpts.append(it)
            except:
                continue
    ckpts = sorted(ckpts)
    print("found ckpts", ckpts)
    state_dict = None
    for n, it in enumerate(ckpts):
        model_path = os.path.join(path, '{}.pkl'.format(it))
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            # net.load_state_dict(checkpoint['model_state_dict'])
            state_dict = smooth_dict(state_dict, checkpoint['model_state_dict'], n, alpha=alpha)
            print('Successfully loaded model at iteration {}'.format(it))
        except:
            raise Exception(f'No valid model found at iteration {it}, path {model_path}')
    return state_dict

def smooth_dict(d, d0, n=None, alpha=None):
    """ Smooth with arithmetic average (if n not None) or geometric average (if alpha not None) """
    assert int(n is None) + int(alpha is None) == 1
    if d is None:
        assert n is None or n == 0 # must be first iteration
        return d0

    if n is not None:
        avg_fn = lambda x, y: (x * n + y) / (n+1)
    else:
        avg_fn = lambda x, y: alpha * x + (1. - alpha) * y
    return _bin_op_dict(d, d0, avg_fn)

def _bin_op_dict(d0, d1, op):
    """ Apply binary operator recursively to two dictionaries with matching keys """
    if isinstance(d0, dict) and isinstance(d1, dict):
        assert d0.keys() == d1.keys(), "Dictionaries must hvae matching keys"
        return {
            k: _bin_op_dict(d0[k], d1[k], op) for k in d0.keys()
        }
    elif not isinstance(d0, dict) and not isinstance(d1, dict):
        return op(d0, d1)
    else: raise Exception("Dictionaries must match keys")


def print_size(net, verbose=False):
    """
    Print the number of parameters of a network
    """

    if net is not None and isinstance(net, torch.nn.Module):
        # module_parameters = filter(lambda p: p.requires_grad, net.parameters())
        module_parameters = list(filter(lambda p: p[1].requires_grad, net.named_parameters()))

        if verbose:
            for n, p in module_parameters:
                print(n, p.numel())

        params = sum([np.prod(p.size()) for n, p in module_parameters])
        print("{} Parameters: {:.6f}M".format(
            net.__class__.__name__, params / 1e6), flush=True)


# Utilities for diffusion models

def std_normal(size):
    """
    Generate the standard Gaussian variable of a certain size
    """

    return torch.normal(0, 1, size=size).cuda()


def calc_diffusion_step_embedding(diffusion_steps, diffusion_step_embed_dim_in):
    """
    Embed a diffusion step $t$ into a higher dimensional space
    E.g. the embedding vector in the 128-dimensional space is
    [sin(t * 10^(0*4/63)), ... , sin(t * 10^(63*4/63)), cos(t * 10^(0*4/63)), ... , cos(t * 10^(63*4/63))]

    Parameters:
    diffusion_steps (torch.long tensor, shape=(batchsize, 1)):
                                diffusion steps for batch data
    diffusion_step_embed_dim_in (int, default=128):
                                dimensionality of the embedding space for discrete diffusion steps

    Returns:
    the embedding vectors (torch.tensor, shape=(batchsize, diffusion_step_embed_dim_in)):
    """

    assert diffusion_step_embed_dim_in % 2 == 0

    half_dim = diffusion_step_embed_dim_in // 2
    _embed = np.log(10000) / (half_dim - 1)
    _embed = torch.exp(torch.arange(half_dim) * -_embed).cuda()
    _embed = diffusion_steps * _embed
    diffusion_step_embed = torch.cat((torch.sin(_embed),
                                      torch.cos(_embed)), 1)

    return diffusion_step_embed


def calc_diffusion_hyperparams(T, beta_0, beta_T, beta=None, fast=False):
    """
    Compute diffusion process hyperparameters

    Parameters:
    T (int):                    number of diffusion steps
    beta_0 and beta_T (float):  beta schedule start/end value,
                                where any beta_t in the middle is linearly interpolated

    Returns:
    a dictionary of diffusion hyperparameters including:
        T (int), Beta/Alpha/Alpha_bar/Sigma (torch.tensor on cpu, shape=(T, ))
        These cpu tensors are changed to cuda tensors on each individual gpu
    """

    if fast and beta is not None:
        Beta = torch.tensor(beta)
        T = len(beta)
    else:
        Beta = torch.linspace(beta_0, beta_T, T)
    Alpha = 1 - Beta
    Alpha_bar = Alpha + 0
    Beta_tilde = Beta + 0
    for t in range(1, T):
        Alpha_bar[t] *= Alpha_bar[t-1]  # \bar{\alpha}_t = \prod_{s=1}^t \alpha_s
        Beta_tilde[t] *= (1-Alpha_bar[t-1]) / (1-Alpha_bar[t])  # \tilde{\beta}_t = \beta_t * (1-\bar{\alpha}_{t-1}) / (1-\bar{\alpha}_t)
    Sigma = torch.sqrt(Beta_tilde)  # \sigma_t^2  = \tilde{\beta}_t

    _dh = {}
    _dh["T"], _dh["Beta"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"] = T, Beta.cuda(), Alpha.cuda(), Alpha_bar.cuda(), Sigma
    diffusion_hyperparams = _dh
    # for key in diffusion_hyperparams:
    #     if key != "T":
    #         diffusion_hyperparams[key] = diffusion_hyperparams[key].cuda()
    return diffusion_hyperparams


def sampling(net, size, diffusion_hyperparams, condition=None):
    """
    Perform the complete sampling step according to p(x_0|x_T) = \prod_{t=1}^T p_{\theta}(x_{t-1}|x_t)

    Parameters:
    net (torch network):            the wavenet model
    size (tuple):                   size of tensor to be generated,
                                    usually is (number of audios to generate, channels=1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors

    Returns:
    the generated audio(s) in torch.tensor, shape=size
    """

    _dh = diffusion_hyperparams
    T, Alpha, Alpha_bar, Sigma = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]
    assert len(Alpha) == T
    assert len(Alpha_bar) == T
    assert len(Sigma) == T
    assert len(size) == 3

    print('begin sampling, total number of reverse steps = %s' % T)

    x = std_normal(size)
    with torch.no_grad():
        for t in tqdm(range(T-1, -1, -1)):
            diffusion_steps = (t * torch.ones((size[0], 1))).cuda()  # use the corresponding reverse step
            epsilon_theta = net((x, diffusion_steps,), mel_spec=condition)  # predict \epsilon according to \epsilon_\theta
            x = (x - (1-Alpha[t])/torch.sqrt(1-Alpha_bar[t]) * epsilon_theta) / torch.sqrt(Alpha[t])  # update x_{t-1} to \mu_\theta(x_t)
            if t > 0:
                x = x + Sigma[t] * std_normal(size)  # add the variance term to x_{t-1}
    return x


def training_loss(net, loss_fn, audio, diffusion_hyperparams, mel_spec=None):
    """
    Compute the training loss of epsilon and epsilon_theta

    Parameters:
    net (torch network):            the wavenet model
    loss_fn (torch loss function):  the loss function, default is nn.MSELoss()
    X (torch.tensor):               training data, shape=(batchsize, 1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors

    Returns:
    training loss
    """

    _dh = diffusion_hyperparams
    T, Alpha_bar = _dh["T"], _dh["Alpha_bar"]

    # audio = X
    B, C, L = audio.shape  # B is batchsize, C=1, L is audio length
    diffusion_steps = torch.randint(T, size=(B,1,1)).cuda()  # randomly sample diffusion steps from 1~T
    z = std_normal(audio.shape)
    transformed_X = torch.sqrt(Alpha_bar[diffusion_steps]) * audio + torch.sqrt(1-Alpha_bar[diffusion_steps]) * z  # compute x_t from q(x_t|x_0)
    epsilon_theta = net((transformed_X, diffusion_steps.view(B,1),), mel_spec=mel_spec)  # predict \epsilon according to \epsilon_\theta
    return loss_fn(epsilon_theta, z)


def local_directory(model_config, diffusion_config, dataset_config, output_directory):
    # tensorboard_directory = train_config['tensorboard_directory']
    # ckpt_path = output_directory # train_config['output_directory']

    # generate experiment (local) path
    if model_config['sashimi']:
        model_name = "{}_d{}_n{}_pool_{}_expand{}_ff{}".format(
            "unet" if model_config["unet"] else "snet",
            model_config["d_model"],
            model_config["n_layers"],
            len(model_config["pool"]),
            model_config["expand"],
            model_config["ff"],
            # model_config["channels"],
            # diffusion_config["T"],
            # diffusion_config["beta_T"],
        )
    else:
        model_name = "wnet_h{}_d{}".format(
            model_config["res_channels"],
            model_config["num_res_layers"],
            # diffusion_config["T"],
            # diffusion_config["beta_T"],
        )
    diffusion_name = f"_T{diffusion_config['T']}_betaT{diffusion_config['beta_T']}"
    if model_config["unconditional"]:
        data_name = ""
    else:
        data_name = f"_L{dataset_config['segment_length']}_hop{dataset_config['hop_length']}"
    local_path = model_name + diffusion_name + data_name + f"_{'uncond' if model_config['unconditional'] else 'cond'}"

    # Get shared output_directory ready
    output_directory = os.path.join('exp', local_path, output_directory)
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    print("output directory", output_directory, flush=True)
    return local_path, output_directory
