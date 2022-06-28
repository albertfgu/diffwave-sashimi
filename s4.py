""" Standalone version of Structured (Sequence) State Space (S4) model. """


import logging
from functools import partial
import math
import numpy as np
from scipy import special as ss
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as U
# from pytorch_lightning.utilities import rank_zero_only
from einops import rearrange, repeat
from omegaconf import DictConfig
import opt_einsum as oe

contract = oe.contract


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    # for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
    #     setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger
log = get_logger(__name__)

""" Cauchy kernel """

try: # Try CUDA extension
    from extensions.cauchy.cauchy import cauchy_mult
    has_cauchy_extension = True
except:
    log.warn(
        "CUDA extension for cauchy multiplication not found. Install by going to extensions/cauchy/ and running `python setup.py install`. This should speed up end-to-end training by 10-50%"
    )
    has_cauchy_extension = False

try: # Try pykeops
    import pykeops
    from pykeops.torch import Genred
except ImportError:
    if not has_cauchy_extension:
        log.error(
            "Install at least one of pykeops or the cauchy_mult extension."
        )

def _broadcast_dims(*tensors):
    max_dim = max([len(tensor.shape) for tensor in tensors])
    tensors = [tensor.view((1,)*(max_dim-len(tensor.shape))+tensor.shape) for tensor in tensors]
    return tensors

def cauchy_conj(v, z, w, num=2, denom=2):
    """ Pykeops version """
    if num == 1:
        expr_num = 'z * ComplexReal(v) - Real2Complex(ComplexReal(v)*ComplexReal(w) + ComplexImag(v)*ComplexImag(w))'
    elif num == 2:
        expr_num = 'z * ComplexReal(v) - Real2Complex(Sum(v * w))'
    else: raise NotImplementedError

    if denom == 1:
        expr_denom = 'ComplexMult(z-Real2Complex(ComplexReal(w)), z-Real2Complex(ComplexReal(w))) + Real2Complex(Square(ComplexImag(w)))'
    elif denom == 2:
        expr_denom = 'ComplexMult(z-w, z-Conj(w))'
    else: raise NotImplementedError

    cauchy_mult = Genred(
        f'ComplexDivide({expr_num}, {expr_denom})',
        # expr_num,
        # expr_denom,
        [
            'v = Vj(2)',
            'z = Vi(2)',
            'w = Vj(2)',
        ],
        reduction_op='Sum',
        axis=1,
        # dtype='float32' if v.dtype == torch.cfloat else 'float64',
    )

    v, z, w = _broadcast_dims(v, z, w)
    v = torch.view_as_real(v)
    z = torch.view_as_real(z)
    w = torch.view_as_real(w)

    r = 2*cauchy_mult(v, z, w, backend='GPU')
    return torch.view_as_complex(r)

_conj = lambda x: torch.cat([x, x.conj()], dim=-1)

""" simple nn.Module components """

def Activation(activation=None, dim=-1):
    if activation in [ None, 'id', 'identity', 'linear' ]:
        return nn.Identity()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'relu':
        return nn.ReLU()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation in ['swish', 'silu']:
        return nn.SiLU()
    elif activation == 'glu':
        return nn.GLU(dim=dim)
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise NotImplementedError("hidden activation '{}' is not implemented".format(activation))

def get_initializer(name, activation=None):
    if activation in [ None, 'id', 'identity', 'linear', 'modrelu' ]:
        nonlinearity = 'linear'
    elif activation in ['relu', 'tanh', 'sigmoid']:
        nonlinearity = activation
    elif activation in ['gelu', 'swish', 'silu']:
        nonlinearity = 'relu' # Close to ReLU so approximate with ReLU's gain
    else:
        raise NotImplementedError(f"get_initializer: activation {activation} not supported")

    if name == 'uniform':
        initializer = partial(torch.nn.init.kaiming_uniform_, nonlinearity=nonlinearity)
    elif name == 'normal':
        initializer = partial(torch.nn.init.kaiming_normal_, nonlinearity=nonlinearity)
    elif name == 'xavier':
        initializer = torch.nn.init.xavier_normal_
    elif name == 'zero':
        initializer = partial(torch.nn.init.constant_, val=0)
    elif name == 'one':
        initializer = partial(torch.nn.init.constant_, val=1)
    else:
        raise NotImplementedError(f"get_initializer: initializer type {name} not supported")

    return initializer

class TransposedLinear(nn.Module):
    """ Linear module on the second-to-last dimension """

    def __init__(self, d_input, d_output, bias=True):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(d_output, d_input))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5)) # nn.Linear default init
        # nn.init.kaiming_uniform_(self.weight, nonlinearity='linear') # should be equivalent

        if bias:
            self.bias = nn.Parameter(torch.empty(d_output, 1))
            bound = 1 / math.sqrt(d_input)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = 0.0

    def forward(self, x):
        return contract('... u l, v u -> ... v l', x, self.weight) + self.bias

def LinearActivation(
        d_input, d_output, bias=True,
        zero_bias_init=False,
        transposed=False,
        initializer=None,
        activation=None,
        activate=False, # Apply activation as part of this module
        weight_norm=False,
        **kwargs,
    ):
    """ Returns a linear nn.Module with control over axes order, initialization, and activation """

    # Construct core module
    linear_cls = TransposedLinear if transposed else nn.Linear
    if activation == 'glu': d_output *= 2
    linear = linear_cls(d_input, d_output, bias=bias, **kwargs)

    # Initialize weight
    if initializer is not None:
        get_initializer(initializer, activation)(linear.weight)

    # Initialize bias
    if bias and zero_bias_init:
        nn.init.zeros_(linear.bias)

    # Weight norm
    if weight_norm:
        linear = nn.utils.weight_norm(linear)

    if activate and activation is not None:
        activation = Activation(activation, dim=-2 if transposed else -1)
        linear = nn.Sequential(linear, activation)
    return linear

""" Misc functional utilities """

def krylov(L, A, b, c=None, return_power=False):
    """
    Compute the Krylov matrix (b, Ab, A^2b, ...) using the squaring trick.

    If return_power=True, return A^{L-1} as well
    """
    # TODO There is an edge case if L=1 where output doesn't get broadcasted, which might be an issue if caller is expecting broadcasting semantics... can deal with it if it arises

    x = b.unsqueeze(-1) # (..., N, 1)
    A_ = A

    AL = None
    if return_power:
        AL = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
        _L = L-1

    done = L == 1
    # loop invariant: _L represents how many indices left to compute
    while not done:
        if return_power:
            if _L % 2 == 1: AL = A_ @ AL
            _L //= 2

        # Save memory on last iteration
        l = x.shape[-1]
        if L - l <= l:
            done = True
            _x = x[..., :L-l]
        else: _x = x

        _x = A_ @ _x
        x = torch.cat([x, _x], dim=-1) # there might be a more efficient way of ordering axes
        if not done: A_ = A_ @ A_

    assert x.shape[-1] == L

    if c is not None:
        x = torch.einsum('...nl, ...n -> ...l', x, c)
    x = x.contiguous() # WOW!!
    if return_power:
        return x, AL
    else:
        return x

def power(L, A, v=None):
    """ Compute A^L and the scan sum_i A^i v_i

    A: (..., N, N)
    v: (..., N, L)
    """

    I = torch.eye(A.shape[-1]).to(A) # , dtype=A.dtype, device=A.device)

    powers = [A]
    l = 1
    while True:
        if L % 2 == 1: I = powers[-1] @ I
        L //= 2
        if L == 0: break
        l *= 2
        powers.append(powers[-1] @ powers[-1])

    if v is None: return I

    # Invariants:
    # powers[-1] := A^l
    # l := largest po2 at most L

    # Note that an alternative divide and conquer to compute the reduction is possible and can be embedded into the above loop without caching intermediate powers of A
    # We do this reverse divide-and-conquer for efficiency reasons:
    # 1) it involves fewer padding steps for non-po2 L
    # 2) it involves more contiguous arrays

    # Take care of edge case for non-po2 arrays
    # Note that this initial step is a no-op for the case of power of 2 (l == L)
    k = v.size(-1) - l
    v_ = powers.pop() @ v[..., l:]
    v = v[..., :l]
    v[..., :k] = v[..., :k] + v_

    # Handle reduction for power of 2
    while v.size(-1) > 1:
        v = rearrange(v, '... (z l) -> ... z l', z=2)
        v = v[..., 0, :] + powers.pop() @ v[..., 1, :]
    return I, v.squeeze(-1)


""" HiPPO utilities """

def transition(measure, N, **measure_args):
    """ A, B transition matrices for different measures

    measure: the type of measure
      legt - Legendre (translated)
      legs - Legendre (scaled)
      glagt - generalized Laguerre (translated)
      lagt, tlagt - previous versions of (tilted) Laguerre with slightly different normalization
    """
    # Laguerre (translated)
    if measure == 'lagt':
        b = measure_args.get('beta', 1.0)
        A = np.eye(N) / 2 - np.tril(np.ones((N, N)))
        B = b * np.ones((N, 1))
    elif measure == 'tlagt':
        # beta = 1 corresponds to no tilt
        b = measure_args.get('beta', 1.0)
        A = (1.-b)/2 * np.eye(N) - np.tril(np.ones((N, N)))
        B = b * np.ones((N, 1))
    # Generalized Laguerre
    # alpha 0, beta small is most stable (limits to the 'lagt' measure)
    # alpha 0, beta 1 has transition matrix A = [lower triangular 1]
    elif measure == 'glagt':
        alpha = measure_args.get('alpha', 0.0)
        beta = measure_args.get('beta', 0.01)
        A = -np.eye(N) * (1 + beta) / 2 - np.tril(np.ones((N, N)), -1)
        B = ss.binom(alpha + np.arange(N), np.arange(N))[:, None]

        L = np.exp(.5 * (ss.gammaln(np.arange(N)+alpha+1) - ss.gammaln(np.arange(N)+1)))
        A = (1./L[:, None]) * A * L[None, :]
        B = (1./L[:, None]) * B * np.exp(-.5 * ss.gammaln(1-alpha)) * beta**((1-alpha)/2)
    # Legendre (translated)
    elif measure == 'legt':
        Q = np.arange(N, dtype=np.float64)
        R = (2*Q + 1) ** .5
        j, i = np.meshgrid(Q, Q)
        A = R[:, None] * np.where(i < j, (-1.)**(i-j), 1) * R[None, :]
        B = R[:, None]
        A = -A
    # LMU: equivalent to LegT up to normalization
    elif measure == 'lmu':
        Q = np.arange(N, dtype=np.float64)
        R = (2*Q + 1)[:, None] # / theta
        j, i = np.meshgrid(Q, Q)
        A = np.where(i < j, -1, (-1.)**(i-j+1)) * R
        B = (-1.)**Q[:, None] * R
    # Legendre (scaled)
    elif measure == 'legs':
        q = np.arange(N, dtype=np.float64)
        col, row = np.meshgrid(q, q)
        r = 2 * q + 1
        M = -(np.where(row >= col, r, 0) - np.diag(q))
        T = np.sqrt(np.diag(2 * q + 1))
        A = T @ M @ np.linalg.inv(T)
        B = np.diag(T)[:, None]
        B = B.copy() # Otherwise "UserWarning: given NumPY array is not writeable..." after torch.as_tensor(B)
    else:
        raise NotImplementedError

    return A, B

def rank_correction(measure, N, rank=1, dtype=torch.float):
    """ Return low-rank matrix L such that A + L is normal """

    if measure == 'legs':
        assert rank >= 1
        p = torch.sqrt(.5+torch.arange(N, dtype=dtype)).unsqueeze(0) # (1 N)
    elif measure == 'legt':
        assert rank >= 2
        p = torch.sqrt(1+2*torch.arange(N, dtype=dtype)) # (N)
        p0 = p.clone()
        p0[0::2] = 0.
        p1 = p.clone()
        p1[1::2] = 0.
        p = torch.stack([p0, p1], dim=0) # (2 N)
    elif measure == 'lagt':
        assert rank >= 1
        p = .5**.5 * torch.ones(1, N, dtype=dtype)
    else: raise NotImplementedError

    d = p.size(0)
    if rank > d:
        p = torch.stack([p, torch.zeros(N, dtype=dtype).repeat(rank-d, d)], dim=0) # (rank N)
    return p


def nplr(measure, N, rank=1, dtype=torch.float):
    """ Return w, p, q, V, B such that
    (w - p q^*, B) is unitarily equivalent to the original HiPPO A, B by the matrix V
    i.e. A = V[w - p q^*]V^*, B = V B
    """
    A, B = transition(measure, N)
    A = torch.as_tensor(A, dtype=dtype) # (N, N)
    B = torch.as_tensor(B, dtype=dtype)[:, 0] # (N,)

    p = rank_correction(measure, N, rank=rank, dtype=dtype)
    Ap = A + torch.sum(p.unsqueeze(-2)*p.unsqueeze(-1), dim=-3)
    w, V = torch.linalg.eig(Ap) # (..., N) (..., N, N)
    # V w V^{-1} = A

    # Only keep one of the conjugate pairs
    w = w[..., 0::2].contiguous()
    V = V[..., 0::2].contiguous()

    V_inv = V.conj().transpose(-1, -2)

    B = contract('ij, j -> i', V_inv, B.to(V)) # V^* B
    p = contract('ij, ...j -> ...i', V_inv, p.to(V)) # V^* p


    return w, p, p, B, V


""" Final S4 Module, and simplified but slower version for testing/exposition """

class OptimModule(nn.Module):
    """ Interface for Module that allows registering buffers/parameters with configurable optimizer hyperparameters """

    def register(self, name, tensor, trainable=0, lr=None, wd=None, repeat=1):
        """Utility method: register a tensor as a buffer or trainable parameter"""

        if trainable == 0:
            self.register_buffer(name, tensor)
        elif trainable == 1:
            self.register_parameter(name, nn.Parameter(tensor))
        elif trainable == 2:
            if repeat > 1:
                tensor = tensor.repeat(repeat, *(1,) * len(tensor.shape))
            self.register_parameter(name, nn.Parameter(tensor))
        else:
            raise NotImplementedError

        optim = {}
        if trainable and lr is not None:
            optim["lr"] = lr
            # setattr(getattr(self, name), '_lr', lr)
        if trainable and wd is not None:
            optim["weight_decay"] = wd
            # setattr(getattr(self, name), '_wd', wd)
        if len(optim) > 0:
            setattr(getattr(self, name), "_optim", optim)


class SSKernelNPLR(OptimModule):
    """Stores a representation of and computes the SSKernel function K_L(A^dt, B^dt, C) corresponding to a discretized state space, where A is Normal + Low Rank (NPLR)

    The class name stands for 'State-Space SSKernel for Normal Plus Low-Rank'.
    The parameters of this function are as follows.

    A: (... N N) the state matrix
    B: (... N) input matrix
    C: (... N) output matrix
    dt: (...) timescales / discretization step size
    p, q: (... P N) low-rank correction to A, such that Ap=A+pq^T is a normal matrix

    The forward pass of this Module returns:
    (... L) that represents represents FFT SSKernel_L(A^dt, B^dt, C)

    """

    @torch.no_grad()
    def _process_C(self, L, double_length=False):
        C = torch.view_as_complex(self.C)
        self._setup(setup_C=False)
        dA = self.dA
        dA_L = power(L, dA)
        # I = torch.eye(dA.size(-1)).to(dA)
        N = C.size(-1)
        # Multiply C by I - dA_L
        C_ = C[..., 0, :]
        # C_ = torch.cat([C_, C_.conj()], dim=-1)
        C_ = _conj(C_)
        prod = contract("... m n, ... n -> ... m", dA_L.conj().transpose(-1, -2), C_)
        if double_length:  # Multiply by I + dA_L instead
            C_ = C_ + prod
        else:
            C_ = C_ - prod
        C_ = C_[..., :N]
        # self.C[..., 0, :, :].copy_(torch.view_as_real(C_))
        self.C.copy_(torch.view_as_real(C_[..., None, :]))

    def _nodes(self, L, dtype, device):
        # Cache FFT nodes and their "unprocessed" them with the bilinear transform
        # nodes = torch.tensor(np.exp(-2j * np.pi / (L)), dtype=torch.cfloat, device=Ap.device) # \omega_{2L}
        nodes = torch.tensor(
            np.exp(-2j * np.pi / (L)), dtype=dtype, device=device
        )  # \omega_{2L}
        nodes = nodes ** torch.arange(0, L // 2 + 1, device=device)
        z = 2 * (1 - nodes) / (1 + nodes)
        return nodes, z

    def __init__(
        self,
        L, w, p, q, B, C,
        log_dt,
        trainable=None,
        lr=None,
        setup_C=False,
        keops=False,
        hurwitz=False,
    ):
        """
        L: Maximum length; this module computes SSKernel function of length L
        A: (..., N, N) represented by diag(w) - pq^*
        B: (..., N)
        C: (..., c, N) system is 1-D to c-D
        dt: (...)
        p: (..., r, N) low-rank correction to A
        q: (..., r, N)
        """

        super().__init__()
        self.keops = keops
        self.hurwitz = hurwitz

        # Rank of low-rank correction
        if q is not None: assert p.shape[-2] == q.shape[-2]
        self.rank = p.shape[-2]
        self.L = L

        # Broadcast everything to shape of C
        B = B.unsqueeze(-2)  # (..., 1, N)
        if q is not None:
            if len(C.shape) > len(q.shape):
                q = q.repeat(C.shape[:-2] + (1, 1)) # TODO do with einops
        if len(C.shape) > len(p.shape):
            p = p.repeat(C.shape[:-2] + (1, 1))
        if len(C.shape) > len(B.shape):
            B = B.repeat(C.shape[:-2] + (1, 1))

        if L is not None:
            nodes, z = self._nodes(L, dtype=w.dtype, device=w.device)
            self.register_buffer("nodes", torch.view_as_real(nodes))
            self.register_buffer("z", torch.view_as_real(z))

        # Register parameters
        # TODO figure out conventions for registering P and Q
        # trainable should be 0 or 2 by default (everything should train the H dimension)
        # TODO broadcast p, q, B to same dim as C, remove repeat argument
        if trainable is None:
            trainable = DictConfig({"A": 0, "B": 0, "C": 1, "P": 0, "Q": 0, "dt": 1})
        if lr is None:
            lr = DictConfig({"A": None, "B": None, "C": None, "P": None, "Q": None, "dt": None})
        repeat = C.size(0)
        self.register("log_dt", log_dt, trainable.dt, lr.dt, 0.0)
        if self.hurwitz:
            log_w_real = torch.log(-w.real)
            w_imag = w.imag
            self.register("log_w_real", log_w_real, trainable.A, lr.A, 0.0, repeat=repeat)
            self.register("w_imag", w_imag, trainable.A, lr.A, 0.0, repeat=repeat)
        else:
            self.register("w", torch.view_as_real(w), trainable.A, lr.A, 0.0, repeat=repeat)
        self.register("B", torch.view_as_real(B), trainable.B, lr.B, 0.0, repeat=repeat)
        self.register("C", torch.view_as_real(C), trainable.C, lr.C)
        self.register("p", torch.view_as_real(p), trainable.P, lr.P, 0.0, repeat=repeat)
        if q is not None:
            self.register("q", torch.view_as_real(q), trainable.Q, lr.Q, 0.0, repeat=repeat)
        else:
            self.q = None

        self._is_diagonalized = False

        if setup_C:
            self._process_C(L)

    def _w(self):
        # Get the internal w (diagonal) parameter
        if self.hurwitz:
            # TODO maybe faster way to do this
            w_real = -torch.exp(self.log_w_real)
            w_imag = self.w_imag
            w = w_real + 1j * w_imag
        else:
            w = torch.view_as_complex(self.w)  # (..., N)
        return w

    def forward(self, state=None, rate=1.0, L=None):
        """
        state: (..., s, N) extra tensor that augments B
        rate: sampling rate factor

        returns: (..., c+s, L)
        """
        # if L is not None: raise NotImplementedError

        # TODO: handle potential length doubling logic so that max_len doesn't need to be passed in
        while rate == 1.0 and L is not None and L > self.L:
            log.info(f"S4: Doubling length from L = {self.L} to {2*self.L}")
            self.double_length()

        if L is None:
            L = self.L
        if rate == 1.0: # [AG 22-02-02 I forget what this is for but I remember it gave me a headache so don't touch]
            L = self.L
        else:
            rate = self.L / L
        dt = torch.exp(self.log_dt) * rate
        B = torch.view_as_complex(self.B)
        C = torch.view_as_complex(self.C)
        p = torch.view_as_complex(self.p)
        q = p if self.q is None else torch.view_as_complex(self.q)
        w = self._w()
        # z = torch.view_as_complex(self.z) # (..., L)

        # TODO adjust based on rate times normal max length
        if L == self.L:
            nodes = torch.view_as_complex(self.nodes)
            z = torch.view_as_complex(self.z)  # (..., L)
        else:
            nodes, z = self._nodes(L, dtype=w.dtype, device=w.device)

        # Augment B
        if state is not None:  # TODO have not tested since p/q refactor
            # Have to "unbilinear" the state to put it into the same "type" as B
            # Compute (I + dt/2 A) @ state
            s = state.transpose(0, 1)  # (H B N)
            # p = B[..., 1:, :]  # (... r N)
            # q = C[..., 1:, :]  # (... r N)

            # Calculate contract('... s n, ... r n, ... r m -> ... s m', sV, qV.conj(), pV), but take care of conjugate symmetry
            sA = (
                s * w.unsqueeze(-2)
                - (2 + 0j) * (s @ q.conj().transpose(-1, -2)).real @ p
            )
            s = s / dt.unsqueeze(-1).unsqueeze(-1) + sA / 2

            B = torch.cat([s, B], dim=-2)  # (..., 2+s, N)

        # Incorporate dt into A
        w = w * dt.unsqueeze(-1)  # (... N)

        # Stack B and p, C and q for convenient batching
        B = torch.cat([B, p], dim=-2) # (..., 1+s+r, N)
        C = torch.cat([C, q], dim=-2) # (..., c+r, N)

        # Incorporate B and C batch dimensions
        v = B.unsqueeze(-3) * C.unsqueeze(-2).conj()  # (..., c+r, 1+s+r, N)
        w = w[..., None, None, :]  # (..., 1, 1, N)
        z = z[..., None, None, :]  # (..., 1, 1, L)

        # Calculate resolvent at nodes
        if not self.keops and has_cauchy_extension:
            r = cauchy_mult(v, z, w, symmetric=True)
        else:
            r = cauchy_conj(v, z, w)
        r = r * dt[..., None, None, None]  # (..., c+r, 1+s+r, L)

        # Low-rank Woodbury correction
        if self.rank == 1:
            k_f = r[..., :-1, :-1, :] - r[..., :-1, -1:, :] * r[..., -1:, :-1, :] / (
                1 + r[..., -1:, -1:, :]
            )
        elif self.rank == 2:
            r00 = r[..., : -self.rank, : -self.rank, :]
            r01 = r[..., : -self.rank, -self.rank :, :]
            r10 = r[..., -self.rank :, : -self.rank, :]
            r11 = r[..., -self.rank :, -self.rank :, :]
            det = (1 + r11[..., :1, :1, :]) * (1 + r11[..., 1:, 1:, :]) - r11[
                ..., :1, 1:, :
            ] * r11[..., 1:, :1, :]
            s = (
                r01[..., :, :1, :] * (1 + r11[..., 1:, 1:, :]) * r10[..., :1, :, :]
                + r01[..., :, 1:, :] * (1 + r11[..., :1, :1, :]) * r10[..., 1:, :, :]
                - r01[..., :, :1, :] * (r11[..., :1, 1:, :]) * r10[..., 1:, :, :]
                - r01[..., :, 1:, :] * (r11[..., 1:, :1, :]) * r10[..., :1, :, :]
            )
            s = s / det
            k_f = r00 - s
        else:
            r00 = r[..., : -self.rank, : -self.rank, :]
            r01 = r[..., : -self.rank, -self.rank :, :]
            r10 = r[..., -self.rank :, : -self.rank, :]
            r11 = r[..., -self.rank :, -self.rank :, :]
            r11 = rearrange(r11, "... a b n -> ... n a b")
            r11 = torch.linalg.inv(torch.eye(self.rank, device=r.device) + r11)
            r11 = rearrange(r11, "... n a b -> ... a b n")
            k_f = r00 - torch.einsum(
                "... i j n, ... j k n, ... k l n -> ... i l n", r01, r11, r10
            )

        # Final correction for the bilinear transform
        k_f = k_f * 2 / (1 + nodes)

        k = torch.fft.irfft(k_f)  # (..., c, 1+s, L)
        if state is not None:
            k_state = k[..., :, :-1, :]  # (..., c, s, L) # TODO what happens when c > 1
            # k_state = k_state.transpose(0, 1)
            k_state = rearrange(k_state, '... c s l -> s ... c l')
            k_B = k[..., :, -1, :]  # (..., L)
            return k_B.to(torch.float), k_state.to(torch.float)
        else:
            return k.squeeze(-2).to(torch.float)

    @torch.no_grad()
    def double_length(self):
        self._process_C(self.L, double_length=True)

        self.L *= 2
        dtype = torch.view_as_complex(self.w).dtype
        nodes, z = self._nodes(self.L, dtype=dtype, device=self.w.device)
        self.register_buffer("nodes", torch.view_as_real(nodes))
        self.register_buffer("z", torch.view_as_real(z))

    @torch.no_grad()
    def _check(self):
        """Check if A, B, C parameters and vanilla SSKernel construction can be recovered"""

        self._setup(setup_C=True)

        K = krylov(self.L, self.dA, self.dB, self.dC.conj())

        diff = K - self.forward(L=self.L)
        print("checking SSKernel construction", torch.sum(diff ** 2))

    def _setup(self, setup_C=True):
        # w = _conj(torch.view_as_complex(self.w))
        w = _conj(self._w())
        B = _conj(torch.view_as_complex(self.B))
        C = _conj(torch.view_as_complex(self.C))
        C = C.conj()
        # p = B[..., -1, :]
        # q = C[..., -1, :]
        B = B[..., 0, :]
        C = C[..., 0, :]
        p = _conj(torch.view_as_complex(self.p))
        q = _conj(torch.view_as_complex(self.q))
        # TODO for now we're only handling the case where p and q are rank 1
        p = p[..., 0, :]
        q = q[..., 0, :]
        q = q.conj()
        dt = torch.exp(self.log_dt)
        d = (2.0 / dt.unsqueeze(-1) - w).reciprocal()  # (H, N)
        r = (1 + contract("... n, ... n, ... n -> ...", q, d, p)).reciprocal()
        # A_f = torch.diag_embed(2./dt[:, None] + w) - contract('... n, ... m -> ... n m', p, q)
        # A_b = torch.diag_embed(d) - contract('... p, ... p, ..., ... q, ... q -> ... p q', d, p, r, q, d)
        # dA = A_b @ A_f

        self.step_params = {
            "d": d,
            "r": r.unsqueeze(-1) * d * q,
            # 'r': r,
            "p": p,
            "q": q,
            "B": B,
            "d1": 2.0 / dt.unsqueeze(-1) + w,
        }
        N = d.size(-1)
        H = dt.size(-1)

        state = torch.eye(N, dtype=w.dtype, device=w.device).unsqueeze(-2)
        u = w.new_zeros(H)
        dA = self.step_state_linear(u, state)
        dA = rearrange(dA, "n h m -> h m n")
        self.dA = dA
        u = w.new_ones(H)
        state = w.new_zeros(N // 2)
        dB = self.step_state_linear(u, state)
        dB = _conj(dB)
        self.dB = dB

        if setup_C:
            # dA_L = power(self.L, dA)
            # I = torch.eye(dA.size(-1)).to(dA)
            # dC = torch.linalg.solve(
            #     I - dA_L.transpose(-1, -2).conj(), 
            #     C.conj().unsqueeze(-1),
            # ).squeeze(-1)
            dA_L = power(self.L, dA.to(torch.complex128))
            I = torch.eye(dA.size(-1)).to(dA_L)
            dC = torch.linalg.solve(
                I - dA_L.transpose(-1, -2).conj(), 
                C.conj().unsqueeze(-1).to(torch.complex128),
            ).squeeze(-1)
            self.C__ = C
            if len(torch.where(dC.isnan())[0]):
                print("NaN in dC")
                # Compute the eigendecomposition of the A matrix
                L, V = torch.linalg.eig(dA)
            self.dC = dC.to(torch.complex64)

    def _setup_diagonal_representation(self):
        # Eigendecomposition of the A matrix
        L, V = torch.linalg.eig(self.dA)
        V_inv = torch.linalg.inv(V)
        # Check that the eigendedecomposition is correct
        # torch.dist(V @ torch.diag_embed(L) @ V_inv, self.dA)

        # Change the parameterization to diagonalize
        self.dA = torch.diag_embed(L)
        self.dA_diag = L
        self.dB = contract('h n m, h m -> h n', V_inv, self.dB)
        # self.dC = contract('h n, h n m -> h m', self.dC, V)
        self.dC = contract('h n m, h n -> h m', V.conj(), self.dC)

        self._is_diagonalized = True

    def step_state_linear(self, u=None, state=None):
        """Version of the step function that has time O(N) instead of O(N^2) per step. Unfortunately, as currently implemented it's about 2x slower because it calls several sequential operations. Perhaps a fused CUDA kernel implementation would be much faster"""
        N = self.step_params["d"].size(-1)
        H = self.log_dt.size(-1)

        if u is None:
            u = torch.zeros(H, dtype=torch.float, device=self.log_dt.device)
        if state is None:
            state = torch.zeros(H, N, dtype=torch.cfloat, device=self.log_dt.device)

        conj = state.size(-1) != N
        step_params = self.step_params.copy()
        if conj:
            assert state.size(-1) == N // 2
            step_params = {k: v[..., : N // 2] for k, v in step_params.items()}
        d1 = step_params["d1"]  # (H N)
        p = step_params["p"]  # (H N)
        q = step_params["q"]  # (H N)
        B = step_params["B"]  # (H N)
        r = step_params["r"]
        d = step_params["d"]  # (H N)
        # dC = self.step_params['dC'] # (H N)
        state = state.to(d1)

        if conj:
            new_state = (
                2 * p * torch.sum(q * state, dim=-1, keepdim=True).real
            )  # conjugated version
        else:
            new_state = contract("... n, ... m, ... m -> ... n", p, q, state)  # (B H N)
        new_state = d1 * state - new_state
        new_state = new_state + 2.0 * B * u.unsqueeze(-1)  # (B H N)
        if conj:
            A_ = (
                2 * p * torch.sum(r * new_state, dim=-1, keepdim=True).real
            )  # conj version
        else:
            A_ = contract("... p, ... q, ... q -> ... p", p, r, new_state)  # (B H N)
        new_state = d * (new_state - A_)

        return new_state

    def step_state(self, u, state):
        state = state.to(self.dA)
        conj = state.size(-1) != self.dA.size(-1)
        if conj:
            state = _conj(state)

        if not hasattr(self, 'state_contraction_1'):

            if self._is_diagonalized:
                self.state_contraction_1 = contract_expression(
                    "h n, b h n -> b h n",
                    self.dA_diag.shape,
                    state.shape,
                )
            else:
                self.state_contraction_1 = contract_expression(
                    "h m n, b h n -> b h m",
                    self.dA.shape,
                    state.shape,
                )
        
        if not hasattr(self, 'state_contraction_2'):
            self.state_contraction_2 = contract_expression(
                "h n, b h -> b h n",
                self.dB.shape,
                u.shape,
            )

        if self._is_diagonalized:
            # next_state = self.state_contraction_1(self.dA_diag, state) + self.state_contraction_2(self.dB, u)
            next_state = self.dA_diag * state + self.state_contraction_2(self.dB, u)
        else:
            next_state = self.state_contraction_1(self.dA, state) + self.state_contraction_2(self.dB, u)
        
        # next_state = contract("h m n, b h n -> b h m", self.dA, state) + contract(
        #     "h n, b h -> b h n", self.dB, u
        # )
        if conj:
            next_state = next_state[..., : state.size(-1) // 2]
        return next_state

    def step(self, u, state, linear=False, approximate_C=False):

        N = self.step_params["d"].size(-1)
        conj = state.size(-1) != N

        if linear:
            new_state = self.step_state_linear(u, state)
        else:
            new_state = self.step_state(u, state)

        if approximate_C:
            # C is approximated using data in the equation y_t = x_t C
            assert hasattr(self, 'dC_'), "C approximation is missing."
            out = contract("... n, ... n -> ...", self.dC_, new_state)
            return out.to(torch.float), new_state

        if conj:
            assert state.size(-1) == N // 2
            # dC = self.dC[..., 0::2].conj()
            dC = self.dC[..., : N // 2].conj()
            out = 2 * torch.sum(dC * new_state, dim=-1).real  # conj version
        else:
            if not hasattr(self, 'contraction_1'):
                self.contraction_1 = contract_expression(
                    "... n, ... n -> ...",
                    self.dC.shape,
                    new_state.shape,
                )
            out = self.contraction_1(self.dC.conj(), new_state)            
            # out = contract("... n, ... n -> ...", self.dC.conj(), new_state)
        return out.to(torch.float), new_state






class HippoSSKernel(nn.Module):
    """Wrapper around SSKernelNPLR that generates A, B, C, dt according to HiPPO arguments."""

    def __init__(
        self,
        N,
        H,
        L=None,
        measure="legs",
        rank=1,
        dt_min=0.001,
        dt_max=0.1,
        channels=1,
        trainable=None,
        lr=None,
        mode="nplr",  # 'slow' for complex naive version, 'real' for real naive version
        tied_lr=False,
        length_correction=False, # Multiply by I-A|^L after initialization
        precision=1, # 1 (single) or 2 (double) for the kernel
        cache=False, # Cache kernel at eval time
        resample=False,  # if given inputs of different lengths, adjust the sampling rate
        keops=False, # Force pykeops
        hurwitz=False,
    ):
        super().__init__()
        self.N = N
        self.H = H
        L = L or 1
        self.precision = precision
        dtype = torch.double if self.precision == 2 else torch.float
        self.rate = None if resample else 1.0

        # Set default trainable and lr parameters
        self.trainable = DictConfig(
            {
                "A": 0,
                "B": 0,
                "C": 1,
                "P": 0, # TODO fix for NPLR vs slow
                "Q": 0,
                "dt": 0,
            }
        )
        if trainable is not None:
            self.trainable.update(trainable)
        self.lr = DictConfig(
            {
                "A": 1e-3,
                "B": 1e-3,
                "C": None,
                "P": 1e-3,
                "Q": 1e-3,
                "dt": 1e-3,
            }
        )
        if lr is not None:
            self.lr.update(lr)

        # Generate dt
        self.log_dt = torch.rand(self.H, dtype=dtype) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)

        # Compute the preprocessed representation
        if mode == "real":  # Testing purposes only
            # Generate A, B
            A, B = hippo.transition(measure, N)
            A = torch.as_tensor(A, dtype=dtype)
            B = torch.as_tensor(B, dtype=dtype)[:, 0]

            # Generate C
            C = torch.randn(self.H, self.N, dtype=dtype)

            self.krylov = SSKernelSlow(
                L, A, B, C, self.log_dt, trainable=self.trainable, lr=self.lr
            )
        else:
            # Generate low rank correction p for the measure
            if measure == 'random':
                w, p, q, B, _ = nplr('legs', N, rank, dtype=dtype)
                w = torch.randn_like(w, dtype=torch.cfloat)
                p = torch.randn_like(p, dtype=torch.cfloat)
                q = torch.randn_like(q, dtype=torch.cfloat)
                B = torch.randn_like(B, dtype=torch.cfloat)
            else:
                w, p, q, B, _ = nplr(measure, N, rank, dtype=dtype)
            cdtype = torch.cfloat if dtype == torch.float else torch.cdouble
            C = torch.randn(self.H, channels, self.N // 2, dtype=cdtype)
            if mode == "nplr":
                if tied_lr: q = None
                self.krylov = SSKernelNPLR(
                    L,
                    w,
                    p,
                    q,
                    B,
                    C,
                    self.log_dt,
                    trainable=self.trainable,
                    lr=self.lr,
                    setup_C=length_correction,
                    keops=keops,
                    hurwitz=hurwitz,
                )
            elif mode == "slow":  # Testing only
                A = torch.diag_embed(_conj(w)) - contract(
                    "... r p, ... r q -> ... p q", _conj(p), _conj(q).conj()
                )
                self.krylov = SSKernelSlow(
                    L,
                    A,
                    _conj(B),
                    _conj(C),
                    self.log_dt,
                    trainable=self.trainable,
                    lr=self.lr,
                )

        # Cached tensors
        self.K = None
        self.cache = cache

    def forward(self, state=None, L=None):
        """
        state: (B, H, N)
        """

        if state is not None:
            k, k_state = self.krylov(
                state=state, rate=self.rate, L=L
            )  # (B, H, L) (B, H, N)
            return k, k_state
        else:
            # Calculate K if needed
            if not self.training and self.K is not None and self.K.size(-1) == L:
                k = self.K
            else:
                k = self.krylov(rate=self.rate, L=L).to(torch.float)

            # Store K if needed
            if self.cache and not self.training:
                self.K = k
            else:  # If training, parameter will change after backprop so make sure to recompute on next pass
                self.K = None
            return k

    @torch.no_grad()
    def next_state(self, state, u):
        """
        state: (..., N)
        u: (..., L)

        Returns: (..., N)
        """

        self.krylov._setup()
        dA, dB = self.krylov.dA, self.krylov.dB

        conj = state.size(-1) != dA.size(-1)
        if conj:
            state = _conj(state)

        v = dB.unsqueeze(-1) * u.flip(-1).unsqueeze(-2)  # (..., N, L)
        AL, v = power(u.size(-1), dA, v)
        next_state = contract("... m n, ... n -> ... m", AL, state)
        next_state = next_state + v

        if conj:
            next_state = next_state[..., : next_state.size(-1) // 2]
        return next_state

    def step(self, u, state, **kwargs):
        return self.krylov.step(u, state, **kwargs)

    def double_length(self):
        self.krylov.double_length()

class S4(nn.Module):

    def __init__(
            self,
            H,
            l_max=None,
            # Arguments for SSM Kernel
            d_state=64,
            measure='legs',
            dt_min=0.001,
            dt_max=0.1,
            rank=1,
            trainable=None,
            lr=None,
            length_correction=False,
            stride=1,
            weight_decay=0.0, # weight decay on the SS Kernel
            precision=1,
            cache=False, # Cache the SS Kernel during evaluation
            # Arguments for FF
            activation='gelu', # activation in between SS and FF
            postact=None, # activation after FF
            weight_norm=False, # weight normalization on FF
            initializer=None, # initializer on FF
            input_linear=False,
            hyper_act=None,
            dropout=0.0,
            transposed=True, # axis ordering (B, L, D) or (B, D, L)
            resample=False,
            use_state=False,
            verbose=False,
            mode='nplr',
            keops=False,
            bidirectional=False,
            channels=1,
        ):
        """
        d_state: the dimension of the state, also denoted by N
        l_max: the maximum sequence length, also denoted by L
          if this is not known at model creation, or inconvenient to pass in,
          set l_max=None and length_correction=True
        dropout: standard dropout argument
        transposed: choose backbone axis ordering of (B, L, D) or (B, D, L) [B=batch size, L=sequence length, D=feature dimension]

        Other options are all experimental and should not need to be configured
        """

        super().__init__()
        if verbose:
            import src.utils.train
            log = src.utils.train.get_logger(__name__)
            log.info(f"Constructing s4 (H, N, L) = ({H}, {d_state}, {l_max})")

        self.h = H
        self.n = d_state if d_state > 0 else H
        self.stride = stride
        if l_max is not None and stride > 1:
            assert l_max % stride == 0
            l_max = l_max // self.stride
        self.cache = cache
        self.weight_decay = weight_decay
        self.transposed = transposed
        self.resample = resample
        self.bidirectional = bidirectional
        self.channels = channels
        self.L = l_max

        self.D = nn.Parameter(torch.randn(self.h, channels))

        if self.bidirectional:
            channels *= 2

        # SSM Kernel
        self.kernel = HippoSSKernel(self.n, self.h, l_max, dt_min=dt_min, dt_max=dt_max, measure=measure, rank=rank, trainable=trainable, lr=lr, length_correction=length_correction, precision=precision, cache=cache, mode=mode, resample=resample, keops=keops, channels=channels)
        # if self.bidirectional:
        #     self.kernel2 = HippoSSKernel(self.n, self.h, l_max, dt_min=dt_min, dt_max=dt_max, measure=measure, rank=rank, trainable=trainable, lr=lr, length_correction=length_correction, precision=precision, cache=cache, mode=mode, resample=resample, keops=keops, channels=channels)

        self.K = None # Cache the computed convolution filter if possible (during evaluation)

        # optional multiplicative modulation
        self.hyper = hyper_act is not None
        if self.hyper:
            self.hyper_linear = LinearActivation(
                self.h,
                self.h,
                transposed=True,
                initializer=initializer,
                activation=hyper_act,
                activate=True,
                weight_norm=weight_norm,
            )


        self.activation = Activation(activation)
        dropout_fn = nn.Dropout2d if self.transposed else nn.Dropout
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        self.output_linear = LinearActivation(
            # 2*self.h if self.bidirectional else self.h,
            self.h*self.channels,
            self.h,
            transposed=self.transposed,
            initializer=initializer,
            activation=postact,
            activate=True,
            weight_norm=weight_norm,
        )

        if use_state:
            self._initial_state = nn.Parameter(torch.zeros(self.h, self.n))


    def forward(self, u, state=None, cache=None, **kwargs): # absorbs return_output and transformer src mask
        """
        u: (B H L) if self.transposed else (B L H)
        state: (H N) never needed unless you know what you're doing

        Returns: same shape as u
        """
        if not self.transposed: u = u.transpose(-1, -2)
        L = u.size(-1)

        # Compute SS Kernel
        assert state is None
        k = self.kernel(L=self.L)

        k = k[..., :self.L]

        # Convolution

        if self.bidirectional:
            C = k.size(-2) // 2
            k = F.pad(k[..., :C, :], (0, 2*L-self.L)) \
                    + F.pad(k[..., C:, :].flip(-1), (2*L-self.L, 0)) \

        # TODO can padd to L + self.L
        k_f = torch.fft.rfft(k, n=2*L) # (H C L)
        u_f = torch.fft.rfft(u, n=2*L) # (B H L)
        y_f = k_f * u_f.unsqueeze(-2) # (B H C L)
        y = torch.fft.irfft(y_f, n=2*L)[..., :L] # (B H C L)

        # Compute D term in state space equation - essentially a skip connection
        # y = y + u * self.D.unsqueeze(-1)
        y = y + u.unsqueeze(-2) * self.D.unsqueeze(-1)

        # bidirectional attempt 2
        # if self.bidirectional:
        #     k2 = self.kernel2(L=L)[..., :L]
        #     k2_f = torch.fft.rfft(k2.flip(-1), n=2*L)
        #     y2_f = k2_f * u_f.unsqueeze(-2)
        #     y2 = torch.fft.irfft(y2_f, n=2*L)[..., L-1:-1]
        #     y2 = y2 + u.unsqueeze(-2) * self.D.unsqueeze(-1)
        #     y = torch.cat([y, y2], dim=-2)



        # Compute state update
        if state is not None:
            y = y + k_state[..., :L]
            next_state = self.kernel.next_state(state, u)
        else:
            next_state = None

        # Optional hyper-network multiplication
        if self.hyper:
            hyper = self.hyper_linear(u)
            y = hyper * y

        # Reshape to flatten channels
        y = rearrange(y, '... h c l -> ... (h c) l')

        y = self.dropout(self.activation(y))

        if not self.transposed: y = y.transpose(-1, -2)

        y = self.output_linear(y)

        return y
