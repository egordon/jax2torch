# https://gist.github.com/mattjj/e8b51074fed081d765d2f3ff90edf0e9

import torch
from torch.utils import dlpack as torch_dlpack

import jax
from jax import dlpack as jax_dlpack
import jax.numpy as jnp
from jax.tree_util import tree_map

from inspect import signature
from functools import wraps

# TODO: figure out how to use UnTypedStorage
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

def j2t(x_jax):
    # to_dlpack is now deprecated, can pass in directly
    x_torch = torch_dlpack.from_dlpack(x_jax)
    return x_torch

def t2j(x_torch):
    # Needs to be detached before DLPack
    x_torch = x_torch.detach().contiguous() # https://github.com/google/jax/issues/8082
    # Unwrap Grad-Tracking Tensor: https://github.com/pytorch/pytorch/issues/91810
    if torch._C._functorch.is_gradtrackingtensor(x_torch):
        x_unwrap = torch._C._functorch.get_unwrapped(x_torch)
        x_jax = jnp.array(x_unwrap.storage().tolist()).reshape(x_unwrap.shape)
    else:
        x_jax = jax_dlpack.from_dlpack(x_torch)
    return x_jax

def tree_t2j(x_torch):
    return tree_map(lambda t: t2j(t) if isinstance(t, torch.Tensor) else t, x_torch)

def tree_j2t(x_jax):
    return tree_map(lambda t: j2t(t) if isinstance(t, jnp.ndarray) else t, x_jax)

def jax2torch(fn):
    @wraps(fn)
    def inner(*args, **kwargs):
        class JaxFun(torch.autograd.Function):
            @staticmethod
            def forward(*args):
                args = tree_t2j(args)
                y_, _ = jax.vjp(fn, *args)
                return tree_j2t(y_)

            @staticmethod
            def setup_context(ctx, inputs, _):
                jaxargs = tree_t2j(inputs)
                _, ctx.fun_vjp = jax.vjp(fn, *jaxargs)
                ctx.batch_vjp = jax.vmap(ctx.fun_vjp)

            @staticmethod
            def backward(ctx, *grad_args):
                # Check for batched tensor and unwrap
                batch_args = grad_args if len(grad_args) > 1 else grad_args[0]
                if torch._C._functorch.is_batchedtensor(batch_args):
                    level = torch._C._functorch.maybe_get_level(batch_args)
                    bdim = torch._C._functorch.maybe_get_bdim(batch_args)
                    unwrap_args = torch._C._functorch.get_unwrapped(batch_args)
                    grads = ctx.batch_vjp(tree_t2j(unwrap_args))
                    grads = tuple(map(lambda t: t if isinstance(t, jnp.ndarray) else None, grads))
                    ret_unwrap = tree_j2t(grads)
                    return tuple(torch._C._functorch._add_batch_dim(ret, bdim, level) for ret in ret_unwrap)
                # Normal operation
                grad_args = tree_t2j(grad_args) if len(grad_args) > 1 else t2j(grad_args[0])
                grads = ctx.fun_vjp(grad_args)
                grads = tuple(map(lambda t: t if isinstance(t, jnp.ndarray) else None, grads))
                return tree_j2t(grads)

        sig = signature(fn)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        return JaxFun.apply(*bound.arguments.values())
    return inner
