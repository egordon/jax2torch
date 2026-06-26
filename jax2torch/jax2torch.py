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

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)


def j2t(x_jax):
    # to_dlpack is now deprecated, can pass in directly
    x_torch = torch_dlpack.from_dlpack(x_jax)
    return x_torch


def t2j(x_torch):
    # Needs to be detached before DLPack
    x_torch = x_torch.detach().contiguous()  # https://github.com/google/jax/issues/8082
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
            generate_vmap_rule = True

            @staticmethod
            def forward(*args):
                # Vmap behavior
                # TODO: Make General
                if torch._C._functorch.is_batchedtensor(args[0]):
                    level = torch._C._functorch.maybe_get_level(args[0])
                    bdim = torch._C._functorch.maybe_get_bdim(args[0])
                    unwrap_args = []
                    for arg in args:
                        if torch._C._functorch.is_batchedtensor(arg):
                            unwrap_args.append(torch._C._functorch.get_unwrapped(arg))
                        else:
                            unwrap_args.append(arg)
                    y_ = jax.vmap(fn)(*tree_t2j(unwrap_args))
                    ret_unwrap = tree_j2t(y_)
                    return torch._C._functorch._add_batch_dim(ret_unwrap, bdim, level)

                # Normal Behavior
                jax_args = tree_t2j(args)
                y_ = fn(*jax_args)
                # y_, _ = jax.vjp(fn, *args)
                return tree_j2t(y_)

            @staticmethod
            def setup_context(ctx, inputs, outputs):
                # Vmap behavior
                # TODO: Make General
                if torch._C._functorch.is_batchedtensor(inputs[0]):
                    level = torch._C._functorch.maybe_get_level(inputs[0])
                    bdim = torch._C._functorch.maybe_get_bdim(inputs[0])
                    # unwrap_args = [torch._C._functorch.get_unwrapped(inp) for inp in inputs]
                    unwrap_args = []
                    for arg in inputs:
                        if torch._C._functorch.is_batchedtensor(arg):
                            unwrap_args.append(torch._C._functorch.get_unwrapped(arg))
                        else:
                            unwrap_args.append(arg)
                    jaxargs = tree_t2j(unwrap_args)
                    ctx.fun_vjp = jax.vjp(jax.vmap(fn), *jaxargs)[1]
                    ctx.batch_vjp = ctx.fun_vjp
                    ctx.batch_size = jaxargs[0].shape[0]
                elif any(inp.requires_grad for inp in inputs):
                    # Normal Behavior
                    jaxargs = tree_t2j(inputs)
                    ctx.fun_vjp = jax.vjp(fn, *jaxargs)[1]
                    ctx.batch_vjp = jax.vmap(ctx.fun_vjp)
                    ctx.batch_size = 0

            @staticmethod
            def backward(ctx, *grad_args):
                # Check for batched tensor and unwrap
                batch_args = grad_args if len(grad_args) > 1 else grad_args[0]
                # Vmap operation
                if torch._C._functorch.is_batchedtensor(batch_args):
                    level = torch._C._functorch.maybe_get_level(batch_args)
                    bdim = torch._C._functorch.maybe_get_bdim(batch_args)
                    unwrap_args = torch._C._functorch.get_unwrapped(batch_args)
                    batch_vjp = ctx.batch_vjp
                    for _ in range(level - 1):
                        unwrap_args = torch._C._functorch.get_unwrapped(unwrap_args)
                        batch_vjp = jax.vmap(batch_vjp)
                    jaxargs = tree_t2j(unwrap_args)
                    ## TODO: HACK handle independent batch dimension
                    ## 39 == self.space.n_x + n_outs
                    """
                    if ctx.batch_size > 0 and level > 1 and jaxargs.shape[0] % ctx.batch_size == 0 and jaxargs.shape[0] // ctx.batch_size == 39:
                        newargs = jnp.transpose(jnp.diagonal(jaxargs.reshape((ctx.batch_size, -1) + jaxargs.shape[1:]), axis1=0, axis2=2), axes=(0, 2, 1))
                        grads_new = batch_vjp(newargs)
                        grads_new = tree_j2t(grads_new)
                        # Reshape as input
                        rets_new = []
                        for grad in grads_new:
                            if grad.ndim == 4:
                                ret_trans = grad.transpose(1, 2).transpose(2, 3)
                            elif grad.ndim == 3:
                                ret_trans = grad.transpose(1, 2)
                            rets_new.append(torch.diag_embed(ret_trans, dim1=0, dim2=2).reshape((jaxargs.shape[0],) + grad.shape[1:]))
                        rets = rets_new
                    ## END HACK
                    else:
                    """
                    grads = batch_vjp(jaxargs)
                    grads = tuple(
                        map(lambda t: t if isinstance(t, jnp.ndarray) else None, grads)
                    )
                    rets = tree_j2t(grads)

                    for lvl in range(level):
                        rets = tuple(
                            torch._C._functorch._add_batch_dim(ret, bdim, lvl + 1)
                            for ret in rets
                        )
                    return rets
                # Normal operation
                grad_args = (
                    tree_t2j(grad_args) if len(grad_args) > 1 else t2j(grad_args[0])
                )
                grads = ctx.fun_vjp(grad_args)
                grads = tuple(
                    map(lambda t: t if isinstance(t, jnp.ndarray) else None, grads)
                )
                return tree_j2t(grads)

        sig = signature(fn)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        return JaxFun.apply(*bound.arguments.values())

    return inner
