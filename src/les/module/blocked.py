"""Bounded-storage multipole sums, recomputed through force-loss backward."""
from functools import partial

import torch
from torch.utils.checkpoint import checkpoint

from .make_kernels import make_kernels

# Default dense budget used by Ewald/PME: a graph whose pair (real space) or
# atom-by-mode (reciprocal) count fits is evaluated as one ordinary autograd
# block. Recomputing it through checkpointed VJPs would only add launch and
# triple-forward overhead without saving meaningful memory.
DEFAULT_DENSE_PAIRS = 65536


def _vjp(function, primals, cotangents):
    higher_order = torch.is_grad_enabled()
    with torch.enable_grad():
        # Separate clones take partials without following shared upstream dependencies.
        inputs = tuple(t.clone().requires_grad_(True) for t in primals)
        outputs = function(*inputs)
        gradients = torch.autograd.grad(outputs, inputs, cotangents,
                                        create_graph=higher_order, allow_unused=True)
    return tuple(torch.zeros_like(t) if g is None else g for t, g in zip(primals, gradients))


class _Recompute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, function, *args):
        ctx.function = function
        ctx.save_for_backward(*args)
        return function(*args)

    @staticmethod
    def backward(ctx, *cotangents):
        args = (ctx.function, ctx.saved_tensors, cotangents)
        if torch.is_grad_enabled():
            gradients = checkpoint(_vjp, *args, use_reentrant=False)
        else:
            gradients = _vjp(*args)
        return (None, *gradients)


def _reciprocal_block(r, q, u, quad, k, weight, potential, field):
    phase = r @ k.T
    cosine, sine = phase.cos(), phase.sin()
    if quad is not None:
        kk = (k[:, :, None] * k[:, None, :]).reshape(len(k), 9)
        qk2 = (quad.reshape(len(r) * q.shape[1], 9) @ kk.T).reshape(len(r), q.shape[1], len(k))
        amplitude = q[:, :, None] - .5 * qk2
    else:
        amplitude = q[:, :, None]
    if u is not None:
        uk = u @ k.T
        real = (amplitude * cosine[:, None] - uk * sine[:, None]).sum(0)
        imag = (amplitude * sine[:, None] + uk * cosine[:, None]).sum(0)
    else:
        real = (amplitude * cosine[:, None]).sum(0)
        imag = (amplitude * sine[:, None]).sum(0)
    energy = ((real.square() + imag.square()) * weight).sum(1)
    phi = q * 0
    electric = q.unsqueeze(-1).expand(*q.shape, 3) * 0 if u is None else u * 0
    if potential:
        phi = 2 * ((real * weight) @ cosine.T + (imag * weight) @ sine.T).T
    if field:
        electric = 2 * torch.einsum('ncm,md->ncd',
            weight * (real[None] * sine[:, None] - imag[None] * cosine[:, None]), k)
    return energy, phi, electric


def _run_reciprocal_block(has_u, has_quad, potential, field, *tensors):
    idx = 0
    r = tensors[idx]; idx += 1
    q = tensors[idx]; idx += 1
    u = tensors[idx] if has_u else None
    if has_u:
        idx += 1
    quad = tensors[idx] if has_quad else None
    if has_quad:
        idx += 1
    k = tensors[idx]; idx += 1
    weight = tensors[idx]
    return _reciprocal_block(r, q, u, quad, k, weight, potential, field)


def reciprocal_multipoles(r, q, u, quad, k, weight, potential, field, block_size=1024, dense_pairs=0):
    has_u, has_quad = u is not None, quad is not None
    zero_sum = r.sum() + q.sum() + k.sum() + weight.sum()
    if has_u:
        zero_sum = zero_sum + u.sum()
    if has_quad:
        zero_sum = zero_sum + quad.sum()
    zero = zero_sum * 0
    u_zero = q.new_zeros(*q.shape, 3) if not has_u else u * 0
    result = (q.sum(0) * 0 + zero, q * 0 + zero, u_zero + zero)
    if 0 < len(k) and len(r) * len(k) <= dense_pairs:
        out = _reciprocal_block(r, q, u, quad, k, weight, potential, field)
        return tuple(a + b for a, b in zip(result, out))
    function = partial(_run_reciprocal_block, has_u, has_quad, potential, field)
    # An empty block preserves the cross-input zero Hessian for force-loss backward.
    for start in range(0, max(1, len(k)), block_size):
        args = [r, q]
        if has_u:
            args.append(u)
        if has_quad:
            args.append(quad)
        args.extend([k[start:start + block_size], weight[start:start + block_size]])
        out = _Recompute.apply(function, *args)
        result = tuple(a + b for a, b in zip(result, out))
    return result


def _real_block(rs, rt, qs, qt, us, ut, Qs, Qt, source, target, sigma, norm, field, has_u, has_quad):
    mask = (torch.arange(len(rs), device=rs.device)[:, None] + source
            != torch.arange(len(rt), device=rt.device)[None] + target)
    fqq, fqu, fuu, fQu, fQQ = make_kernels(rs, sigma, norm, compute_u=has_u or field,
                                         compute_Q=has_quad, r_target=rt, mask_off=mask)
    phi_q = fqq.T @ qs
    phi = phi_q
    energy = (.5 * qt * phi_q).sum(0)
    electric = (torch.einsum('ic,ija->jca', qs, fqu) if field
                else (qt.unsqueeze(-1).expand(*qt.shape, 3) * 0 if ut is None else ut * 0))
    if has_u:
        phi_u = torch.einsum('ica,ija->jc', us, fqu)
        Eu = torch.einsum('ijab,ica->jcb', fuu, us)
        phi = phi + phi_u
        energy = energy + (qt * phi_u).sum(0) - .5 * (ut * Eu).sum((0, 2))
        if field:
            electric = electric + Eu
    if has_quad:
        phi_Q = .5 * torch.einsum('icab,ijab->jc', Qs, fuu)
        EQ = .5 * torch.einsum('icab,ijabd->jcd', Qs, fQu)
        phi = phi + phi_Q
        energy = energy + (qt * phi_Q).sum(0)
        if has_u:
            energy = energy - (ut * EQ).sum((0, 2))
        energy = energy + .125 * torch.einsum('icab,ijabde,jcde->c', Qs, fQQ, Qt)
        if field:
            electric = electric + EQ
    return energy, phi, electric


def _run_real_block(source, target, sigma, norm, field, has_u, has_quad, *tensors):
    idx = 0
    rs = tensors[idx]; idx += 1
    rt = tensors[idx]; idx += 1
    qs = tensors[idx]; idx += 1
    qt = tensors[idx]; idx += 1
    if has_u:
        us = tensors[idx]; idx += 1
        ut = tensors[idx]; idx += 1
    else:
        us = ut = None
    if has_quad:
        Qs = tensors[idx]; idx += 1
        Qt = tensors[idx]; idx += 1
    else:
        Qs = Qt = None
    return _real_block(rs, rt, qs, qt, us, ut, Qs, Qt, source, target, sigma, norm, field, has_u, has_quad)


def real_multipoles(r, q, u, quad, sigma, norm, field, block_size=64, dense_pairs=0):
    has_u, has_quad = u is not None, quad is not None
    zero_sum = r.sum() + q.sum()
    if has_u:
        zero_sum = zero_sum + u.sum()
    if has_quad:
        zero_sum = zero_sum + quad.sum()
    energy = q.sum(0) * 0 + zero_sum * 0
    if 0 < len(r) and len(r) * len(r) <= dense_pairs:
        out = _real_block(r, r, q, q, u, u, quad, quad, 0, 0, sigma, norm, field, has_u, has_quad)
        return energy + out[0], out[1], out[2]
    potentials, fields = [], []
    for target in range(0, len(r), block_size):
        stop = target + block_size
        phi = q[target:stop] * 0
        electric = q[target:stop].unsqueeze(-1).expand(*q[target:stop].shape, 3) * 0 if not has_u else u[target:stop] * 0
        for source in range(0, len(r), block_size):
            end = source + block_size
            function = partial(_run_real_block, source, target, sigma, norm, field, has_u, has_quad)
            args = [r[source:end], r[target:stop], q[source:end], q[target:stop]]
            if has_u:
                args.extend([u[source:end], u[target:stop]])
            if has_quad:
                args.extend([quad[source:end], quad[target:stop]])
            out = _Recompute.apply(function, *args)
            energy, phi, electric = energy + out[0], phi + out[1], electric + out[2]
        potentials.append(phi)
        fields.append(electric)
    return energy, torch.cat(potentials), torch.cat(fields)
