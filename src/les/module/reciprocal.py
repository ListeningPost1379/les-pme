"""Exact monopole sum with bounded storage through force-loss backward."""

import torch
from torch.utils.checkpoint import checkpoint


def _spectrum(r, q, k):
    phase = r @ k.T
    cosine, sine = phase.cos(), phase.sin()
    return cosine, sine, q.T @ cosine, q.T @ sine


def _gradient_block(r, q, k, weight, grad_energy):
    cosine, sine, real, imag = _spectrum(r, q, k)
    real_grad = 2 * grad_energy[:, None] * weight * real
    imag_grad = 2 * grad_energy[:, None] * weight * imag
    grad_q = cosine @ real_grad.T + sine @ imag_grad.T
    grad_phase = cosine * (q @ imag_grad) - sine * (q @ real_grad)
    return (
        grad_phase @ k,
        grad_q,
        grad_phase.T @ r,
        (grad_energy[:, None] * (real.square() + imag.square())).sum(0),
    )


class _ReciprocalEnergy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, q, k, weight):
        ctx.save_for_backward(r, q, k, weight)
        energy = q.new_zeros(q.shape[1])
        for start in range(0, len(k), 1024):
            _, _, real, imag = _spectrum(r, q, k[start:start + 1024])
            energy = energy + ((real.square() + imag.square()) * weight[start:start + 1024]).sum(1)
        return energy

    @staticmethod
    def backward(ctx, grad_energy):
        r, q, k, weight = ctx.saved_tensors
        if len(k) == 0:
            # A zero force still needs a graph for force-loss backward.
            zero = (r.sum() + q.sum() + k.sum() + weight.sum() + grad_energy.sum()) * 0
            return tuple(torch.zeros_like(tensor) + zero for tensor in (r, q, k, weight))
        grad_r, grad_q = torch.zeros_like(r), torch.zeros_like(q)
        grad_k, grad_weight = [], []
        for start in range(0, len(k), 1024):
            args = (r, q, k[start:start + 1024], weight[start:start + 1024], grad_energy)
            if torch.is_grad_enabled():
                # Checkpoint the analytic force too: force training retains its graph.
                dr, dq, dk, dw = checkpoint(_gradient_block, *args, use_reentrant=False)
            else:
                dr, dq, dk, dw = _gradient_block(*args)
            grad_r, grad_q = grad_r + dr, grad_q + dq
            grad_k.append(dk)
            grad_weight.append(dw)
        return (
            grad_r,
            grad_q,
            torch.cat(grad_k),
            torch.cat(grad_weight),
        )


reciprocal_energy = _ReciprocalEnergy.apply
