from typing import Dict
import torch
from torch import nn
from typing import List, Optional

def grad(
    y: torch.Tensor,
    x: torch.Tensor,
    training: bool = True,
    retain_graph: Optional[bool] = None,
) -> torch.Tensor:
    """
    a wrapper for the gradient calculation
    alow multiple dimensional and/or complex y
    y: [n_graphs, ] or [n_graphs, dim_y]
    x: [n_nodes, :]
    """
    if retain_graph is None:
        retain_graph = training
    if y.is_complex():
        get_imag = True
    else:
        get_imag = False

    if len(y.shape) == 1:
        grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(y)]
        gradient_real = torch.autograd.grad(
            outputs=[y],  # [n_graphs, ]
            inputs=[x],  # [n_nodes, 3]
            grad_outputs=grad_outputs,
            retain_graph=(retain_graph or get_imag),  # Make sure the graph is not destroyed during training
            create_graph=training,  # Create graph for second derivative
            allow_unused=True,  # For complete dissociation turn to true
        )[0]  # [n_nodes, 3]
        if gradient_real is None:
            gradient_real = torch.zeros_like(x)
        if get_imag:
            gradient_imag = torch.autograd.grad(
                outputs=[y/1j],  # [n_graphs, ]
                inputs=[x],  # [n_nodes, 3]
                grad_outputs=grad_outputs,
                retain_graph=retain_graph,  # Make sure the graph is not destroyed during training
                create_graph=training,  # Create graph for second derivative
                allow_unused=True,  # For complete dissociation turn to true
            )[0]  # [n_nodes, 3]
            if gradient_imag is None:
                gradient_imag = torch.zeros_like(x)
        else:
            gradient_imag = torch.tensor(0.0, dtype=x.dtype, device=x.device)
    else:
        dim_y = y.shape[1] 
        grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(y[:,0])]
        grad_list_real = []
        for i in range(dim_y):
            if y[:, i].requires_grad:
                g = torch.autograd.grad(
                    outputs=[y[:, i]],  # [n_graphs, ]
                    inputs=[x],         # [n_nodes, 3]
                    grad_outputs=grad_outputs,
                    retain_graph=(retain_graph or (i < dim_y - 1) or get_imag),
                    create_graph=training, # Create graph for second derivative
                    allow_unused=True, # For complete dissociation turn to true
                )[0]
            else:
                g = None
            if g is None:
                g = torch.zeros_like(x)
            grad_list_real.append(g)
        gradient_real = torch.stack(grad_list_real, dim=2)  # [n_nodes, 3, dim_y]
        # if y is complex, we need to calculate the imaginary part
        if get_imag:
            grad_list_imag = []
            for i in range(dim_y):
                if y[:, i].requires_grad:
                    g = torch.autograd.grad(
                        outputs=[y[:, i]/1j], # [n_graphs, ]
                        inputs=[x], # [n_nodes, 3]
                        grad_outputs=grad_outputs,
                        retain_graph=(retain_graph or (i < dim_y - 1)), # Make sure the graph is not destroyed during training
                        create_graph=training, # Create graph for second derivative
                        allow_unused=True, # For complete dissociation turn to true
                    )[0]
                else:
                    g = None
                if g is None:
                    g = torch.zeros_like(x)
                grad_list_imag.append(g)
            gradient_imag = torch.stack(grad_list_imag, dim=2)  # [n_nodes, 3, dim_y]
        else:
            gradient_imag = torch.tensor(0.0, dtype=x.dtype, device=x.device)

    if get_imag:
        return gradient_real + 1j * gradient_imag
    else:
        return gradient_real
