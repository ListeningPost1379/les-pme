import torch
from torch import nn
from typing import Dict, Any, Union, Optional

from .module.blocked import DEFAULT_DENSE_PAIRS

from .module import (
    Atomwise,
    Ewald,
    Ewald_vectorized,
    PME,
    BEC,
    FixedCharges,
    AtomicAlpha,
)

__all__ = ['Les']

class Les(nn.Module):

    __constants__ = ['use_fixed_atomic_charges', 'use_atomic_alpha', 'use_atomwise', 'use_epsilon_r_scaling']
    def __init__(self, les_arguments: Union[Dict[str, Any], str] = {}):
        """
        LES model for long-range interations
        """
        super().__init__()

        if isinstance(les_arguments, str):
            import yaml
            with open(les_arguments, 'r') as file:
                les_arguments = yaml.safe_load(file)
                if les_arguments is None:
                    les_arguments = {}

        self._parse_arguments(les_arguments)

        self.atomwise: nn.Module = (
            Atomwise(
                n_layers=self.n_layers,
                n_hidden=self.n_hidden,
                add_linear_nn=self.add_linear_nn,
                output_scaling_factor=self.output_scaling_factor,
            )
            if self.use_atomwise
            else _DummyAtomwise()
        )

        if self.use_fixed_atomic_charges:
            self.fixed_charges = FixedCharges(normalization_factor=self.fixed_atomic_charges_scaling_factor)
        if self.use_atomic_alpha:
            self.atomic_alpha = AtomicAlpha()

        reciprocal_solver = les_arguments.get('reciprocal_solver', 'direct')
        if reciprocal_solver not in ('direct', 'original', 'blocked', 'pme'):
            raise ValueError("reciprocal_solver must be 'direct', 'original', 'blocked' or 'pme'")
        compute_dtype = les_arguments.get('compute_dtype', 'auto')
        if compute_dtype not in ('auto', 'float32', 'float64'):
            raise ValueError("compute_dtype must be 'auto', 'float32' or 'float64'")
        if (compute_dtype != 'auto' or reciprocal_solver in ('original', 'blocked', 'pme')) and les_arguments.get('reciprocal_mode', 'legacy') not in ('legacy', 'upstream'):
            raise ValueError('Explicit solver/precision selection requires the legacy reciprocal kernel')
        if reciprocal_solver != 'pme' and any(key in les_arguments for key in ('pme_mesh', 'pme_cutoff_dtype')):
            raise ValueError('PME options require reciprocal_solver=pme')
        dense_pairs = les_arguments.get('dense_pairs', DEFAULT_DENSE_PAIRS)
        if reciprocal_solver == 'pme':
            if self.is_periodic is not None or les_arguments.get('reciprocal_mode', 'legacy') not in ('legacy', 'upstream'):
                raise ValueError('PME requires automatic boundaries and the legacy reciprocal kernel')
            self.ewald = PME(mesh=les_arguments.get('pme_mesh', 'auto'), sigma=self.sigma,
                             dl=self.dl, remove_self_interaction=self.remove_self_interaction,
                             compute_dtype=compute_dtype,
                             pme_cutoff_dtype=les_arguments.get('pme_cutoff_dtype', 'float64'),
                             dense_pairs=dense_pairs,
                             use_epsilon_r_scaling=self.use_epsilon_r_scaling)
        elif self.is_periodic is not None:
            if compute_dtype != 'auto' or reciprocal_solver == 'blocked':
                raise ValueError('Explicit precision and blocked require automatic boundaries (is_periodic=None)')
            self.ewald = Ewald_vectorized( # torch.compile compatible
                sigma=self.sigma, # default 1.0
                dl=self.dl, # default 2.0
                is_periodic=self.is_periodic, # True: periodic, False: non-periodic
                N_max=self.N_max, # default 10, recommend N_max * dl > cell norm for accuracy
                remove_self_interaction=self.remove_self_interaction,
                use_epsilon_r_scaling=self.use_epsilon_r_scaling,
                )
            self.ewald.reciprocal_solver = reciprocal_solver
        else:
            self.ewald = Ewald( # legacy module, not torch.compile compatible, supports mixed datasets
                sigma=self.sigma,
                dl=self.dl,
                remove_self_interaction=self.remove_self_interaction,
                use_epsilon_r_scaling=self.use_epsilon_r_scaling,
                reciprocal_solver=reciprocal_solver,
                compute_dtype=compute_dtype,
                dense_pairs=dense_pairs,
                )

        self.bec = BEC(
             remove_mean=self.remove_mean,
             epsilon_factor=self.epsilon_factor,
             )

    def _parse_arguments(self, les_arguments: Dict[str, Any]):
        """
        Parse arguments for LES model
        """
        self.n_layers = les_arguments.get('n_layers', 3)
        self.n_hidden = les_arguments.get('n_hidden', [32, 16])
        self.add_linear_nn = les_arguments.get('add_linear_nn', True)
        self.output_scaling_factor = les_arguments.get('output_scaling_factor', 0.1)

        self.sigma = les_arguments.get('sigma', 1.0)
        self.dl = les_arguments.get('dl', 2.0)
        self.remove_self_interaction = les_arguments.get('remove_self_interaction', True)


        self.is_periodic = les_arguments.get('is_periodic', None) # set is_periodic (True/False) to use the vectorized, torch.compile
        self.N_max = les_arguments.get('N_max', 10)

        self.remove_mean = les_arguments.get('remove_mean', True)
        self.epsilon_factor = les_arguments.get('epsilon_factor', 1.)
        self.use_atomwise = les_arguments.get('use_atomwise', False)
        self.use_fixed_atomic_charges = les_arguments.get('use_fixed_atomic_charges', False)
        self.fixed_atomic_charges_scaling_factor = les_arguments.get('fixed_atomic_charges_scaling_factor', 0.5)
        self.use_atomic_alpha = les_arguments.get('use_atomic_alpha', False)
        self.use_epsilon_r_scaling = les_arguments.get('use_epsilon_r_scaling', False)

    def __setstate__(self, state: Dict[str, Any]):
        # Backward compatibility
        for key, default in {
            'use_atomwise': False,
            'use_fixed_atomic_charges': False,
            'use_atomic_alpha': False,
            'use_epsilon_r_scaling': False,
            'is_periodic': None,
            'N_max': 10,
        }.items():
            state.setdefault(key, default)
        super().__setstate__(state)

    def forward(self,
               positions: torch.Tensor, # [n_atoms, 3]
               cell: torch.Tensor, # [batch_size, 3, 3]
               e_ext: Optional[torch.Tensor]= None,
               desc: Optional[torch.Tensor]= None, # [n_atoms, n_features]
               latent_charges: Optional[torch.Tensor] = None, # [n_atoms, ]
               latent_dipoles: Optional[torch.Tensor] = None, # [n_atoms, 3]
               latent_quads: Optional[torch.Tensor] = None, # [n_atoms, 3, 3]
               latent_kappas: Optional[torch.Tensor] = None, # [n_atoms, ]
               latent_alphas: Optional[torch.Tensor] = None, # [n_atoms, ]
               atomic_numbers: Optional[torch.Tensor] = None, # [n_atoms, ]
               batch: Optional[torch.Tensor] = None,
               compute_energy: bool = True,
               compute_field: bool = False,
               compute_bec: bool = False,
               bec_output_index: Optional[int] = None, # option to compute BEC components along only one direction
               ) -> Dict[str, Optional[torch.Tensor]]:
        # check the input shapes
        if batch is None:
            batch = torch.zeros(positions.shape[0], dtype=torch.int64, device=positions.device)

        if latent_charges is not None:
            # check the shape of latent charges
            assert latent_charges.shape[0] == positions.shape[0]
        elif desc is not None and latent_charges is None:
            if not self.use_atomwise:
                raise ValueError("desc must be provided and use_atomwise must be True if latent_charges is not provided")
            # compute the latent charges
            assert desc.shape[0] == positions.shape[0]
            latent_charges = self.atomwise(desc, batch)
        else:
            raise ValueError("Either desc or latent_charges must be provided")

        if atomic_numbers is not None and self.use_fixed_atomic_charges:
            fixed_q = self.fixed_charges(atomic_numbers).to(dtype=latent_charges.dtype)
            if latent_charges.dim() == 2:
                fixed_q = fixed_q.unsqueeze(1)
            latent_charges = latent_charges + fixed_q

        if atomic_numbers is not None and self.use_atomic_alpha and latent_alphas is not None:
            baseline_alphas = self.atomic_alpha(atomic_numbers).to(dtype=latent_alphas.dtype)
            if latent_alphas.dim() == 1:
                latent_alphas = latent_alphas + baseline_alphas
            elif latent_alphas.dim() == 2:
                latent_alphas = latent_alphas + baseline_alphas.unsqueeze(1)
            elif latent_alphas.dim() == 3:
                eye = torch.eye(3, device=baseline_alphas.device, dtype=latent_alphas.dtype).unsqueeze(0)
                latent_alphas = latent_alphas + baseline_alphas[:, None, None] * eye
            elif latent_alphas.dim() == 4:
                eye = torch.eye(3, device=baseline_alphas.device, dtype=latent_alphas.dtype).unsqueeze(0).unsqueeze(0)
                latent_alphas = latent_alphas + baseline_alphas[:, None, None, None] * eye


        # compute the long-range interactions
        if compute_energy:
            E_lr, q_induced, u_induced = self.ewald(q=latent_charges,
                              u=latent_dipoles,
                              kappa=latent_kappas,
                              alpha=latent_alphas,
                              quad=latent_quads,
                              r=positions,
                              cell=cell,
                              batch=batch,
                              compute_field=compute_field,
                              e_ext = e_ext,
                              )
        else:
            E_lr, q_induced, u_induced = None, None, None

        if latent_alphas is not None and u_induced is not None:
            if latent_dipoles is not None:
                if latent_dipoles.dim() == 2 and u_induced.dim() == 3 and u_induced.shape[1] == 1:
                    u_induced = u_induced.squeeze(1)
                elif latent_dipoles.dim() == 3 and u_induced.dim() == 2:
                    u_induced = u_induced.unsqueeze(1)
                assert latent_dipoles.shape == u_induced.shape, 'latent_dipoles dimension error'
                latent_dipoles = latent_dipoles + u_induced
            else:
                latent_dipoles = u_induced

        if latent_kappas is not None and q_induced is not None:
            if latent_charges.dim() == 1 and q_induced.dim() == 2 and q_induced.shape[1] == 1:
                q_induced = q_induced.squeeze(1)
            elif latent_charges.dim() == 2 and q_induced.dim() == 1:
                q_induced = q_induced.unsqueeze(1)
            assert latent_charges.shape == q_induced.shape, 'latent_charges dimension error'
            latent_charges = latent_charges + q_induced

        # compute the BEC
        if compute_bec:
            bec = self.bec(q=latent_charges,
                           u=latent_dipoles,
                           r=positions,
                           cell=cell,
                           batch=batch,
                           output_index=bec_output_index,
		           )
        else:
            bec = None

        output = {
            'E_lr': E_lr,
            'latent_charges': latent_charges,
            'latent_dipoles': latent_dipoles,
            'latent_quads': latent_quads,
            'latent_alphas': latent_alphas,
            'BEC': bec,
            }
        return output

class _DummyAtomwise(nn.Module):
    def forward(self, desc: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        raise ValueError("set use_atomwise to True to use Atomwise module")
