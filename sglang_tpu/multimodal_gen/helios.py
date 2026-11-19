# SPDX-License-Identifier: Apache-2.0
"""Helios on TPU workers: RoPE without stride-2 lanes, DMD steps without host reads.

Runtime patch, tracked debt:
- Why: off CUDA, Helios rotates q and k with ``apply_rotary_emb_transposed``,
  which reads the head dim and the tables with stride-2 slices and writes the
  result back with strided scatters; torch_tpu re-lays out those lane slices
  in every block. Helios-Distilled's DMD step reads two device scalars back to
  the host per step (``torch.full`` filled from the step's timestep and the
  next one) and finds each sigma with an int64 ``torch.argmin`` on the device;
  torch_tpu eager aborts int64 reductions such as ``mask.sum()``
  ("Unsupported conversion from vmreg/vreg to U64").
  The UniPC schedule (Helios-Base, Helios-Mid) built on the TPU in float64
  ends in NaN where the CPU's last sigma is 0, so every Helios-Base video is
  black (NaN latents).
- How: :func:`install`
  (1) rebinds ``apply_rotary_emb_transposed`` in the Helios DiT module, for
      bfloat16 TPU tensors with float32 tables, to ``x * cos + (x @ R) * sin``
      with R the signed pair swap of ``sglang_tpu.multimodal_gen.rope`` and
      the tables' full-width halves. Helios tables repeat each frequency
      across its lane pair, so those halves hold the lanes the stock form
      reads: the same float32 products and sums, rounded once to bfloat16;
  (2) runs ``HeliosScheduler.set_timesteps`` for DMD and UniPC schedules on
      the host, copies the timesteps and sigmas (float64, as stock) to the TPU
      and keeps a host copy of the timesteps;
  (3) runs ``HeliosScheduler.step_dmd`` with each sigma index found on the
      host from that copy, with the stock rule: the timestep cast to int64,
      then the first nearest schedule entry. The device runs only the
      elementwise update.
- Upstream PR or issue: none.
- Remove when: (1) the diffusion kernel registry accepts a platform-provided
  Helios rotary, or torch_tpu lowers stride-2 lane slices without a relayout;
  (2) torch_tpu builds the float64 schedules as the CPU does, and (2) and (3)
  the stock ``step_dmd`` reads no device scalars and torch_tpu eager lowers
  int64 reductions.
"""

from __future__ import annotations

import torch

from sglang.multimodal_gen.runtime.models.dits import helios as dit
from sglang.multimodal_gen.runtime.models.schedulers import scheduling_helios
from sglang_tpu.multimodal_gen.rope import _pair_swap_matrix

# Devices that take these paths.
_DEVICE_TYPES = ("tpu",)

_Scheduler = scheduling_helios.HeliosScheduler
_STOCK_ROTARY = dit.apply_rotary_emb_transposed
_STOCK_SET_TIMESTEPS = _Scheduler.set_timesteps
_STOCK_STEP_DMD = _Scheduler.step_dmd


def apply_rotary_emb_transposed(hidden_states, freqs_cis):
    """``apply_rotary_emb_transposed`` as ``x * cos + (x @ R) * sin``."""
    x = hidden_states
    if (
        x.device.type not in _DEVICE_TYPES
        or x.dtype != torch.bfloat16
        or freqs_cis.dtype != torch.float32
    ):
        return _STOCK_ROTARY(hidden_states, freqs_cis)
    cos, sin = freqs_cis.unsqueeze(-2).chunk(2, dim=-1)
    rot = _pair_swap_matrix(x.shape[-1], x.dtype, x.device)
    return (x.float() * cos + torch.matmul(x, rot).float() * sin).to(x.dtype)


def set_timesteps(
    self,
    num_inference_steps,
    stage_index=None,
    device=None,
    sigmas=None,
    mu=None,
    is_amplify_first_chunk=False,
):
    """DMD and UniPC ``HeliosScheduler.set_timesteps`` on the host, copied to TPU
    devices."""
    on_tpu = (
        self.config.scheduler_type in ("dmd", "unipc")
        and device is not None
        and torch.device(device).type in _DEVICE_TYPES
    )
    _STOCK_SET_TIMESTEPS(
        self,
        num_inference_steps,
        stage_index,
        "cpu" if on_tpu else device,
        sigmas,
        mu,
        is_amplify_first_chunk,
    )
    self._tpu_host_schedule = self.timesteps if on_tpu else None
    if on_tpu:
        self.timesteps = self.timesteps.to(device)
        self.sigmas = self.sigmas.to(device)


def step_dmd(
    self,
    model_output,
    timestep=None,
    sample=None,
    return_dict=True,
    cur_sampling_step=0,
    dmd_noisy_tensor=None,
    dmd_sigmas=None,
    dmd_timesteps=None,
    all_timesteps=None,
    **kwargs,
):
    """``HeliosScheduler.step_dmd`` with the sigma indices found on the host.

    The stage passes the scheduler's own schedule, with ``timestep`` its entry
    ``cur_sampling_step``, as the stock step's next-timestep lookup assumes;
    any other schedule takes the stock step.
    """
    host_timesteps = getattr(self, "_tpu_host_schedule", None)
    if (
        host_timesteps is None
        or dmd_sigmas is not self.sigmas
        or dmd_timesteps is not self.timesteps
        or all_timesteps is not self.timesteps
    ):
        return _STOCK_STEP_DMD(
            self,
            model_output,
            timestep,
            sample,
            return_dict,
            cur_sampling_step,
            dmd_noisy_tensor,
            dmd_sigmas,
            dmd_timesteps,
            all_timesteps,
            **kwargs,
        )
    batch_size = model_output.shape[0]

    def sigma_at(sigmas, step):
        target = host_timesteps[step].to(torch.long)
        index = int(torch.argmin((host_timesteps - target).abs()))
        return sigmas[index].expand(batch_size).reshape(-1, 1, 1, 1, 1)

    # convert_flow_pred_to_x0, in float64.
    sigma_t = sigma_at(dmd_sigmas.double(), cur_sampling_step)
    prev_sample = sample.double() - sigma_t * model_output.double()
    prev_sample = prev_sample.to(model_output.dtype)
    if cur_sampling_step < len(all_timesteps) - 1:
        # add_noise toward the next timestep.
        sigma = sigma_at(dmd_sigmas, cur_sampling_step + 1)
        prev_sample = (1 - sigma) * prev_sample + sigma * dmd_noisy_tensor
        prev_sample = prev_sample.type_as(dmd_noisy_tensor)
    if not return_dict:
        return (prev_sample,)
    return scheduling_helios.HeliosSchedulerOutput(prev_sample=prev_sample)


def install() -> None:
    dit.apply_rotary_emb_transposed = apply_rotary_emb_transposed
    _Scheduler.set_timesteps = set_timesteps
    _Scheduler.step_dmd = step_dmd
