"""Aliases core still needs for TPU MoE kernels in :mod:`sglang_tpu`.

``mxfp4_tpu_util`` sizes MXFP4 MoE weights against the gmm_v2 tiling. Core may not
import the plugin, so it reaches the two kernels it needs through the aliases in
this package. Delete them once that weight preparation sits behind a platform hook.
"""
