# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
"""This module provides efficiency improvements over torch's einsum through caching."""

import functools
from typing import Any, Callable, List, Sequence

import opt_einsum
import torch
from gatr.utils.misc import minimum_autocast_precision
from gatr.utils.cache import gatr_cache


@minimum_autocast_precision(torch.float32)
def _einsum_with_path(equation: str, *operands: torch.Tensor, path: List[int]) -> torch.Tensor:
    """Computes einsum with a given contraction path."""

    # Justification: For the sake of performance, we need direct access to torch's private methods.

    # pylint:disable-next=protected-access
    return torch._VF.einsum(equation, operands, path=path)  # type: ignore[attr-defined]

@minimum_autocast_precision(torch.float32)
def _einsum_with_path_ignored(equation: str, *operands: torch.Tensor, **kwargs: Any):
    """Calls torch.einsum whilst dropping all kwargs.

    Allows use of hard-coded optimal contraction paths in `gatr_einsum_with_path` for
    non-compiling code whilst dropping the optimal contraction path for compiling code.
    """
    return torch.einsum(equation, *operands)

@minimum_autocast_precision(torch.float32)
def _cached_einsum(equation: str, *operands: torch.Tensor) -> torch.Tensor:
    """Computes einsum whilst caching the optimal contraction path.

    Inspired by upstream
    https://github.com/pytorch/pytorch/blob/v1.13.0/torch/functional.py#L381.
    """
    op_shape = tuple(op.shape for op in operands)
    path = _get_cached_path_for_equation_and_shapes(equation=equation, op_shape=op_shape)

    return _einsum_with_path(equation, *operands, path=path)

@minimum_autocast_precision(torch.float32)
@functools.lru_cache(maxsize=None)
def _get_cached_path_for_equation_and_shapes(
    equation: str, op_shape: Sequence[torch.Tensor]
) -> List[int]:
    """Provides shape-based caching of the optimal contraction path."""
    tupled_path = opt_einsum.contract_path(equation, *op_shape, optimize="optimal", shapes=True)[0]

    return [item for pair in tupled_path for item in pair]

_gatr_einsum = _cached_einsum
_gatr_einsum_with_path = _einsum_with_path

@minimum_autocast_precision(torch.float32)
def gatr_einsum(equation: str, *operands: torch.Tensor):
    """Computes torch.einsum with contraction path caching if enabled (and compilation is not used).

    Cf. `enable_cached_einsum` for more context.
    """
    return _gatr_einsum(equation, *operands)

@minimum_autocast_precision(torch.float32)
def gatr_einsum_with_path(equation: str, *operands: torch.Tensor, path: List[int]):
    """Computes einsum with a given contraction path (which is ignored when using compilation).

    Cf. `enable_cached_einsum` for more context.
    """
    return _gatr_einsum_with_path(equation, *operands, path=path)

@minimum_autocast_precision(torch.float32)
def enable_cached_einsum(flag: bool) -> None:
    """Selects whether to use caching of optimal paths in einsum contraction computations.

    When using torch.compile (torch==2.2.1), if we specify the precomputed paths when calling
    `torch._VF.einsum(equation, operands, path=path)`, the compiler errors out.

    Thus, users who wish to use `torch.compile` need to disable caching of einsum
    by calling `enable_cached_einsum(False)`.

    By default, caching is used, as we currently expect less users to use compilation.
    """
    global _gatr_einsum
    global _gatr_einsum_with_path
    if flag:
        _gatr_einsum = _cached_einsum
        _gatr_einsum_with_path = _einsum_with_path
    else:
        _gatr_einsum = torch.einsum
        _gatr_einsum_with_path = _einsum_with_path_ignored
