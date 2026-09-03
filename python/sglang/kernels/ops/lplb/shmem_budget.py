"""Shared-memory budget accounting for the fused IPM kernel.

Fused layout (fp32), one block per LP, all state in shared memory::

    A        NC * NV       constraint matrix     (resident)
    ax2      NC * NV       A scaled by x^2       (resident)
    c        NV            cost vector           (resident)
    x        NV            IPM state             (resident)
    ata      NC * NC       KKT matrix / Cholesky factor
    rhs      NC            ax2c, then delta
    d        NV            aliased with r = A.T @ delta

    S_bytes = 4 * (2*NC*NV + NC*NC + 4*NV + 2*NC + 1) + 4

The last four bytes are the ``bool`` flag plus structure padding. Keep this
formula in sync with ``ipm_smem`` in ``csrc/lplb/ipm.cuh``. The previous
formula omitted ``ax2`` and therefore allowed oversized kernels to reach the
CUDA launch and fail with ``invalid argument``.

Dynamic shared-memory cap per block (with opt-in via
``cudaFuncAttributeMaxDynamicSharedMemorySize``):

    A100   SM_80   164 KB   practical 160 KB
    H100   SM_90   227 KB   practical 223 KB  <- default target
    H200   SM_90   227 KB
    H20    SM_90   227 KB
    B200   SM_100  228 KB   practical 224 KB
"""

from __future__ import annotations

from dataclasses import dataclass

# Per-block slack reserved for cuBLASDx workspace and CUDA runtime state.
_RUNTIME_PAD_BYTES = 256

# fp32
_BYTES_PER_ELEM = 4

# Practical per-block dynamic shmem caps (bytes)
GPU_BUDGETS_BYTES: dict[str, int] = {
    "a100": 160 * 1024,
    "h100": 223 * 1024,
    "h200": 223 * 1024,
    "h20": 223 * 1024,
    "b200": 224 * 1024,
}


@dataclass(frozen=True)
class ShmemBreakdown:
    nc: int
    nv: int
    a_bytes: int
    c_bytes: int
    x_bytes: int
    ata_bytes: int
    rhs_bytes: int
    d_bytes: int
    pad_bytes: int

    @property
    def total_bytes(self) -> int:
        return (
            self.a_bytes
            + self.c_bytes
            + self.x_bytes
            + self.ata_bytes
            + self.rhs_bytes
            + self.d_bytes
            + self.pad_bytes
        )

    def as_kib(self) -> float:
        return self.total_bytes / 1024.0


def shmem_bytes(nc: int, nv: int, bytes_per_elem: int = _BYTES_PER_ELEM) -> int:
    """Exact ``sizeof(ipm_smem<NC, NV>)`` plus runtime safety padding."""
    struct_bytes = bytes_per_elem * (
        2 * nc * nv + nc * nc + 4 * nv + 2 * nc + 1
    ) + 4
    return struct_bytes + _RUNTIME_PAD_BYTES


def breakdown(
    nc: int, nv: int, bytes_per_elem: int = _BYTES_PER_ELEM
) -> ShmemBreakdown:
    """Per-array byte breakdown — useful for debugging shmem pressure."""
    b = bytes_per_elem
    return ShmemBreakdown(
        nc=nc,
        nv=nv,
        a_bytes=b * 2 * nc * nv,
        c_bytes=b * nv,
        x_bytes=b * nv,
        ata_bytes=b * nc * nc,
        rhs_bytes=b * 2 * nc,
        d_bytes=b * 2 * nv,
        pad_bytes=b + 4 + _RUNTIME_PAD_BYTES,
    )


def gpu_budget_bytes(gpu: str) -> int:
    key = gpu.lower()
    if key not in GPU_BUDGETS_BYTES:
        raise ValueError(
            f"unknown gpu '{gpu}', expected one of {sorted(GPU_BUDGETS_BYTES)}"
        )
    return GPU_BUDGETS_BYTES[key]


def fits(nc: int, nv: int, gpu: str = "h100") -> bool:
    return shmem_bytes(nc, nv) <= gpu_budget_bytes(gpu)


def assert_fits(nc: int, nv: int, gpu: str = "h100") -> None:
    """Raise if the fused kernel will not fit on the target GPU."""
    used = shmem_bytes(nc, nv)
    cap = gpu_budget_bytes(gpu)
    if used > cap:
        raise ValueError(
            f"fused IPM kernel needs {used/1024:.1f} KiB of shared memory for "
            f"NC={nc}, NV={nv}, but {gpu} allows {cap/1024:.1f} KiB/block. "
            f"Either reduce problem size or switch to a tiled design."
        )


def max_nc_for_nv(nv: int, gpu: str = "h100") -> int:
    """Largest NC that fits for a given NV. Solves
        4 * (NC^2 + (2*NV+2)*NC + 4*NV + 1) + 4 + pad <= cap
    via the quadratic formula (monotone in NC). Returns 0 if even NC=1 overflows.
    """
    cap = gpu_budget_bytes(gpu)
    b = _BYTES_PER_ELEM
    # cap - pad >= b * (NC^2 + (NV+1)*NC + 3*NV)
    rhs = (cap - _RUNTIME_PAD_BYTES - 4) / b - 4 * nv - 1
    if rhs <= 0:
        return 0
    # NC^2 + (2*NV+2)*NC - rhs <= 0
    import math

    linear = 2 * nv + 2
    disc = linear**2 + 4 * rhs
    nc_max = int((-linear + math.sqrt(disc)) / 2.0)
    while nc_max > 0 and shmem_bytes(nc_max, nv) > cap:
        nc_max -= 1
    return max(nc_max, 0)


def report(nc: int, nv: int, gpu: str = "h100") -> str:
    """Human-readable summary — used by kernels on init for logging."""
    bd = breakdown(nc, nv)
    cap = gpu_budget_bytes(gpu)
    status = "FITS" if bd.total_bytes <= cap else "OVER BUDGET"
    return (
        f"[shmem] NC={nc} NV={nv} gpu={gpu} | "
        f"A={bd.a_bytes/1024:.1f}K "
        f"ata={bd.ata_bytes/1024:.1f}K "
        f"rest={(bd.c_bytes+bd.x_bytes+bd.rhs_bytes+bd.d_bytes)/1024:.1f}K | "
        f"total={bd.total_bytes/1024:.1f}K / {cap/1024:.1f}K  {status}"
    )
