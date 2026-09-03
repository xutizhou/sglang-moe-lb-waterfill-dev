from sglang.kernels.ops.lplb.shmem_budget import fits, shmem_bytes


def test_ipm_shmem_budget_counts_both_nc_by_nv_matrices():
    nc, nv = 126, 256
    expected_struct_bytes = 4 * (
        2 * nc * nv + nc * nc + 4 * nv + 2 * nc + 1
    ) + 4

    assert shmem_bytes(nc, nv) == expected_struct_bytes + 256
    assert not fits(nc, nv, gpu="h20")


def test_ipm_r64_shape_fits_h20():
    assert fits(54, 120, gpu="h20")
