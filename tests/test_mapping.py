import xarray as xr
import numpy as np
import scipy.sparse as sp
import pytest
from pathlib import Path
from mom6_forge.mapping import (
    compute_cressman_weights,
    dst_to_source,
    source_to_dst,
    regrid_dataset_via_cressman,
    _make_subgrid_points,
    regrid_with_subsampling,
    write_mapping_file,
)
from mom6_forge._supergrid import haversine
from mom6_forge.grid import Grid
from mom6_forge import mapping

from utils import fetch_inputdata


@pytest.fixture(scope="module")
def tx2_3_mesh():
    """tx2_3v2 ESMF mesh, fetched from CESM input data if not available locally."""
    return fetch_inputdata("share/meshes/tx2_3v2_230415_ESMFmesh.nc")


@pytest.fixture(scope="module")
def rof_ocn_meshes():
    """(runoff, ocean) ESMF mesh pair for end-to-end runoff mapping."""
    return (
        fetch_inputdata("share/meshes/rx1_nomask_181022_ESMFmesh.nc"),
        fetch_inputdata("share/meshes/gx1v7_151008_ESMFmesh.nc"),
    )


def make_synthetic_grids():
    """Tiny 8x8 source, 2x2 destination — all ocean, known depths."""
    src_lon = np.linspace(0.5, 7.5, 8)
    src_lat = np.linspace(0.5, 7.5, 8)
    src_lon_2d, src_lat_2d = np.meshgrid(src_lon, src_lat)

    # constant depth so remapped value is exactly known
    src_depth = np.full((8, 8), 1000.0)

    src_ds = xr.Dataset(
        {"depth": (["lat", "lon"], src_depth)},
        coords={"lon": src_lon, "lat": src_lat},
    )

    # 2x2 destination centred in the source domain
    dst_lon = np.array([[2.0, 6.0], [2.0, 6.0]])
    dst_lat = np.array([[2.0, 2.0], [6.0, 6.0]])
    dst_area = np.full((2, 2), 1.2e11)
    dst_mask = np.ones((2, 2), dtype=bool)

    dst_ds = xr.Dataset(
        {
            "lon": (["y", "x"], dst_lon),
            "lat": (["y", "x"], dst_lat),
            "area": (["y", "x"], dst_area),
            "mask": (["y", "x"], dst_mask),
        }
    )

    return src_ds, dst_ds


def test_compute_cressman_weights_correctness():
    src_ds, dst_ds = make_synthetic_grids()
    ds_w = compute_cressman_weights(src_ds, dst_ds, smooth_scl=2.0)

    # --- shape metadata is present and consistent ---
    assert ds_w.sizes["n_a"] == 64, "n_a should be 8*8=64"
    assert ds_w.sizes["n_b"] == 4, "n_b should be 2*2=4"
    assert int(ds_w["nj_a"].values) == 8
    assert int(ds_w["ni_a"].values) == 8
    assert int(ds_w["nj_b"].values) == 2
    assert int(ds_w["ni_b"].values) == 2

    # --- weight sums to 1 for every filled destination cell ---

    row = ds_w["row"].values - 1
    col = ds_w["col"].values - 1
    data = ds_w["S"].values
    S = sp.coo_matrix(
        (data, (row, col)), shape=(ds_w.sizes["n_b"], ds_w.sizes["n_a"])
    ).tocsr()

    weight_sums = np.asarray(S.sum(axis=1)).ravel()
    filled = ~ds_w["unfilled"].values
    assert np.allclose(
        weight_sums[filled], 1.0, atol=1e-6
    ), f"Weight sums not ~1: {weight_sums}"

    # --- no negative weights ---
    assert (ds_w["S"].values >= 0).all(), "Negative weights found"

    # --- constant field reproduces exactly ---
    out = S @ np.ones(ds_w.sizes["n_a"])
    assert np.allclose(
        out[filled], 1.0, atol=1e-6
    ), f"Constant field not reproduced: {out}"

    # --- remapped depth of constant 1000m field is 1000m ---
    depth_out = S @ np.ones(ds_w.sizes["n_a"]) * 1000.0
    assert np.allclose(
        depth_out[filled], 1000.0, atol=1e-3
    ), f"Depth not reproduced: {depth_out}"

    # --- no unfilled cells (all ocean, generous radius) ---
    assert not ds_w["unfilled"].values.any(), "Unexpected unfilled cells"

    # pick dst cell (0,0) — centred at lon=2, lat=2
    dst_flat = 0
    row_vec = S.getrow(dst_flat)
    src_indices = row_vec.nonzero()[1]
    weights = np.asarray(row_vec[0, src_indices].todense()).ravel()

    # compute great-circle distances from dst centre to each contributing source pixel
    dst_lon = ds_w["xc_b"].values[dst_flat]
    dst_lat = ds_w["yc_b"].values[dst_flat]
    src_lons = ds_w["xc_a"].values[src_indices]
    src_lats = ds_w["yc_a"].values[src_indices]

    distances = haversine(src_lats, src_lons, dst_lat, dst_lon, R=6.371e6)

    # sort by distance and check weights are non-increasing
    order = np.argsort(distances)
    sorted_weights = weights[order]
    sorted_distances = distances[order]

    assert np.all(
        np.diff(sorted_weights) <= 1e-10
    ), f"Weights not monotonically decreasing with distance:\n  distances={sorted_distances}\n  weights={sorted_weights}"


def test_regrid_dataset_via_cressman_smoke(tmp_path):
    src_ds, dst_ds = make_synthetic_grids()

    weights_path = tmp_path / "weights.nc"
    output_path = tmp_path / "regridded.nc"

    depth_dst, unfilled = regrid_dataset_via_cressman(
        src_ds,
        dst_ds,
        weights_path=weights_path,
        output_path=output_path,
        write_to_file=True,
    )

    # --- returned arrays have right shape ---
    assert depth_dst.depth.shape == (2, 2), f"Wrong shape: {depth_dst.depth.shape}"
    assert unfilled.shape == (2, 2), f"Wrong shape: {unfilled.shape}"

    # --- weights file was written ---
    assert weights_path.exists(), "Weights file not written"

    # --- no unfilled cells ---
    assert not unfilled.any(), "Unexpected unfilled cells"

    # --- output file written if requested ---
    assert output_path.exists(), "Output file not written"


def test_smoke_weight_lookups():
    """
    Smoke test for source_to_dst and dst_to_source using a tiny synthetic grid
    where the correct answer is known analytically.
    """

    # --- tiny 4x4 source grid, 2x2 destination grid ---
    src_lon = np.array([0.0, 1.0, 2.0, 3.0])
    src_lat = np.array([0.0, 1.0, 2.0, 3.0])
    src_lon_2d, src_lat_2d = np.meshgrid(src_lon, src_lat)

    # all ocean, depth = 1000m everywhere so weights are easy to reason about
    src_depth = np.full((4, 4), 1000.0)

    src_ds = xr.Dataset(
        {"depth": (["lat", "lon"], src_depth)},
        coords={"lon": src_lon, "lat": src_lat},
    )

    # 2x2 destination grid centred on the source grid
    dst_lon = np.array([[1.0, 2.0], [1.0, 2.0]])
    dst_lat = np.array([[1.0, 1.0], [2.0, 2.0]])
    dst_area = np.full((2, 2), 1.2e11)  # ~approx area for 1° cell in m²
    dst_mask = np.ones((2, 2), dtype=bool)

    dst_ds = xr.Dataset(
        {
            "lon": (["y", "x"], dst_lon),
            "lat": (["y", "x"], dst_lat),
            "area": (["y", "x"], dst_area),
            "mask": (["y", "x"], dst_mask),
        }
    )

    # --- compute weights ---
    print("Building synthetic weight dataset...")
    ds_w = compute_cressman_weights(
        src_ds, dst_ds, smooth_scl=0.5
    )  # use a small smoothing scale to get more localized weights and a more interesting test case
    print(
        f"  n_s={ds_w.sizes['n_s']}, n_a={ds_w.sizes['n_a']}, n_b={ds_w.sizes['n_b']}"
    )
    print(f"  src_shape=({ds_w['nj_a'].values}, {ds_w['ni_a'].values})")
    print(f"  dst_shape=({ds_w['nj_b'].values}, {ds_w['ni_b'].values})")

    # --- test 1: dst_to_source ---
    print("\n--- dst_to_source(ds_w, (0, 0)) ---")
    src_indices, weights = dst_to_source(ds_w, (0, 0))
    assert len(src_indices) > 0, "dst cell (0,0) should have source pixels"
    assert np.isclose(
        weights.sum(), 1.0, atol=1e-6
    ), f"weights should sum to 1, got {weights.sum()}"
    print(f"  PASS: {len(src_indices)} source pixels, weight sum={weights.sum():.6f}")

    # --- test 2: source_to_dst ---
    print("\n--- source_to_dst(ds_w, (1, 1)) ---")
    # source pixel (1,1) is at lon=1, lat=1 — right on top of dst cell (0,0)
    # so it should have a high weight toward that cell
    dst_indices, weights = source_to_dst(ds_w, (1, 1))
    assert len(dst_indices) > 0, "source pixel (1,1) should feed at least one dst cell"
    assert all(w > 0 for w in weights), "all weights should be positive"
    print(f"  PASS: feeds {len(dst_indices)} dst cells")

    # --- test 3: round-trip consistency ---
    print("\n--- Round-trip consistency ---")
    # every source pixel that dst (0,0) draws from should list (0,0) as a destination
    src_indices_fwd, _ = dst_to_source(ds_w, (0, 0))
    for src_flat in src_indices_fwd:
        src_2d = np.unravel_index(src_flat, (4, 4))
        dst_indices_back, _ = source_to_dst(ds_w, src_2d)
        assert (
            0 in dst_indices_back
        ), f"src {src_2d} feeds dst (0,0) fwd but not bwd — inconsistency!"
    print(f"  PASS: all {len(src_indices_fwd)} source pixels point back to dst (0,0)")

    # --- test 4: constant field reproduces exactly ---
    print("\n--- Constant field reproduction ---")
    import scipy.sparse as sp

    row = ds_w["row"].values - 1
    col = ds_w["col"].values - 1
    data = ds_w["S"].values
    S = sp.coo_matrix(
        (data, (row, col)), shape=(ds_w.sizes["n_b"], ds_w.sizes["n_a"])
    ).tocsr()
    out = S @ np.ones(ds_w.sizes["n_a"])
    assert np.allclose(
        out[dst_mask.ravel()], 1.0, atol=1e-6
    ), f"constant field not reproduced: {out}"
    print(f"  PASS: constant field → {out} (all ~1.0 for ocean cells)")


def test_make_subgrid_points(get_simple_grid):
    # Test with a simple 2x2 grid and 2 sub-points per cell
    nx_sub = ny_sub = 2
    grid = get_simple_grid
    sub_lon, sub_lat = _make_subgrid_points(
        grid.qlon.values, grid.qlat.values, nx_sub, ny_sub
    )

    expected_sub_lon = np.array(
        [
            [[[4 / 3, 5 / 3], [4 / 3, 5 / 3]], [[7 / 3, 8 / 3], [7 / 3, 8 / 3]]],
            [[[4 / 3, 5 / 3], [4 / 3, 5 / 3]], [[7 / 3, 8 / 3], [7 / 3, 8 / 3]]],
        ]
    )
    expected_sub_lat = np.array(
        [
            [[[4 / 3, 4 / 3], [5 / 3, 5 / 3]], [[4 / 3, 4 / 3], [5 / 3, 5 / 3]]],
            [[[7 / 3, 7 / 3], [8 / 3, 8 / 3]], [[7 / 3, 7 / 3], [8 / 3, 8 / 3]]],
        ]
    )

    assert np.allclose(
        sub_lon, expected_sub_lon
    ), "Sub-grid longitudes do not match expected values."
    assert np.allclose(
        sub_lat, expected_sub_lat
    ), "Sub-grid latitudes do not match expected values."


def test_smoke_seams_and_global_make_subgrid_points(
    get_dateline_seam_grid, get_PM_seam_grid, get_simple_global_grid
):
    # Test with a simple 2x2 grid and 2 sub-points per cell
    nx_sub = ny_sub = 2
    grid = get_dateline_seam_grid
    sub_lon, sub_lat = _make_subgrid_points(
        grid.qlon.values, grid.qlat.values, nx_sub, ny_sub
    )
    grid = get_PM_seam_grid
    sub_lon, sub_lat = _make_subgrid_points(
        grid.qlon.values, grid.qlat.values, nx_sub, ny_sub
    )
    grid = get_simple_global_grid
    sub_lon, sub_lat = _make_subgrid_points(
        grid.qlon.values, grid.qlat.values, nx_sub, ny_sub
    )


def test_regrid_with_subsampling(get_simple_grid):
    # Test with a simple 2x2 grid and 2 sub-points per cell with data that lands exactly on the sub points (subtracted by 0.1 to show snapping to sub points)
    nx_sub = ny_sub = 2
    grid = get_simple_grid
    lon = [4 / 3, 5 / 3, 7 / 3, 8 / 3]
    lat = [4 / 3, 5 / 3, 7 / 3, 8 / 3]
    input_ds = xr.Dataset(
        {
            "data": (
                ["lon", "lat"],
                [
                    np.arange(1, 5, 1),
                    np.arange(1, 5, 1),
                    np.arange(1, 5, 1),
                    np.arange(1, 5, 1),
                ],
            )
        },
        coords={
            "lon": (["lon"], [x - 0.1 for x in lon]),
            "lat": (["lat"], [x - 0.1 for x in lat]),
        },
    )
    ds, _ = regrid_with_subsampling(
        input_ds, grid.qlon.values, grid.qlat.values, nx_sub, ny_sub
    )
    assert ds["data"].shape == (2, 2, 2, 2), "Output shape is incorrect."
    expected_data = np.array(
        [[[[1, 1], [2, 2]], [[1, 1], [2, 2]]], [[[3, 3], [4, 4]], [[3, 3], [4, 4]]]]
    )
    assert np.allclose(
        ds["data"].values, expected_data
    ), "Regridded data does not match expected values."


def test_regrid_with_subsampling_time_dim(get_simple_grid):
    nx_sub = ny_sub = 2
    grid = get_simple_grid
    lon = [4 / 3, 5 / 3, 7 / 3, 8 / 3]
    lat = [4 / 3, 5 / 3, 7 / 3, 8 / 3]
    spatial_data = np.array(
        [
            np.arange(1, 5, 1),
            np.arange(1, 5, 1),
            np.arange(1, 5, 1),
            np.arange(1, 5, 1),
        ],
        dtype=float,
    )
    nt = 2
    input_ds = xr.Dataset(
        {"data": (["time", "lon", "lat"], np.stack([spatial_data] * nt))},
        coords={
            "lon": (["lon"], [x - 0.1 for x in lon]),
            "lat": (["lat"], [x - 0.1 for x in lat]),
        },
    )
    ds, _ = regrid_with_subsampling(
        input_ds, grid.qlon.values, grid.qlat.values, nx_sub, ny_sub
    )
    assert ds["data"].shape == (
        nt,
        2,
        2,
        2,
        2,
    ), "Output shape with time dim is incorrect."
    expected_spatial = np.array(
        [[[[1, 1], [2, 2]], [[1, 1], [2, 2]]], [[[3, 3], [4, 4]], [[3, 3], [4, 4]]]]
    )
    for t in range(nt):
        assert np.allclose(
            ds["data"].values[t], expected_spatial
        ), f"Regridded data at t={t} does not match expected values."


# ---------------------------------------------------------------------------
# write_mapping_file mesh-shape lookup
# ---------------------------------------------------------------------------


def _write_esmf_mesh(grid, path):
    grid.supergrid.to_esmf_mesh(str(path), mask="all_unmasked")
    return path


def test_write_mapping_file_uses_shape_lookup_not_full_reconstruction(
    tmp_path, monkeypatch
):
    """write_mapping_file only ever needs each mesh's (nx, ny) shape - it must not
    reconstruct full mesh geometry (Topo.from_esmf_mesh) just to get that shape."""
    src_grid = Grid(
        resolution=1.0, xstart=0.0, lenx=3.0, ystart=0.0, leny=2.0, name="src_shape"
    )
    dst_grid = Grid(
        resolution=1.0, xstart=0.0, lenx=2.0, ystart=0.0, leny=2.0, name="dst_shape"
    )
    src_path = _write_esmf_mesh(src_grid, tmp_path / "src.nc")
    dst_path = _write_esmf_mesh(dst_grid, tmp_path / "dst.nc")

    def _boom(*args, **kwargs):
        raise AssertionError("write_mapping_file must not call Topo.from_esmf_mesh")

    monkeypatch.setattr("mom6_forge.topo.Topo.from_esmf_mesh", _boom)

    weights_coo = sp.coo_matrix(([1.0, 1.0], ([0, 1], [0, 1])), shape=(4, 6))
    out_path = tmp_path / "out.nc"
    write_mapping_file(
        src_mesh=str(src_path),
        dst_mesh=str(dst_path),
        filename=out_path,
        weights_coo=weights_coo,
    )

    ds = xr.open_dataset(out_path)
    assert list(ds["src_grid_dims"].values) == [3, 2]
    assert list(ds["dst_grid_dims"].values) == [2, 2]


# ---------------------------------------------------------------------------
# runoff mapping: flatten_to_mesh, coastline masking
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# flatten_to_mesh
# ---------------------------------------------------------------------------


def test_flatten_to_mesh_c_order():
    """flatten_to_mesh must use row-major (C) ordering."""
    field_2d = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    )
    result = mapping.flatten_to_mesh(field_2d)
    np.testing.assert_array_equal(result, [1, 2, 3, 4, 5, 6])


def test_flatten_to_mesh_dataarray():
    """flatten_to_mesh accepts xr.DataArray and returns a numpy array."""
    da = xr.DataArray(
        np.array([[10, 20], [30, 40]]),
        dims=("nlat", "nlon"),
    )
    result = mapping.flatten_to_mesh(da)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, [10, 20, 30, 40])


def test_flatten_to_mesh_roundtrip():
    """Reshape to (ny, nx) then flatten_to_mesh must recover the original 1D array."""
    original_1d = np.arange(24)
    field_2d = original_1d.reshape((4, 6), order="C")
    recovered = mapping.flatten_to_mesh(field_2d)
    np.testing.assert_array_equal(recovered, original_1d)


# ---------------------------------------------------------------------------
# grid_from_esmf_mesh / flatten_to_mesh roundtrip  (integration, real mesh)
# ---------------------------------------------------------------------------


def test_grid_from_esmf_mesh_flatten_to_mesh_mask_roundtrip(tx2_3_mesh):
    """grid_from_esmf_mesh followed by flatten_to_mesh must recover the
    original elementMask exactly."""
    mesh = xr.open_dataset(tx2_3_mesh)

    original_mask_1d = mesh["elementMask"].values  # shape (n_elements,)

    # 1D -> 2D
    grid_2d = mapping.grid_from_esmf_mesh(mesh)
    mask_2d = grid_2d["mask"]  # xr.DataArray, shape (ny, nx)

    # 2D -> 1D using the standardized helper
    recovered_mask_1d = mapping.flatten_to_mesh(mask_2d)

    assert recovered_mask_1d.shape == original_mask_1d.shape, (
        f"Shape mismatch: got {recovered_mask_1d.shape}, "
        f"expected {original_mask_1d.shape}"
    )
    np.testing.assert_array_equal(
        recovered_mask_1d,
        original_mask_1d,
        err_msg="Roundtrip 1D->2D->1D changed elementMask values",
    )


def test_generate_esmf_map_via_xesmf_coastline_masking_only_coastal_nonzero(
    monkeypatch,
):
    """When coastline_masking=True, nonzero destination rows in generated
    weights should only occur on coastal destination cells."""

    src_grid = xr.Dataset(
        data_vars={"mask": (("nlat", "nlon"), np.ones((2, 2), dtype=int))},
        coords={
            "lon": (("nlat", "nlon"), np.array([[0.0, 1.0], [2.0, 3.0]])),
            "lat": (("nlat", "nlon"), np.array([[10.0, 10.0], [11.0, 11.0]])),
        },
    )
    dst_grid = xr.Dataset(
        data_vars={"mask": (("nlat", "nlon"), np.ones((2, 3), dtype=int))},
        coords={
            "lon": (("nlat", "nlon"), np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])),
            "lat": (
                ("nlat", "nlon"),
                np.array([[40.0, 40.0, 40.0], [41.0, 41.0, 41.0]]),
            ),
        },
    )
    coastline_mask = xr.DataArray(
        np.array([[0, 1, 0], [0, 1, 0]], dtype=int),
        dims=("nlat", "nlon"),
    )

    def _fake_grid_from_esmf_mesh(mesh):
        return src_grid if "src" in str(mesh) else dst_grid

    def _fake_extract_coastline_mask(grid):
        return coastline_mask

    class _FakeRegridder:
        def __init__(self, ds_in, ds_out, **kwargs):
            # Build synthetic weights with nonzero rows only on active dst cells.
            active_rows = np.flatnonzero(
                mapping.flatten_to_mesh(ds_out["mask"].data != 0)
            )
            src_cols = np.arange(active_rows.size) % ds_in["mask"].size

            class _FakeSparse:
                def __init__(self, rows, cols):
                    self.data = np.ones(rows.size, dtype=float)
                    self.coords = np.vstack([rows, cols])
                    self.shape = (ds_out["mask"].size, ds_in["mask"].size)

            class _FakeWeights:
                def __init__(self, rows, cols):
                    self.data = _FakeSparse(rows, cols)

            self.weights = _FakeWeights(active_rows, src_cols)

    captured = {}

    def _fake_write_mapping_file(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(mapping, "grid_from_esmf_mesh", _fake_grid_from_esmf_mesh)
    monkeypatch.setattr(mapping, "extract_coastline_mask", _fake_extract_coastline_mask)
    monkeypatch.setattr(mapping, "is_mesh_cyclic_x", lambda *_: False)
    monkeypatch.setattr(mapping.xe, "Regridder", _FakeRegridder)
    monkeypatch.setattr(mapping, "write_mapping_file", _fake_write_mapping_file)

    mapping.generate_ESMF_map_via_xesmf(
        src_mesh_path="src_mesh.nc",
        dst_mesh_path="dst_mesh.nc",
        mapping_file="out.nc",
        method="nearest_d2s",
        map_overlap=False,
        coastline_masking=True,
    )

    assert "weights" in captured
    w = captured["weights"].data

    coastal_flat = np.flatnonzero(mapping.flatten_to_mesh(coastline_mask.data == 1))
    nonzero_rows = np.unique(w.coords[0])

    # Ensure all nonzero destination rows are coastal cells.
    assert set(nonzero_rows).issubset(set(coastal_flat))

    # And every non-coastal destination row has zero entries.
    row_nnz = np.bincount(w.coords[0], minlength=dst_grid["mask"].size)
    for row in set(range(dst_grid["mask"].size)) - set(coastal_flat):
        assert row_nnz[row] == 0


def test_generate_esmf_map_via_xesmf_coastline_masking_tx2_3v2(tmp_path, tx2_3_mesh):
    """Integration test on tx2_3v2 mesh: with coastline_masking=True,
    nonzero destination rows must be coastal cells."""

    mapping_file = tmp_path / "tx2_3v2_coast_nn.nc"

    mapping.generate_ESMF_map_via_xesmf(
        src_mesh_path=tx2_3_mesh,
        dst_mesh_path=tx2_3_mesh,
        mapping_file=mapping_file,
        method="nearest_d2s",
        area_normalization=False,
        map_overlap=False,
        coastline_masking=True,
    )

    ds_map = xr.open_dataset(mapping_file)
    dst_mesh = xr.open_dataset(tx2_3_mesh)
    try:
        dst_grid = mapping.grid_from_esmf_mesh(dst_mesh)

        # In ESMF map files, `row` indexes destination cells (1-based).
        mapped_dst_rows = ds_map["row"].data.astype(np.int64) - 1

        coastal_mask_2d = mapping.extract_coastline_mask(dst_grid)
        coastal_mask_1d = mapping.flatten_to_mesh(coastal_mask_2d == 1).astype(bool)
        coastal_rows = np.flatnonzero(coastal_mask_1d)

        # All mapped destination rows must be coastal.
        assert set(np.unique(mapped_dst_rows)).issubset(set(coastal_rows))

        # Guard against malformed row indexing before bincount.
        assert np.all(mapped_dst_rows >= 0)

        # All non-coastal destination rows must have zero entries in the weights.
        row_counts = np.bincount(mapped_dst_rows, minlength=dst_grid["mask"].size)
        for row in set(range(dst_grid["mask"].size)) - set(coastal_rows):
            assert row_counts[row] == 0
    finally:
        ds_map.close()
        dst_mesh.close()


def test_generate_esmf_map_via_xesmf_nonfast_path(monkeypatch):
    """Non-fast path should keep the original write_mapping_file(weights=...) flow."""

    grid = xr.Dataset(
        data_vars={"mask": (("nlat", "nlon"), np.ones((2, 2), dtype=int))},
        coords={
            "lon": (("nlat", "nlon"), np.array([[0.0, 1.0], [2.0, 3.0]])),
            "lat": (("nlat", "nlon"), np.array([[0.0, 0.0], [1.0, 1.0]])),
        },
    )

    def _fake_grid_from_esmf_mesh(mesh):
        return grid

    class _FakeRegridder:
        def __init__(self, ds_in, ds_out, **kwargs):
            self.weights = "sentinel-weights"

    captured = {}

    def _fake_write_mapping_file(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(mapping, "grid_from_esmf_mesh", _fake_grid_from_esmf_mesh)
    monkeypatch.setattr(mapping, "is_mesh_cyclic_x", lambda *_: False)
    monkeypatch.setattr(mapping.xe, "Regridder", _FakeRegridder)
    monkeypatch.setattr(mapping, "write_mapping_file", _fake_write_mapping_file)

    mapping.generate_ESMF_map_via_xesmf(
        src_mesh_path="src_mesh.nc",
        dst_mesh_path="dst_mesh.nc",
        mapping_file="out.nc",
        method="nearest_s2d",
        map_overlap=False,
        coastline_masking=True,
    )

    assert captured.get("weights") == "sentinel-weights"
    assert "weights_coo" not in captured


# ---------------------------------------------------------------------------
# gen_rof_maps end-to-end  (integration, real meshes)
# ---------------------------------------------------------------------------


def test_gen_rof_maps_end_to_end(tmp_path, rof_ocn_meshes):
    """Full runoff mapping pipeline on rx1 -> gx1v7.

    Covers generate_ESMF_map_via_xesmf with coastline masking,
    compute_smoothing_weights (topography-aware BFS), and write_mapping_file.
    """
    rof_mesh, ocn_mesh = rof_ocn_meshes
    rmax = fold = 500.0

    mapping.gen_rof_maps(
        rof_mesh, ocn_mesh, tmp_path, "rx1_to_g17", rmax=rmax, fold=fold
    )

    nn_path = tmp_path / "rx1_to_g17_nn.nc"
    sm_path = tmp_path / f"rx1_to_g17_r{rmax:.0f}_f{fold:.0f}_nnsm.nc"
    assert nn_path.exists(), f"missing {nn_path}"
    assert sm_path.exists(), f"missing {sm_path}"

    nn = xr.open_dataset(nn_path)
    sm = xr.open_dataset(sm_path)

    # Both maps describe the same source and destination grids.
    for v in ("src_grid_dims", "dst_grid_dims"):
        np.testing.assert_array_equal(nn[v].values, sm[v].values)

    n_dst = int(np.prod(nn["dst_grid_dims"].values))
    n_src = int(np.prod(nn["src_grid_dims"].values))

    for ds, label in ((nn, "nearest neighbor"), (sm, "smoothed")):
        row, col, S = ds["row"].values, ds["col"].values, ds["S"].values
        assert row.shape == col.shape == S.shape, label
        # Mapping files are 1-based and must stay inside the grids.
        assert row.min() >= 1 and row.max() <= n_dst, label
        assert col.min() >= 1 and col.max() <= n_src, label
        assert np.all(np.isfinite(S)), f"non-finite weights in {label} map"
        assert np.all(S >= 0.0), f"negative weights in {label} map"

    # Smoothing spreads each runoff cell over a neighborhood, so the smoothed
    # map must have strictly more nonzeros than the nearest-neighbor map.
    assert sm["S"].size > nn["S"].size

    # Smoothing redistributes runoff; it must not create or destroy any.
    # Column sums are preserved per source cell, weighted by destination area.
    area_b = nn["area_b"].values
    tot_nn = np.bincount(
        nn["col"].values - 1,
        weights=nn["S"].values * area_b[nn["row"].values - 1],
        minlength=n_src,
    )
    tot_sm = np.bincount(
        sm["col"].values - 1,
        weights=sm["S"].values * area_b[sm["row"].values - 1],
        minlength=n_src,
    )
    active = tot_nn > 0
    np.testing.assert_allclose(
        tot_sm[active],
        tot_nn[active],
        rtol=1e-10,
        err_msg="smoothing did not conserve area-weighted runoff",
    )

    nn.close()
    sm.close()
