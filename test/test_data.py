# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT
"""Tests for cutout data preparation."""

from types import SimpleNamespace

import pandas as pd
import pytest
import xarray as xr

from atlite import data


@pytest.mark.parametrize("existing_file", [False, True])
def test_prepare_writes_each_module_and_cleans_up(tmp_path, monkeypatch, existing_file):
    path = tmp_path / "cutout.nc"
    initial = xr.Dataset(attrs={"prepared_features": []})
    if existing_file:
        initial.to_netcdf(path)
        initial = xr.open_dataset(path)
    cutout = SimpleNamespace(
        prepared=False, module=["first", "second"], data=initial, path=path
    )
    index = pd.MultiIndex.from_tuples(
        [("first", "feature_first"), ("second", "feature_second")],
        names=["module", "feature"],
    )
    target = pd.Series(["var_first", "var_second"], index=index)
    monkeypatch.setattr(data, "available_features", lambda modules: target)
    monkeypatch.setattr(
        data,
        "get_features",
        lambda cutout, module, features, **kwargs: xr.Dataset(
            {f"var_{module}": (("x",), [1.0])}
        ),
    )
    cleaned = []
    for module in cutout.module:
        monkeypatch.setitem(
            data.datamodules,
            module,
            SimpleNamespace(cleanup=lambda tmpdir, module=module: cleaned.append(module)),
        )

    result = data.cutout_prepare(cutout, tmpdir=tmp_path)

    try:
        assert result is cutout
        assert set(cutout.data.data_vars) == {"var_first", "var_second"}
        assert set(cutout.data.attrs["prepared_features"]) == {
            "feature_first",
            "feature_second",
        }
        assert cleaned == ["first", "second"]
    finally:
        cutout.data.close()


@pytest.mark.parametrize("fail", [False, True])
def test_prepare_cleanup_when_skipped_or_failed(tmp_path, monkeypatch, fail):
    variables = {} if fail else {"var_first": (("x",), [1.0])}
    initial = xr.Dataset(variables, attrs={"prepared_features": []})
    cutout = SimpleNamespace(
        prepared=False, module=["first"], data=initial, path=tmp_path / "cutout.nc"
    )
    index = pd.MultiIndex.from_tuples(
        [("first", "feature_first")], names=["module", "feature"]
    )
    monkeypatch.setattr(
        data, "available_features", lambda modules: pd.Series(["var_first"], index=index)
    )
    cleaned = []
    monkeypatch.setitem(
        data.datamodules,
        "first",
        SimpleNamespace(cleanup=lambda tmpdir: cleaned.append(tmpdir)),
    )

    def fail_get_features(*args, **kwargs):
        raise RuntimeError("download failed")

    monkeypatch.setattr(data, "get_features", fail_get_features)
    if fail:
        with pytest.raises(RuntimeError, match="download failed"):
            data.cutout_prepare(cutout, tmpdir=tmp_path)
    else:
        assert data.cutout_prepare(cutout, tmpdir=tmp_path) is cutout
    assert cleaned == [tmp_path]
