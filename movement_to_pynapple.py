"""Convert a movement xarray.Dataset into pynapple time series.

Convenience converter inspired by pynapple-org/pynapple#617. Every data
variable in the movement dataset (``position``, ``confidence``, and any
derived variables you added such as ``head_position`` / ``head_direction``)
becomes a pynapple object, keyed by its name:

- a variable with only a ``time`` dimension  -> ``nap.Tsd``
- a variable with extra dimensions           -> ``nap.TsdFrame``
  (the extra dims are flattened into columns, e.g. ``x_left_ear``)
"""

import itertools

import pynapple as nap
import xarray as xr


def movement_to_nap(ds: xr.Dataset, individual: str | None = None) -> dict:
    """Convert a movement dataset to a dict of pynapple time series.

    Parameters
    ----------
    ds : xarray.Dataset
        A movement poses or bboxes dataset (optionally with derived variables).
    individual : str, optional
        Which individual to export. Required only if the dataset holds more
        than one; a single-individual dataset is selected automatically.

    Returns
    -------
    dict
        Maps each data-variable name to a ``nap.Tsd`` or ``nap.TsdFrame``.
    """
    t = ds.coords["time"].values
    return {
        name: _da_to_pynapple(da, individual, t)
        for name, da in ds.data_vars.items()
    }


def _da_to_pynapple(da, individual, t):
    if "individual" in da.dims:
        if da.sizes["individual"] == 1:
            da = da.isel(individual=0, drop=True)
        elif individual is None:
            raise ValueError(
                f"Dataset has {da.sizes['individual']} individuals; "
                "pass `individual=` to pick one."
            )
        else:
            da = da.sel(individual=individual, drop=True)

    # Order as (time, <other dims...>) so rows are time, rest becomes columns.
    other_dims = [d for d in da.dims if d != "time"]
    da = da.transpose("time", *other_dims)

    if not other_dims:
        return nap.Tsd(t=t, d=da.values)

    d = da.values.reshape(len(t), -1)
    # Column labels: cartesian product of the other dims' coord values.
    coord_vals = [da.coords[dim].values.tolist() for dim in other_dims]
    columns = ["_".join(map(str, combo)) for combo in itertools.product(*coord_vals)]
    return nap.TsdFrame(t=t, d=d, columns=columns)
