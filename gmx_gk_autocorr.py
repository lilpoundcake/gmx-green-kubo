#!/usr/bin/env python3
"""Calculate the autocorrelation of the pressure tensor elements from GROMACS
MD simulations and the viscosity via the Green-Kubo equation, plus the hybrid
Green-Kubo (hGK) pipeline of Meel & Mogurampelly.

Two Green-Kubo workflows are available through CLI subcommands:

* ``gk``      — single-trajectory classical Green-Kubo following Prass
                 et al. (https://doi.org/10.1021/acs.jcim.3c00947).
* ``acf``     — per-trajectory unbiased SACF + Green-Kubo integral following
                 Meel & Mogurampelly
                 (https://doi.org/10.1021/acs.jpclett.5c03863) — zero-padded
                 FFT, log-spaced output. Supports multiple xvg inputs.
* ``average`` — mean / standard deviation across multiple per-trajectory
                 SACF + viscosity files.
* ``scan``    — stretched-exponential tail-fit window convergence scan
                 (helps choose τ_up).
* ``fit``     — final tail-extrapolated viscosity using a chosen fit window.

V, T, dt and C(0) auto-load from the ``acf_mean.dat`` metadata header
written by ``average`` for both ``scan`` and ``fit``.

Every plot trace is capped at ``--plot-points`` (default 1500) samples
before being serialised to Plotly, so HTML output size is independent of
the input trajectory length. Use ``--plot-format png`` or ``svg`` (needs
kaleido) for static images instead of interactive HTML.

For backwards compatibility, invoking the script with two positional file
arguments (no subcommand) routes to ``gk``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import plotly.graph_objects as go
from scipy import fft as _sfft

__version__ = "0.2.0"

KB = 1.380649e-23  # Boltzmann constant in J/K


def _rebuild_time_axis(n: int, dt_ps: float) -> np.ndarray:
    """Return an (N,) array of times in ps assuming uniform spacing ``dt_ps``.

    Used by the --dt override: some xvg files report frame indices (0, 1,
    2, …) instead of physical times. The fix is to ignore that column and
    construct time = arange(N) * dt explicitly.
    """
    return np.arange(n, dtype=np.float64) * float(dt_ps)


def read_xvg(
    path: Path, dt_override_ps: float | None = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a GROMACS .xvg energy file produced by `gmx energy`.

    Returns
    -------
    time : (N,) array of times in ps
    temperature : (N,) array of temperatures in K
    pressure_tensor : (9, N) array with rows ordered
        Pxx, Pxy, Pxz, Pyx, Pyy, Pyz, Pzx, Pzy, Pzz (in bar)

    If ``dt_override_ps`` is given, the time axis is reconstructed as
    ``arange(N) * dt_override_ps`` and the file's time column is ignored.
    Use this when the input xvg's first column is frame indices or in the
    wrong units.
    """
    raw = np.loadtxt(path, comments=("@", "#"))
    if raw.ndim != 2 or raw.shape[1] < 12:
        raise ValueError(
            f"{path}: expected at least 12 columns (time, T, P + 9 tensor components), "
            f"got shape {raw.shape}"
        )
    if dt_override_ps is not None:
        time = _rebuild_time_axis(raw.shape[0], dt_override_ps)
    else:
        time = raw[:, 0]
    temperature = raw[:, 1]
    # raw[:, 2] is the total pressure column from gmx energy - not used here
    pressure_tensor = raw[:, 3:12].T  # shape (9, N): Pxx,Pxy,Pxz,Pyx,Pyy,Pyz,Pzx,Pzy,Pzz
    return time, temperature, pressure_tensor


def read_volume_from_gro(path: Path) -> float:
    """Return the box volume in nm^3 from the last line of a .gro file.

    A cubic box is assumed: the last line contains the box edge as the first
    whitespace-separated token (in nm) and the volume is edge**3.
    """
    last_line = None
    with path.open() as fh:
        for raw_line in fh:
            stripped = raw_line.strip()
            if stripped:
                last_line = stripped
    if last_line is None:
        raise ValueError(f"{path} contained no readable lines")
    edge_str = last_line.split()[0]
    edge = float(edge_str)
    return edge ** 3


def symmetrize_pressure_tensor(pt: np.ndarray) -> np.ndarray:
    """Return the six symmetrized pressure tensor combinations used for the
    Green-Kubo viscosity.

    Input rows: Pxx, Pxy, Pxz, Pyx, Pyy, Pyz, Pzx, Pzy, Pzz (shape (9, N)).
    Output rows (shape (6, N)):
        (Pxy + Pyx) / 2
        (Pxz + Pzx) / 2
        (Pyz + Pzy) / 2
        (Pxx - Pyy) / 2
        (Pxx - Pzz) / 2
        (Pyy - Pzz) / 2
    """
    Pxx, Pxy, Pxz, Pyx, Pyy, Pyz, Pzx, Pzy, Pzz = pt
    return np.stack(
        [
            0.5 * (Pxy + Pyx),
            0.5 * (Pxz + Pzx),
            0.5 * (Pyz + Pzy),
            0.5 * (Pxx - Pyy),
            0.5 * (Pxx - Pzz),
            0.5 * (Pyy - Pzz),
        ]
    )


def _check_sample_mean(
    channels: np.ndarray,
    names: Tuple[str, ...],
    subtract_mean: bool,
    verbose: bool,
) -> None:
    """Print per-channel sample means and warn about systematic bias.

    Each off-diagonal channel of the pressure tensor has ``<P_αβ>_eq = 0``
    in theory, so a finite-sample mean larger than ``3 * σ/√N`` (where σ
    is the sample standard deviation and N is the trajectory length) is
    a sign that either the signal isn't equilibrated or the data has a
    real systematic stress. Either way, the integrated Green-Kubo
    viscosity picks up a ``μ²·V·t/(kB·T)`` linear-in-t bias that is *not*
    physical viscosity. ``--subtract-mean`` removes it.
    """
    means = channels.mean(axis=1)
    stds = channels.std(axis=1)
    N = channels.shape[1]
    sem = stds / np.sqrt(N)  # naive std-of-mean (treats samples as iid)
    z = np.abs(means) / np.maximum(sem, 1e-30)
    if verbose:
        print("Per-channel sample means (bar):")
        for name, m, s, e, zi in zip(names, means, stds, sem, z):
            print(
                f"  {name:>4}: mean = {m:+9.4f}  std = {s:8.2f}  "
                f"std-of-mean ≈ {e:7.4f}  |mean|/sem ≈ {zi:6.1f}"
            )
    suspicious = (z > 3) & (np.abs(means) > 1e-3)
    if suspicious.any() and not subtract_mean:
        bad = ", ".join(n for n, b in zip(names, suspicious) if b)
        print(
            f"WARNING: channels {{{bad}}} have sample means well above "
            f"sampling noise. The Green-Kubo integral picks up a "
            f"μ²·V·t/(kB·T) bias from each. Re-run with --subtract-mean "
            f"to use the fluctuation form δP = P − <P> (the FDT-correct "
            f"convention). Without it, the result tracks the bias, not "
            f"the physical viscosity."
        )


def autocorrelation_fft(x: np.ndarray, subtract_mean: bool = False) -> np.ndarray:
    """Circular autocorrelation via FFT (matches the original C++ tool by default).

    Computes  c[k] = IDFT(|FFT(x)|**2 / N)[k]  on the raw signal — i.e.
    an estimator of the second moment ``<P(0)·P(t)>``, not the covariance
    ``<δP(0)·δP(t)>``. This is the convention used by gkvisco.cpp's
    ``m_do_wk`` and by every Green-Kubo Python tool surveyed (omidshy/aMD,
    sergey-kruchinin/viscosity, argha1992/Viscosity_Green_Kubo), and it
    relies on the theoretical identity ``<P_αβ>_eq = 0`` for off-diagonal
    components.

    On finite trajectories where the sample mean is significantly non-zero
    (large enough that it can't be explained by sampling noise on a
    true-zero-mean signal), the raw correlator picks up a bias of order
    ``μ²·V·t/(kB·T)`` in the integrated viscosity. Pass
    ``subtract_mean=True`` (CLI: ``--subtract-mean``) to switch to the
    fluctuation form ``δx = x - mean(x)`` that eliminates this bias.

    Performance: uses scipy.fft.rfft / irfft with ``workers=-1`` to
    parallelise across CPU cores; rfft halves the time and memory vs the
    complex FFT used by the original implementation. Output is
    bit-equivalent (within IEEE rounding) to the numpy.fft path.
    """
    N = x.size
    arr = (x - x.mean()) if subtract_mean else x
    # rfft on a length-N real signal gives an N//2+1-long complex spectrum;
    # |X|^2/N is real; irfft of that gives a length-N real array.
    X = _sfft.rfft(arr, n=N, workers=-1)
    psd = (X.real * X.real + X.imag * X.imag) / N
    return _sfft.irfft(psd, n=N, workers=-1)


def green_kubo_running_integral(
    autocorr: np.ndarray, factor: float, half_len: int
) -> np.ndarray:
    """Trapezoidal running integral of each autocorrelation curve.

    Parameters
    ----------
    autocorr : (n_channels, N) array of autocorrelation curves (in bar^2)
    factor : prefactor that converts the integrated autocorrelation (bar^2 * ps)
             into a viscosity in mPa*s, i.e.
             factor = V[nm^3] * 1e-26 / (KB * T[K]) * dt[ps]
    half_len : number of leading time points to integrate over (= N/2 in the
               reference tool)

    Returns
    -------
    out : (n_channels + 1, half_len - 1) array — rows 0..n_channels-1 are the
          per-channel running integrals at t = dt, 2*dt, ..., (half_len-1)*dt,
          row n_channels is the running mean of the per-channel curves
          (matching the "average" column in viscosity.dat).
    """
    if half_len < 2:
        raise ValueError("half_len must be >= 2 to integrate at least one step")
    increments = 0.5 * (autocorr[:, : half_len - 1] + autocorr[:, 1:half_len]) * factor
    running = np.cumsum(increments, axis=1)
    running_avg = running.mean(axis=0, keepdims=True)
    return np.vstack([running, running_avg])


def write_metadata(path: Path, volume_nm3: float, temperature_avg_K: float) -> None:
    """Write a one-line metadata file with the box volume and mean temperature."""
    with path.open("w") as fh:
        fh.write("# avg_volume_nm3    avg_temperature_K\n")
        fh.write(f"{volume_nm3}    {temperature_avg_K}\n")


def write_autocorr(path: Path, dt: float, autocorr: np.ndarray, half_len: int) -> None:
    """Write the classical-GK ACF table.

    8 columns: time(ps), 6 symmetrised-channel ACFs (bar²), channel-average.
    """
    times = (np.arange(half_len) + 1) * dt
    channels = autocorr[:, :half_len]
    average = channels.mean(axis=0, keepdims=True)
    data = np.column_stack([times, channels.T, average.T])
    np.savetxt(path, data, fmt="%.6g", delimiter="    ")


def write_visco(path: Path, dt: float, viscosity: np.ndarray) -> None:
    """Write the classical-GK running-viscosity table.

    8 columns: time(ps), 6 channel running integrals (mPa·s), channel-average.
    """
    n_rows = viscosity.shape[1]
    times = (np.arange(1, n_rows + 1)) * dt
    data = np.column_stack([times, viscosity.T])
    np.savetxt(path, data, fmt="%.6g", delimiter="    ")


def _write_plot(
    fig: go.Figure, path: Path, plot_format: str
) -> Path:
    """Write a Plotly figure as HTML, PNG, or SVG.

    For HTML, plotly.js is loaded from CDN to keep the file size minimal.
    PNG and SVG require ``kaleido`` (``pip install '.[static-plots]'`` or
    install via ``environment.yml``).
    """
    path = Path(path)
    if plot_format == "html":
        fig.write_html(path, include_plotlyjs="cdn")
        return path
    if plot_format in ("png", "svg"):
        out = path.with_suffix(f".{plot_format}")
        try:
            fig.write_image(out)
        except (ImportError, ValueError) as exc:
            raise RuntimeError(
                f"static-image output ({plot_format!r}) needs kaleido: "
                f"`pip install '.[static-plots]'` or `micromamba install "
                f"'python-kaleido<1.0'`. Underlying error: {exc}"
            ) from exc
        return out
    raise ValueError(
        f"plot_format must be one of 'html', 'png', 'svg' (got {plot_format!r})"
    )


def plot_curve(
    path: Path,
    time: np.ndarray,
    y: np.ndarray,
    title: str,
    x_label: str,
    y_label: str,
    plot_points: int = 1500,
    plot_format: str = "html",
) -> Path:
    """Write a single-trace plot of `y` vs `time`.

    Linear x-axis is assumed; trace is capped at ``plot_points`` samples
    via linear-spaced indices.
    """
    time_p, y_p = _subsample_for_plot(time, y, plot_points, log_axis=False)
    fig = go.Figure(
        data=[go.Scatter(x=time_p, y=y_p, mode="lines", name=y_label)]
    )
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        template="plotly_white",
        hovermode="x unified",
    )
    return _write_plot(fig, path, plot_format)


def plot_traces(
    path: Path,
    traces: Iterable[dict],
    title: str,
    x_label: str,
    y_label: str,
    x_log: bool = False,
    y_log: bool = False,
    plot_points: int = 1500,
    plot_format: str = "html",
) -> Path:
    """Write a multi-trace plot.

    Each entry in `traces` is a dict with keys 'x', 'y', 'name'; optional
    keys: 'mode' (default 'lines'), 'dash' (e.g. 'dash'), 'fill', 'fillcolor',
    'showlegend'.

    Every (x, y) pair is independently capped at ``plot_points`` —
    log-spaced indices when ``x_log`` is True, linear otherwise.
    """
    figs = []
    for tr in traces:
        x_p, y_p = _subsample_for_plot(tr["x"], tr["y"], plot_points, log_axis=x_log)
        line = {}
        if "dash" in tr:
            line["dash"] = tr["dash"]
        kwargs = dict(
            x=x_p, y=y_p, name=tr["name"], mode=tr.get("mode", "lines"),
        )
        if line:
            kwargs["line"] = line
        for k in ("fill", "fillcolor", "showlegend"):
            if k in tr:
                kwargs[k] = tr[k]
        figs.append(go.Scatter(**kwargs))
    fig = go.Figure(data=figs)
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        template="plotly_white",
        hovermode="x unified",
    )
    if x_log:
        fig.update_xaxes(type="log")
    if y_log:
        fig.update_yaxes(type="log")
    return _write_plot(fig, path, plot_format)


# ---------------------------------------------------------------------------
# Hybrid Green-Kubo (hGK) framework
#   Implementation of the workflow described in Meel & Mogurampelly,
#   J. Phys. Chem. Lett. 2026, 17, 14, 4016-4022.
# ---------------------------------------------------------------------------

# Indices into the 6-row off-diagonal pressure block expected by the hGK
# framework (matches pressure_components.xvg column order produced by
# `gmx energy` selecting Pres-XY, Pres-XZ, Pres-YZ, Pres-YX, Pres-ZX, Pres-ZY).
HGK_CHANNEL_NAMES = ("pxy", "pxz", "pyz", "pyx", "pzx", "pzy")


def read_pressure_components_xvg(
    path: Path,
    dt_override_ps: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Read an .xvg file with off-diagonal pressure tensor components.

    Supports two layouts:
      * 7 columns (hGK pressure_components.xvg):
            time, Pxy, Pxz, Pyz, Pyx, Pzx, Pzy
      * >= 12 columns (legacy gmx_gk_autocorr energy.xvg):
            time, Temperature, Pressure,
            Pxx, Pxy, Pxz, Pyx, Pyy, Pyz, Pzx, Pzy, Pzz
        — the six off-diagonals are extracted and reordered to the hGK layout.

    Returns
    -------
    time : (N,) array in ps
    channels : (6, N) array in bar, in order
        Pxy, Pxz, Pyz, Pyx, Pzx, Pzy.

    If ``dt_override_ps`` is given, the time axis is reconstructed as
    ``arange(N) * dt_override_ps`` and the file's time column is ignored.
    Use this when the input xvg's first column is frame indices instead
    of ps (e.g. column shows 0, 1, …, 49999 for a 50 000-step trajectory).
    """
    raw = np.loadtxt(path, comments=("@", "#"))
    if raw.ndim != 2:
        raise ValueError(f"{path}: failed to parse as a 2-D table (got shape {raw.shape})")
    n_cols = raw.shape[1]
    if dt_override_ps is not None:
        time = _rebuild_time_axis(raw.shape[0], dt_override_ps)
    else:
        time = raw[:, 0]
    if n_cols >= 12:
        # legacy energy.xvg layout
        # columns: t, T, P, Pxx, Pxy, Pxz, Pyx, Pyy, Pyz, Pzx, Pzy, Pzz
        Pxy = raw[:, 4]
        Pxz = raw[:, 5]
        Pyx = raw[:, 6]
        Pyz = raw[:, 8]
        Pzx = raw[:, 9]
        Pzy = raw[:, 10]
        channels = np.stack([Pxy, Pxz, Pyz, Pyx, Pzx, Pzy])
    elif n_cols >= 7:
        channels = raw[:, 1:7].T
    else:
        raise ValueError(
            f"{path}: expected 7+ columns (hGK layout) or 12+ columns "
            f"(legacy energy.xvg), got {n_cols}"
        )
    return time, channels


def unbiased_autocorrelation_fft(
    x: np.ndarray, subtract_mean: bool = False
) -> np.ndarray:
    """Linear unbiased autocorrelation via zero-padded FFT (matches the hGK
    reference by default).

    Returns  r[k] = (1/(N-k)) * sum_{j=0}^{N-1-k} x[j] x[j+k]  on the raw
    signal — i.e. an estimator of ``<P(0)·P(t)>``, matching the SACF used
    by Meel & Mogurampelly's gk_viscosity_fft.py. This relies on the
    theoretical identity ``<P_αβ>_eq = 0`` for off-diagonal components.

    Pass ``subtract_mean=True`` (CLI: ``--subtract-mean``) to switch to
    the fluctuation form ``δx = x - mean(x)``, which is the form the
    fluctuation-dissipation theorem actually prescribes and which removes
    the linear-in-t bias that appears when the sample mean is
    significantly non-zero (a real risk on short or non-equilibrated
    trajectories).

    Performance: zero-padding is handled by passing ``n=2*N`` to rfft
    (no explicit ``np.zeros(2*N)`` allocation), the rfft path is ~2× the
    speed of the previous complex FFT, and ``workers=-1`` parallelises
    across CPU cores.
    """
    n = x.size
    arr = (x - x.mean()) if subtract_mean else x
    # rfft with n=2*n is equivalent to FFT of a zero-padded signal, but
    # avoids the explicit np.zeros(2*n) allocation and uses the real-FFT
    # symmetry (half the work and half the memory of complex FFT).
    f = _sfft.rfft(arr, n=2 * n, workers=-1)
    psd = f.real * f.real + f.imag * f.imag
    acf_full = _sfft.irfft(psd, n=2 * n, workers=-1)[:n]
    # acf_full[k] equals sum_{j=0}^{N-1-k} arr[j] arr[j+k] exactly
    # (zero-padding suppresses the wrap-around contribution); divide by
    # (N - k) for the unbiased estimator.
    return acf_full / np.arange(n, 0, -1)


def hgk_compute_per_run(
    time: np.ndarray,
    channels: np.ndarray,
    volume_nm3: float,
    temperature_K: float,
    subtract_mean: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Compute the per-trajectory hGK SACF and running viscosity integral.

    Parameters
    ----------
    time : (N,) array in ps (uniform spacing assumed).
    channels : (6, N) off-diagonal pressure tensor components in bar.
    volume_nm3, temperature_K : ensemble V and T.

    Returns
    -------
    sacf_per_channel : (6, N) unbiased ACF in bar^2 for each off-diagonal channel.
    sacf_avg : (N,) channel-averaged ACF in bar^2.
    poft0 : float — value of sacf_avg at zero lag (i.e. C(0) in bar^2).
    eta_per_channel : (6, N) running Green-Kubo integral per channel (mPa·s).
        Note: integration is a left Riemann sum (cumsum * dt * prefactor) to
        match the hGK reference implementation; for large N the trapezoidal
        correction is negligible relative to noise.

    The prefactor used is dt(s) * 1e-14 * V(nm^3) / (kB * T) so that the
    resulting integral is in mPa·s when SACF is in bar^2 and dt is in s.
    """
    if time.size < 4:
        raise ValueError(f"Need at least 4 frames, got {time.size}")
    dt_ps = float(time[1] - time[0])
    dt_s = dt_ps * 1e-12
    prefactor = dt_s * 1e-14 * volume_nm3 / (KB * temperature_K)

    # Batched FFT across all 6 channels at once (single scipy call with
    # workers=-1 — orders of magnitude faster than a Python loop calling
    # numpy.fft six times).
    n = channels.shape[1]
    arr = channels - channels.mean(axis=1, keepdims=True) if subtract_mean else channels
    f = _sfft.rfft(arr, n=2 * n, axis=1, workers=-1)
    psd = f.real * f.real + f.imag * f.imag
    acf_full = _sfft.irfft(psd, n=2 * n, axis=1, workers=-1)[:, :n]
    sacf_per_channel = acf_full / np.arange(n, 0, -1)

    sacf_avg = sacf_per_channel.mean(axis=0)
    poft0 = float(sacf_avg[0])
    eta_per_channel = prefactor * np.cumsum(sacf_per_channel, axis=1)
    return sacf_per_channel, sacf_avg, poft0, eta_per_channel


def log_subsample_indices(n: int, num: int = 10000) -> np.ndarray:
    """Indices for a log-spaced subsample of size up to ``num`` from a length-n array."""
    if n <= 1:
        return np.array([0]) if n == 1 else np.array([], dtype=int)
    return np.unique(np.logspace(0, np.log10(n - 1), num=num, dtype=int))


def _linear_subsample_indices(n: int, num: int) -> np.ndarray:
    """Evenly-spaced subsample of size up to ``num`` from a length-n array."""
    if n <= 1:
        return np.array([0]) if n == 1 else np.array([], dtype=int)
    if n <= num:
        return np.arange(n, dtype=int)
    return np.unique(np.linspace(0, n - 1, num=num, dtype=int))


def _subsample_for_plot(
    x: np.ndarray, y: np.ndarray, n_max: int, log_axis: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """Cap a Plotly trace at ``n_max`` points.

    Plotly serialises every (x, y) value to JSON inside the HTML, so the
    file size is linear in the trace length. For visualisation purposes,
    1–2k points per trace is plenty, and the raw data files are kept at
    full density in the corresponding ``.dat`` outputs.

    Picks log-spaced indices for log-x plots (so density is preserved on
    a log scale) and linear-spaced indices otherwise.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if n_max <= 0 or x.size <= n_max:
        return x, y
    if log_axis:
        idx = log_subsample_indices(x.size, num=n_max)
    else:
        idx = _linear_subsample_indices(x.size, n_max)
    return x[idx], y[idx]


HGK_METADATA_TAG = "hGK_metadata:"


def format_hgk_metadata(meta: dict) -> str:
    """Serialise an hGK metadata dict into a single comment line.

    Example output:
        # hGK_metadata: volume_nm3=121.734 temperature_K=303.0 timeperframe_ps=0.001 p0_bar2=33375.834 n_runs=1
    """
    pairs = " ".join(f"{k}={v!r}" if isinstance(v, str) else f"{k}={v:.10g}"
                     for k, v in meta.items())
    return f"# {HGK_METADATA_TAG} {pairs}"


def parse_hgk_metadata(path: Path) -> dict:
    """Read the hGK metadata header line from a poft/etaoft/avg* file.

    Returns an empty dict if no metadata line is present.
    """
    meta: dict = {}
    with path.open() as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            if HGK_METADATA_TAG in line:
                body = line.split(HGK_METADATA_TAG, 1)[1].strip()
                for tok in body.split():
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        try:
                            meta[k] = float(v)
                        except ValueError:
                            meta[k] = v
                break
    return meta


def write_hgk_poft(
    path: Path,
    time: np.ndarray,
    sacf_avg: np.ndarray,
    sacf_per_channel: np.ndarray,
    indices: np.ndarray,
    metadata: dict | None = None,
) -> None:
    """Write a log-spaced normalised SACF table (8 columns)."""
    poft0 = sacf_avg[0]
    per_channel_norm = sacf_per_channel / sacf_per_channel[:, 0:1]
    data = np.column_stack(
        [time, sacf_avg / poft0, per_channel_norm.T]
    )[indices]
    header_lines = []
    if metadata:
        header_lines.append(format_hgk_metadata(metadata))
    header_lines.append(
        f"# p0 = {poft0:.4f}; time(ps) pavg "
        + " ".join(HGK_CHANNEL_NAMES)
    )
    np.savetxt(path, data, header="\n".join(header_lines), comments="", fmt="%.8f")


def write_hgk_etaoft(
    path: Path,
    time: np.ndarray,
    eta_per_channel: np.ndarray,
    indices: np.ndarray,
    metadata: dict | None = None,
) -> None:
    """Write a log-spaced running viscosity table in mPa·s (8 columns)."""
    eta_avg = eta_per_channel.mean(axis=0)
    data = np.column_stack([time, eta_avg, eta_per_channel.T])[indices]
    header_lines = []
    if metadata:
        header_lines.append(format_hgk_metadata(metadata))
    header_lines.append(
        "# time(ps) etaavg "
        + " ".join("eta_" + name for name in HGK_CHANNEL_NAMES)
    )
    np.savetxt(path, data, header="\n".join(header_lines), comments="", fmt="%.8f")


def hgk_average_runs(
    run_paths: list[Path],
    acf_name: str = "acf.dat",
    viscosity_name: str = "viscosity.dat",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Average hGK per-run outputs across a set of run directories.

    Returns
    -------
    time : (M,) time axis (taken from the first run; assumed identical).
    mean_poft, std_poft : (M, n_cols-1) arrays.
    mean_eta, std_eta : (M, n_cols-1) arrays.
    """
    if not run_paths:
        raise ValueError("No run directories provided")

    poft_list = []
    eta_list = []
    time = None
    for run in run_paths:
        p_path = run / acf_name
        e_path = run / viscosity_name
        if not p_path.is_file():
            raise FileNotFoundError(p_path)
        if not e_path.is_file():
            raise FileNotFoundError(e_path)
        p = np.loadtxt(p_path, comments="#")
        e = np.loadtxt(e_path, comments="#")
        if time is None:
            time = p[:, 0]
        elif p.shape[0] != time.size:
            raise ValueError(
                f"{p_path}: row count {p.shape[0]} differs from first run ({time.size})"
            )
        poft_list.append(p[:, 1:])
        eta_list.append(e[:, 1:])

    poft_stack = np.stack(poft_list)  # (n_runs, M, n_cols-1)
    eta_stack = np.stack(eta_list)
    mean_poft = poft_stack.mean(axis=0)
    std_poft = poft_stack.std(axis=0, ddof=0)
    mean_eta = eta_stack.mean(axis=0)
    std_eta = eta_stack.std(axis=0, ddof=0)
    return time, mean_poft, std_poft, mean_eta, std_eta


def write_hgk_avg(
    path: Path,
    time: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    prefix: str,
    metadata: dict | None = None,
) -> None:
    """Write a single averaged file (time + mean cols + std cols)."""
    data = np.column_stack([time, mean, std])
    n_cols = mean.shape[1]
    lines = []
    if metadata:
        lines.append(format_hgk_metadata(metadata))
    lines.append(
        "# Time(ps) "
        + " ".join(f"avg{prefix}{i + 1}" for i in range(n_cols))
        + " "
        + " ".join(f"std{prefix}{i + 1}" for i in range(n_cols))
    )
    np.savetxt(path, data, header="\n".join(lines), comments="", fmt="%.8e")


def stretched_exponential(t: np.ndarray, a0: float, a1: float, a2: float) -> np.ndarray:
    """Stretched-exponential model used by the hGK SACF tail fit."""
    return a0 * np.exp(-((t / a1) ** a2))


def hgk_fit_tail(
    tau: np.ndarray,
    sacf: np.ndarray,
    tau_low: float,
    tau_upper: float,
) -> Tuple[np.ndarray, int, int]:
    """Fit a stretched exponential to the SACF on [tau_low, tau_upper].

    Returns ``(popt, tau_low_id, tau_upper_id)``. Raises RuntimeError if the
    optimizer fails to converge.
    """
    from scipy.optimize import curve_fit

    if tau_low >= tau_upper:
        raise ValueError("tau_low must be < tau_upper")
    if tau_upper > tau[-1]:
        raise ValueError(f"tau_upper={tau_upper} exceeds SACF range (max {tau[-1]})")
    tau_low_id = int(np.searchsorted(tau, tau_low))
    tau_upper_id = int(np.searchsorted(tau, tau_upper))
    xdata = tau[tau_low_id:tau_upper_id]
    ydata = sacf[tau_low_id:tau_upper_id]
    if xdata.size < 4:
        raise ValueError("Need at least 4 points inside the fit window")
    popt, _ = curve_fit(
        stretched_exponential,
        xdata, ydata,
        bounds=(0, np.inf),
        maxfev=100000,
    )
    return popt, tau_low_id, tau_upper_id


def _hgk_tail_prefactor(
    fine_dt_ps: float, volume_nm3: float, temperature_K: float
) -> float:
    """Prefactor that converts a normalised-SACF cumulative sum on a grid of
    spacing ``fine_dt_ps`` (ps) into a viscosity in mPa·s.

    Both the discrete sum and the prefactor must use the same time step,
    which is the *integration grid step* of the analytical stretched-
    exponential tail — not the sampling rate of the original MD data.
    """
    fine_dt_s = fine_dt_ps * 1e-12
    return fine_dt_s * 1e-14 * volume_nm3 / (KB * temperature_K)


def hgk_window_scan(
    tau: np.ndarray,
    sacf_norm: np.ndarray,
    eta_avg: np.ndarray,
    volume_nm3: float,
    temperature_K: float,
    p0_bar2: float,
    tau_low_ps: float,
    step: int = 10,
    fine_dt_ps: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sweep the upper bound of the stretched-exponential fit window.

    Returns parallel arrays (window_lengths, viscosities, dvis/dwindow).
    Mirrors hGK_scan.py: starts at ``tau_low_id + 10``, steps by ``step`` indices.
    Both the analytical-tail integration grid step and the unit prefactor
    are controlled by ``fine_dt_ps`` (independent of the original MD
    sampling rate).
    """
    from scipy.optimize import curve_fit

    prefactor = _hgk_tail_prefactor(fine_dt_ps, volume_nm3, temperature_K)

    if tau_low_ps >= tau[-1]:
        raise ValueError("tau_low exceeds SACF time range.")
    tau_low_id = int(np.searchsorted(tau, tau_low_ps))
    start_id = tau_low_id + step
    indices = range(start_id, len(tau), step)

    window_lengths: list[float] = []
    viscosities: list[float] = []

    base_eta_at_tau_low = float(eta_avg[tau_low_id])

    for tau_upper_id in indices:
        tau_upper = tau[tau_upper_id]
        xdata = tau[tau_low_id:tau_upper_id]
        ydata = sacf_norm[tau_low_id:tau_upper_id]
        if xdata.size < 4:
            continue
        try:
            popt, _ = curve_fit(
                stretched_exponential, xdata, ydata,
                bounds=(0, np.inf), maxfev=100000,
            )
        except RuntimeError:
            continue
        t_tail = np.arange(tau[tau_low_id], tau[-1], fine_dt_ps)
        sacf_tail = stretched_exponential(t_tail, *popt)
        etan = p0_bar2 * prefactor * np.cumsum(sacf_tail) + base_eta_at_tau_low
        viscosities.append(float(etan[-1]))
        window_lengths.append(tau_upper - tau_low_ps)

    if not viscosities:
        raise RuntimeError("All stretched-exponential fits failed")

    window_lengths_arr = np.array(window_lengths)
    viscosities_arr = np.array(viscosities)
    gradient_arr = np.gradient(viscosities_arr, window_lengths_arr)
    return window_lengths_arr, viscosities_arr, gradient_arr


def hgk_final_extrapolation(
    tau: np.ndarray,
    sacf_norm: np.ndarray,
    eta_avg: np.ndarray,
    volume_nm3: float,
    temperature_K: float,
    p0_bar2: float,
    tau_low_ps: float,
    tau_up_ps: float,
    tau_cut_ps: float,
    fine_dt_ps: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Final viscosity estimate using a tail extrapolated stretched exponential.

    Returns ``(t_tail, sacf_tail_fit, eta_tail, popt)`` — the tail time axis
    in ps, the fitted SACF on that axis, the integrated running viscosity in
    mPa·s, and the best-fit stretched-exponential parameters. The
    integration grid step ``fine_dt_ps`` controls both the analytical-tail
    discretisation and the unit-conversion prefactor.
    """
    if tau_low_ps >= tau_up_ps:
        raise ValueError("tau_low must be smaller than tau_up")
    if tau_up_ps > tau[-1]:
        raise ValueError("tau_up exceeds SACF time range")
    popt, tau_low_id, _ = hgk_fit_tail(tau, sacf_norm, tau_low_ps, tau_up_ps)

    prefactor = _hgk_tail_prefactor(fine_dt_ps, volume_nm3, temperature_K)
    t_tail = np.arange(tau_low_ps, tau_cut_ps, fine_dt_ps)
    sacf_tail = stretched_exponential(t_tail, *popt)
    eta_tail = p0_bar2 * prefactor * np.cumsum(sacf_tail) + float(eta_avg[tau_low_id])
    return t_tail, sacf_tail, eta_tail, popt


def run_gk(
    xvg_path: Path,
    gro_path: Path,
    output_dir: Path,
    output_prefix: str,
    dt_override_ps: float | None = None,
    subtract_mean: bool = False,
    plot: bool = True,
    plot_points: int = 1500,
    plot_format: str = "html",
    verbose: bool = False,
) -> None:
    """Classical Green-Kubo from one trajectory.

    Bit-exact with the original C++ gmx_gk_autocorr by default. Pass
    ``subtract_mean=True`` (CLI: ``--subtract-mean``) to switch to the
    FDT-correct fluctuation form, which removes a μ²·V·t/(kB·T) bias
    that contaminates the integral when the off-diagonal stress has a
    sample mean significantly above sampling noise.
    """
    if verbose:
        print(f"Reading energy data from {xvg_path}")
        if dt_override_ps is not None:
            print(
                f"Overriding time axis: dt = {dt_override_ps} ps "
                f"(input xvg's time column ignored)"
            )
    time, temperature, pressure_tensor = read_xvg(xvg_path, dt_override_ps=dt_override_ps)
    if time.size < 4:
        raise ValueError(
            f"{xvg_path}: need at least 4 frames to compute an autocorrelation, "
            f"got {time.size}"
        )

    if verbose:
        print(f"Reading volume from {gro_path}")
    volume = read_volume_from_gro(gro_path)

    dt = float(time[1] - time[0])
    temperature_avg = float(temperature.mean())

    if verbose:
        print(f"Frames: {time.size}")
        print(f"Time step:      {dt} ps")
        print(f"Volume:         {volume} nm^3")
        print(f"Avg. temperature: {temperature_avg} K")

    sym = symmetrize_pressure_tensor(pressure_tensor)
    N = sym.shape[1]
    _check_sample_mean(
        sym,
        names=("XY+YX", "XZ+ZX", "YZ+ZY", "XX-YY", "XX-ZZ", "YY-ZZ"),
        subtract_mean=subtract_mean,
        verbose=verbose,
    )

    if verbose:
        print(
            f"Computing autocorrelations via FFT (N = {N})"
            f"{' on δP = P − <P>' if subtract_mean else ''}"
        )
    # Batched circular ACF across all 6 channels in one scipy.fft call.
    arr = sym - sym.mean(axis=1, keepdims=True) if subtract_mean else sym
    X = _sfft.rfft(arr, n=N, axis=1, workers=-1)
    psd = (X.real * X.real + X.imag * X.imag) / N
    autocorr = _sfft.irfft(psd, n=N, axis=1, workers=-1)

    half_len = N // 2
    factor = volume * 1e-26 / (KB * temperature_avg) * dt

    if verbose:
        print(f"Integrating Green-Kubo running integral up to t = {half_len * dt} ps")
    viscosity = green_kubo_running_integral(autocorr, factor, half_len)

    output_dir.mkdir(parents=True, exist_ok=True)
    acf_path = output_dir / f"{output_prefix}acf.dat"
    viscosity_path = output_dir / f"{output_prefix}viscosity.dat"
    metadata_path = output_dir / f"{output_prefix}metadata.dat"

    write_autocorr(acf_path, dt, autocorr, half_len)
    write_visco(viscosity_path, dt, viscosity)
    write_metadata(metadata_path, volume, temperature_avg)

    if verbose:
        print(f"Wrote {acf_path}")
        print(f"Wrote {viscosity_path}")
        print(f"Wrote {metadata_path}")

    if plot:
        autocorr_times = (np.arange(half_len) + 1) * dt
        autocorr_avg = autocorr[:, :half_len].mean(axis=0)
        acf_plot_path = plot_curve(
            output_dir / f"{output_prefix}acf.html",
            autocorr_times,
            autocorr_avg,
            title="Pressure-tensor autocorrelation (average of 6 channels)",
            x_label="Time (ps)",
            y_label="Average ACF (bar²)",
            plot_points=plot_points,
            plot_format=plot_format,
        )

        visco_times = (np.arange(viscosity.shape[1]) + 1) * dt
        visco_avg = viscosity[-1]  # last row is the running average
        viscosity_plot_path = plot_curve(
            output_dir / f"{output_prefix}viscosity.html",
            visco_times,
            visco_avg,
            title="Green-Kubo running viscosity (average of 6 channels)",
            x_label="Time (ps)",
            y_label="Viscosity (mPa·s)",
            plot_points=plot_points,
            plot_format=plot_format,
        )

        if verbose:
            print(f"Wrote {acf_plot_path}")
            print(f"Wrote {viscosity_plot_path}")


# ---------------------------------------------------------------------------
# hGK subcommand runners
# ---------------------------------------------------------------------------


def _resolve_volume_temperature(
    xvg_path: Path,
    gro_path: Path | None,
    volume_arg: float | None,
    temperature_arg: float | None,
    verbose: bool,
) -> Tuple[float, float]:
    """Determine V (nm^3) and T (K) from CLI args or auxiliary files.

    Priority order:
      1. Explicit ``--volume`` / ``--temperature`` flags.
      2. ``--gro`` file (volume) and the temperature column of the legacy
         12-column energy.xvg (temperature averaged across frames).
    """
    volume = volume_arg
    temperature = temperature_arg

    if volume is None and gro_path is not None:
        volume = read_volume_from_gro(gro_path)
        if verbose:
            print(f"Read volume {volume} nm^3 from {gro_path}")

    if temperature is None:
        # Try to recover temperature from the legacy energy.xvg layout.
        try:
            raw = np.loadtxt(xvg_path, comments=("@", "#"))
        except Exception:
            raw = None
        if raw is not None and raw.ndim == 2 and raw.shape[1] >= 12:
            temperature = float(np.mean(raw[:, 1]))
            if verbose:
                print(f"Averaged temperature {temperature} K from {xvg_path}")

    if volume is None:
        raise ValueError(
            "Volume not provided. Pass --volume or --gro pointing to a cubic .gro file."
        )
    if temperature is None:
        raise ValueError(
            "Temperature not provided. Pass --temperature; legacy energy.xvg "
            "layout (with a Temperature column) is also auto-detected."
        )
    return float(volume), float(temperature)


def _subdir_name_for_xvg(xvg_path: Path) -> str:
    """Choose a sensible per-run subdirectory name for an input xvg file.

    Uses the parent directory's name when the file lives in a meaningful
    folder (e.g. ``run3/pressure.xvg`` → ``run3``); otherwise falls back to
    the file stem.
    """
    parent = xvg_path.resolve().parent.name
    if parent in ("", ".", "/"):
        return xvg_path.stem
    return parent


def _process_single_hgk_run(
    xvg_path: Path,
    output_dir: Path,
    output_prefix: str,
    volume_arg: float | None,
    temperature_arg: float | None,
    gro_path: Path | None,
    log_points: int,
    plot: bool,
    plot_points: int,
    plot_format: str,
    verbose: bool,
    dt_override_ps: float | None = None,
    subtract_mean: bool = False,
) -> dict:
    """Run the per-trajectory hGK step on a single xvg file.

    Returns the metadata dict written to the output headers.
    """
    if verbose:
        print(f"Reading pressure components from {xvg_path}")
        if dt_override_ps is not None:
            print(
                f"Overriding time axis: dt = {dt_override_ps} ps "
                f"(input xvg's time column ignored)"
            )
    time, channels = read_pressure_components_xvg(
        xvg_path, dt_override_ps=dt_override_ps
    )
    if time.size < 4:
        raise ValueError(f"{xvg_path}: need at least 4 frames, got {time.size}")

    volume, temperature = _resolve_volume_temperature(
        xvg_path, gro_path, volume_arg, temperature_arg, verbose
    )

    dt = float(time[1] - time[0])
    if verbose:
        print(f"Frames: {time.size}  dt = {dt} ps")
        print(f"Volume: {volume} nm^3  Temperature: {temperature} K")
    _check_sample_mean(
        channels,
        names=("Pxy", "Pxz", "Pyz", "Pyx", "Pzx", "Pzy"),
        subtract_mean=subtract_mean,
        verbose=verbose,
    )
    if verbose:
        print(
            "Computing unbiased SACF via zero-padded FFT for 6 channels"
            f"{' on δP = P − <P>' if subtract_mean else ''}"
        )

    sacf_per_channel, sacf_avg, poft0, eta_per_channel = hgk_compute_per_run(
        time, channels, volume, temperature, subtract_mean=subtract_mean
    )
    eta_avg = eta_per_channel.mean(axis=0)

    metadata = {
        "volume_nm3": volume,
        "temperature_K": temperature,
        "timeperframe_ps": dt,
        "p0_bar2": poft0,
        "n_runs": 1,
        "source": xvg_path.name,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    acf_path = output_dir / f"{output_prefix}acf.dat"
    viscosity_path = output_dir / f"{output_prefix}viscosity.dat"

    indices = log_subsample_indices(time.size, num=log_points)
    write_hgk_poft(acf_path, time, sacf_avg, sacf_per_channel, indices, metadata)
    write_hgk_etaoft(viscosity_path, time, eta_per_channel, indices, metadata)

    if verbose:
        print(f"Wrote {acf_path}")
        print(f"Wrote {viscosity_path}")
        print(f"C(0) = {poft0:.4f} bar^2")

    if plot:
        plot_traces(
            output_dir / f"{output_prefix}acf.html",
            [
                {
                    "x": time[indices][1:],
                    "y": (sacf_avg / poft0)[indices][1:],
                    "name": "Normalised SACF (avg of 6)",
                }
            ],
            title=f"Normalised SACF (C(0) = {poft0:.3f} bar²)",
            x_label="Time (ps)",
            y_label="C(t) / C(0)",
            x_log=True,
            plot_points=plot_points,
            plot_format=plot_format,
        )
        plot_traces(
            output_dir / f"{output_prefix}viscosity.html",
            [
                {
                    "x": time[indices][1:],
                    "y": eta_avg[indices][1:],
                    "name": "Running viscosity (avg of 6)",
                }
            ],
            title="Green-Kubo running viscosity (per-trajectory unbiased SACF)",
            x_label="Time (ps)",
            y_label="η(t) (mPa·s)",
            x_log=True,
            plot_points=plot_points,
            plot_format=plot_format,
        )

    return metadata


def run_hgk_run(
    xvg_paths: list[Path],
    output_dir: Path,
    output_prefix: str,
    volume_arg: float | None,
    temperature_arg: float | None,
    gro_path: Path | None,
    log_points: int = 10000,
    plot: bool = True,
    plot_points: int = 1500,
    plot_format: str = "html",
    verbose: bool = False,
    dt_override_ps: float | None = None,
    subtract_mean: bool = False,
) -> None:
    """Per-trajectory hGK step on one or more input xvg files.

    With a single xvg file, the outputs are written directly to ``output_dir``
    (backwards-compatible single-run behaviour). With multiple files, each
    file is processed into a per-run subdirectory of ``output_dir`` named
    after the file's parent folder (or its stem if the parent is unclear).

    V and T can come from explicit CLI flags, from a .gro file (volume), or
    from the Temperature column of a legacy 12-column energy.xvg.
    """
    if not xvg_paths:
        raise ValueError("acf: at least one input .xvg file is required")

    if len(xvg_paths) == 1:
        _process_single_hgk_run(
            xvg_paths[0], output_dir, output_prefix,
            volume_arg, temperature_arg, gro_path,
            log_points, plot, plot_points, plot_format, verbose,
            dt_override_ps=dt_override_ps,
            subtract_mean=subtract_mean,
        )
        return

    if verbose:
        print(f"Processing {len(xvg_paths)} trajectories into {output_dir}")
    subdir_names = [_subdir_name_for_xvg(p) for p in xvg_paths]
    if len(set(subdir_names)) != len(subdir_names):
        # Disambiguate by appending an index.
        subdir_names = [f"{n}_{i + 1}" for i, n in enumerate(subdir_names)]
    for xvg_path, name in zip(xvg_paths, subdir_names):
        sub_out = output_dir / name
        if verbose:
            print(f"\n=== {name} ({xvg_path}) ===")
        _process_single_hgk_run(
            xvg_path, sub_out, output_prefix,
            volume_arg, temperature_arg, gro_path,
            log_points, plot, plot_points, plot_format, verbose,
            dt_override_ps=dt_override_ps,
            subtract_mean=subtract_mean,
        )
    if verbose:
        print(
            f"\nDone. To average: "
            f"gmx_gk_autocorr.py average "
            + " ".join(str(output_dir / n) for n in subdir_names)
            + f" -o {output_dir}"
        )


def _merge_run_metadata(run_dirs: list[Path], acf_name: str) -> dict:
    """Combine per-run metadata into a single averaged record.

    V, T, dt are taken from the first run and a warning is printed if any
    subsequent run disagrees beyond a small tolerance. ``p0_bar2`` is the
    mean of per-run C(0) values — this is what ``scan`` / ``fit`` use as
    the absolute-scale prefactor for the (normalised) averaged SACF.
    """
    merged: dict = {}
    p0_values: list[float] = []
    first_meta: dict | None = None
    for run in run_dirs:
        meta = parse_hgk_metadata(run / acf_name)
        if not meta:
            continue
        if first_meta is None:
            first_meta = meta
        else:
            for key in ("volume_nm3", "temperature_K", "timeperframe_ps"):
                if key in meta and key in first_meta:
                    a, b = first_meta[key], meta[key]
                    if a and abs(a - b) / abs(a) > 1e-6:
                        print(
                            f"warning: {run}/{acf_name}: {key}={b} differs from "
                            f"first run ({a})"
                        )
        if "p0_bar2" in meta:
            p0_values.append(meta["p0_bar2"])
    if first_meta is None:
        return merged
    for key in ("volume_nm3", "temperature_K", "timeperframe_ps"):
        if key in first_meta:
            merged[key] = first_meta[key]
    if p0_values:
        merged["p0_bar2"] = float(np.mean(p0_values))
        merged["p0_bar2_std"] = float(np.std(p0_values, ddof=0))
    merged["n_runs"] = len(run_dirs)
    return merged


def run_hgk_avg(
    run_dirs: list[Path],
    output_dir: Path,
    output_prefix: str,
    acf_name: str = "acf.dat",
    viscosity_name: str = "viscosity.dat",
    plot: bool = True,
    plot_points: int = 1500,
    plot_format: str = "html",
    verbose: bool = False,
) -> None:
    """Average SACF and Green-Kubo integrals across multiple runs."""
    if verbose:
        print(f"Averaging across {len(run_dirs)} runs")
        for r in run_dirs:
            print(f"  {r}")
    time, mean_poft, std_poft, mean_eta, std_eta = hgk_average_runs(
        run_dirs, acf_name=acf_name, viscosity_name=viscosity_name
    )

    metadata = _merge_run_metadata(run_dirs, acf_name)
    if metadata and verbose:
        v = metadata.get("volume_nm3")
        t = metadata.get("temperature_K")
        dt = metadata.get("timeperframe_ps")
        p0 = metadata.get("p0_bar2")
        p0_std = metadata.get("p0_bar2_std")
        print(
            f"Metadata: V = {v} nm^3, T = {t} K, dt = {dt} ps, "
            f"⟨C(0)⟩ = {p0:.4f} ± {p0_std:.4f} bar² over {metadata.get('n_runs')} runs"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    acf_mean_path = output_dir / f"{output_prefix}acf_mean.dat"
    viscosity_mean_path = output_dir / f"{output_prefix}viscosity_mean.dat"
    write_hgk_avg(acf_mean_path, time, mean_poft, std_poft, prefix="P", metadata=metadata)
    write_hgk_avg(viscosity_mean_path, time, mean_eta, std_eta, prefix="eta", metadata=metadata)
    if verbose:
        print(f"Wrote {acf_mean_path}")
        print(f"Wrote {viscosity_mean_path}")

    if plot:
        # Column 0 in mean/std arrays is the average across the 6 channels.
        avg_sacf = mean_poft[:, 0]
        std_sacf = std_poft[:, 0]
        avg_eta_mean = mean_eta[:, 0]
        std_eta_mean = std_eta[:, 0]
        mask = time > 0
        plot_traces(
            output_dir / f"{output_prefix}acf_mean.html",
            [
                {
                    "x": time[mask],
                    "y": (avg_sacf + std_sacf)[mask],
                    "name": "mean + 1σ",
                    "mode": "lines",
                    "dash": "dot",
                    "showlegend": True,
                },
                {
                    "x": time[mask],
                    "y": (avg_sacf - std_sacf)[mask],
                    "name": "mean − 1σ",
                    "mode": "lines",
                    "dash": "dot",
                    "fill": "tonexty",
                    "fillcolor": "rgba(99,110,250,0.15)",
                    "showlegend": True,
                },
                {
                    "x": time[mask],
                    "y": avg_sacf[mask],
                    "name": f"mean across {len(run_dirs)} runs",
                },
            ],
            title="Averaged normalised SACF",
            x_label="Time (ps)",
            y_label="⟨C(t)/C(0)⟩",
            x_log=True,
            plot_points=plot_points,
            plot_format=plot_format,
        )
        plot_traces(
            output_dir / f"{output_prefix}viscosity_mean.html",
            [
                {
                    "x": time[mask],
                    "y": (avg_eta_mean + std_eta_mean)[mask],
                    "name": "mean + 1σ",
                    "mode": "lines",
                    "dash": "dot",
                },
                {
                    "x": time[mask],
                    "y": (avg_eta_mean - std_eta_mean)[mask],
                    "name": "mean − 1σ",
                    "mode": "lines",
                    "dash": "dot",
                    "fill": "tonexty",
                    "fillcolor": "rgba(99,110,250,0.15)",
                },
                {
                    "x": time[mask],
                    "y": avg_eta_mean[mask],
                    "name": f"mean across {len(run_dirs)} runs",
                },
            ],
            title="Averaged running viscosity",
            x_label="Time (ps)",
            y_label="⟨η(t)⟩ (mPa·s)",
            x_log=True,
            plot_points=plot_points,
            plot_format=plot_format,
        )


def _load_avg_files(
    acf_mean_path: Path, viscosity_mean_path: Path
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load acf_mean.dat / viscosity_mean.dat and return (tau, sacf_norm, eta_avg, eta_table).

    The first non-time column is the channel-averaged value (per the layout
    produced by ``acf`` and ``average``). Comment lines (starting with
    '#') are skipped automatically, so files with or without the hGK
    metadata header are both supported.
    """
    poft = np.loadtxt(acf_mean_path, comments="#")
    eta = np.loadtxt(viscosity_mean_path, comments="#")
    if poft.shape[0] != eta.shape[0]:
        raise ValueError(
            f"Row count mismatch: {acf_mean_path} has {poft.shape[0]} rows, "
            f"{viscosity_mean_path} has {eta.shape[0]}"
        )
    tau = poft[:, 0]
    sacf_norm = poft[:, 1]
    eta_avg = eta[:, 1]
    return tau, sacf_norm, eta_avg, eta


def _resolve_hgk_fit_params(
    acf_mean_path: Path,
    volume_arg: float | None,
    temperature_arg: float | None,
    p0_arg: float | None,
    verbose: bool,
) -> Tuple[float, float, float]:
    """Auto-load (V, T, C(0)) from the acf_mean header, with CLI overrides.

    A CLI flag wins over a header value. Missing parameters trigger an error
    so the user understands which value is needed. The original MD sampling
    rate (``timeperframe_ps``) is read for diagnostic display only — the
    analytical-tail integral in ``scan`` / ``fit`` is set by
    ``--fine-dt``, not by the original data spacing.
    """
    meta = parse_hgk_metadata(acf_mean_path)
    if verbose and meta:
        n = meta.get("n_runs")
        print(
            f"Loaded metadata from {acf_mean_path}: "
            f"V={meta.get('volume_nm3')} nm^3, T={meta.get('temperature_K')} K, "
            f"data dt={meta.get('timeperframe_ps')} ps, "
            f"C(0)={meta.get('p0_bar2')} bar^2"
            + (f" (averaged across {int(n)} runs)" if n else "")
        )
    def _pick(name: str, cli: float | None, key: str) -> float:
        if cli is not None:
            return cli
        if key in meta:
            return float(meta[key])
        raise ValueError(
            f"{name} not provided and not present in {acf_mean_path}'s metadata. "
            f"Pass --{name.lower().replace(' ', '-')} on the command line."
        )
    volume = _pick("volume", volume_arg, "volume_nm3")
    temperature = _pick("temperature", temperature_arg, "temperature_K")
    p0 = _pick("p0", p0_arg, "p0_bar2")
    return volume, temperature, p0


def run_hgk_scan(
    acf_mean_path: Path,
    viscosity_mean_path: Path,
    output_dir: Path,
    output_prefix: str,
    volume: float,
    temperature: float,
    p0_bar2: float,
    tau_low_ps: float,
    step: int = 10,
    fine_dt_ps: float = 1e-3,
    plot: bool = True,
    plot_points: int = 1500,
    plot_format: str = "html",
    verbose: bool = False,
) -> None:
    """Stretched-exponential fit-window convergence scan."""
    if verbose:
        print(f"Loading averaged data from {acf_mean_path} and {viscosity_mean_path}")
    tau, sacf_norm, eta_avg, _ = _load_avg_files(acf_mean_path, viscosity_mean_path)

    if verbose:
        print(
            f"Scanning fit windows: tau_low = {tau_low_ps} ps, "
            f"step = {step}, fine_dt = {fine_dt_ps} ps, "
            f"V = {volume} nm^3, T = {temperature} K, "
            f"C(0) = {p0_bar2} bar^2"
        )
    windows, viscosities, gradients = hgk_window_scan(
        tau, sacf_norm, eta_avg,
        volume_nm3=volume,
        temperature_K=temperature,
        p0_bar2=p0_bar2,
        tau_low_ps=tau_low_ps,
        step=step,
        fine_dt_ps=fine_dt_ps,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    scan_path = output_dir / f"{output_prefix}window_scan.dat"
    data = np.column_stack([windows, viscosities, gradients])
    np.savetxt(
        scan_path, data,
        header=(
            "# Fit window length (ps)  Extrapolated viscosity (mPa.s)  "
            "Slope (d eta / d tau)"
        ),
        comments="",
        fmt="%.8e",
    )
    if verbose:
        print(f"Wrote {scan_path}")

    if plot:
        plot_traces(
            output_dir / f"{output_prefix}window_scan.html",
            [
                {
                    "x": windows, "y": viscosities,
                    "name": "Viscosity vs window length",
                },
            ],
            title="Fit-window convergence scan",
            x_label="Fit window length τᵤ − τₗ (ps)",
            y_label="Extrapolated viscosity (mPa·s)",
            plot_points=plot_points,
            plot_format=plot_format,
        )


def run_hgk_final(
    acf_mean_path: Path,
    viscosity_mean_path: Path,
    output_dir: Path,
    output_prefix: str,
    volume: float,
    temperature: float,
    p0_bar2: float,
    tau_low_ps: float,
    tau_up_ps: float,
    tau_cut_ps: float,
    fine_dt_ps: float = 1e-3,
    log_points: int = 50,
    plot: bool = True,
    plot_points: int = 1500,
    plot_format: str = "html",
    verbose: bool = False,
) -> None:
    """Final tail-extrapolated viscosity from one chosen fit window."""
    if verbose:
        print(f"Loading averaged data from {acf_mean_path} and {viscosity_mean_path}")
    tau, sacf_norm, eta_avg, _ = _load_avg_files(acf_mean_path, viscosity_mean_path)

    t_tail, sacf_tail, eta_tail, popt = hgk_final_extrapolation(
        tau, sacf_norm, eta_avg,
        volume_nm3=volume,
        temperature_K=temperature,
        p0_bar2=p0_bar2,
        tau_low_ps=tau_low_ps,
        tau_up_ps=tau_up_ps,
        tau_cut_ps=tau_cut_ps,
        fine_dt_ps=fine_dt_ps,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{output_prefix}viscosity_final.dat"
    log_idx = log_subsample_indices(len(t_tail), num=log_points)
    np.savetxt(
        out_path,
        np.column_stack([t_tail[log_idx], eta_tail[log_idx]]),
        header="Time(ps)  Extrapolated viscosity integral (mPa.s)",
        fmt="%.8e",
    )
    final_eta = float(eta_tail[-1])
    if verbose:
        print(
            f"Fit popt: a0={popt[0]:.4g}  tau={popt[1]:.4g} ps  beta={popt[2]:.4g}"
        )
        print(f"Final viscosity at τ_cut = {tau_cut_ps} ps: {final_eta:.4f} mPa·s")
        print(f"Wrote {out_path}")

    if plot:
        # The plot helpers cap each trace at `plot_points` (default 1500),
        # so the multi-million-point fine-grid tail is automatically
        # log-subsampled for the HTML.
        mask_fit = (tau >= tau_low_ps) & (tau <= tau_up_ps)
        mask_data = tau > 0
        plot_traces(
            output_dir / f"{output_prefix}acf_fit.html",
            [
                {
                    "x": tau[mask_data], "y": sacf_norm[mask_data],
                    "name": "Data (averaged normalised SACF)",
                },
                {
                    "x": t_tail, "y": sacf_tail,
                    "name": "Stretched-exp fit (extrapolated)",
                    "dash": "dash",
                },
                {
                    "x": tau[mask_fit], "y": sacf_norm[mask_fit],
                    "name": f"Fit window [{tau_low_ps}, {tau_up_ps}] ps",
                    "mode": "markers",
                },
            ],
            title=(
                f"SACF tail fit — a0={popt[0]:.3g}, τ={popt[1]:.3g} ps, β={popt[2]:.3g}"
            ),
            x_label="Time (ps)",
            y_label="C(t)/C(0)",
            x_log=True,
            plot_points=plot_points,
            plot_format=plot_format,
        )
        plot_traces(
            output_dir / f"{output_prefix}viscosity_final.html",
            [
                {"x": tau, "y": eta_avg, "name": "Running ⟨η(t)⟩ (data)"},
                {
                    "x": t_tail, "y": eta_tail,
                    "name": "Tail extrapolation",
                    "dash": "dash",
                },
            ],
            title=(
                f"hGK final viscosity = {final_eta:.4f} mPa·s "
                f"(τ_low={tau_low_ps}, τ_up={tau_up_ps}, τ_cut={tau_cut_ps} ps)"
            ),
            x_label="Time (ps)",
            y_label="η(t) (mPa·s)",
            x_log=True,
            plot_points=plot_points,
            plot_format=plot_format,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_common_output_args(p: argparse.ArgumentParser, default_prefix: str) -> None:
    p.add_argument(
        "-o", "--output-dir", type=Path, default=Path("."),
        help="Directory for output files (default: current dir).",
    )
    p.add_argument(
        "-p", "--output-prefix", default=default_prefix,
        help=f'Filename prefix for output files (default: "{default_prefix}").',
    )
    p.add_argument(
        "--no-plot", dest="plot", action="store_false",
        help="Skip writing plots (data files are still produced).",
    )
    p.add_argument(
        "--plot-points", type=int, default=1500,
        help=(
            "Maximum number of points per Plotly trace (default: 1500). "
            "Capping makes HTML output size independent of the input "
            "trajectory length; the data files (.dat) keep their full "
            "density."
        ),
    )
    p.add_argument(
        "--plot-format", choices=["html", "png", "svg"], default="html",
        help=(
            "Plot output format. 'html' (default) writes interactive Plotly "
            "files using a CDN-loaded plotly.js. 'png'/'svg' write static "
            "images and need kaleido (pip install '.[static-plots]' or "
            "via environment.yml)."
        ),
    )
    p.add_argument(
        "-v", "--verbose", action="store_true",
        help="Print progress information to stdout.",
    )


class _ConfigArgParser(argparse.ArgumentParser):
    """ArgumentParser that accepts ``@file`` config files with friendly syntax.

    A line in a config file may contain:
      * a single argument or flag (one token per line — argparse's default);
      * a ``--flag value`` pair on the same line (split on whitespace);
      * positional arguments interleaved on their own lines;
      * blank lines, and ``#`` line comments / trailing ``#`` comments.

    Example ``fit`` config file::

        # Metadata overrides (otherwise read from acf_mean.dat header)
        --volume       121.734    # nm^3
        --temperature  303.0      # K

        # Fit window
        --tau-low 0.2
        --tau-up  5.0
        --tau-cut 5000
    """

    def convert_arg_line_to_args(self, arg_line: str) -> list[str]:
        # Strip trailing comments and surrounding whitespace.
        line = arg_line.split("#", 1)[0].strip()
        if not line:
            return []
        # Split into tokens on whitespace so ``--flag value`` works on one line.
        return line.split()


def build_parser() -> argparse.ArgumentParser:
    parser = _ConfigArgParser(
        prog="gmx-gk-autocorr",
        fromfile_prefix_chars="@",
        description=(
            "Green-Kubo viscosity from GROMACS MD simulations. Two workflows: "
            "(1) 'gk' computes the classical Green-Kubo viscosity from a "
            "single trajectory; (2) 'acf' + 'average' + 'scan' + 'fit' is the "
            "hybrid Green-Kubo pipeline that averages multiple trajectories "
            "and tail-extrapolates the SACF for a converged viscosity. Any "
            "argument may be loaded from a config file with @file (one arg "
            "per line, or 'flag value' per line; '#' starts a comment)."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<subcommand>")

    # gk -------------------------------------------------------------------
    p_gk = subparsers.add_parser(
        "gk",
        help="Classical Green-Kubo viscosity from one trajectory.",
        description=(
            "Compute the autocorrelation of pressure tensor elements and the "
            "Green-Kubo running viscosity from one GROMACS NVT trajectory."
        ),
    )
    p_gk.add_argument(
        "xvg", type=Path,
        help=(
            "GROMACS .xvg energy file. Columns required: time, Temperature, "
            "Pressure, Pres-XX, Pres-XY, Pres-XZ, Pres-YX, Pres-YY, Pres-YZ, "
            "Pres-ZX, Pres-ZY, Pres-ZZ."
        ),
    )
    p_gk.add_argument(
        "gro", type=Path,
        help="GROMACS .gro structure file (only the last line, cubic box edge in nm, is read).",
    )
    p_gk.add_argument(
        "--dt", dest="dt_override_ps", type=float, default=0.002,
        help=(
            "Per-frame time step in ps used to (re)build the time axis "
            "(default: 0.002). The time array is computed as arange(N) * dt "
            "and the input xvg's first column is ignored. Pass the value "
            "matching the integration step of your MD setup."
        ),
    )
    p_gk.add_argument(
        "--subtract-mean", dest="subtract_mean", action="store_true",
        help=(
            "Subtract the sample mean from each channel before computing the "
            "autocorrelation, i.e. compute <δP(0)·δP(t)> instead of "
            "<P(0)·P(t)>. The FDT-correct convention. Off by default (matches "
            "the original C++ tool); turn on if the verbose diagnostic reports "
            "channel means significantly above sampling noise."
        ),
    )
    _add_common_output_args(p_gk, default_prefix="")

    # acf ------------------------------------------------------------------
    p_acf = subparsers.add_parser(
        "acf",
        help="Per-trajectory unbiased SACF + Green-Kubo integral (hybrid step 1).",
        description=(
            "Compute the linear unbiased SACF (zero-padded FFT) over the six "
            "off-diagonal pressure components and the running Green-Kubo "
            "integral. With multiple xvg inputs, each trajectory is "
            "processed into its own subdirectory of -o (named after the file's "
            "parent folder), suitable for direct consumption by 'average'. "
            "V and T are auto-detected from the xvg + .gro pair when "
            "possible (volume from a cubic .gro, temperature from the legacy "
            "12-column energy.xvg layout)."
        ),
    )
    p_acf.add_argument(
        "xvg", type=Path, nargs="+",
        help=(
            "One or more pressure-components .xvg files (or a shell glob). "
            "Two layouts accepted: hGK pressure_components.xvg (7 columns) "
            "or legacy 12-column energy.xvg — off-diagonals are extracted "
            "automatically."
        ),
    )
    p_acf.add_argument("--gro", type=Path, default=None,
                      help="Cubic .gro file to derive the volume.")
    p_acf.add_argument("--volume", type=float, default=None,
                      help="Volume in nm^3 (overrides --gro).")
    p_acf.add_argument("--temperature", type=float, default=None,
                      help="Temperature in K (auto-detected for legacy energy.xvg).")
    p_acf.add_argument(
        "--log-points", type=int, default=10000,
        help="Approximate number of log-spaced points in the output (default: 10000).",
    )
    p_acf.add_argument(
        "--dt", dest="dt_override_ps", type=float, default=0.002,
        help=(
            "Per-frame time step in ps used to (re)build the time axis "
            "(default: 0.002). The time array is computed as arange(N) * dt "
            "and the input xvg's first column is ignored. Pass the value "
            "matching the integration step of your MD setup."
        ),
    )
    p_acf.add_argument(
        "--subtract-mean", dest="subtract_mean", action="store_true",
        help=(
            "Subtract the sample mean from each channel before computing the "
            "SACF, i.e. work with δP = P − <P>. The FDT-correct convention. "
            "Off by default (matches the hGK reference); turn on if the "
            "verbose diagnostic reports channel means significantly above "
            "sampling noise."
        ),
    )
    _add_common_output_args(p_acf, default_prefix="")

    # average --------------------------------------------------------------
    p_avg = subparsers.add_parser(
        "average",
        help="Average per-trajectory SACFs and viscosities across independent runs.",
        description=(
            "Combine acf.dat / viscosity.dat from N independent trajectories "
            "into acf_mean.dat and viscosity_mean.dat with mean ± standard "
            "deviation columns."
        ),
    )
    p_avg.add_argument(
        "runs", nargs="+", type=Path,
        help=(
            "One or more run directories, each containing acf.dat and "
            "viscosity.dat."
        ),
    )
    p_avg.add_argument("--acf-name", default="acf.dat",
                       help="Filename of per-run SACF file (default: acf.dat).")
    p_avg.add_argument("--viscosity-name", default="viscosity.dat",
                       help="Filename of per-run running-integral file (default: viscosity.dat).")
    _add_common_output_args(p_avg, default_prefix="")

    def _add_hgk_metadata_overrides(p: argparse.ArgumentParser) -> None:
        p.add_argument("--volume", type=float, default=None,
                       help="Volume in nm^3 (overrides acf_mean.dat metadata).")
        p.add_argument("--temperature", type=float, default=None,
                       help="Temperature in K (overrides metadata).")
        p.add_argument("--p0", type=float, default=None,
                       help="C(0) of the SACF in bar^2 (overrides metadata's averaged C(0)).")

    # scan -----------------------------------------------------------------
    p_scan = subparsers.add_parser(
        "scan",
        help="Stretched-exponential fit-window convergence scan.",
        description=(
            "Sweep the upper bound τᵤ of the SACF tail fit and report the "
            "resulting extrapolated viscosity vs window length. Useful to "
            "diagnose convergence and to pick a τᵤ for 'fit'. V, T and "
            "C(0) are auto-loaded from acf_mean.dat's metadata header; pass "
            "--volume / --temperature / --p0 to override. The analytical "
            "tail integral runs on a fine grid set by --fine-dt (default "
            "0.001 ps)."
        ),
    )
    p_scan.add_argument("acf_mean", type=Path,
                       help="acf_mean.dat (from 'average') or acf.dat (one run).")
    p_scan.add_argument("viscosity_mean", type=Path,
                       help="viscosity_mean.dat (from 'average') or viscosity.dat (one run).")
    p_scan.add_argument("--tau-low", type=float, required=True,
                       help="Lower bound of the SACF tail fit window in ps.")
    p_scan.add_argument("--step", type=int, default=10,
                       help="Index step between fit-window upper bounds (default: 10).")
    p_scan.add_argument("--fine-dt", type=float, default=1e-3,
                       help="Fine-grid step for the analytical tail integral in ps (default: 0.001).")
    _add_hgk_metadata_overrides(p_scan)
    _add_common_output_args(p_scan, default_prefix="")

    # fit ------------------------------------------------------------------
    p_final = subparsers.add_parser(
        "fit",
        help="Final tail-extrapolated viscosity from a chosen fit window.",
        description=(
            "Fit a stretched exponential to the SACF on [τ_low, τ_up], "
            "extrapolate to τ_cut, and integrate to obtain the final "
            "viscosity. V, T and C(0) are auto-loaded from acf_mean.dat's "
            "metadata header; pass --volume / --temperature / --p0 to "
            "override. The analytical tail integral runs on a fine grid set "
            "by --fine-dt (default 0.001 ps)."
        ),
    )
    p_final.add_argument("acf_mean", type=Path,
                        help="acf_mean.dat (from 'average') or acf.dat (one run).")
    p_final.add_argument("viscosity_mean", type=Path,
                        help="viscosity_mean.dat (from 'average') or viscosity.dat (one run).")
    p_final.add_argument("--tau-low", type=float, required=True,
                        help="Lower bound of the SACF tail fit window in ps.")
    p_final.add_argument("--tau-up", type=float, required=True,
                        help="Upper bound of the SACF tail fit window in ps.")
    p_final.add_argument("--tau-cut", type=float, required=True,
                        help="Upper bound for tail extrapolation in ps (e.g. 5000).")
    p_final.add_argument("--fine-dt", type=float, default=1e-3,
                        help="Fine-grid step for the tail integral in ps (default: 0.001).")
    p_final.add_argument("--log-points", type=int, default=50,
                        help="Log-spaced points in viscosity_final.dat (default: 50).")
    _add_hgk_metadata_overrides(p_final)
    _add_common_output_args(p_final, default_prefix="")

    return parser


def _is_legacy_invocation(argv: list[str]) -> bool:
    """Detect the legacy ``gmx_gk_autocorr.py energy.xvg conf.gro`` form."""
    if not argv:
        return False
    first = argv[0]
    if first.startswith("-") or first.startswith("@"):
        # @file expands into real arguments handled below; never auto-wrap.
        return False
    known = {"gk", "acf", "average", "scan", "fit"}
    return first not in known


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if _is_legacy_invocation(argv):
        # Preserve the original positional-argument call as the 'gk' subcommand.
        argv = ["gk", *argv]

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0

    try:
        if args.command == "gk":
            run_gk(
                xvg_path=args.xvg,
                gro_path=args.gro,
                output_dir=args.output_dir,
                output_prefix=args.output_prefix,
                dt_override_ps=args.dt_override_ps,
                subtract_mean=args.subtract_mean,
                plot=args.plot,
                plot_points=args.plot_points,
                plot_format=args.plot_format,
                verbose=args.verbose,
            )
        elif args.command == "acf":
            run_hgk_run(
                xvg_paths=args.xvg,
                output_dir=args.output_dir,
                output_prefix=args.output_prefix,
                volume_arg=args.volume,
                temperature_arg=args.temperature,
                gro_path=args.gro,
                log_points=args.log_points,
                plot=args.plot,
                plot_points=args.plot_points,
                plot_format=args.plot_format,
                verbose=args.verbose,
                dt_override_ps=args.dt_override_ps,
                subtract_mean=args.subtract_mean,
            )
        elif args.command == "average":
            run_hgk_avg(
                run_dirs=args.runs,
                output_dir=args.output_dir,
                output_prefix=args.output_prefix,
                acf_name=args.acf_name,
                viscosity_name=args.viscosity_name,
                plot=args.plot,
                plot_points=args.plot_points,
                plot_format=args.plot_format,
                verbose=args.verbose,
            )
        elif args.command in ("scan", "fit"):
            volume, temperature, p0 = _resolve_hgk_fit_params(
                args.acf_mean,
                args.volume, args.temperature, args.p0,
                args.verbose,
            )
            if args.command == "scan":
                run_hgk_scan(
                    acf_mean_path=args.acf_mean,
                    viscosity_mean_path=args.viscosity_mean,
                    output_dir=args.output_dir,
                    output_prefix=args.output_prefix,
                    volume=volume,
                    temperature=temperature,
                    p0_bar2=p0,
                    tau_low_ps=args.tau_low,
                    step=args.step,
                    fine_dt_ps=args.fine_dt,
                    plot=args.plot,
                    plot_points=args.plot_points,
                    plot_format=args.plot_format,
                    verbose=args.verbose,
                )
            else:
                run_hgk_final(
                    acf_mean_path=args.acf_mean,
                    viscosity_mean_path=args.viscosity_mean,
                    output_dir=args.output_dir,
                    output_prefix=args.output_prefix,
                    volume=volume,
                    temperature=temperature,
                    p0_bar2=p0,
                    tau_low_ps=args.tau_low,
                    tau_up_ps=args.tau_up,
                    tau_cut_ps=args.tau_cut,
                    fine_dt_ps=args.fine_dt,
                    log_points=args.log_points,
                    plot=args.plot,
                    plot_points=args.plot_points,
                    plot_format=args.plot_format,
                    verbose=args.verbose,
                )
        else:
            parser.error(f"unknown subcommand {args.command!r}")
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
