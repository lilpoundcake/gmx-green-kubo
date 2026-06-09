# gmx-green-kubo

A pure-Python CLI that computes the shear viscosity of a liquid from the
pressure-tensor autocorrelation of an equilibrium MD simulation. Two
workflows are exposed through a single tool:

- **Classical Green-Kubo** (`gk`) — direct integration of the symmetrised
  pressure tensor autocorrelation, following the formulation used by Prass
  et al., [J. Chem. Inf. Model. 2023, 63, 6957](https://doi.org/10.1021/acs.jcim.3c00947).
- **Hybrid Green-Kubo / hGK** (`acf`, `average`, `scan`,
  `fit`) — multi-run averaging plus a stretched-exponential tail fit
  that extrapolates the integral past the noisy long-time region, following
  Meel & Mogurampelly,
  [J. Phys. Chem. Lett. 2026, 17, 4016](https://doi.org/10.1021/acs.jpclett.5c03863).

Both modes consume GROMACS `.xvg` output, share the same `@file` config
mechanism, and produce interactive Plotly HTML plots alongside the
numerical results. Static PNG/SVG output is opt-in via `--plot-format`
(needs `kaleido`). Plot file size is independent of the input trajectory
length — every trace is capped at `--plot-points` samples (default 1500),
so even a multi-GB `.xvg` produces ~50 KB plot HTMLs.

## Inspired by

- The C++ tool **`gmx_gk_autocorr`** by Prass et al. — [github.com/MolSimGroup/gmx_gk_autocorr](https://github.com/MolSimGroup/gmx_gk_autocorr)
  ([doi:10.1021/acs.jcim.3c00947](https://doi.org/10.1021/acs.jcim.3c00947))
  — the classical Green-Kubo (`gk`) subcommand is a Python port that
  produces bit-exact output on the original reference dataset.
- The **hybrid Green-Kubo (hGK)** framework by Meel & Mogurampelly
  ([doi:10.1021/acs.jpclett.5c03863](https://doi.org/10.1021/acs.jpclett.5c03863))
  — the `acf` / `average` / `scan` / `fit` pipeline reimplements its
  multi-trajectory averaging and stretched-exponential tail
  extrapolation in a single CLI.


## Theory

The Green-Kubo expression for the shear viscosity of an isotropic liquid is

$$\eta = \lim_{t\to\infty} \eta(t) = \frac{V}{k_{\mathrm{B}} T} \int_0^{\infty} \left\langle P_{\alpha\beta}(0)\,P_{\alpha\beta}(t) \right\rangle \, dt$$

where $V$ is the box volume, $T$ the temperature and $P_{\alpha\beta}$ an
off-diagonal Cartesian component of the instantaneous pressure tensor. To
reduce statistical noise the running integral is averaged over the six
independent stress components.

**`gk` workflow.** Uses the six symmetrised combinations introduced by
Daivis & Evans:

- $(P_{xy}+P_{yx})/2$, $(P_{xz}+P_{zx})/2$, $(P_{yz}+P_{zy})/2$,
- $(P_{xx}-P_{yy})/2$, $(P_{xx}-P_{zz})/2$, $(P_{yy}-P_{zz})/2$.

The autocorrelation of each is computed with an FFT and the six running
integrals are averaged. This matches the original C++ tool's formulation
bit-for-bit on its example dataset.

**`acf` / `average` / `scan` / `fit` workflow.** Uses the six raw off-diagonal components
($P_{xy}, P_{xz}, P_{yz}, P_{yx}, P_{zx}, P_{zy}$) and the *unbiased*
linear autocorrelation estimator (zero-padded FFT, divided by $N-k$ pairs
at each lag) rather than the circular one. Multiple trajectories are
averaged; the SACF tail is then fitted with a stretched exponential

$$C_{\mathrm{tail}}(t) = a_0 \, \exp\left[-(t/\tau)^{\beta}\right],$$

and the tail integral is computed analytically (numerically, on a fine
grid) up to a chosen $\tau_{\mathrm{cut}}$. The final viscosity is

$$\eta = \eta(\tau_{\mathrm{low}}) + C(0) \int_{\tau_{\mathrm{low}}}^{\tau_{\mathrm{cut}}} C_{\mathrm{tail}}(t)\,dt,$$

where $\eta(\tau_{\mathrm{low}})$ is the running integral of the measured
SACF evaluated at the fit-window's lower bound. This decouples the final
estimate from the noisy long-time tail of any single trajectory.


## Installation

The recommended setup uses the bundled `environment.yml` (conda/micromamba)
and then installs the CLI via pip:

```bash
micromamba create -f environment.yml      # or: conda env create -f environment.yml
micromamba activate green-kubo
pip install .                              # or: pip install -e .  (editable)
```

After installation, the `gmx-gk-autocorr` command is on `$PATH`:

```bash
gmx-gk-autocorr --version
gmx-gk-autocorr --help
```

The script also runs directly with `python3 gmx_gk_autocorr.py …` without
installation, if you already have NumPy / SciPy / Plotly available.

The conda/micromamba env includes `kaleido` by default, so
`--plot-format png|svg` works out of the box. For the pip-only path,
opt in with `pip install '.[static-plots]'`.

### Requirements

- Python ≥ 3.9
- NumPy ≥ 1.20
- SciPy ≥ 1.7 — used by `scan` / `fit` for the stretched-exponential fit.
- Plotly ≥ 5.0 — interactive HTML plots; disable with `--no-plot`.
- Kaleido ≥ 0.2 *(optional)* — required only for `--plot-format png|svg`.
  Pre-installed via `environment.yml`. For the pip-only path:
  `pip install '.[static-plots]'`.
- Polars ≥ 1.0 *(optional)* — auto-detected. Speeds up I/O on xvg files
  with single-character column separators (5–12× on the bundled hGK
  `pressure_components.xvg`). Variable-whitespace GROMACS output is
  unaffected and continues through `np.loadtxt`.
- PyFFTW ≥ 0.13 *(optional)* — auto-detected. Plugged in as a scipy.fft
  backend at import time; gives ~1.5–2× faster FFT than the bundled
  pocketfft.
- All three optional extras: `pip install '.[fast,static-plots]'` (or
  use `environment.yml` which already lists them).


## Quick start

```bash
# Classical Green-Kubo on one trajectory (--dt defaults to 0.002 ps)
gmx-gk-autocorr gk energy.xvg conf.gro -v

# Static PNG plots instead of interactive HTML (needs kaleido)
gmx-gk-autocorr gk energy.xvg conf.gro --plot-format png -v

# Hybrid Green-Kubo from five independent trajectories
gmx-gk-autocorr acf run*/pressure_components.xvg \
    --gro conf.gro --temperature 303.0 --dt 0.002 -o output/ -v
gmx-gk-autocorr average output/run* -o output/ -v
gmx-gk-autocorr scan output/acf_mean.dat output/viscosity_mean.dat \
    --tau-low 0.2 -v                       # diagnose convergence
gmx-gk-autocorr fit output/acf_mean.dat output/viscosity_mean.dat \
    --tau-low 0.2 --tau-up 5.0 --tau-cut 5000 -v
```


## Subcommands

The five subcommands form two independent workflows. The hybrid pipeline
chains four subcommands; outputs of one feed the next:

```
  [ 1 trajectory ]                      [ N trajectories ]
         │                                      │
         ▼                                      ▼
     ┌──────┐                                ┌──────┐
     │  gk  │                                │  acf │   (per run)
     └──┬───┘                                └──┬───┘
        │ acf.dat                              │ acf.dat
        │ viscosity.dat                        │ viscosity.dat
        │ metadata.dat                         ▼
   final η(t) ready              ┌────────────────────┐
                                  │      average       │
                                  └────────┬───────────┘
                                            │ acf_mean.dat
                                            │ viscosity_mean.dat
                                            │
                              ┌─────────────┴────────────┐
                              ▼                          ▼
                          ┌──────┐                   ┌─────┐
                          │ scan │                   │ fit │
                          └──┬───┘                   └──┬──┘
                              │ window_scan.dat          │ viscosity_final.dat
                              │ (diagnostics)            │ + html plots
                              │                          │ (final η)
                              ▼                          ▼
                       choose τ_up                 final viscosity
```

All subcommands share these flags:

| Flag | Description |
|---|---|
| `-o`, `--output-dir DIR` | Output directory (default: current dir). |
| `-p`, `--output-prefix STR` | Filename prefix (default: empty). |
| `--no-plot` | Skip writing plots; data files are still produced. |
| `--plot-points N` | Cap on points per Plotly trace (default 1500). Subsamples log-spaced on log-x plots, linear-spaced otherwise. Data files (`.dat`) keep full density. |
| `--plot-format {html,png,svg}` | Plot output format. `html` (default) writes interactive Plotly. `png`/`svg` write static images via `kaleido`. |
| `-v`, `--verbose` | Print progress to stdout. |

### `gk` — classical Green-Kubo (one trajectory)

```bash
gmx-gk-autocorr gk energy.xvg conf.gro [options]
# Equivalent shorthand (no subcommand keyword):
gmx-gk-autocorr energy.xvg conf.gro
```

| Argument | Description |
|---|---|
| `energy.xvg` | GROMACS `gmx energy` output. Required columns in order: time, Temperature, Pressure, Pres-XX, Pres-XY, Pres-XZ, Pres-YX, Pres-YY, Pres-YZ, Pres-ZX, Pres-ZY, Pres-ZZ. |
| `conf.gro` | GROMACS structure file. Only the last line (cubic box edge, nm) is read. |
| `--dt PS` | *(optional)* override the per-frame time step in ps. When set, the time array is rebuilt as `arange(N) * dt` and the xvg's first column is ignored. Use this only when the xvg's time column is broken (frame indices, wrong units). |
| `--subtract-mean` | *(optional)* compute the ACF on `δP = P − ⟨P⟩` instead of the raw signal. Off by default to match the original C++ tool. Turn on if the verbose diagnostic reports `bias/signal > 0.1` for any channel. |
| `--max-lag-fraction F` | Stop the running integral at fraction F of the trajectory length (default 0.5 = N/2, the hard upper bound of the circular-ACF method). For long trajectories (≫100 ps) the SACF has decayed long before N/2 and integrating further just integrates noise. Try `0.001–0.01` on multi-ns trajectories to keep the integral near the SACF-decay region. |

Outputs:

| File | Contents |
|---|---|
| `acf.dat` | Time (ps), 6 channel ACFs, average — units bar². Row 0 is t=0 (the lag-0 value, C(0)). |
| `viscosity.dat` | Time (ps), 6 channel Green-Kubo integrals, average — units mPa·s. |
| `metadata.dat` | Average volume (nm³) and temperature (K). |
| `acf.html`, `viscosity.html` | Interactive Plotly plots. |

> **Reading `viscosity.dat` for long trajectories.** The viscosity is
> the **plateau** of the running η(t) curve, not the value at the last
> row. After the SACF decays (typically ≲10 ps for water-like liquids)
> the running integral is a random walk around the plateau — it can
> drift positive or negative, especially with `--subtract-mean`. The
> tool prints the **peak value of η(t) and its time** in verbose mode
> (`-v`) as a single-number estimate; open `viscosity.html` for the
> full curve.
>
> For multi-ns trajectories you should pass `--max-lag-fraction` to
> stop the integration in the SACF-decay region instead of letting it
> walk through ns of noise. Rule of thumb: `--max-lag-fraction
> (100 ps / T_trajectory)` — e.g. on a 100 ns trajectory pass
> `--max-lag-fraction 0.001`. Alternatively use the `acf` →
> `average` → `scan` → `fit` pipeline, which fits and analytically
> extrapolates the tail instead of just integrating it.

### `acf` — per-trajectory hGK step

```bash
gmx-gk-autocorr acf XVG [XVG ...] [options]
```

| Argument / Flag | Description |
|---|---|
| `xvg` (positional, 1+) | One or more `.xvg` files (shell glob expanded). Two layouts accepted: 7-column hGK `pressure_components.xvg` (time, Pxy, Pxz, Pyz, Pyx, Pzx, Pzy) or 12-column legacy `energy.xvg` (off-diagonals auto-extracted). |
| `--gro PATH` | Cubic `.gro` file → volume. |
| `--volume V` | Volume in nm³ (overrides `--gro`). |
| `--temperature T` | Temperature in K (auto-detected when the .xvg has a Temperature column). |
| `--log-points N` | Log-spaced sample size for the output (default 10000). |
| `--dt PS` | Per-frame time step in ps (default `0.002`). Applied uniformly to all input xvgs. The time array is rebuilt as `arange(N) * dt` and the xvg's first column is ignored. |
| `--subtract-mean` | *(optional)* compute the unbiased SACF on `δP = P − ⟨P⟩` instead of the raw signal. Off by default to match the hGK reference; turn on if the verbose diagnostic warns about channel means. |
| `--jobs N`, `-j N` | Process multiple xvg files in parallel via ProcessPoolExecutor (default 1 = sequential). `0` or negative uses all available cores. Each worker single-threads its FFT to avoid over-subscription. Memory scales linearly with N — rule of thumb ~3 GB per worker at 5 M frames. |

With one input the outputs go into `-o` directly. With multiple inputs each
trajectory is processed into its own subdirectory of `-o`, auto-named after
the input file's parent folder (e.g. `run3/pressure.xvg → output/run3/`).

Outputs per run:

| File | Contents |
|---|---|
| `acf.dat` | Log-spaced normalised SACF (8 columns: time, avg, 6 channels). Header carries V, T, dt and C(0). |
| `viscosity.dat` | Log-spaced running viscosity in mPa·s (8 columns: time, avg, 6 channels). |
| `acf.html`, `viscosity.html` | Interactive plots. |

### `average` — multi-trajectory averaging

```bash
gmx-gk-autocorr average RUN_DIR [RUN_DIR ...] [options]
```

| Argument / Flag | Description |
|---|---|
| `runs` (positional, 1+) | Directories containing `acf.dat` and `viscosity.dat`. |
| `--acf-name`, `--viscosity-name` | Override the expected per-run filenames (defaults `acf.dat`, `viscosity.dat`). |

Reads each run's metadata, warns on V/T mismatch, averages C(0) across runs.
Writes:

| File | Contents |
|---|---|
| `acf_mean.dat` | Time + per-column mean + per-column standard deviation. Metadata header carries the mean C(0) and the run-to-run spread. |
| `viscosity_mean.dat` | Same layout for the running viscosity. |
| `acf_mean.html`, `viscosity_mean.html` | Plots with ±1σ shaded bands. |

### `scan` — fit-window convergence scan

```bash
gmx-gk-autocorr scan acf_mean.dat viscosity_mean.dat \
    --tau-low T_LOW [options]
```

Sweeps the upper bound τᵤ of the stretched-exponential fit window and
reports the resulting extrapolated viscosity vs window length. Use this to
diagnose convergence and pick `--tau-up` for `fit`.

| Flag | Description |
|---|---|
| `--tau-low` (required) | Lower bound of the SACF tail fit window (ps). |
| `--step N` | Index step between scan upper bounds (default 10). |
| `--fine-dt` | Fine-grid step (ps) for both the analytical tail integral and the unit prefactor (default 0.001). |
| `--volume`, `--temperature`, `--p0` | Optional overrides; default values are taken from `acf_mean.dat`'s metadata header. |

Outputs:

| File | Contents |
|---|---|
| `window_scan.dat` | Three columns: fit-window length (ps), extrapolated viscosity (mPa·s), dη/dτ. |
| `window_scan.html` | Interactive plot of η vs window length. |

### `fit` — final tail-extrapolated viscosity

```bash
gmx-gk-autocorr fit acf_mean.dat viscosity_mean.dat \
    --tau-low T_LOW --tau-up T_UP --tau-cut T_CUT [options]
```

Fits a stretched exponential to the SACF on `[τ_low, τ_up]`, extrapolates
to `τ_cut`, and integrates to obtain the final viscosity.

| Flag | Description |
|---|---|
| `--tau-low` (required) | Lower bound of the fit window (ps). |
| `--tau-up` (required) | Upper bound of the fit window (ps). |
| `--tau-cut` (required) | Upper bound for the tail extrapolation (ps), e.g. 5000. |
| `--fine-dt` | Fine-grid step (ps) for both the analytical tail integral and the unit prefactor (default 0.001). |
| `--log-points` | Log-spaced points in the tail output (default 50). |
| `--volume`, `--temperature`, `--p0` | Optional overrides; default values are taken from `acf_mean.dat`'s metadata header. |

Outputs:

| File | Contents |
|---|---|
| `viscosity_final.dat` | Log-spaced extrapolated η(t); last row is the final viscosity. |
| `acf_fit.html` | SACF data with the fit and extrapolated tail overlaid. |
| `viscosity_final.html` | Running η(t) with the extrapolated tail overlaid. |


## Plot file size and static images

Plotly serialises every trace's (x, y) values into the HTML, so file size
scales linearly with the number of points unless capped. To keep HTML
output small regardless of input size, every plot trace is subsampled to
**`--plot-points`** samples (default 1500) before reaching Plotly:

- **log-x plots** (`acf`, `average`, `fit`) — log-spaced indices, so density
  is preserved across multiple decades of time.
- **linear-x plots** (`gk`, `scan`) — evenly-spaced indices.

The numerical `.dat` files are unaffected; they keep their full or
`--log-points`-controlled density. Example: for a 7 GB input `.xvg`, plot
HTMLs are ~50 KB after this change (previously ~500 MB).

To produce static images instead of interactive HTML, pass
`--plot-format png` (or `svg`). This needs `kaleido` — already in
`environment.yml`; pip-only users can `pip install '.[static-plots]'`.
PNG outputs are typically ~30–80 KB and useful for headless or
publication contexts.

```bash
# Tiny HTML (default)
gmx-gk-autocorr gk energy.xvg conf.gro -o out/

# Even tinier static PNG
gmx-gk-autocorr gk energy.xvg conf.gro --plot-format png -o out/

# More points for a smoother curve on a very long trajectory
gmx-gk-autocorr gk energy.xvg conf.gro --plot-points 5000 -o out/
```


## Choosing τ_low, τ_up and τ_cut

The three fit parameters control the stretched-exponential tail extrapolation
of the hGK pipeline. Each has a specific physical meaning and a diagnostic
plot you can use to pick it. As a rule of thumb (good first guess for simple
liquids near room T): **`--tau-low 0.2 --tau-up 5 --tau-cut 5000`** (units: ps).

| Parameter | Meaning | Typical range | How to choose |
|---|---|---|---|
| `--tau-low` | Lower bound of the SACF fit window | 0.1–0.5 ps | Lowest time at which the SACF is smooth and monotonically decaying (past the cage-rattling oscillations). |
| `--tau-up` | Upper bound of the SACF fit window | 1–20 ps | The time where the **scan plateau** is reached — read from `scan`. |
| `--tau-cut` | How far to extrapolate the fitted tail | 10³–10⁴ ps | Large enough that the running η has flattened — read from `viscosity_final.html`. |

### Step 1 — pick `τ_low` from `acf_mean.html`

Open the averaged SACF plot (log x-axis). You'll see two regimes:

1. **Early dynamics** (~< 0.05 ps for water): oscillations from
   intermolecular vibrations. Do *not* include these in the fit.
2. **Slow tail**: smooth, monotonic decay. This is what gets fitted.

Set `τ_low` to the time where the curve first becomes monotonically
decaying. The choice is not very sensitive — anywhere inside the smooth
region works.

### Step 2 — pick `τ_up` from `scan`

Run a scan with `τ_low` chosen and no fit-window upper bound:

```bash
gmx-gk-autocorr scan output/acf_mean.dat output/viscosity_mean.dat --tau-low 0.2 -v
```

Open `window_scan.html`. The plot has three characteristic regions:

```
η (mPa·s)
  │            ___________
  │           /           \____      ← noise region (η drifts as the fit chases noise)
  │          /                       ← PLATEAU — pick τ_up here
  │     ____/
  │    /                             ← unstable region (window too short to constrain)
  │   /
  └────────────────────────────────► fit-window length τ_up − τ_low (ps)
```

Pick `τ_up = τ_low + <plateau window length>`. If there is no clear
plateau, the averaged SACF is still too noisy — run more independent
trajectories and re-`average` them.

### Step 3 — pick `τ_cut` and verify convergence

Set `τ_cut` to something comfortably long (e.g. 5000 ps) and run
`fit`. Open `viscosity_final.html`: the dashed extrapolated η(t)
curve **must visibly flatten** before reaching `τ_cut`. If it's still
rising at the end, increase `τ_cut` and re-run.

A simple convergence check is to run `fit` twice with e.g.
`--tau-cut 2000` and `--tau-cut 5000`. If the two final η values differ
by less than ~0.1 %, you've converged.

### When the method breaks down

- **No clear plateau in `scan`** → not enough independent
  trajectories; the averaged SACF tail is too noisy.
- **η changes a lot with `τ_cut`** → `τ_cut` is too small; multiply by 10.
- **Different `τ_low` gives very different η** → a single stretched
  exponential doesn't describe your tail well; try a narrower
  `[τ_low, τ_up]` or verify that `τ_low` is past the oscillatory regime.
- **Very stretched fit (β ≪ 0.5) with tiny τ** → the fit is fragile;
  change `τ_low` by ±50 % as a sensitivity check and confirm the final η
  moves by less than a few %.


## Sample-mean bias and `--subtract-mean`

### The symptom

`gk` and `acf` on the same trajectory give wildly different viscosities
(e.g. **5 vs 14 mPa·s**, with the ratio close to 1:2). The per-channel
viscosities printed by both tools are tens of mPa·s instead of the
fraction-of-a-mPa·s expected for water-like liquids.

This is not a bug in the algorithms — it's a finite-trajectory artifact
of the input.

### The cause

Both algorithms estimate

$$\eta(t) = \frac{V}{k_{\mathrm{B}}T} \int_0^{t} \big\langle P_{\alpha\beta}(0)\, P_{\alpha\beta}(t') \big\rangle\, dt'$$

with the **raw** correlator $\langle P\,P \rangle$. The theoretical
identity used to justify this is $\langle P_{\alpha\beta}\rangle_{\mathrm{eq}} = 0$
for off-diagonal stress: at true equilibrium the off-diagonal pressure
has zero mean, so $\langle P(0) P(t) \rangle = \mathrm{Cov}(P(0), P(t))$
and the integral gives the viscosity.

On a **finite trajectory**, that identity holds only approximately.
Write $P = \mu + \delta P$ where $\mu$ is the sample mean and $\delta P$
is the fluctuation:

$$\big\langle P(0)P(t) \big\rangle_{\mathrm{sample}} \;=\; \mu^2 \;+\; \big\langle \delta P(0)\,\delta P(t) \big\rangle$$

The $\mu^2$ floor is *constant* in $t$, so it integrates to a
**linear-in-t bias** in the running viscosity:

$$\eta_{\mathrm{bias}}(t) \;=\; \frac{V}{k_{\mathrm{B}}T}\, \mu^{2}\, t$$

This is *not* viscosity — it's just a unit-bearing leftover from a
non-vanishing sample mean. On any data where $\mu$ has not been
sufficiently averaged to zero (short trajectories, non-equilibrium
systems, systematic pressure-control artifacts), this bias dominates
the result.

| Algorithm | Integration window | Bias contribution |
|---|---|---|
| `gk` | t = N·dt/2 | $V\mu^2(N\cdot dt/2)/(k_{\mathrm{B}}T)$ |
| `acf` (single-traj) | t = N·dt | twice the gk bias + tail noise |

That ratio is exactly the 1:2 pattern in the 5-vs-14 symptom.

### How to detect it

Run with `-v`. The tool prints the sample mean, sample standard
deviation, and approximate standard-of-mean for every channel, and
emits a warning when the sample mean is more than 3 standard-of-means
above zero:

```
Per-channel sample means (bar):
  Pxy: mean = +0.9399  std =   39.92  std-of-mean ≈  0.1031  |mean|/sem ≈   9.1
  Pxz: mean = +0.9389  std =   39.95  std-of-mean ≈  0.1031  |mean|/sem ≈   9.1
  ...
WARNING: channels {Pxy, Pxz, …} have sample means well above sampling
noise. The Green-Kubo integral picks up a μ²·V·t/(kB·T) bias from each.
Re-run with --subtract-mean …
```

The check uses a naïve $\sigma/\sqrt{N}$ estimator that ignores
autocorrelation, so on short trajectories it is over-eager (it can fire
even when the true mean is zero and the apparent mean is sampling
noise). Use it as a prompt to inspect your data, not as a hard verdict.

### The fix

Pass `--subtract-mean` (available on `gk` and `acf`). With the flag,
the autocorrelation is computed on $\delta P = P - \langle P \rangle$
instead of $P$ — the fluctuation form that the
fluctuation-dissipation theorem actually prescribes:

```bash
gmx-gk-autocorr acf my.xvg --gro conf.gro --temperature 303 \
    --dt 0.002 --subtract-mean -o out/ -v
```

After this, the bias is gone:

- $\eta_{\mathrm{computed}}(t) \approx \eta_{\mathrm{true}}$ (modulo
  long-time noise)
- `gk` and `acf` agree with each other to within tail-extrapolation
  noise — that's the test that the fix worked
- the residual scatter at long times is real statistical noise that
  needs more trajectories or the hGK pipeline's tail extrapolation
  (`scan` + `fit`)

### Watch out: negative running viscosity on long trajectories

After the bias is removed, the SACF is the autocorrelation of pure
fluctuations and decays to zero at long lag. Past the decay (typically
≲ 10 ps), every additional lag contributes pure noise around zero, and
the cumulative integral becomes a **random walk** — it can drift to
negative values purely from noise accumulation, with no physical
meaning.

On a 100 ns trajectory with `gk --subtract-mean`, the default integration
caps at N/2 ≈ 50 ns — that's ~50 000 ps of noise after a ~10 ps SACF
decay. The last row of `viscosity.dat` (at t = 50 ns) is essentially
that random walk's endpoint, *not* the viscosity.

**The viscosity is the plateau value**, not the cap value. Two fixes:

1. **Cap the integration earlier** with `--max-lag-fraction`:

   ```bash
   # On a 100 ns trajectory: cap at 100 ps (= 0.1% of trajectory)
   gmx-gk-autocorr gk md.xvg md.gro --subtract-mean --max-lag-fraction 0.001 -v
   ```

   `-v` will then print the `Running η peak: <value> at t = <t> ps`
   line, which is your viscosity estimate.

2. **Use the hybrid pipeline** (`acf` → `average` → `scan` → `fit`)
   which fits a stretched exponential to the SACF tail and integrates
   the fit analytically to a chosen `τ_cut`. This is the right
   workflow for any multi-ns trajectory.

### Why it's not the default

Two reasons:

1. **Bit-exactness with the references we ported.** The original C++
   `gmx_gk_autocorr` and the hGK reference Python scripts
   (Meel & Mogurampelly) both use raw $\langle P\,P\rangle$. Keeping
   the default identical preserves byte-for-byte reproducibility of
   both:
   - `gk` on `example_files/` ≡ C++ `SELFvisco.xvg` (`max abs diff = 0`)
   - `acf+average+fit` on `spce_water/` ≈ 0.6585 mPa·s (was 0.6637 before
     the noisy-tail truncation and trapezoidal integration fixes; same
     ~0.66 mPa·s ballpark as the hGK paper)
2. **Short-trajectory bias trade-off.** On truly-zero-mean
   well-equilibrated data, subtracting the noisy sample mean removes a
   small amount of real signal (the low-frequency Fourier component
   contaminated by sampling noise). For the bundled `spce_water/`
   example (10 ps × 5 runs), the FDT-correct flag drops η from
   0.6585 → 0.4115 mPa·s, the difference being mostly this
   short-trajectory artifact rather than the bias.

The right workflow is therefore: **run without the flag first**,
inspect the verbose output, and turn the flag on if the diagnostic says
your channel means are too big to be sampling noise. The published
references (textbooks, Allen & Tildesley §4, Frenkel & Smit, FDT
derivation) all agree that the fluctuation form is the strictly
correct one.


## Config files (`@file`)

Any argument can be loaded from a config file with argparse's `@file`
syntax. Each line may carry a single token, a `flag value` pair, or any
mix; `#` starts a comment.

`system.cfg`:

```
# Shared simulation parameters (overrides for acf_mean.dat metadata)
--volume       121.734    # nm^3
--temperature  303.0      # K
```

`fit.cfg`:

```
--tau-low 0.2
--tau-up  5.0     # ps
--tau-cut 5000    # ps
```

Run:

```bash
gmx-gk-autocorr fit acf_mean.dat viscosity_mean.dat @system.cfg @fit.cfg -v
```

Multiple `@file`s are concatenated in order, and plain CLI flags can be
mixed in alongside them. This subsumes the reference hGK workflow's
`in.viscosity` / `in.hGK_scan` / `in.hGK_final` input files.


## Units

All time values across the CLI and output files are in **picoseconds** —
`--tau-low`, `--tau-up`, `--tau-cut`, `--fine-dt`, time axes in `acf.dat` /
`viscosity.dat` / etc. Internal conversions to SI seconds happen in the
Green-Kubo prefactor; you should never need to think about them.

| Quantity | CLI / file unit |
|---|---|
| Time (τ, dt, t) | ps |
| Volume | nm³ |
| Temperature | K |
| Pressure tensor element | bar |
| SACF C(t), C(0) | bar² |
| Viscosity η | mPa·s |

## Auto-detected parameters

Whenever it can, the tool fills in the four numbers needed by the Green-Kubo
prefactor automatically — you rarely need to pass them on the command line:

| Parameter | Source | Override flag |
|---|---|---|
| Volume V | last line of `--gro` (cubic box edge → V = edge³) | `--volume` |
| Temperature T | Temperature column of a 12-column `energy.xvg` | `--temperature` |
| Sample step dt | `--dt` flag (default `0.002` ps) — the xvg's time column is *not* read | `--dt PS` |
| Zero-lag SACF C(0) | computed by `acf`, averaged by `average`, embedded in `acf_mean.dat` header | `--p0` |

`scan` and `fit` read V, T and C(0) directly from `acf_mean.dat`'s metadata
header, so a pipeline like `acf → average → fit` needs no V/T/C(0) flags
after the first step.

> **Note on `--fine-dt`.** The fine-grid step used by `scan` / `fit` is
> *not* tied to the simulation's sampling rate — it discretises the
> analytical stretched-exponential fit, not the data. The default 0.001 ps
> works for any liquid where the fitted relaxation time τ ≳ 0.01 ps;
> decrease it if `fit` reports a sub-0.005 ps τ.

> **Note on `--dt` (gk / acf only).** The time axis is **always** rebuilt
> from this flag as `arange(N) * dt`; the xvg's first column is ignored.
> The default is **0.002 ps**, which matches the typical leap-frog MD
> integration step. If your trajectory was sampled at a different rate,
> pass the matching value — e.g. `--dt 0.001` for the bundled
> `spce_water/` example, or `--dt 0.004` for the bundled `example_files/`
> dataset. This is intentionally an explicit user input: reading the
> xvg's time column was the source of the previous "50k points labelled
> as ps" bug.

> **Note on `--subtract-mean` (gk / acf only).** By default the ACF is
> computed on the raw pressure-tensor signal, matching the original
> C++ tool and the hGK reference. If `gk` and `acf` disagree by a
> factor close to 2× on the same data, or if per-channel viscosities
> are unreasonably large, the cause is almost certainly a non-zero
> sample mean producing a linear-in-t bias. See **Sample-mean bias and
> `--subtract-mean`** below for a full explanation and the fix.


## Example data

The repository ships two ready-to-run datasets.

### `gk` example — `example_files/`

A short NVT energy file plus a cubic box `.gro`. Sampling rate dt =
0.004 ps, so override the default:

```bash
gmx-gk-autocorr gk example_files/shortenergy.xvg example_files/confout.gro \
    --dt 0.004 -v

# Same data, static PNG plots instead of interactive HTML
gmx-gk-autocorr gk example_files/shortenergy.xvg example_files/confout.gro \
    --dt 0.004 --plot-format png -v
```

### Hybrid example — `spce_water/`

Five independent SPCE-water trajectories (V = 121.734 nm³, T = 303 K, dt =
0.001 ps), one `pressure_components.xvg` per `runN/` subdir. Full hybrid
pipeline on this set:

```bash
# This dataset was sampled at dt = 0.001 ps, so override the default 0.002 ps:
gmx-gk-autocorr acf spce_water/run*/pressure_components.xvg \
    --volume 121.734 --temperature 303.0 --dt 0.001 -o spce_out/ -v
gmx-gk-autocorr average spce_out/run* -o spce_out/ -v
gmx-gk-autocorr scan spce_out/acf_mean.dat spce_out/viscosity_mean.dat \
    --tau-low 0.2 -v
gmx-gk-autocorr fit spce_out/acf_mean.dat spce_out/viscosity_mean.dat \
    --tau-low 0.2 --tau-up 5 --tau-cut 5000 -v
# → Final viscosity at τ_cut = 5000.0 ps: 0.6585 mPa·s
```


## Citations

If you use the classical (`gk`) workflow, please cite the publication that
introduced its underlying implementation:
[doi:10.1021/acs.jcim.3c00947](https://doi.org/10.1021/acs.jcim.3c00947).

If you use the hybrid Green-Kubo (`acf` / `average` / `scan` / `fit`) workflow, please cite:

> Meel, A. K.; Mogurampelly, S.  
> *A hybrid Green-Kubo (hGK) framework for calculating viscosity from short MD simulations*  
> J. Phys. Chem. Lett. **2026**, 17, 4016–4022. [doi:10.1021/acs.jpclett.5c03863](https://doi.org/10.1021/acs.jpclett.5c03863)
