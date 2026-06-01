# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment setup

The project uses a conda-lock environment. Install once:

```bash
conda config --add channels conda-forge && conda config --set channel_priority strict
conda install -n base -c conda-forge conda-lock
conda-lock install -n greenbubble_gls --platform osx-arm64 envs/locks/conda-lock-osx-arm64.yml
conda activate greenbubble_gls
```

API tokens for data retrieval are loaded from `.env` (gitignored). Copy `.env.example` → `.env` and fill in your tokens before running the preprocessing steps.

## Running the model

**Snakemake workflow (recommended):**
```bash
snakemake -n                                  # dry-run: preview execution plan
snakemake --cores 4                           # full run
snakemake --cores 4 build_network             # single rule only
snakemake --cores 4 --forcerun solve_network  # force re-run a rule
```

**Standalone script (original, no Snakemake):**
```bash
python greenbubble_main.py
```

## Configuration system

Three layered YAML pairs (default → user override, deep-merged at import):

| Default file | User override | Purpose |
|---|---|---|
| `config/config.default.yaml` | `config/config.yaml` | Run targets, CO2 cost, energy year, n_flags, solver, stochastic/RH settings |
| `config/n_config.default.yaml` | `config/n_config.yaml` | Per-technology greenfield/brownfield capacity, ramp limits, min-load, `options:` section |
| `config/plots_config.default.yaml` | *(no override shown)* | Plot aesthetics |

User override files are gitignored. Only override the keys that differ from defaults.

`scripts/config.py` loads all files at import time and exposes module-level variables (`c.n_flags`, `c.stochastic`, `c.optimization`, `c.n_config`, `c.n_options`, …). Every script imports this module rather than parsing YAML directly.

## Snakemake DAG

```
retrieve_tech_data  ──┐
preprocess_inputs   ──┤
                       ├─► prepare_inputs ─► build_network ─► solve_network ─► plot_results
```

- Intermediate files land in `resources/` and `data/Inputs_{year}/` (both gitignored).
- Final outputs go to `outputs/single_analysis/{network_name}/plots/` and `/networks/`.
- The `onstart` hook in `Snakefile` checks the remote SHA of the technology-data CSV and auto-invalidates the local copy if it has changed.

## Output naming convention

The network name (and output folder) is constructed automatically from active flags and targets:

```
{B_}{H_}{RE_}{H2_}{MEOH_}{METH_}{SN_}{ST_}CO2_{co2}_{tD|tP}_H2_{h2}_MeOH_{meoh}_CH4_{ch4}_{year}_El_{el}_{DET|STC}[_TR_{res}]_{run_name}
```

`_TR_{res}` (e.g. `_TR_4h`) is appended only when `clustering.temporal.resolution` is set.

`build_network_name()` in `Snakefile` and `file_name_network()` in `scripts/helpers.py` both produce this string (one for Snakemake path resolution before any rule runs, one for the standalone script after the network is built).

## Key script responsibilities

| Script | Role |
|---|---|
| `scripts/parameters.py` | Central constants: all file paths, API token reads, snapshot index |
| `scripts/config.py` | Config loader; exposes all YAML values as module-level variables |
| `scripts/prepare_network.py` | Core PyPSA network builder (`build_network`) and technology flag dependency resolver (`network_dependencies`) |
| `scripts/helpers.py` | Annuity/cost functions, market price builders, custom Linopy constraints, solve wrapper (`build_model_solve_network`), export utilities |
| `scripts/technology_inputs.py` | Project-specific techno-economic data not in technology-data catalogue; uses CoolProp for compressor and dryer physics |
| `scripts/preprocessing.py` | Fetches and processes external market data (electricity prices, CO2 intensity, NG prices, capacity factors) |
| `scripts/solver_profiles.py` | Named HiGHS and Gurobi option dictionaries; default is `highs-fast` for local runs |
| `scripts/create_stoch_scenarios.py` | Couples multiple annual networks into a single stochastic LP |
| `scripts/snakemake_*.py` | Thin Snakemake wrappers that call the above modules with `snakemake.input/output` paths |

## Network architecture

The PyPSA network is structured as an industrial energy hub with internal buses for each carrier:
- **Energy carriers**: `El` (electricity), `H2`, `CO2`, `bioCH4`, `Methanol`, `Heat MT`, `Heat DH`, `Heat LT`
- **Technology agents** controlled by `n_flags` booleans: `biogas`, `central_heat`, `renewables`, `electrolysis`, `meoh`, `methanation`, `symbiosis`, `storage`
- **`symbiosis`** flag connects all internal carrier buses across agents; disabling it isolates each plant.
- **External market links**: electricity grid (buy/sell), NG grid (buy/sell), district heating (sell), biomass markets — all price-taker.

### Greenfield vs brownfield (`n_config`)

Each technology entry in `n_config.yaml` has three controlling parameters:

| `initial capacity` | `expansion` | `residual cost factor` | Mode |
|---|---|---|---|
| 0 | true | 0 | Pure greenfield |
| >0 | false | 0 | Brownfield fixed, sunk cost |
| >0 | false | >0 | Brownfield fixed, residual CAPEX charged |
| >0 | true | 0/> | Mixed: existing (EXI_ component) + expandable |

Existing capacity is added as a separate `EXI_`-prefixed component whose capital cost = `residual_cost_factor × annualised_cost`.

### Custom constraints

`add_max_RE_sales_constraint` (in `helpers.py`) enforces that electricity exported to the grid ≤ `max_RE_to_grid` × total RE consumed on-site. This is added to the Linopy model inside `build_model_solve_network` before solving.

## Optimization modes

- **Deterministic** (default): single annual LP, one energy year.
- **Stochastic** (`stochastic.stochastic: true`): multi-scenario LP with scenario coupling; EVPI calculated automatically when `EVPI: true`. MILP (committable links) is incompatible with stochastic mode.
- **Rolling horizon** (`rolling_horizon.enabled: true`): dispatch-only on a fixed-capacity network loaded from `rolling_horizon.network_path`; capacity expansion is bypassed entirely.

## Technology cost data

Base techno-economic data comes from a fork of the [technology-data](https://github.com/BertoGBG/technology-data) repository (`data/technology-data/outputs/costs_{year_EU}.csv`). Project-specific overrides (compressors, biomass dryer, GLS-specific equipment) are defined in `scripts/technology_inputs.py` as `tech_inputs` dict keyed by `("technology", "parameter")` tuples and merged via `helpers.merge_into_costs`.
