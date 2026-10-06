# GreenBubble tutorials

Ready-to-run configuration sets for the tutorial series in the documentation
(`docs/tutorial_*.rst`). Each subfolder holds a `config.yaml` and `n_config.yaml`
that **override** the committed defaults in `config/*.default.yaml`.

## How to run a tutorial

```bash
cp tutorials/<tutorial>/config.yaml   config/config.yaml
cp tutorials/<tutorial>/n_config.yaml config/n_config.yaml
snakemake --cores 4
```

`config/config.yaml` and `config/n_config.yaml` are gitignored user-override
files — copying a tutorial set over them is non-destructive to the repo (but it
overwrites your own current overrides, so back them up first if needed).

| Folder | Tutorial | Driver | Notes |
|---|---|---|---|
| `1_greenfield_demand` | 1.1 | demand | greenfield, biomethanation only, 10-y payback |
| `1_greenfield_price`  | 1.2 | price  | same, price-driven (all three products) |
| `2_brownfield`        | 2   | price  | existing biogas/wind/solar + residual cost, district heating |
| `2_brownfield_heat`   | 2b  | demand | brownfield with heat pump, TES DH and DH sales; electrolysis fixed |
| `3_rolling_horizon`   | 3   | demand | dispatch-only on the Tutorial 2b network (run `2_brownfield_heat` first) |
| `4_stochastic`        | 4   | price  | brownfield across 3 scenarios (pure LP: no committable, ramp limits null) |

All tutorials use the default HiGHS solver (no licence needed) and a coarse
temporal resolution so each solves in about five minutes on a laptop:
`8h` for Tutorials 1-3 and `24h` for the stochastic Tutorial 4, which holds
three years in one LP. Tutorial 3 requires the solved Tutorial 2b network;
its `rolling_horizon.network_path` points at the default Tutorial 2b output path.
