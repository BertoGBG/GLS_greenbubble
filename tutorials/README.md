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
| `3_rolling_horizon`   | 3   | price  | dispatch-only on the Tutorial 2 network (run T2 first) |
| `4_stochastic`        | 4   | price  | brownfield across 3 scenarios (pure LP: no committable, ramp limits null) |

All tutorials use `clustering.temporal.resolution: 3h` and the default HiGHS
solver so they solve quickly. Tutorial 3 requires the solved Tutorial 2 network;
set its `rolling_horizon.network_path` to match your Tutorial 2 output path.
