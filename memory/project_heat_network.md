---
name: Heat Network Extension
description: Steam/HT heat network implementation plan and progress for GreenBubble model
type: project
---

Phase A implemented on branch `heat_integration` (2026-04-24).

**What was done (Phase A — infrastructure):**
- `config/config.yaml` + `config_default/config.yaml`: added `heat_integration` section with `delta_T_min: 10.0` [°C] and `steam_pressure_levels` (Steam HP 40 bar, Steam MP 10 bar, Steam LP 3 bar)
- `scripts/technology_inputs.py`: extended `symbiosis_data` with `T_return`, `P_return`, `open_loop` fields on all heat entries; added `Heat HT max` (350°C/60 bar), `Steam HP`, `Steam MP`, `Steam LP` entries
- `scripts/helpers.py`: added `enthalpy_splits()` (CoolProp-based interval fractions, phase-change aware) and `hx_capital_cost_per_MW()` (phase-aware HX annualised cost)
- `scripts/prepare_network.py`: modified `add_methanolisation_cap_exp` to add `Heat HT` bus (direction +1, supplying to network), `bus6`/`efficiency6` on the methanolisation Link, using `GL_eff.at["Heat HT", "Methanol plant"]` = 0.22

**Why:** The Methanol plant in the GreenLab Excel file has a `Heat HT` row (350–200°C) with value 0.5 MW per MW CO2 input (normalised: 0.22). This is high-temperature heat that was not captured in the model before.

**Phase B (not yet done — network topology):**
- Add `Heat HT`, `Steam HP/MP/LP` buses to the PyPSA network
- Add condenser/evaporator Links between steam buses and hot-water buses
- Replace existing process→bus Links with multi-output Links using CoolProp `enthalpy_splits` fractions
- Add makeup-water Generator for open-loop steam consumers (future: steam dryer process)
- Use `hx_capital_cost_per_MW()` instead of generic "DH heat exchanger" cost for high-T connections

**Key design decisions:**
- Steam pressure levels fixed (not LP decision variables): pressure is derived from saturation T; only relevant for steam turbine work output, which this model doesn't have
- `delta_T_min = 10°C` in config (was not previously configurable)
- `open_loop = False` for all current entries; will be `True` for future steam dryer
- `enthalpy_splits` uses supply-pressure for vapour phase, saturation-line for condensate

**How to apply:** When adding new heat streams or processes, use `enthalpy_splits()` to compute multi-output Link efficiencies. Use `hx_capital_cost_per_MW()` for capital cost. Set `open_loop=True` in symbiosis_data for any steam reactant consumers and add a makeup-water Generator.
