# SPDX-License-Identifier: MIT
"""Snakemake wrapper: nos_intersect (Tier 3a) of the staged NOS pipeline.

Intersects all mga.robustness.years' adaptive hulls and finds the Chebyshev
centre of the intersection -- the vendored geometry path
(scripts.vendor.near_opt_geometry, via scripts.near_optimal.chebyshev_centre /
intersect_hulls), mirroring the reference implementation's
intersect_near_optimal.py. See rules/near_optimal_staged.smk.
"""
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts import config as c
from scripts import near_optimal as nos

mga = c.mga

per_year_points = {}
years = list(mga["robustness"]["years"].keys())
for year, path in zip(years, snakemake.input.points):
    per_year_points[year] = pd.read_csv(path)

keys = list(next(iter(per_year_points.values())).columns)

hulls = [pts.to_numpy() for pts in per_year_points.values() if len(pts) >= len(keys) + 1]
if len(hulls) < 1:
    raise RuntimeError(
        f"Not enough years produced a usable hull (need >= {len(keys) + 1} points each; "
        f"got {[len(p) for p in per_year_points.values()]})."
    )

cheb = nos.chebyshev_centre(hulls, keys=keys)
print(f"[nos_intersect] Chebyshev radius={cheb['radius']}, feasible={cheb['feasible']}")

intersection_pts = nos.intersect_hulls(hulls) if len(hulls) >= 2 else hulls[0]
pd.DataFrame(intersection_pts, columns=keys).to_csv(snakemake.output.intersection, index=False)

centre_out = {
    "years": years,
    "dimensions": keys,
    "chebyshev_radius": cheb["radius"],
    "feasible_intersection": bool(cheb["feasible"]),
    "centre": (cheb["centre"].to_dict() if cheb["centre"] is not None else None),
}
with open(snakemake.output.centre, "w") as f:
    json.dump(centre_out, f, indent=2, default=str)

weight_by = mga.get("dimension_weight", "capacity")
plot_unit, plot_scale = ("GW", 1e3) if weight_by == "capacity" else ("M EUR/y", 1e6)

Path(snakemake.output.plot).parent.mkdir(parents=True, exist_ok=True)
nos.plot_robustness(per_year_points, cheb["centre"], keys, snakemake.output.plot,
                     scale=plot_scale, unit=plot_unit)

print(f"[nos_intersect] centre -> {snakemake.output.centre}, plot -> {snakemake.output.plot}")
