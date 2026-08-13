# SPDX-License-Identifier: MIT
"""Snakemake wrapper: nos_cost_bound of the staged NOS pipeline.

Shared robustness budget anchor c* = max_i c_opt(i) across mga.robustness.years
(Grochowicz et al. 2023, eq. 9), mirroring the reference implementation's
calc_obj_bound.py -- that script bakes the (1+eps) margin into its output
directly; GreenBubble instead applies (1+eps)/(1-eps) (sign-robust, see
scripts.near_optimal._add_budget_constraint) inside each downstream solve's
own budget constraint, so this rule only needs to compute the shared max(objs).
See rules/near_optimal_staged.smk.
"""
objs = []
for f in snakemake.input.objs:
    with open(f) as fh:
        objs.append(float(fh.read()))

c_star = max(objs)
with open(snakemake.output.bound, "w") as f:
    f.write(str(c_star))

print(f"[nos_cost_bound] c* = max({objs}) = {c_star}")
