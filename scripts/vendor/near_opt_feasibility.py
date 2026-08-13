# SPDX-FileCopyrightText: 2022 Koen van Greevenbroek & Aleksander Grochowicz
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# ---------------------------------------------------------------------------
# GREENBUBBLE NOTE (not part of the original file)
# ---------------------------------------------------------------------------
# Vendored from https://github.com/aleks-g/intersecting-near-opt-spaces
# (`workflow/scripts/solve_operations.py` and `workflow/scripts/utilities.py`),
# the reference implementation for
#   Grochowicz, A., van Greevenbroek, K., Benth, F.E. & Zeyringer, M. (2023).
#   "Intersecting Near-Optimal Spaces: European Power Systems with More
#   Resilience to Weather Variability." Energy Economics.
#   https://doi.org/10.1016/j.eneco.2022.106496
# Source commit: e89318ecbef92a1bd2cd7fbb1da5fcca53a54b81 (`main`, 2022-11-03).
# Vendored 2026-08-13.
#
# This file is deliberately kept under the ORIGINAL GPL-3.0-or-later licence
# and attribution -- it is not relicensed as part of GreenBubble (MIT). Only
# import from here where that licence is acceptable for your use of
# GreenBubble; do not merge this code into MIT-licensed modules.
#
# Only the two functions below are ported: `set_extendable_false` (from
# solve_operations.py) and `compute_feasibility_criteria` (from utilities.py)
# -- the generic, PyPSA-version-and-domain-agnostic pieces of their
# "simultaneous multi-year feasibility check" (validate_robust /
# solve_operations.py + summarise_feasibility.py in their Snakefile). Their
# `set_nom_to_opt` (copying capacities within ONE network) and
# `add_load_shedding` (assumes a country-level model where every AC bus needs
# a slack generator) are NOT reused verbatim: GreenBubble applies a realised
# design's capacities ACROSS separate per-year network files rather than
# within one combined multi-year network, and slack generators are placed at
# every bus carrying a Load component (GreenBubble has no generic "AC bus"
# concept) rather than a hardcoded AC-carrier filter -- both GreenBubble-
# specific and implemented in scripts/near_optimal.py instead, not here.
# `network_lopf`/`pypsa.linopf` (their pre-Linopy solve calls) are replaced by
# GreenBubble's own `n.optimize` (Linopy) solve, for the same reason
# `_solve_mga_in_direction` re-derives rather than reuses their per-direction
# solve -- ported logic, not literal reuse, where the PyPSA API differs.
# ---------------------------------------------------------------------------

"""Vendored feasibility-check primitives for the multi-year robustness validation.

For this given network, all components are set to be non-extendable so that
re-solving finds only the operational dispatch, not new capacity (from their
``solve_operations.py``). Feasibility is then read back as load-shedding
curtailment (from their ``utilities.py``).
"""

import pandas as pd
import pypsa
from pypsa.descriptors import nominal_attrs


def set_extendable_false(n: pypsa.Network) -> None:
    """Set all technologies in `n` to non-extendable.

    Modifies the argument `n`.
    """
    for c, attr in nominal_attrs.items():
        n.df(c)[attr + "_extendable"] = False


def compute_feasibility_criteria(n: pypsa.Network, name: str) -> pd.DataFrame:
    """Compute feasibility in terms of load curtailment.

    Compute the total curtailment, the percent of load curtailed,
    accounting for numerical instabilities (curtailment starts with
    load shedding above 1 MW).

    Parameters
    ----------
    n: pypsa.Network
        Network to be validated.
    name: str
        Name of network.

    Returns
    -------
    feasibility: pd.DataFrame
        Dataframe storing the values obtained from the feasibility criteria.
    """
    # Read out the total load curtailed.
    load_shedding = n.generators_t["p"].filter(like="load shedding").sum(axis=1)
    # Read out the total load in the network.
    total_load = n.loads_t["p"].sum().sum()
    # Filter out load curtailment above 1 MW in a node to avoid taking
    # numerical instability into account.
    filtered_shedding = load_shedding[load_shedding > 1]
    total_curtailment = filtered_shedding.sum()
    relative_curtailment = total_curtailment / total_load if total_load else float("nan")

    columns = [
        "Total curtailment",
        "Relative curtailment",
    ]
    values = [
        total_curtailment,
        relative_curtailment,
    ]
    return pd.DataFrame([values], columns=columns, index=[name])
