import pypsa
import pypsatopo
import matplotlib as mpl
import re
import math
import pandas as pd
import calendar
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import calendar
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import os
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Optional


# ---- INPUTS PLOTS ----

def _ldc(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    return np.sort(s.values)[::-1]


def _scenario_list_from_tcols(df: pd.DataFrame):
    cols = df.columns
    if isinstance(cols, pd.MultiIndex) and "scenario" in cols.names:
        return list(cols.get_level_values("scenario").unique())
    return []


def _series_by_mi_col(df: pd.DataFrame, scen, name):
    """df has MultiIndex columns ('scenario','name') -> return Series for (scen,name) if exists."""
    if not isinstance(df.columns, pd.MultiIndex):
        return df[name] if name in df.columns else None
    if scen is None:
        scen = df.columns.get_level_values("scenario").unique()[0]
    key = (scen, name)
    return df[key] if key in df.columns else None


def _available_names_mi_cols(df: pd.DataFrame, scen):
    """Return list of available component names for scenario scen from MultiIndex columns."""
    if not isinstance(df.columns, pd.MultiIndex):
        return list(df.columns)
    if scen is None:
        scen = df.columns.get_level_values("scenario").unique()[0]
    return list(df.xs(scen, level="scenario", axis=1).columns)


def _pick_first_match(candidates, selector):
    """
    selector can be:
      - str exact
      - dict: {"contains": "..."} or {"regex": r"..."}
      - callable: fn(name)->bool
    Returns matched name or None.
    """
    if selector is None:
        return None

    if isinstance(selector, str):
        return selector if selector in candidates else None

    if isinstance(selector, dict):
        if "contains" in selector:
            token = selector["contains"]
            for c in candidates:
                if token in c:
                    return c
            return None
        if "regex" in selector:
            pat = re.compile(selector["regex"])
            for c in candidates:
                if pat.search(c):
                    return c
            return None

    if callable(selector):
        for c in candidates:
            try:
                if selector(c):
                    return c
            except Exception:
                continue
        return None

    return None


def _generator_mc_series(n, scen, gen_name):
    """
    Return generator marginal cost series over snapshots if possible.
    Priority:
      1) n.generators_t.marginal_cost (time-varying)
      2) n.generators.marginal_cost (static) expanded to snapshots
    Works with stochastic MultiIndex (scenario,name) on generators index.
    """
    # 1) time-varying
    gt = getattr(n, "generators_t", None)
    if gt is not None and hasattr(gt, "marginal_cost"):
        df = gt.marginal_cost
        s = _series_by_mi_col(df, scen, gen_name)
        if s is not None:
            return s

    # 2) static marginal_cost
    if hasattr(n, "generators") and "marginal_cost" in n.generators.columns:
        g = n.generators
        if isinstance(g.index, pd.MultiIndex) and "scenario" in g.index.names:
            if scen is None:
                scen = g.index.get_level_values("scenario").unique()[0]
            key = (scen, gen_name)
            if key in g.index:
                mc = g.loc[key, "marginal_cost"]
            else:
                return None
        else:
            if gen_name not in g.index:
                return None
            mc = g.loc[gen_name, "marginal_cost"]

        # expand to snapshots
        return pd.Series(mc, index=n.snapshots)

    return None


def plot_ldc_inputs_by_scenario(
    n,
    outpath=None,
    title="Input duration curves by scenario",
    ncols=3,
    figsize_per_panel=(5.6, 4.6),
    # link selectors: list of {"label":..., "selector":..., "ls":..., "lw":..., "show_chosen": bool}
    price_links=None,
    # generator MC selectors: list of {"label":..., "selector":..., "ls":..., "lw":..., "show_chosen": bool}
    price_gens=None,
    # CF gens exact names: list of {"label":..., "name":..., "ls":..., "lw":...}
    cf_gens=None,
):
    """
    Subplot per scenario; deterministic -> single plot.

    Left axis: prices from links_t.marginal_cost + generator marginal_cost (t or static)
    Right axis: CF from generators_t.p_max_pu
    Skips anything missing per scenario/config.
    """

    # Defaults matching
    price_links = price_links or [
        {"label": "Electricity price", "selector": {"contains": "DK1_to_El_"}, "ls": "-", "lw": 1.8},
        {"label": "NG price",          "selector": {"regex": r"_NG boiler$"},  "ls": "-", "lw": 1.8},
    ]
    price_gens = price_gens or [
        # Examples: add what you want; these are OPTIONAL and skipped if missing
        {"label": "Grid gen MC", "selector": "Grid gen", "ls": "-.", "lw": 1.8, "show_chosen": False},
        {"label": "NG grid MC",  "selector": "NG grid",  "ls": "-.", "lw": 1.8, "show_chosen": False},
    ]
    cf_gens = cf_gens or [
        {"label": "Wind CF",  "name": "onshorewind", "ls": "--", "lw": 1.8},
        {"label": "Solar CF", "name": "solar",       "ls": "--", "lw": 1.8},
    ]

    scenarios = _scenario_list_from_tcols(n.generators_t.p_max_pu)
    if not scenarios:
        scenarios = [None]

    n_panels = len(scenarios)
    ncols = min(ncols, n_panels)
    nrows = int(np.ceil(n_panels / ncols))

    fig_w = figsize_per_panel[0] * ncols
    fig_h = figsize_per_panel[1] * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    cmap = mpl.colormaps["Dark2"]

    # Color pools
    price_total = len(price_links) + len(price_gens)
    price_colors = [cmap(i % cmap.N) for i in range(price_total)]
    cf_colors    = [cmap((i + 5) % cmap.N) for i in range(len(cf_gens))]

    def _panel(ax, scen):
        ax2 = ax.twinx()
        handles, labels = [], []
        any_price, any_cf = False, False

        mc_links = n.links_t.marginal_cost
        pmaxpu = n.generators_t.p_max_pu

        # ----- LINK marginal costs (LEFT axis, solid)
        link_candidates = _available_names_mi_cols(mc_links, scen)

        color_idx = 0
        for item in price_links:
            chosen = _pick_first_match(link_candidates, item.get("selector"))
            if chosen is None:
                color_idx += 1
                continue

            s = _series_by_mi_col(mc_links, scen, chosen)
            y = _ldc(s) if s is not None else None
            if y is None:
                color_idx += 1
                continue

            x = np.linspace(0, 100, len(y))
            h, = ax.plot(
                x, y,
                color=price_colors[color_idx],
                linestyle="-",  # FORCE solid
                linewidth=item.get("lw", 1.8),
            )

            handles.append(h)
            labels.append(item["label"])
            any_price = True
            color_idx += 1

        # ----- GENERATOR marginal costs (LEFT axis, solid)
        if isinstance(n.generators.index, pd.MultiIndex) and "scenario" in n.generators.index.names:
            gen_candidates = list(n.generators.xs(
                scen if scen is not None else n.generators.index.get_level_values("scenario").unique()[0],
                level="scenario"
            ).index)
        else:
            gen_candidates = list(n.generators.index)

        for item in price_gens:
            chosen = _pick_first_match(gen_candidates, item.get("selector"))
            if chosen is None:
                color_idx += 1
                continue

            s = _generator_mc_series(n, scen, chosen)
            y = _ldc(s) if s is not None else None
            if y is None:
                color_idx += 1
                continue

            x = np.linspace(0, 100, len(y))
            h, = ax.plot(
                x, y,
                color=price_colors[color_idx],
                linestyle="-",  # FORCE solid
                linewidth=item.get("lw", 1.8),
            )

            handles.append(h)
            labels.append(item["label"])
            any_price = True
            color_idx += 1

        # ----- CF generators (RIGHT axis, dashed)
        for i, item in enumerate(cf_gens):
            s = _series_by_mi_col(pmaxpu, scen, item["name"])
            y = _ldc(s) if s is not None else None
            if y is None:
                continue

            x = np.linspace(0, 100, len(y))
            h, = ax2.plot(
                x, y,
                color=cf_colors[i],
                linestyle="--",  # FORCE dashed
                linewidth=item.get("lw", 1.8),
            )

            handles.append(h)
            labels.append(item["label"])
            any_cf = True

        scen_label = "deterministic" if scen is None else str(scen)
        ax.set_title(f"Scenario: {scen_label}")
        ax.set_xlabel("Percent of hours (%)")

        if any_price:
            ax.set_ylabel("Price (€/MWh)")
        if any_cf:
            ax2.set_ylabel("Capacity factor (-)")
            ax2.set_ylim(0, 1)

        ax.grid(True, alpha=0.25)

        return handles, labels


    legend_map = {}  # label -> handle (first occurrence)

    for i, scen in enumerate(scenarios):
        handles, labels = _panel(axes[i], scen)
        for h, l in zip(handles, labels):
            legend_map.setdefault(l, h)

    for j in range(n_panels, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, y=1.02)

    leg = None
    if legend_map:
        H = list(legend_map.values())
        L = list(legend_map.keys())

        ncol = min(5, len(L))

        leg = fig.legend(
            H, L,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=ncol,
            frameon=False,
            fontsize=9,
        )

    # --- dynamic spacing so legend never overlaps subplots
    fig.canvas.draw()
    bottom = 0.06
    if leg is not None:
        bbox = leg.get_window_extent(fig.canvas.get_renderer())
        bbox_fig = bbox.transformed(fig.transFigure.inverted())
        bottom = bbox_fig.height + 0.03

    fig.tight_layout(rect=[0, bottom, 1, 1])

    if outpath:
        fig.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ---- RESULTS


# --- NETWORK TOPOLOGY
def extract_deterministic_from_stochastic(n, scenario=None, slice_timeseries=False):
    """
    Slice ONE scenario out of a stochastic PyPSA network and return a "normal"
    deterministic-like network (no MultiIndex in component tables).

    Works even if n.scenario_weightings is a read-only property.
    """

    # --- pick scenario if not provided
    if scenario is None:
        if hasattr(n, "scenario_weightings") and n.scenario_weightings is not None:
            try:
                if len(n.scenario_weightings.index) > 0:
                    scenario = n.scenario_weightings.index[0]
            except Exception:
                pass

        if scenario is None:
            for comp in ["buses", "generators", "links", "loads", "stores"]:
                df = getattr(n, comp, None)
                if isinstance(df, pd.DataFrame) and isinstance(df.index, pd.MultiIndex) and "scenario" in df.index.names:
                    scenario = df.index.get_level_values("scenario")[0]
                    break

    n_det = n

    # --- helper: robust xs for scenario labels that might be int/str mismatch
    def _xs_scenario_index(df, scen):
        if not (isinstance(df.index, pd.MultiIndex) and "scenario" in df.index.names):
            return df
        try:
            return df.xs(scen, level="scenario")
        except Exception:
            scen_vals = df.index.get_level_values("scenario").astype(str)
            key = str(scen)
            if key in set(scen_vals):
                lab = df.index.get_level_values("scenario")[scen_vals == key][0]
                return df.xs(lab, level="scenario")
            return df

    def _xs_scenario_cols(df, scen):
        if not (isinstance(df.columns, pd.MultiIndex) and "scenario" in df.columns.names):
            return df
        try:
            return df.xs(scen, level="scenario", axis=1)
        except Exception:
            scen_vals = df.columns.get_level_values("scenario").astype(str)
            key = str(scen)
            if key in set(scen_vals):
                lab = df.columns.get_level_values("scenario")[scen_vals == key][0]
                return df.xs(lab, level="scenario", axis=1)
            return df

    # --- slice static component tables
    for comp in ["buses", "carriers", "generators", "links", "loads", "stores",
                 "lines", "transformers", "transformers2w", "transformers3w"]:
        if not hasattr(n_det, comp):
            continue
        df = getattr(n_det, comp)
        if isinstance(df, pd.DataFrame):
            df2 = _xs_scenario_index(df, scenario)
            setattr(n_det, comp, df2)

    # --- optionally slice time series tables
    if slice_timeseries:
        for comp_t_name in ["buses_t", "generators_t", "links_t", "stores_t", "loads_t"]:
            comp_t = getattr(n_det, comp_t_name, None)
            if comp_t is None:
                continue
            for attr, df in vars(comp_t).items():
                if isinstance(df, pd.DataFrame):
                    df2 = _xs_scenario_cols(df, scenario)
                    setattr(comp_t, attr, df2)

    for private_name in ["_scenario_weightings", "_scenarios"]:
        if hasattr(n_det, private_name):
            try:
                delattr(n_det, private_name)
            except Exception:
                pass

    return n_det

def _filter_network_for_topo(n, threshold: float = 0.5):
    """Remove near-zero-capacity components and orphaned buses from n in-place.

    Modifies n directly (no copy) — matches the original optimal_network_only()
    approach which avoids PyPSA's restriction on copying a network with an
    attached solver model.  The network must already be exported before calling.

    Uses p_nom_opt / e_nom_opt (OPT) when available, falls back to p_nom / e_nom (PRE).
    Also removes buses that become orphaned after component removal.
    """
    comp_map = [
        ("generators",    "p_nom", "p_nom_opt", "Generator"),
        ("links",         "p_nom", "p_nom_opt", "Link"),
        ("stores",        "e_nom", "e_nom_opt", "Store"),
        ("storage_units", "p_nom", "p_nom_opt", "StorageUnit"),
    ]
    total_removed = 0
    for attr, nom_col, opt_col, cls_name in comp_map:
        df = getattr(n, attr, None)
        if df is None or df.empty:
            continue
        cap_col = opt_col if opt_col in df.columns else nom_col
        if cap_col not in df.columns:
            continue
        to_remove = df.index[df[cap_col].fillna(0.0).abs() < threshold].tolist()
        for name in to_remove:
            n.remove(cls_name, name)
        total_removed += len(to_remove)

    # Remove buses no longer referenced by any remaining component
    bus_ok = set()
    for bus_col in ("bus", "bus0", "bus1", "bus2", "bus3", "bus4"):
        for attr in ("generators", "links", "stores", "storage_units", "loads"):
            df = getattr(n, attr, None)
            if df is not None and not df.empty and bus_col in df.columns:
                bus_ok.update(df[bus_col].dropna().values)
    orphan_buses = [b for b in n.buses.index if b not in bus_ok]
    for b in orphan_buses:
        n.remove("Bus", b)

    if total_removed or orphan_buses:
        print(f"[print_network] removed {total_removed} near-zero components, "
              f"{len(orphan_buses)} orphaned buses (threshold={threshold})")
    return n


def print_network(n, n_flags, nc_path, network_name, suffix, plot_folder, is_stochastic):
    # function that prints .svg of network topology with pypsatopo

    if not n_flags.get("print", False):
        return None

    if nc_path is None:
        print("[WARN] No nc_path provided; skipping network plot.")
        return None

    if is_stochastic:
        n_plot = pypsa.Network(nc_path)
        n_plot = extract_deterministic_from_stochastic(
            n_plot, scenario=None, slice_timeseries=False
        )
    else:
        # Always reload from saved NC to avoid any in-memory state issues with pypsatopo
        n_plot = pypsa.Network(nc_path)

    filename = f"{network_name}{suffix}.svg"
    svg_path = os.path.join(plot_folder, filename)

    pypsatopo.generate(
        n_plot,
        file_output=svg_path,
        negative_efficiency=False,
        carrier_color=True,
    )
    print(f"✅ PyPSA network plotted to: {svg_path}")
    return svg_path

# ---- Save optimal capacities:

def save_opt_capacity_components(
    n_opt,
    network_comp_allocation,
    file_path,
):
    """
    Saves optimal capacities + annualized capex for allocated assets.

    Solver-noise filtering is handled upstream by zero_small_capacities()
    in snakemake_plot.py, so no per-component threshold is needed here.
    """

    # -------- helpers --------
    def detect_levels(mi: pd.MultiIndex):
        names = list(mi.names)
        scenario_level = "scenario" if "scenario" in names else names[0]
        name_level = "name" if "name" in names else names[-1]
        return scenario_level, name_level

    def first_scenario(mi: pd.MultiIndex, scenario_level: str):
        # handle empty index safely
        if mi is None or len(mi) == 0:
            return None
        try:
            sc = mi.get_level_values(scenario_level)
        except (KeyError, IndexError):
            return None
        sc = pd.Index(sc).drop_duplicates()
        return sc[0] if len(sc) else None

    def slice_first_scenario_df(df: pd.DataFrame):
        # empty df => return as-is
        if df is None or df.empty:
            return df if df is not None else pd.DataFrame(), None

        # no MultiIndex => nothing to slice
        if not isinstance(df.index, pd.MultiIndex):
            return df, None

        sc_level, _ = detect_levels(df.index)
        sc0 = first_scenario(df.index, sc_level)
        if sc0 is None:
            # MultiIndex exists but no scenarios (or empty) => return unchanged
            return df, None

        return df.xs(sc0, level=sc_level), sc0

    def norm_unit(u):
        if u is None or (isinstance(u, float) and np.isnan(u)):
            return None
        s = str(u).strip()
        if not s:
            return None
        s_low = s.lower().replace(" ", "")
        if s_low == "mw":
            return "MW"
        if s_low == "mwh":
            return "MWh"
        if s_low in {"t/h", "tph", "tperh"}:
            return "t/h"
        if s_low in {"t", "ton", "tonne", "tonnes"}:
            return "t"
        return s

    # -------- slice static tables --------
    gens, sc0 = slice_first_scenario_df(n_opt.generators)
    links, sc1 = slice_first_scenario_df(n_opt.links)
    stores, sc2 = slice_first_scenario_df(n_opt.stores)

    # NEW: storage units
    sus, sc3 = slice_first_scenario_df(n_opt.storage_units) if hasattr(n_opt, "storage_units") else (pd.DataFrame(), None)

    chosen_scenario = next((x for x in [sc0, sc1, sc2, sc3] if x is not None), None)

    if isinstance(n_opt.buses.index, pd.MultiIndex):
        sc_level_b, _ = detect_levels(n_opt.buses.index)
        if chosen_scenario is None:
            chosen_scenario = first_scenario(n_opt.buses.index, sc_level_b)
        buses_static = n_opt.buses.xs(chosen_scenario, level=sc_level_b)
    else:
        buses_static = n_opt.buses

    def unit_of_bus(bus):
        if buses_static is None or buses_static.empty:
            return None
        b = bus[-1] if isinstance(bus, tuple) else bus
        if "unit" in buses_static.columns and b in buses_static.index:
            u = buses_static.at[b, "unit"]
            return None if pd.isna(u) else u
        return None

    # -------- build rows --------
    rows = []

    gen_opt = "p_nom_opt" if "p_nom_opt" in gens.columns else None
    link_opt = "p_nom_opt" if "p_nom_opt" in links.columns else None
    store_opt = "e_nom_opt" if "e_nom_opt" in stores.columns else None
    su_opt = "p_nom_opt" if (sus is not None and not sus.empty and "p_nom_opt" in sus.columns) else None

    for plant, alloc in (network_comp_allocation or {}).items():
        alloc = alloc or {}

        # Generators
        if gen_opt:
            for g in alloc.get("generators", []) or []:
                if g not in gens.index:
                    continue
                bus = gens.at[g, "bus"]
                u = unit_of_bus(bus)
                cap = float(gens.at[g, gen_opt])
                cc = float(gens.at[g, "capital_cost"]) if "capital_cost" in gens.columns else np.nan
                rows.append({
                    "plant": plant,
                    "component": "generator",
                    "asset": str(g),
                    "capacity": cap,
                    "Fixed cost (€/y)": cap * cc,
                    "reference inlet": bus,
                    "unit": norm_unit(u),
                })

        # Links
        if link_opt:
            for l in alloc.get("links", []) or []:
                if l not in links.index:
                    continue
                bus0 = links.at[l, "bus0"]
                u = unit_of_bus(bus0)
                cap = float(links.at[l, link_opt])
                cc = float(links.at[l, "capital_cost"]) if "capital_cost" in links.columns else np.nan
                rows.append({
                    "plant": plant,
                    "component": "link",
                    "asset": str(l),
                    "capacity": cap,
                    "Fixed cost (€/y)": cap * cc,
                    "reference inlet": bus0,
                    "unit": norm_unit(u),
                })

        # Stores
        if store_opt:
            for s in alloc.get("stores", []) or []:
                if s not in stores.index:
                    continue
                bus = stores.at[s, "bus"]
                u = unit_of_bus(bus)
                cap = float(stores.at[s, store_opt])
                cc = float(stores.at[s, "capital_cost"]) if "capital_cost" in stores.columns else np.nan
                rows.append({
                    "plant": plant,
                    "component": "store",
                    "asset": str(s),
                    "capacity": cap,
                    "Fixed cost (€/y)": cap * cc,
                    "reference inlet": bus,
                    "unit": norm_unit(u),
                })

        # Storage Units
        if su_opt:
            for su in alloc.get("storage_units", []) or []:
                if su not in sus.index:
                    continue
                bus = sus.at[su, "bus"]
                u = unit_of_bus(bus)
                cap = float(sus.at[su, su_opt])
                cc = float(sus.at[su, "capital_cost"]) if "capital_cost" in sus.columns else np.nan
                e_cap = np.nan
                if "max_hours" in sus.columns:
                    mh = sus.at[su, "max_hours"]
                    if mh is not None and not pd.isna(mh):
                        e_cap = float(cap) * float(mh)
                rows.append({
                    "plant": plant,
                    "component": "storage_unit",
                    "asset": str(su),
                    "capacity": cap,
                    "energy_capacity": e_cap,
                    "Fixed cost (€/y)": cap * cc,
                    "reference inlet": bus,
                    "unit": norm_unit(u),
                })

    df = pd.DataFrame(rows)

    out = Path(file_path)
    if out.suffix.lower() != ".csv":
        out = out.with_suffix(".csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    df.attrs["chosen_scenario"] = str(chosen_scenario) if chosen_scenario is not None else None
    return df


def save_full_component_csv(n, network_comp_allocation, file_path, num_tol=1e-3, comp_tech_map=None):
    """Save a comprehensive human-readable table of all optimal-capacity components.

    Output: full_component_table.csv — one row per component.

    Filters components whose optimal capacity < num_tol (solver numerical noise only).
    For stochastic networks the first scenario slice is used for static tables;
    variable costs are summed across all snapshots weighted by snapshot_weightings.
    """

    # ---- helpers ----
    def _slice_static(df):
        if df is None or df.empty:
            return df if df is not None else pd.DataFrame()
        if not isinstance(df.index, pd.MultiIndex):
            return df
        return df.xs(df.index.get_level_values(0)[0], level=0)

    def _get_buses_static():
        if isinstance(n.buses.index, pd.MultiIndex):
            return n.buses.xs(n.buses.index.get_level_values(0)[0], level=0)
        return n.buses

    buses_s = _get_buses_static()

    def _unit_of_bus(bus_name):
        b = bus_name[-1] if isinstance(bus_name, tuple) else bus_name
        if "unit" in buses_s.columns and b in buses_s.index:
            u = buses_s.at[b, "unit"]
            if pd.notna(u) and str(u).strip():
                return str(u).strip()
        return "MW"

    # ---- reverse lookup: component name -> plant ----
    comp_to_plant: dict = {}
    for plant, alloc in (network_comp_allocation or {}).items():
        for kind_key in ("generators", "links", "stores", "storage_units"):
            for cname in (alloc or {}).get(kind_key, []) or []:
                comp_to_plant[cname] = plant

    _tech_map = comp_tech_map or {}

    # ---- snapshot weightings ----
    sw = n.snapshot_weightings
    weights = sw.get("generators", sw.get("objective", sw.iloc[:, 0]))
    total_hours = float(weights.sum())

    def _weighted_sum(ts):
        if ts is None:
            return 0.0
        return float(ts.reindex(weights.index).fillna(0.0).mul(weights).sum())

    def _get_ts(ts_df, name):
        if ts_df is None or ts_df.empty:
            return None
        if isinstance(ts_df.columns, pd.MultiIndex):
            matches = [col for col in ts_df.columns if col[-1] == name]
            return ts_df[matches].sum(axis=1) / max(len(matches), 1) if matches else None
        return ts_df[name] if name in ts_df.columns else None

    def _capacity_factor(dispatch_wsum, cap):
        if cap <= 0 or total_hours <= 0:
            return np.nan
        return round(dispatch_wsum / (cap * total_hours), 3)

    def _vre_curtailment(name, cap):
        """Curtailment fraction = (available - produced) / available for VRE generators."""
        p_max_pu_df = getattr(n.generators_t, "p_max_pu", None)
        if p_max_pu_df is None:
            return np.nan
        ts_avail = _get_ts(p_max_pu_df, name)
        if ts_avail is None:
            return np.nan
        ts_p = _get_ts(n.generators_t.get("p"), name)
        if ts_p is None:
            return np.nan
        avail_wsum = _weighted_sum(ts_avail * cap)
        prod_wsum  = _weighted_sum(ts_p)
        if avail_wsum <= 0:
            return np.nan
        return round((avail_wsum - prod_wsum) / avail_wsum, 3)

    def _is_vre(carrier):
        c = str(carrier).lower()
        return any(k in c for k in ("wind", "solar", "pv"))

    def _effectively_expandable(name, row, ext_col, min_col, max_col):
        """Return False for EXI_ components pinned at p_nom_min=p_nom_max.

        _force_exi_capex_into_objective sets p_nom_extendable=True on EXI_
        components that carry a residual capital cost, so the cost enters
        the PyPSA objective.  Those components are effectively fixed (the
        optimizer has no capacity choice).  Exposing them as Expandable=True
        in the CSV confuses users, so we report them as non-expandable.
        """
        if not bool(row.get(ext_col, False)):
            return False
        if not str(name).startswith("EXI_"):
            return True
        lo = float(row.get(min_col, 0.0))
        hi = float(row.get(max_col, np.inf))
        if np.isfinite(hi) and abs(lo - hi) < 1e-6:
            return False
        return True

    # ---- static slices ----
    gens_s   = _slice_static(n.generators)
    links_s  = _slice_static(n.links)
    stores_s = _slice_static(n.stores)
    sus_s    = _slice_static(n.storage_units) if hasattr(n, "storage_units") else pd.DataFrame()

    rows = []

    # --- Generators ---
    cap_col = "p_nom_opt" if "p_nom_opt" in gens_s.columns else "p_nom"
    for name, row in gens_s.iterrows():
        cap = float(row.get(cap_col, 0.0))
        if cap < num_tol:
            continue
        cc      = float(row.get("capital_cost", 0.0))
        mc      = float(row.get("marginal_cost", 0.0))
        bus     = str(row.get("bus", ""))
        carrier = row.get("carrier", "")
        ts      = _get_ts(n.generators_t.get("p"), name)
        disp    = _weighted_sum(ts)
        var     = mc * disp
        rows.append({
            "Plant":                          comp_to_plant.get(name, "Unallocated"),
            "Component":                      "Generator",
            "Asset":                          name,
            "Cost input":                     _tech_map.get(name, ""),
            "Carrier":                        carrier,
            "Reference inlet":                bus,
            "Unit":                           _unit_of_bus(bus),
            "Expandable":                     _effectively_expandable(name, row, "p_nom_extendable", "p_nom_min", "p_nom_max"),
            "Initial capacity":               round(float(row.get("p_nom", 0.0)), 3),
            "Optimal capacity":               round(cap, 3),
            "Optimal energy capacity":        np.nan,
            "Capacity factor":                _capacity_factor(disp, cap),
            "Curtailment":                    _vre_curtailment(name, cap) if _is_vre(carrier) else np.nan,
            "Specific fixed cost (€/(unit y))":   round(cc, 2),
            "Specific variable cost (€/(unit h))": round(mc, 4),
            "Fixed cost (€/y)":               round(cap * cc, 0),
            "Variable cost (€/y)":            round(var, 0),
            "Total cost (€/y)":               round(cap * cc + var, 0),
        })

    # --- Links ---
    cap_col = "p_nom_opt" if "p_nom_opt" in links_s.columns else "p_nom"
    for name, row in links_s.iterrows():
        cap  = float(row.get(cap_col, 0.0))
        if cap < num_tol:
            continue
        cc   = float(row.get("capital_cost", 0.0))
        mc   = float(row.get("marginal_cost", 0.0))
        bus0 = str(row.get("bus0", ""))
        ts   = _get_ts(n.links_t.get("p0"), name)
        ts_pos = ts.clip(lower=0) if ts is not None else None
        disp = _weighted_sum(ts_pos)
        var  = mc * disp
        rows.append({
            "Plant":                          comp_to_plant.get(name, "Unallocated"),
            "Component":                      "Link",
            "Asset":                          name,
            "Cost input":                     _tech_map.get(name, ""),
            "Carrier":                        row.get("carrier", ""),
            "Reference inlet":                bus0,
            "Unit":                           _unit_of_bus(bus0),
            "Expandable":                     _effectively_expandable(name, row, "p_nom_extendable", "p_nom_min", "p_nom_max"),
            "Initial capacity":               round(float(row.get("p_nom", 0.0)), 3),
            "Optimal capacity":               round(cap, 3),
            "Optimal energy capacity":        np.nan,
            "Capacity factor":                _capacity_factor(disp, cap),
            "Curtailment":                    np.nan,
            "Specific fixed cost (€/(unit y))":   round(cc, 2),
            "Specific variable cost (€/(unit h))": round(mc, 4),
            "Fixed cost (€/y)":               round(cap * cc, 0),
            "Variable cost (€/y)":            round(var, 0),
            "Total cost (€/y)":              round(cap * cc + var, 0),
        })

    # --- Stores ---
    cap_col = "e_nom_opt" if "e_nom_opt" in stores_s.columns else "e_nom"
    for name, row in stores_s.iterrows():
        cap = float(row.get(cap_col, 0.0))
        if cap < num_tol:
            continue
        cc  = float(row.get("capital_cost", 0.0))
        mc  = float(row.get("marginal_cost", 0.0))
        bus = str(row.get("bus", ""))
        ts  = _get_ts(n.stores_t.get("p"), name)
        ts_pos = ts.clip(lower=0) if ts is not None else None
        disp = _weighted_sum(ts_pos)
        var  = mc * disp
        rows.append({
            "Plant":                          comp_to_plant.get(name, "Unallocated"),
            "Component":                      "Store",
            "Asset":                          name,
            "Cost input":                     _tech_map.get(name, ""),
            "Carrier":                        row.get("carrier", ""),
            "Reference inlet":                bus,
            "Unit":                           _unit_of_bus(bus),
            "Expandable":                     _effectively_expandable(name, row, "e_nom_extendable", "e_nom_min", "e_nom_max"),
            "Initial capacity":               round(float(row.get("e_nom", 0.0)), 3),
            "Optimal capacity":               round(cap, 3),
            "Optimal energy capacity":        np.nan,
            "Capacity factor":                _capacity_factor(disp, cap),
            "Curtailment":                    np.nan,
            "Specific fixed cost (€/(unit y))":   round(cc, 2),
            "Specific variable cost (€/(unit h))": round(mc, 4),
            "Fixed cost (€/y)":               round(cap * cc, 0),
            "Variable cost (€/y)":            round(var, 0),
            "Total cost (€/y)":               round(cap * cc + var, 0),
        })

    # --- StorageUnits ---
    if not sus_s.empty:
        cap_col = "p_nom_opt" if "p_nom_opt" in sus_s.columns else "p_nom"
        for name, row in sus_s.iterrows():
            cap = float(row.get(cap_col, 0.0))
            if cap < num_tol:
                continue
            cc   = float(row.get("capital_cost", 0.0))
            mc   = float(row.get("marginal_cost", 0.0))
            bus  = str(row.get("bus", ""))
            mh   = row.get("max_hours", np.nan)
            e_cap = round(cap * float(mh), 3) if pd.notna(mh) else np.nan
            ts   = _get_ts(n.storage_units_t.get("p"), name)
            ts_pos = ts.clip(lower=0) if ts is not None else None
            disp = _weighted_sum(ts_pos)
            var  = mc * disp
            rows.append({
                "Plant":                          comp_to_plant.get(name, "Unallocated"),
                "Component":                      "StorageUnit",
                "Asset":                          name,
                "Cost input":                     _tech_map.get(name, ""),
                "Carrier":                        row.get("carrier", ""),
                "Reference inlet":                bus,
                "Unit":                           _unit_of_bus(bus),
                "Expandable":                     _effectively_expandable(name, row, "p_nom_extendable", "p_nom_min", "p_nom_max"),
                "Initial capacity":               round(float(row.get("p_nom", 0.0)), 3),
                "Optimal capacity":               round(cap, 3),
                "Optimal energy capacity":        e_cap,
                "Capacity factor":                _capacity_factor(disp, cap),
                "Curtailment":                    np.nan,
                "Specific fixed cost (€/(unit y))":   round(cc, 2),
                "Specific variable cost (€/(unit h))": round(mc, 4),
                "Fixed cost (€/y)":               round(cap * cc, 0),
                "Variable cost (€/y)":            round(var, 0),
                "Total cost (€/y)":               round(cap * cc + var, 0),
            })

    df = pd.DataFrame(rows).sort_values("Plant", kind="stable").reset_index(drop=True)

    # ---- objective function check ----
    total_fixed = df["Fixed cost (€/y)"].sum()
    total_var   = df["Variable cost (€/y)"].sum()
    total_cost  = df["Total cost (€/y)"].sum()
    obj = getattr(n, "objective", None)
    if obj is not None:
        try:
            obj_f = float(obj) + float(getattr(n, "objective_constant", 0.0) or 0.0)
            diff  = total_cost - obj_f
            pct   = 100.0 * diff / obj_f if obj_f != 0 else float("nan")
            print(
                f"[full_component_table] Cost check: "
                f"table total={total_cost:.0f} €/y  |  "
                f"TSC (obj+constant)={obj_f:.0f} €/y  |  "
                f"diff={diff:+.0f} €/y ({pct:+.1f}%)"
            )
        except Exception:
            pass

    # ---- save main CSV ----
    out = Path(file_path)
    if out.suffix.lower() != ".csv":
        out = out.with_suffix(".csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    return df


def save_cost_assumptions_csv(tech_costs_used, file_path):
    """Save the technology cost assumptions used in the model as a CSV.

    tech_costs_used is a DataFrame (subset of tech_costs) with rows indexed
    by technology name and columns of cost/efficiency parameters. Columns that
    are all-NaN for the selected technologies are dropped for readability.
    """
    if tech_costs_used is None or tech_costs_used.empty:
        print("[save_cost_assumptions_csv] No tech_costs_used data available — skipping.")
        return None
    out = Path(file_path)
    if out.suffix.lower() != ".csv":
        out = out.with_suffix(".csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    tech_costs_used.dropna(axis=1, how="all").to_csv(out)
    return tech_costs_used


def save_pypsa_statistics(n, file_path):
    """Save n.statistics() DataFrame as CSV."""
    try:
        stats = n.statistics()
    except Exception as e:
        print(f"[save_pypsa_statistics] n.statistics() failed: {e}")
        return None
    out = Path(file_path)
    if out.suffix.lower() != ".csv":
        out = out.with_suffix(".csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(out)
    return stats


# Filter very small capacities:
def filter_items_by_capacity_threshold(
    n,
    items,
    default_th=1e-3,
    include_exi=True,
    verbose=False,
):
    """
    Returns a NEW items list where each item's selector becomes an explicit allow-list
    of component names that exist and have installed capacity >= th.

    Works for stochastic or deterministic networks:
      - If static tables are scenario-indexed, it uses the first scenario.
      - Otherwise uses the deterministic table.

    Assumes earlier helpers exist:
      - _slice_df_first_scenario
      - _expand_exi
      - _as_list
    """

    def cap_series_one(kind):
        if kind == "Link":
            links = _slice_df_first_scenario(n.links)
            col = "p_nom_opt" if "p_nom_opt" in links.columns else ("p_nom" if "p_nom" in links.columns else None)
            return links[col] if col else None

        if kind == "Generator":
            gens = _slice_df_first_scenario(n.generators)
            col = "p_nom_opt" if "p_nom_opt" in gens.columns else ("p_nom" if "p_nom" in gens.columns else None)
            return gens[col] if col else None

        if kind == "Store":
            stores = _slice_df_first_scenario(n.stores)
            col = "e_nom_opt" if "e_nom_opt" in stores.columns else ("e_nom" if "e_nom" in stores.columns else None)
            return stores[col] if col else None

        # NEW: StorageUnit (power-rated)
        if kind == "StorageUnit":
            sus = _slice_df_first_scenario(n.storage_units)
            col = "p_nom_opt" if "p_nom_opt" in sus.columns else ("p_nom" if "p_nom" in sus.columns else None)
            return sus[col] if col else None

        raise ValueError(kind)

    filtered = []
    dropped = []

    for it in items:
        kind = it["kind"]
        th = float(it.get("th", default_th))

        # normalize selector into list
        sel = it.get("selector")
        wanted = _as_list(sel)
        if include_exi:
            wanted = _expand_exi(wanted)

        caps = cap_series_one(kind)
        if caps is None:
            dropped.append((it.get("label", ""), kind, sel, "no static table"))
            continue

        keep_names = []
        for nm in wanted:
            if nm not in caps.index:
                continue
            v = pd.to_numeric(caps.loc[nm], errors="coerce")
            if pd.isna(v):
                continue
            if abs(v) >= th:
                keep_names.append(nm)

        if keep_names:
            it2 = dict(it)   # copy
            it2["selector"] = keep_names  # explicit allow-list
            filtered.append(it2)
        else:
            dropped.append((it.get("label", ""), kind, sel, f"below th={th} or missing"))

    if verbose and dropped:
        print("Filtered out items:")
        for lab, kind, sel, why in dropped:
            print(f"  - {lab} [{kind}] selector={sel} -> {why}")

    return filtered


def filter_bus_list_mp(n, bus_list, link_th=0.5):
    """Return buses from bus_list that have at least one injecting link with capacity >= link_th.

    Checks every link where the bus appears as bus1..bus5 (output side).
    Buses not present in the network's marginal_price table are also dropped.
    """
    links = _slice_df_first_scenario(n.links) if not n.links.empty else pd.DataFrame()
    cap_col = next((col for col in ("p_nom_opt", "p_nom") if not links.empty and col in links.columns), None)
    mp_cols = set(n.buses_t.marginal_price.columns) if not n.buses_t.marginal_price.empty else set()
    if isinstance(n.buses_t.marginal_price.columns, pd.MultiIndex):
        mp_cols = set(n.buses_t.marginal_price.columns.get_level_values("name"))

    keep, dropped = [], []
    for bus in bus_list:
        # Always keep buses that appear in marginal_price with any content,
        # but only if at least one injecting link is above threshold.
        has_link = False
        if cap_col is not None:
            for bus_col in ("bus1", "bus2", "bus3", "bus4", "bus5"):
                if bus_col not in links.columns:
                    continue
                mask = links[bus_col] == bus
                if mask.any() and (links.loc[mask, cap_col].abs() >= link_th).any():
                    has_link = True
                    break

        if has_link or (not links.empty and cap_col is None):
            keep.append(bus)
        else:
            dropped.append(bus)

    if dropped:
        print(f"[shadow prices] buses filtered out (no injecting link ≥ {link_th} MW): {dropped}")
    return keep


def _bus_net_injection(n, bus):
    """Total power injected INTO `bus` at each snapshot (MW).

    Uses link p0 × efficiency for each output port (positive efficiency only),
    plus generator output and storage-unit discharge at that bus.
    For stochastic networks, uses the first scenario's p0.
    """
    result = pd.Series(0.0, index=n.snapshots)

    # Generators
    if not n.generators.empty and "bus" in n.generators.columns:
        p_gen = getattr(n.generators_t, "p", None)
        if p_gen is not None:
            for g in n.generators.index[n.generators["bus"] == bus]:
                if g in p_gen.columns:
                    result = result.add(p_gen[g].reindex(n.snapshots, fill_value=0.0), fill_value=0.0)

    # Links: injection at each output port = p0 × efficiency_i (if > 0)
    if not n.links.empty:
        p0_raw = getattr(n.links_t, "p0", None)
        if p0_raw is not None and not p0_raw.empty:
            if isinstance(p0_raw.columns, pd.MultiIndex):
                first_scen = p0_raw.columns.get_level_values(0)[0]
                p0_df = p0_raw[first_scen]
            else:
                p0_df = p0_raw
            links = _slice_df_first_scenario(n.links)
            for bus_col, eff_col in [
                ("bus1", "efficiency"),
                ("bus2", "efficiency2"),
                ("bus3", "efficiency3"),
                ("bus4", "efficiency4"),
                ("bus5", "efficiency5"),
            ]:
                if bus_col not in links.columns:
                    continue
                for lk in links.index[links[bus_col] == bus]:
                    if lk not in p0_df.columns:
                        continue
                    eff = float(links.at[lk, eff_col]) if eff_col in links.columns else 1.0
                    if eff > 0:
                        result = result.add(
                            (p0_df[lk] * eff).reindex(n.snapshots, fill_value=0.0),
                            fill_value=0.0,
                        )

    # StorageUnits discharging
    if not n.storage_units.empty and "bus" in n.storage_units.columns:
        p_su = getattr(n.storage_units_t, "p", None)
        if p_su is not None:
            for su in n.storage_units.index[n.storage_units["bus"] == bus]:
                if su in p_su.columns:
                    result = result.add(
                        p_su[su].reindex(n.snapshots, fill_value=0.0).clip(lower=0),
                        fill_value=0.0,
                    )

    # Stores discharging (Store.p > 0 injects into its bus)
    if not n.stores.empty and "bus" in n.stores.columns:
        p_st_raw = getattr(n.stores_t, "p", None)
        if p_st_raw is not None and not p_st_raw.empty:
            if isinstance(p_st_raw.columns, pd.MultiIndex):
                first_scen = p_st_raw.columns.get_level_values(0)[0]
                p_st = p_st_raw[first_scen]
            else:
                p_st = p_st_raw
            stores = _slice_df_first_scenario(n.stores)
            for st in stores.index[stores["bus"] == bus]:
                if st in p_st.columns:
                    result = result.add(
                        p_st[st].reindex(n.snapshots, fill_value=0.0).clip(lower=0),
                        fill_value=0.0,
                    )

    return result.clip(lower=0)


def _energy_weighted_mean(n, bus_list):
    """Return ({bus: energy-weighted mean price}, {bus: annual throughput MWh/y}).

    Weight = energy throughput at the bus each snapshot (q_t × snap_w_t), where
    q_t is the bus net injection. By the bus energy balance the total entering a
    bus equals the total leaving it each snapshot, so weighting by injection or by
    exiting flow gives the same mean. Falls back to duration-weighted mean when a
    bus has no measurable flow. (Single source of truth for the energy-weighted
    mean used by both the CSV/bar chart and the violin overlay.)

    The second return value is the annual throughput (denominator) in MWh/y,
    useful as a second panel in the bar chart.
    """
    mp = n.buses_t.marginal_price
    snap_w = n.snapshot_weightings.get("objective", n.snapshot_weightings.iloc[:, 0])

    means = {}
    throughputs = {}
    for bus in bus_list:
        if isinstance(mp.columns, pd.MultiIndex):
            bus_cols = mp.loc[:, mp.columns.get_level_values("name") == bus]
            if bus_cols.empty:
                continue
            λ = bus_cols.iloc[:, 0]
        else:
            if bus not in mp.columns:
                continue
            λ = mp[bus]

        λ = pd.to_numeric(λ, errors="coerce").reindex(n.snapshots, fill_value=0.0)

        q    = _bus_net_injection(n, bus)
        e_w  = (q * snap_w).reindex(n.snapshots, fill_value=0.0)
        denom = float(e_w.sum())
        throughputs[bus] = denom

        if denom > 1e-6:
            means[bus] = float((λ * e_w).sum() / denom)
        else:
            means[bus] = float((λ * snap_w).sum() / snap_w.sum())

    return means, throughputs


def export_shadow_prices_mean_csv(n, bus_list, out_path):
    """Save energy-weighted mean marginal price and annual throughput for each bus.

    Returns (means, throughputs) dicts — both keyed by bus name.
    Weight = energy injected into the bus at each snapshot (q_t × snap_w_t).
    Falls back to duration-weighted mean when a bus receives no measurable flow.
    """
    means, throughputs = _energy_weighted_mean(n, bus_list)
    rows = [
        {
            "bus": b,
            "energy weighted mean (EUR/MWh)": round(v, 4),
            "annual throughput (MWh/y)": round(throughputs.get(b, 0.0), 1),
        }
        for b, v in means.items()
    ]

    if rows:
        pd.DataFrame(rows).set_index("bus").to_csv(out_path)
        print(f"[shadow prices] energy-weighted mean prices saved → {out_path}")

    return means, throughputs


def plot_shadow_prices_mean_bar(means, out_path, title="Mean shadow prices (energy-weighted)",
                                bus_filter=None, throughput=None):
    """Bar chart of energy-weighted mean marginal prices by bus.

    Parameters
    ----------
    means : dict {bus: float}  (as returned by export_shadow_prices_mean_csv)
    out_path : str or Path
    bus_filter : list[str] or None
        If given, only buses in this list are plotted (preserves order of bus_filter).
    throughput : dict {bus: float} or None
        Annual energy throughput at each bus (MWh/y), used for a second panel
        below the shadow-price bars.  When provided, figure has two rows.
    """
    if not means:
        return

    if bus_filter is not None:
        buses = [b for b in bus_filter if b in means]
    else:
        buses = list(means.keys())
    values = [means[b] for b in buses]

    has_throughput = throughput is not None and any(throughput.get(b, 0) > 0 for b in buses)
    n_rows = 2 if has_throughput else 1
    fig_h = 4.2 * n_rows
    fig_w = max(6, 0.55 * len(buses))

    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(fig_w, fig_h),
        sharex=True,
        constrained_layout=True,
    )
    if n_rows == 1:
        axes = [axes]
    ax = axes[0]

    bars = ax.bar(range(len(buses)), values, color="#2196f3", edgecolor="white", linewidth=0.5)

    # value labels on top of bars
    span = max(abs(v) for v in values) if values else 1.0
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + span * 0.01,
            f"{val:.1f}",
            ha="center", va="bottom", fontsize=8,
        )

    ax.set_ylabel("€/MWh")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.6)

    if has_throughput:
        ax2 = axes[1]
        q_vals = [throughput.get(b, 0.0) for b in buses]
        max_q = max(q_vals) if q_vals else 1.0
        if max_q >= 1_000_000:
            scale, unit = 1e-6, "TWh/y"
        elif max_q >= 1_000:
            scale, unit = 1e-3, "GWh/y"
        else:
            scale, unit = 1.0, "MWh/y"
        q_scaled = [v * scale for v in q_vals]

        bars2 = ax2.bar(range(len(buses)), q_scaled, color="#78909c", edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars2, q_scaled):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(q_scaled) * 0.01,
                f"{val:.1f}",
                ha="center", va="bottom", fontsize=8,
            )
        ax2.set_xticks(range(len(buses)))
        ax2.set_xticklabels(buses, rotation=60, ha="right", fontsize=9)
        ax2.set_ylabel(f"Annual throughput\n({unit})")
        ax2.grid(axis="y", alpha=0.3)
        ax2.axhline(0, color="black", linewidth=0.6)
    else:
        ax.set_xticks(range(len(buses)))
        ax.set_xticklabels(buses, rotation=60, ha="right", fontsize=9)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[shadow prices] mean bar chart saved → {out_path}")


# ---- LCOP BY TECHNOLOGY ----

# Bus slots (in scan order) that can carry a link's main product, paired with
# the efficiency attribute name PyPSA uses for that slot (bus1's is bare
# "efficiency"; bus2+ are "efficiency2", "efficiency3", ...). bus0 is never
# scanned here — by this project's Link convention it is always the primary
# input (the reference flow p0), never an output.
_PRODUCT_BUS_SLOTS = [
    ("bus1", "efficiency"), ("bus2", "efficiency2"),
    ("bus3", "efficiency3"), ("bus4", "efficiency4"),
    ("bus5", "efficiency5"),
]


def _find_product_slot(links, lk, collection_buses):
    """Return (bus_col, eff_col) for whichever bus slot on link `lk` connects
    to a tagged product collection bus (``n.buses['is_product_bus']``), or
    None if none does.

    The main product is not always on bus1 — e.g. biomethanation and
    biomethanation CO2 output their main product via bus2 (bus1 is their
    carbon-source input). Detecting the slot by which bus is actually tagged,
    rather than assuming a fixed bus number, makes this correct for bioCH4,
    H2 and Methanol alike, and for any future product/bus layout.
    """
    for bus_col, eff_col in _PRODUCT_BUS_SLOTS:
        if bus_col not in links.columns:
            continue
        bus = links.at[lk, bus_col]
        if pd.notna(bus) and str(bus) in collection_buses:
            return bus_col, eff_col
    return None


def compute_lcop_by_technology(n, out_csv, out_plot):
    """Compute LCOP, revenue, and annual profit for each technology injecting
    into a product collection bus (tagged is_product_bus=True).

    Definitions (one row per multilink lk with bus1 = collection bus):

      indirect OPEX  = feedstock costs − by-product credits, via KKT:
                     = −Σ_{k≠bus1} eff_k × Σ_t(p0_t × λ_{bus_k,t} × snap_w_t)
                       [eff_0 = −1 for bus0; positive = net cost, typical case]

      LCOP [€/MWh]   = (CAPEX + OPEX + indirect OPEX) / annual production

      revenue main product = eff1 × Σ_t(p0_t × λ_{bus1,t} × snap_w_t)

      net market value = revenue main product − indirect OPEX

      annual profit    = net market value − CAPEX − OPEX
                       = what the market pays you for your product and
                         by-products, minus every cost you incur.
                         ≈ 0 for the marginal (price-setting) technology;
                         > 0 for infra-marginal (lower-cost) technologies.

    Shared components (compressors, storage with their own carrier) are NOT
    included — their cost enters implicitly via the KKT at the interface buses.

    Saves a presentable CSV and a two-panel bar chart.
    Returns results as a DataFrame (column names match the CSV headers).
    """
    import warnings as _warn

    if "is_product_bus" not in n.buses.columns:
        print("[LCOP] no 'is_product_bus' column on buses — skipping")
        return pd.DataFrame()

    collection_buses = set(n.buses.index[n.buses["is_product_bus"].eq(True)])
    if not collection_buses:
        print("[LCOP] no collection buses tagged — skipping")
        return pd.DataFrame()

    links = _slice_df_first_scenario(n.links) if not n.links.empty else pd.DataFrame()
    if links.empty or "bus1" not in links.columns:
        print("[LCOP] no links — skipping")
        return pd.DataFrame()

    product_slots = {}
    for lk in links.index:
        slot = _find_product_slot(links, lk, collection_buses)
        if slot is not None:
            product_slots[lk] = slot
    product_links = list(product_slots.keys())
    if not product_links:
        print("[LCOP] no links inject into collection buses — skipping")
        return pd.DataFrame()

    # ── Statistics ─────────────────────────────────────────────────────────
    try:
        capex_s = n.statistics.capex(groupby=False)
        opex_s  = n.statistics.opex(groupby=False)
    except Exception as e:
        _warn.warn(f"[LCOP] n.statistics failed: {e}")
        capex_s = pd.Series(dtype=float)
        opex_s  = pd.Series(dtype=float)

    def _get_stat(series, name):
        if series.empty:
            return 0.0
        if isinstance(series.index, pd.MultiIndex):
            try:
                val = series.xs(name, level=-1)
                return float(val.sum()) if hasattr(val, "sum") else float(val)
            except KeyError:
                return 0.0
        return float(series.get(name, 0.0))

    snap_w = n.snapshot_weightings.get("objective", n.snapshot_weightings.iloc[:, 0])

    mp = n.buses_t.marginal_price
    if isinstance(mp.columns, pd.MultiIndex):
        mp = mp[mp.columns.get_level_values(0)[0]]

    p0_raw = getattr(n.links_t, "p0", None)
    if p0_raw is not None and not p0_raw.empty:
        p0_df = (
            p0_raw[p0_raw.columns.get_level_values(0)[0]]
            if isinstance(p0_raw.columns, pd.MultiIndex)
            else p0_raw
        )
    else:
        p0_df = pd.DataFrame(index=n.snapshots)

    def _kkt_term(bus, eff, p0):
        """eff × Σ_t(p0_t × λ_{bus,t} × snap_w_t); 0 if bus unavailable."""
        if pd.isna(bus) or str(bus) == "" or str(bus) not in mp.columns:
            return 0.0
        λ = mp[str(bus)].reindex(n.snapshots, fill_value=0.0)
        return float((p0 * float(eff) * λ * snap_w).sum())

    rows = []
    link_names = []
    for lk in product_links:
        capex = _get_stat(capex_s, lk)
        opex  = _get_stat(opex_s,  lk)

        main_bus_col, main_eff_col = product_slots[lk]
        eff1 = float(links.at[lk, main_eff_col]) if main_eff_col in links.columns else 1.0
        p0 = (
            p0_df[lk].reindex(n.snapshots, fill_value=0.0).clip(lower=0)
            if lk in p0_df.columns
            else pd.Series(0.0, index=n.snapshots)
        )
        annual_production = float((p0 * eff1 * snap_w).sum()) if eff1 > 0 else 0.0
        if annual_production <= 0:
            continue

        # ── KKT at bus0 (implicit eff = -1: primary feedstock consumed) ───
        bus0 = links.at[lk, "bus0"] if "bus0" in links.columns else ""
        net_kkt_non_main = _kkt_term(bus0, -1.0, p0)

        # ── KKT at every other bus slot (additional inputs eff<0 and
        # by-products eff>0) — i.e. every slot except bus0 and whichever
        # slot is this link's main product (found above, not always bus1).
        for bus_col, eff_col in _PRODUCT_BUS_SLOTS:
            if bus_col == main_bus_col:
                continue
            if bus_col not in links.columns or eff_col not in links.columns:
                continue
            eff_k = links.at[lk, eff_col]
            if pd.isna(eff_k) or float(eff_k) == 0.0:
                continue
            net_kkt_non_main += _kkt_term(links.at[lk, bus_col], eff_k, p0)

        # indirect OPEX = feedstock costs − by-product credits (positive = net cost)
        indirect_opex = -net_kkt_non_main

        # ── KKT at the main product bus (at the collection bus) ─────────────
        bus1 = links.at[lk, main_bus_col]
        revenue_main = _kkt_term(bus1, eff1, p0)

        net_market_value = revenue_main - indirect_opex
        lcop             = (capex + opex + indirect_opex) / annual_production
        annual_profit    = net_market_value - capex - opex

        product = n.buses.at[bus1, "product"] if ("product" in n.buses.columns and bus1 in n.buses.index) else ""
        link_names.append(lk)
        rows.append({
            "carrier":                      links.at[lk, "carrier"] if "carrier" in links.columns else "",
            "product":                      product,
            "CAPEX (EUR)":                  round(capex, 0),
            "OPEX (EUR)":                   round(opex, 0),
            "indirect OPEX (EUR)":          round(indirect_opex, 0),
            "revenue main product (EUR)":   round(revenue_main, 0),
            "net market value (EUR)":       round(net_market_value, 0),
            "annual production (MWh)":      round(annual_production, 0),
            "LCOP (EUR/MWh)":               round(lcop, 2),
            "annual profit (EUR)":          round(annual_profit, 0),
        })

    if not rows:
        print("[LCOP] no product links with positive output — skipping")
        return pd.DataFrame()

    df = pd.DataFrame(rows, index=pd.Index(link_names, name="link"))

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv)
    print(f"[LCOP] saved → {out_csv}")

    _plot_lcop_bar(df, Path(out_plot))
    return df


def compute_lcop_kkt_by_technology(n, out_csv):
    """KKT-based LCOP: production-weighted average shadow price at the product bus.

    For each technology s injecting into a product collection bus (bus1):

        LCOP_kkt_s = Σ_t( w_t · η₁ · p0_{s,t} · π_{bus1,t} )
                     ─────────────────────────────────────────
                          Σ_t( w_t · η₁ · p0_{s,t} )

    where π_{bus1,t} is the nodal shadow price (marginal_price) at bus1.

    The theory (see docs/economics.rst, section "Levelized Cost of Product
    (LCOP) and shadow prices") shows that at optimum this equals the
    cost-based LCOP from compute_lcop_by_technology for the marginal
    (price-setting) technology; infra-marginal technologies show LCOP_cost <
    LCOP_kkt, the gap being their profit margin. This function lets you
    verify that relationship numerically.

    Returns a DataFrame indexed by link name with columns:
        carrier, product, annual_production_MWh, LCOP_cost (EUR/MWh),
        LCOP_kkt (EUR/MWh), diff (cost − kkt), π_bus1_mean, π_bus1_std
    """
    import warnings as _warn

    if "is_product_bus" not in n.buses.columns:
        print("[LCOP-KKT] no 'is_product_bus' column — skipping")
        return pd.DataFrame()

    collection_buses = set(n.buses.index[n.buses["is_product_bus"].eq(True)])
    if not collection_buses:
        print("[LCOP-KKT] no collection buses tagged — skipping")
        return pd.DataFrame()

    links = _slice_df_first_scenario(n.links) if not n.links.empty else pd.DataFrame()
    if links.empty or "bus1" not in links.columns:
        print("[LCOP-KKT] no links — skipping")
        return pd.DataFrame()

    product_slots = {}
    for lk in links.index:
        slot = _find_product_slot(links, lk, collection_buses)
        if slot is not None:
            product_slots[lk] = slot
    product_links = list(product_slots.keys())
    if not product_links:
        print("[LCOP-KKT] no links inject into collection buses — skipping")
        return pd.DataFrame()

    snap_w = n.snapshot_weightings.get("objective", n.snapshot_weightings.iloc[:, 0])
    mp = n.buses_t.marginal_price
    if isinstance(mp.columns, pd.MultiIndex):
        mp = mp[mp.columns.get_level_values(0)[0]]

    p0_raw = getattr(n.links_t, "p0", None)
    p0_df = (
        (p0_raw[p0_raw.columns.get_level_values(0)[0]]
         if isinstance(p0_raw.columns, pd.MultiIndex) else p0_raw)
        if p0_raw is not None and not p0_raw.empty
        else pd.DataFrame(index=n.snapshots)
    )

    # --- also pull cost-based LCOP for comparison ---
    try:
        cost_df = compute_lcop_by_technology.__wrapped__(n) if hasattr(
            compute_lcop_by_technology, "__wrapped__") else None
    except Exception:
        cost_df = None
    # Simpler: re-read the already-saved cost CSV if it exists alongside out_csv
    cost_csv = Path(out_csv).parent / "lcop_by_technology.csv"
    if cost_df is None and cost_csv.exists():
        try:
            cost_df = pd.read_csv(cost_csv, index_col=0)
        except Exception:
            cost_df = None

    rows = []
    link_names = []
    for lk in product_links:
        main_bus_col, main_eff_col = product_slots[lk]
        bus1 = links.at[lk, main_bus_col]
        eff1 = float(links.at[lk, main_eff_col]) if main_eff_col in links.columns else 1.0

        p0 = (
            p0_df[lk].reindex(n.snapshots, fill_value=0.0).clip(lower=0)
            if lk in p0_df.columns
            else pd.Series(0.0, index=n.snapshots)
        )
        annual_production = float((p0 * eff1 * snap_w).sum())
        if annual_production <= 0:
            continue

        if bus1 not in mp.columns:
            _warn.warn(f"[LCOP-KKT] '{lk}': bus1 '{bus1}' not in marginal_price — skipping")
            continue

        pi = mp[bus1].reindex(n.snapshots, fill_value=0.0)
        # production-weighted average shadow price
        numerator = float((p0 * eff1 * pi * snap_w).sum())
        lcop_kkt  = numerator / annual_production

        # cost-based LCOP for comparison
        lcop_cost = float(cost_df.at[lk, "LCOP (EUR/MWh)"]) if (
            cost_df is not None and lk in cost_df.index and "LCOP (EUR/MWh)" in cost_df.columns
        ) else float("nan")

        product = n.buses.at[bus1, "product"] if (
            "product" in n.buses.columns and bus1 in n.buses.index) else ""

        link_names.append(lk)
        rows.append({
            "carrier":                  links.at[lk, "carrier"] if "carrier" in links.columns else "",
            "product":                  product,
            "annual_production_MWh":    round(annual_production, 0),
            "LCOP_cost (EUR/MWh)":      round(lcop_cost, 4),
            "LCOP_kkt (EUR/MWh)":       round(lcop_kkt, 4),
            "diff cost−kkt (EUR/MWh)":  round(lcop_cost - lcop_kkt, 6) if not np.isnan(lcop_cost) else float("nan"),
            "π_bus1_mean (EUR/MWh)":    round(float(pi.mean()), 4),
            "π_bus1_std (EUR/MWh)":     round(float(pi.std()), 6),
            "π_bus1_prod_weighted (EUR/MWh)": round(lcop_kkt, 4),
        })

    if not rows:
        print("[LCOP-KKT] no product links with positive output — skipping")
        return pd.DataFrame()

    df = pd.DataFrame(rows, index=pd.Index(link_names, name="link"))

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv)
    print(f"[LCOP-KKT] saved → {out_csv}")
    return df


def _plot_lcop_bar(df, out_path):
    """Two-panel bar chart: LCOP [€/MWh] (top) and annual profit [k€] (bottom)."""
    labels  = df.index.tolist()
    lcops   = df["LCOP (EUR/MWh)"].tolist()
    profits = (df["annual profit (EUR)"] / 1e3).tolist()
    if not labels:
        return

    n_tech = len(labels)
    fig, axes = plt.subplots(2, 1, figsize=(max(6, 0.65 * n_tech), 8))

    def _draw_bars(ax, values, ylabel, title, color):
        bars = ax.bar(range(n_tech), values, color=color, edgecolor="white", linewidth=0.5)
        y_range = max(abs(v) for v in values) if values else 1.0
        for bar, val in zip(bars, values):
            ypos = bar.get_height() + y_range * 0.01 if val >= 0 else bar.get_height() - y_range * 0.04
            ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(n_tech))
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        ax.axhline(0, color="black", linewidth=0.6)

    _draw_bars(axes[0], lcops,   "€/MWh",
               "LCOP  =  (CAPEX + OPEX + indirect OPEX) / annual production",
               "#43a047")
    _draw_bars(axes[1], profits, "k€/year",
               "Annual profit  =  revenue main product − indirect OPEX − CAPEX − OPEX",
               "#1565c0")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[LCOP] bar chart saved → {out_path}")


# ---- PAYBACK TIME BY TECHNOLOGY (price mode only) ----

def _payback_years(investment: float, cash_flow: float, discount_rate: float) -> tuple[float, float]:
    """Simple and discounted payback time (years) for one investment.

    Simple payback:      investment / cash_flow
    Discounted payback:  smallest N solving
                          investment = cash_flow · [(1 − (1+r)^-N) / r]
                          i.e. the annuity formula inverted for N:
                          N = −ln(1 − r·investment/cash_flow) / ln(1+r)

    Returns ``(nan, nan)`` if there is no investment to recover, and
    ``(inf, inf)`` if the cash flow never recovers it (``cash_flow <= 0``,
    or — for the discounted case — the perpetuity value ``cash_flow / r``
    is still short of the investment).
    """
    if investment <= 0:
        return float("nan"), float("nan")
    if cash_flow <= 0:
        return float("inf"), float("inf")

    simple = investment / cash_flow

    x = discount_rate * investment / cash_flow
    if x >= 1:
        discounted = float("inf")
    else:
        discounted = -math.log(1 - x) / math.log(1 + discount_rate)

    return simple, discounted


_INVESTMENT_COMP_SPECS = [
    ("Generator",   "generators",    "p_nom"),
    ("Link",        "links",         "p_nom"),
    ("StorageUnit", "storage_units", "p_nom"),
    ("Store",       "stores",        "e_nom"),
]


def _tech_costs_value(tech_costs, row_tech: str, col: str) -> float | None:
    """A single ``tech_costs.at[row_tech, col]`` value, or ``None`` if the row/column
    is missing or non-finite."""
    if row_tech not in tech_costs.index or col not in tech_costs.columns:
        return None
    v = pd.to_numeric(tech_costs.at[row_tech, col], errors="coerce")
    return float(v) if np.isfinite(v) else None


def _tech_annual_fom(tech_costs, row_tech: str) -> float | None:
    """Annual fixed O&M (EUR/MW/year) for one catalogue row: ``investment × FOM%/100``."""
    inv = _tech_costs_value(tech_costs, row_tech, "investment")
    fom_pct = _tech_costs_value(tech_costs, row_tech, "FOM")
    if inv is None or fom_pct is None:
        return None
    return inv * fom_pct / 100.0


def _composite_tech_overrides(tech_costs, n_config, value_fn) -> dict[str, float]:
    """Per-unit-capacity override table (EUR/MW or EUR/MWh) for components whose
    annualised ``capital_cost`` is built in ``scripts/prepare_network.py`` from
    technology-data row(s) that do **not** match the component's own name or
    carrier — so neither ``comp_tech_map`` nor the carrier fallback in
    :func:`_resolve_per_unit_value` can resolve them directly.

    ``value_fn(row_tech) -> float | None`` supplies the per-row value to combine
    (raw ``investment`` for :func:`_composite_investment_overrides`, or annual
    FOM for :func:`_composite_fom_overrides`) — the combination weights
    (``cost factor``, ``distance``, ``max hours``) are identical either way,
    since they come straight from the same ``capital_cost=`` construction in
    ``prepare_network.py`` (only the per-row value being combined differs).

    Keyed by component base name (after stripping an ``EXI_`` prefix), plus
    ``"__SUFFIX__..."`` keys matched against the end of the base name. Only
    entries whose underlying tech_costs row(s) actually exist are included, so
    a missing/renamed catalogue row degrades to "not covered" rather than a
    crash or a silently wrong number.

    This is necessarily a **best-effort, non-exhaustive** table — GreenBubble
    has many composite technologies; only the ones verified against the actual
    ``prepare_network.py`` source are included here. Components not covered
    still show up (with their annualised capex, for materiality) in the
    "not resolved" report from :func:`compute_payback_by_agent`.
    """
    def cf(tech):
        try:
            return float(n_config.at[tech, "cost factor"])
        except Exception:
            return 1.0

    overrides: dict[str, float] = {}

    def _set(name, row_tech, cost_factor_key):
        v = value_fn(row_tech)
        if v is not None:
            overrides[name] = v * cf(cost_factor_key)

    _set("heat pump",                    "industrial heat pump medium temperature", "heat pump")
    _set("El boiler",                    "electric boiler steam",                   "El boiler")
    _set("TES DH storage",               "central water tank storage",              "TES DH")
    _set("TES concrete HX",              "Concrete-discharger",                     "TES concrete")
    _set("TES concrete storage",         "Concrete-store",                          "TES concrete")
    _set("TES concrete El charger",      "Concrete-charger",                        "TES concrete El")
    _set("TES concrete El discharger",   "Concrete-discharger",                     "TES concrete El")
    _set("TES concrete El storage",      "Concrete-store",                          "TES concrete El")

    # H2 pipe / CO2 pipe: fixed = tech.fixed * tech.distance * cost_factor
    for name, row_tech, cf_key in [("H2_pipe", "H2 pipe", "H2 pipe"), ("CO2_pipe", "CO2 gas pipe", "CO2 pipe")]:
        v = value_fn(row_tech)
        dist = _tech_costs_value(tech_costs, row_tech, "distance")
        if v is not None and dist is not None:
            overrides[name] = v * dist * cf(cf_key)

    # battery: two catalogue rows combined (storage/max_hours + inverter) — see add_battery()
    v_store = value_fn("battery storage")
    v_inv = value_fn("battery inverter")
    if v_store is not None and v_inv is not None:
        try:
            max_hours = float(n_config.at["battery", "max hours"])
            overrides["battery"] = (v_store / max_hours + v_inv) * cf("battery")
        except Exception:
            pass

    # "<plant> H2 storage send comp" — hardcoded to the hydrogen compressor row
    # regardless of actual fluid in prepare_network.py::add_HP_storage_aux (a
    # pre-existing quirk, not something this override should "fix").
    _set("__SUFFIX__storage send comp", "hydrogen storage compressor", "H2 compressor")
    # "<plant> H2 HP storage" — confirmed formula; other fluids not verified,
    # deliberately not guessed.
    _set("__SUFFIX__H2 HP storage", "hydrogen storage tank type 1", "H2 HP storage")

    return overrides


def _composite_investment_overrides(tech_costs, n_config) -> dict[str, float]:
    return _composite_tech_overrides(tech_costs, n_config,
                                      lambda t: _tech_costs_value(tech_costs, t, "investment"))


def _composite_fom_overrides(tech_costs, n_config) -> dict[str, float]:
    return _composite_tech_overrides(tech_costs, n_config, lambda t: _tech_annual_fom(tech_costs, t))


def _resolve_per_unit_value(row, comp_tech_map: dict, name: str, overrides: dict, tech_lookup_fn):
    """Resolve a per-unit-capacity value for one component: (1) an exact or
    suffix match in ``overrides`` (composite technologies), (2)
    ``comp_tech_map`` with the component's own carrier as a last-resort
    fallback, resolved via ``tech_lookup_fn(tech) -> float | None``.
    Returns ``None`` if nothing resolves.
    """
    base_name = str(name).removeprefix("EXI_")
    if overrides:
        if base_name in overrides:
            return overrides[base_name]
        for suffix_key, val in overrides.items():
            if suffix_key.startswith("__SUFFIX__") and base_name.endswith(suffix_key.removeprefix("__SUFFIX__")):
                return val

    tech = comp_tech_map.get(name) or (str(row.get("carrier", "")) or None)
    return tech_lookup_fn(tech) if tech else None


def _capacity_opt(row, nom_attr: str) -> float | None:
    """Installed optimal capacity for one component row (``*_opt`` if present, else the static value)."""
    cap_col = f"{nom_attr}_opt" if f"{nom_attr}_opt" in row.index and pd.notna(row.get(f"{nom_attr}_opt")) else nom_attr
    cap = pd.to_numeric(row.get(cap_col), errors="coerce")
    return float(cap) if np.isfinite(cap) and cap > 0 else None


# carrier -> n_config index key, for the few cases where they differ (mirrors
# prepare_network.py's _CARRIER_TO_TECH, but targeting n_config rather than tech_costs).
_EXI_CARRIER_TO_NCONFIG = {"wind": "onwind"}


def _exi_investment_scale(name, carrier, n_config) -> float:
    """Fraction of an EXI_ (brownfield) component's catalogue investment still
    owed, i.e. ``remaining_investment_fraction`` from n_config — mirrors the
    ``rif`` factor in prepare_network.py's ``_exi_capital_cost``, which scales
    that component's ``capital_cost`` in the LP the same way.

    Without this, a brownfield asset's payback investment is its full
    as-new catalogue cost even though the model only ever charges (and only
    ever needs to recover) a residual fraction of it — inflating payback
    time arbitrarily for agents dominated by aged brownfield capacity.
    Returns 1.0 (no scaling) for non-EXI_ components or when the technology/
    fraction can't be resolved in n_config, i.e. falls back to prior behaviour.
    """
    if not str(name).startswith("EXI_") or n_config is None:
        return 1.0
    config_key = _EXI_CARRIER_TO_NCONFIG.get(carrier, carrier)
    if config_key not in n_config.index or "remaining_investment_fraction" not in n_config.columns:
        return 1.0
    rif = n_config.at[config_key, "remaining_investment_fraction"]
    return float(rif) if pd.notna(rif) else 1.0


def _investment_for(row, nom_attr: str, tech_costs, comp_tech_map: dict, name: str,
                     composite_overrides: dict | None = None, n_config=None) -> float | None:
    """Raw upfront investment (EUR) for one component: ``I(tech) × installed capacity``.

    For an ``EXI_`` (brownfield) component, scaled by its ``remaining_investment_fraction``
    (see :func:`_exi_investment_scale`) — the model only owes the residual book value,
    not the full as-new cost.
    """
    inv_per_unit = _resolve_per_unit_value(
        row, comp_tech_map, name, composite_overrides or {},
        lambda tech: _tech_costs_value(tech_costs, tech, "investment"),
    )
    if inv_per_unit is None:
        return None
    cap = _capacity_opt(row, nom_attr)
    if cap is None:
        return None
    scale = _exi_investment_scale(name, row.get("carrier"), n_config)
    return float(inv_per_unit) * cap * scale


def _annual_fom_for(row, nom_attr: str, tech_costs, comp_tech_map: dict, name: str,
                     fom_overrides: dict | None = None) -> float | None:
    """Annual fixed O&M (EUR/year) for one component: ``FOM%(tech) × I(tech) × installed capacity``.

    A real recurring cash outflow (unlike the annualised capital charge, which
    is a bookkeeping construct) — must be subtracted from cash flow for a
    correct payback calculation, separately from the raw investment itself.
    """
    fom_per_unit = _resolve_per_unit_value(
        row, comp_tech_map, name, fom_overrides or {},
        lambda tech: _tech_annual_fom(tech_costs, tech),
    )
    if fom_per_unit is None:
        return None
    cap = _capacity_opt(row, nom_attr)
    return None if cap is None else float(fom_per_unit) * cap


def _plot_payback_bar(df, out_path, total_label="TOTAL", title=None, log_tag="Payback",
                       lifetime_col=None, amortization_label=None, margin_tolerance=0.03):
    """Two-panel figure with one shared legend: discounted payback time
    (left) and capital-cost coverage (right) per row, ``total_label``
    highlighted with a black outline.

    Rows with zero investment (nothing to recoup) are dropped from both
    panels. The two panels tell the same story two ways:

    - **Left (payback, years)**: rows with a non-finite discounted payback
      are still plotted — as a capped, hatched bar at the chart's visual
      ceiling — instead of being silently dropped, since "never pays back"
      is itself a real result. If ``lifetime_col`` is given, a black tick
      marks each row's own technical lifetime.
    - **Right (capital cost coverage, %)**: cash_flow ÷ its own pure
      capital-recovery annuity. Unlike payback, this stays smooth and
      finite right through the optimizer's marginal-pricing condition
      (cash_flow == annualised capital cost, i.e. 100% coverage) instead of
      blowing up to infinity exactly there — the companion view for reading
      "is this agent pulling its own weight" without chasing infinities.

    Rows flagged ``priced at own margin`` (coverage within
    ``margin_tolerance`` of 100%) show a finite payback pinned to their own
    lifetime (marked with a ``*``) rather than the numerically unstable raw
    value — see :func:`compute_payback_by_agent`.
    """
    investment = pd.to_numeric(df["investment (EUR)"], errors="coerce")
    df = df[investment.fillna(0) > 0]
    if df.empty:
        return
    labels = df.index.tolist()
    discounted = pd.to_numeric(df["discounted payback (years)"], errors="coerce")
    cash_flow = pd.to_numeric(df.get("annual cash flow (EUR/y)"), errors="coerce")
    is_finite = np.isfinite(discounted.to_numpy())
    lifetimes = (
        pd.to_numeric(df[lifetime_col], errors="coerce").tolist()
        if lifetime_col is not None and lifetime_col in df.columns
        else [float("nan")] * len(labels)
    )
    coverage = pd.to_numeric(df.get("capital cost coverage (%)"), errors="coerce")
    tol_pct = margin_tolerance * 100

    # -- shared palette across both panels: same color means the same idea
    # ("priced at own margin" and "loss" read identically left and right).
    COLOR_AGENT = "#2f6f8f"
    COLOR_TOTAL = "#c1622d"
    COLOR_SURPLUS = "#3d8f5b"
    COLOR_MARGINAL = "#e3a13f"
    COLOR_SHORTFALL = "#b23b3b"
    COLOR_LOSS = "#9e9e9e"
    HATCH_LOSS = "xx"

    at_margin = (
        df["priced at own margin"].astype(bool).tolist()
        if "priced at own margin" in df.columns
        else [False] * len(labels)
    )

    # -- left panel: payback --
    colors_pb, hatches_pb = [], []
    for i, lbl in enumerate(labels):
        if is_finite[i]:
            if at_margin[i]:
                colors_pb.append(COLOR_MARGINAL)
            else:
                colors_pb.append(COLOR_TOTAL if lbl == total_label else COLOR_AGENT)
            hatches_pb.append(None)
        else:
            colors_pb.append(COLOR_LOSS)
            hatches_pb.append(HATCH_LOSS)

    finite_values = discounted[is_finite].tolist()
    finite_lifetimes = [lt for lt in lifetimes if np.isfinite(lt)]
    cap_height = max(finite_values + finite_lifetimes) * 1.3 if (finite_values or finite_lifetimes) else 1.0
    heights = [discounted.iloc[i] if is_finite[i] else cap_height for i in range(len(labels))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(11, 1.1 * len(labels)), 5))

    bars1 = ax1.bar(range(len(labels)), heights, color=colors_pb, edgecolor="white", linewidth=0.5, zorder=2)
    for bar, hatch in zip(bars1, hatches_pb):
        if hatch:
            bar.set_hatch(hatch)
            bar.set_edgecolor("black")

    y_top_values = list(heights) + finite_lifetimes
    has_lifetime_marker = False
    for bar, lt in zip(bars1, lifetimes):
        if np.isfinite(lt):
            has_lifetime_marker = True
            x0, x1 = bar.get_x(), bar.get_x() + bar.get_width()
            ax1.plot([x0, x1], [lt, lt], color="black", linewidth=2.2, zorder=3)

    y_max = max(y_top_values) if y_top_values else 1.0
    for i, bar in enumerate(bars1):
        label = (f"{discounted.iloc[i]:.1f}*" if at_margin[i] else f"{discounted.iloc[i]:.1f}") \
            if is_finite[i] else "∞ (loss)"
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + y_max * 0.01,
                  label, ha="center", va="bottom", fontsize=8, zorder=4)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=60, ha="right", fontsize=9)
    ax1.set_ylabel("years")
    ax1.set_ylim(top=y_max * 1.18)
    ax1.set_title("Discounted payback time")
    ax1.grid(axis="y", alpha=0.3)

    if amortization_label:
        ax1.text(0.02, 0.97, f"Amortization: {amortization_label}",
                  transform=ax1.transAxes, ha="left", va="top", fontsize=8,
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff8e1", edgecolor="#bdbdbd"))

    # -- right panel: capital cost coverage --
    def _category_color(cov):
        if not np.isfinite(cov):
            return COLOR_LOSS
        if cov < 0:
            return COLOR_LOSS
        if abs(cov - 100) <= tol_pct:
            return COLOR_MARGINAL
        if cov > 100:
            return COLOR_SURPLUS
        return COLOR_SHORTFALL

    cov_mask = coverage.notna()
    cov_labels = [lbl for lbl, ok in zip(labels, cov_mask) if ok]
    cov_values = coverage[cov_mask].tolist()
    if cov_values:
        colors_cov = [_category_color(c) for c in cov_values]
        edge_colors = ["black" if lbl == total_label else "white" for lbl in cov_labels]
        edge_widths = [1.8 if lbl == total_label else 0.5 for lbl in cov_labels]

        bars2 = ax2.bar(range(len(cov_labels)), cov_values, color=colors_cov, edgecolor=edge_colors,
                         linewidth=edge_widths, zorder=2)
        ax2.axhline(100, color="black", linewidth=1.2, linestyle="--", zorder=1)

        y_min2, y_max2 = min(0, min(cov_values)) * 1.1, max(100, max(cov_values)) * 1.15
        for bar, cov in zip(bars2, cov_values):
            va = "bottom" if cov >= 0 else "top"
            offset = (y_max2 - y_min2) * 0.015 * (1 if cov >= 0 else -1)
            ax2.text(bar.get_x() + bar.get_width() / 2, cov + offset, f"{cov:.0f}%",
                      ha="center", va=va, fontsize=8, zorder=4)
        ax2.set_ylim(y_min2, y_max2)
        ax2.set_xticks(range(len(cov_labels)))
        ax2.set_xticklabels(cov_labels, rotation=60, ha="right", fontsize=9)
    ax2.set_ylabel("capital cost coverage (%)")
    ax2.set_title("Capital cost coverage (cash flow ÷ own capital-recovery annuity)")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(title or "Payback and capital cost coverage by agent", y=1.02, fontsize=12)

    legend_handles = [
        Patch(facecolor=COLOR_AGENT, edgecolor="white", label="Agent"),
        Patch(facecolor=COLOR_TOTAL, edgecolor="white", label=total_label),
        Patch(facecolor=COLOR_SURPLUS, edgecolor="white", label="Surplus (coverage > 100%)"),
        Patch(facecolor=COLOR_MARGINAL, edgecolor="white", label=f"Priced at own margin (±{tol_pct:.0f}%)"),
        Patch(facecolor=COLOR_SHORTFALL, edgecolor="white", label="Shortfall (covers opex/FOM, not capital)"),
        Patch(facecolor=COLOR_LOSS, edgecolor="black", hatch=HATCH_LOSS, label="∞ payback (loss)"),
    ]
    if has_lifetime_marker:
        legend_handles.append(Line2D([0], [0], color="black", linewidth=2.2, label="inv. weighted tech. lifetime"))
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, fontsize=8, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.08))

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[{log_tag}] payback + coverage chart saved → {out_path}")


def _agent_capex_fom_lifetime(n, lookup, tech_costs, comp_tech_map, n_config, discount_rate,
                               amortization_period=None, log_tag="Payback"):
    """Investment, FOM and lifetime accumulation per agent — capacity-based and
    scenario-invariant (shared first-stage decisions). Factored out of
    :func:`compute_payback_by_agent` since this part is unrelated to how
    cash flow itself is computed.

    Tracks the technology's own true catalogue lifetime *and*, separately,
    the effective amortization period (``amortization_period`` if set, else
    the true lifetime) — the latter is what the LP's own ``capital_cost``
    actually annuitizes over (see ``helpers.read_costs()`` /
    :ref:`economics-annuity`). These can differ substantially (e.g. a
    10-year ``amortization_period`` against a 25-year technical lifetime),
    and the "capital cost coverage" benchmark must use the *effective*
    period to correctly match the LP's own marginal-pricing condition —
    using the true lifetime instead would silently compare cash flow
    against an annuity the LP was never actually charged.

    Returns
    -------
    (investment_by_agent, fom_by_agent, lifetime_weighted_by_agent,
     lifetime_investment_by_agent, eff_period_weighted_by_agent,
     eff_period_investment_by_agent, pure_annuity_by_agent, coverage_pct)
    """
    from scripts.helpers import annuity

    composite_overrides = _composite_investment_overrides(tech_costs, n_config) if n_config is not None else {}
    fom_overrides = _composite_fom_overrides(tech_costs, n_config) if n_config is not None else {}
    amort_override = (float(amortization_period)
                       if amortization_period not in (None, "", "null") else None)

    investment_by_agent: dict = {}
    fom_by_agent: dict = {}
    lifetime_weighted_by_agent: dict = {}   # Σ(investment_i × lifetime_i), finite-lifetime components only —
                                              # the technology's own TRUE catalogue lifetime (informational:
                                              # "does this pay back within its own useful life").
    lifetime_investment_by_agent: dict = {}  # Σ(investment_i) over that SAME finite-lifetime subset —
                                              # a separate denominator, since some components (e.g. a
                                              # missing `lifetime=` kwarg in prepare_network.py leaves
                                              # PyPSA's inf default attached) have resolved investment
                                              # but no finite lifetime to average in; mixing that
                                              # investment into a denominator that excludes it from the
                                              # numerator would silently understate the average.
    eff_period_weighted_by_agent: dict = {}      # Σ(investment_i × eff_period_i) — eff_period = amortization_period
    eff_period_investment_by_agent: dict = {}    # if set, else the true lifetime. Matches the annuity the LP
                                                  # actually charges; used for the coverage-ratio benchmark
                                                  # and the "priced at own margin" snap target.
    pure_annuity_by_agent: dict = {}         # Σ(investment_i × annuity(eff_period_i, r)) — the PURE capital-
                                              # recovery annuity (no FOM term), matched to the SAME
                                              # finite-eff_period subset. This is the correct benchmark for
                                              # "capital cost coverage": cash_flow here is already net of
                                              # FOM, so comparing it against a FOM-inclusive capital_cost
                                              # (as stored on the network) would double-count FOM.
    total_capex_ground_truth = 0.0
    unresolved = []  # (name, annualised capex)
    for cls, attr, nom in _INVESTMENT_COMP_SPECS:
        df = _slice_df_first_scenario(getattr(n, attr, pd.DataFrame()))
        if df is None or df.empty:
            continue
        ext_col = f"{nom}_extendable"
        if ext_col not in df.columns or "capital_cost" not in df.columns:
            continue
        mask = df[ext_col].astype(bool) & (pd.to_numeric(df["capital_cost"], errors="coerce").fillna(0.0) > 0)
        for name, row in df[mask].iterrows():
            cap_col = f"{nom}_opt" if f"{nom}_opt" in row.index and pd.notna(row.get(f"{nom}_opt")) else nom
            cap = pd.to_numeric(row.get(cap_col), errors="coerce")
            if not np.isfinite(cap) or cap <= 0:
                continue
            annualised = float(cap) * float(row["capital_cost"])
            total_capex_ground_truth += annualised
            agent = lookup.get((cls, str(name)), "Unallocated")

            inv = _investment_for(row, nom, tech_costs, comp_tech_map, name, composite_overrides, n_config)
            if inv is None:
                unresolved.append((name, annualised))
                continue
            investment_by_agent[agent] = investment_by_agent.get(agent, 0.0) + inv
            fom_by_agent[agent] = fom_by_agent.get(agent, 0.0) + (
                _annual_fom_for(row, nom, tech_costs, comp_tech_map, name, fom_overrides) or 0.0)
            lifetime = pd.to_numeric(row.get("lifetime"), errors="coerce")
            if np.isfinite(lifetime) and lifetime > 0:
                lifetime_weighted_by_agent[agent] = lifetime_weighted_by_agent.get(agent, 0.0) + inv * float(lifetime)
                lifetime_investment_by_agent[agent] = lifetime_investment_by_agent.get(agent, 0.0) + inv
                eff_period = amort_override if amort_override is not None else float(lifetime)
                eff_period_weighted_by_agent[agent] = (eff_period_weighted_by_agent.get(agent, 0.0)
                                                        + inv * eff_period)
                eff_period_investment_by_agent[agent] = eff_period_investment_by_agent.get(agent, 0.0) + inv
                pure_annuity_by_agent[agent] = (pure_annuity_by_agent.get(agent, 0.0)
                                                 + inv * annuity(eff_period, discount_rate))

    coverage_pct = (100.0 * (total_capex_ground_truth - sum(a for _, a in unresolved)) / total_capex_ground_truth
                    if total_capex_ground_truth > 0 else float("nan"))
    if unresolved:
        unresolved.sort(key=lambda x: -x[1])
        listed = ", ".join(f"{nm} ({ann:,.0f} EUR/y)" for nm, ann in unresolved[:10])
        more = f", +{len(unresolved) - 10} more" if len(unresolved) > 10 else ""
        print(f"[{log_tag}] investment coverage {coverage_pct:.1f}% of total annualised capex. "
              f"{len(unresolved)} component(s) not resolved: {listed}{more}")
    else:
        print(f"[{log_tag}] investment coverage 100.0% of total annualised capex.")

    return (investment_by_agent, fom_by_agent, lifetime_weighted_by_agent,
            lifetime_investment_by_agent, eff_period_weighted_by_agent,
            eff_period_investment_by_agent, pure_annuity_by_agent, coverage_pct)


def compute_payback_by_agent(n, network_comp_allocation, tech_costs, comp_tech_map, n_config,
                              discount_rate, out_csv, out_plot, amortization_period=None):
    """Simple and discounted payback time per **agent** (the same allocation
    groups used for ``TSC_by_agent`` / :func:`make_global_summary_costs_by_agent`),
    including shared/upstream infrastructure and the revenue attributed to it.

    This aggregates **every** component's investment and cash flow into
    whichever agent ``network_comp_allocation`` assigns it to — e.g. the
    ``biogas`` agent's payback reflects the whole digester + upgrading +
    storage + engine complex and all revenue booked to it, not just one
    isolated link.

    **Handles multiple agents producing the same product.** Cash flow is
    computed per *component* via PyPSA's own shadow-price-based statistics —
    :meth:`n.statistics.revenue` (net of cost on every port, i.e. output
    revenue minus input cost, valued at each bus's own marginal price) minus
    :meth:`n.statistics.opex` (variable dispatch cost) — then summed by agent.
    This is the same accounting PyPSA duality already uses in
    :func:`compute_lcop_by_technology` (verified to reproduce its
    "net market value" exactly), generalised to every component instead of
    just tagged product-bus links.

    This matters because a shared external sale link (e.g. the single
    bioCH4-collection-to-delivery link) is created **once**, by whichever
    producing agent happens to build it first — if a *second* agent later
    also feeds the same collection bus (e.g. catalytic methanation
    alongside biogas upgrading, both selling bioCH4), attributing revenue to
    "whichever component touches the external market" would silently credit
    all of it to the first agent. Per-component shadow-price revenue avoids
    this entirely: each producer earns revenue proportional to its **own**
    throughput at the bus's own price, and the shared sale/delivery link
    itself nets to ~zero (a pure pass-through) — confirmed empirically
    (revenue - opex ≈ 0 to floating-point precision for the bioCH4/H2/
    Methanol collection→delivery links in a real solved network). The same
    reasoning applies to any future carrier producible by more than one
    agent (electricity, if a biogas engine is enabled alongside grid export;
    additional methanol pathways; etc.) with no code changes needed.

    Investment and FOM are capacity-based and scenario-invariant (shared
    first-stage decisions), so they are computed once per component and
    summed by agent. Cash flow is dispatch-based and, for a stochastic
    network, is the probability-weighted **expected** value across scenarios
    (mirroring how ``TSC_by_agent`` reports an expected total). Computed via
    :meth:`n.get_scenario` (a genuine per-scenario flat network, not a view)
    because ``n.statistics.*(groupby=False)`` raises ``TypeError`` outright on
    any scenario-enabled network in the pinned PyPSA 1.0.7 release (confirmed
    by reading ``pypsa/statistics/abstract.py`` — an internal
    ``rename_axis(component_name, axis=0)`` call assumes a flat index).
    Cross-checked on a real solved stochastic network: the weighted
    revenue−opex total matched ``n.objective`` exactly, net of capex.

    Also reports, per agent, the **investment-weighted average technical
    lifetime** — the number a "does this pay back within its useful life?"
    read of the plot needs — alongside whichever ``amortization_period`` is
    actually driving the annuity calculation project-wide (``null`` → each
    technology's own lifetime, shown as "tech lifetime").

    Saves a presentable CSV and a bar chart of discounted payback time
    (technical lifetime marked per bar), plus a ``TOTAL`` row/bar. Returns
    the results as a DataFrame.
    """
    import warnings as _warn

    if tech_costs is None or comp_tech_map is None or network_comp_allocation is None:
        print("[Payback-agent] tech_costs/comp_tech_map/network_comp_allocation not available — skipping")
        return pd.DataFrame()

    lookup = build_allocation_lookup(network_comp_allocation)
    (investment_by_agent, fom_by_agent, lifetime_weighted_by_agent, lifetime_investment_by_agent,
     eff_period_weighted_by_agent, eff_period_investment_by_agent,
     pure_annuity_by_agent, coverage_pct) = _agent_capex_fom_lifetime(
        n, lookup, tech_costs, comp_tech_map, n_config, discount_rate,
        amortization_period=amortization_period, log_tag="Payback-agent")

    # ── Cash flow per agent: per-component shadow-price net value ───────────
    # revenue(component) already nets input cost against output revenue at
    # each port's own marginal price (verified to equal LCOP's "net market
    # value" exactly); opex(component) is the variable dispatch cost on top.
    # Grouping this by agent — rather than summing a shared sale link's opex
    # into whichever agent owns that one link — is what correctly splits
    # revenue when more than one agent produces the same carrier.
    #
    # n.statistics.*(groupby=False) raises TypeError unconditionally on any
    # scenario-enabled network in the pinned PyPSA 1.0.7 release (an internal
    # `rename_axis(component_name, axis=0)` call assumes a flat, non-
    # MultiIndex result — confirmed by reading pypsa/statistics/abstract.py).
    # Workaround: n.get_scenario(name) returns a genuine flat per-scenario
    # Network (has_scenarios=False, not a view/mutation of n) on which the
    # same call works normally; sum each scenario's per-component statistics
    # weighted by its probability. Cross-checked against n.objective on a
    # real solved stochastic network (matched exactly, net of capex).
    if hasattr(n, "scenario_weightings") and n.scenario_weightings is not None and len(n.scenario_weightings) > 0:
        scen_w = n.scenario_weightings["weight"].copy()
        scen_w.index = scen_w.index.astype(str)
        scenario_networks = [(str(s), float(w), n.get_scenario(str(s))) for s, w in scen_w.items()]
    else:
        scenario_networks = [(None, 1.0, n)]

    revenue_by_agent: dict = {}
    opex_by_agent: dict = {}
    for scen, w, n_s in scenario_networks:
        try:
            rev_s = n_s.statistics.revenue(groupby=False, nice_names=False)
            opx_s = n_s.statistics.opex(groupby=False, nice_names=False)
        except Exception as e:
            _warn.warn(f"[Payback-agent] n.statistics.revenue/opex failed"
                       f"{f' for scenario {scen}' if scen else ''}: {e}")
            continue
        for series, target in ((rev_s, revenue_by_agent), (opx_s, opex_by_agent)):
            if series.empty:
                continue
            for (comp_kind, comp_name), val in series.items():
                agent = lookup.get((comp_kind, str(comp_name)), "Unallocated")
                target[agent] = target.get(agent, 0.0) + float(val) * w

    return _assemble_and_save_payback(
        revenue_by_agent, opex_by_agent, investment_by_agent, fom_by_agent,
        lifetime_weighted_by_agent, lifetime_investment_by_agent,
        eff_period_weighted_by_agent, eff_period_investment_by_agent, pure_annuity_by_agent,
        coverage_pct, discount_rate, out_csv, out_plot, amortization_period,
        log_tag="Payback-agent", title="Payback and capital cost coverage by agent")


MARGIN_TOLERANCE = 0.03  # 3% — coverage in [97%, 103%] is treated as "priced at own margin"


def _assemble_and_save_payback(revenue_by_agent, opex_by_agent, investment_by_agent, fom_by_agent,
                                lifetime_weighted_by_agent, lifetime_investment_by_agent,
                                eff_period_weighted_by_agent, eff_period_investment_by_agent,
                                pure_annuity_by_agent, coverage_pct, discount_rate, out_csv, out_plot,
                                amortization_period, log_tag, title):
    """Shared row-assembly, margin-snap, CSV/plot step for
    :func:`compute_payback_by_agent`. Factored out separately since this
    part (payback/coverage formulas, the margin snap, CSV/plot output) is
    independent of how ``revenue_by_agent``/``opex_by_agent`` were computed.

    See :func:`compute_payback_by_agent`'s docstring for the "priced at own
    margin" snap rationale (mathematically, cash_flow == pure capital-
    recovery annuity implies discounted payback == the *effective
    amortization period* exactly — ``amortization_period`` if set, else the
    technology's own lifetime, since that's what the LP's own annuity
    actually uses; MARGIN_TOLERANCE treats a noise-level shortfall near
    that point as such, instead of showing a wildly unstable near-infinite
    number for an economically negligible difference). The reported
    "technical lifetime, investment-weighted" column is always the
    technology's own true catalogue lifetime, regardless of
    ``amortization_period`` — a separate, informational question ("does
    this pay back within its physical life") from the snap target.
    """
    def _weighted_lifetime(agent):
        denom = lifetime_investment_by_agent.get(agent, 0.0)
        if denom <= 0:
            return float("nan")
        return lifetime_weighted_by_agent.get(agent, 0.0) / denom

    def _weighted_eff_period(agent):
        denom = eff_period_investment_by_agent.get(agent, 0.0)
        if denom <= 0:
            return float("nan")
        return eff_period_weighted_by_agent.get(agent, 0.0) / denom

    def _capital_coverage(agent, cash_flow):
        annuity_target = pure_annuity_by_agent.get(agent, 0.0)
        if annuity_target <= 0:
            return float("nan")
        return cash_flow / annuity_target

    def _apply_margin_snap(agent, cash_flow, discounted):
        coverage = _capital_coverage(agent, cash_flow)
        if np.isfinite(coverage) and abs(1.0 - coverage) <= MARGIN_TOLERANCE:
            ep = _weighted_eff_period(agent)
            if np.isfinite(ep):
                return ep, True
        return discounted, False

    # ── Assemble rows ────────────────────────────────────────────────────────
    all_agents = sorted(set(investment_by_agent) | set(revenue_by_agent) | set(opex_by_agent) | set(fom_by_agent))
    rows, row_names = [], []
    for agent in all_agents:
        investment = investment_by_agent.get(agent, 0.0)
        fom = fom_by_agent.get(agent, 0.0)
        cash_flow = revenue_by_agent.get(agent, 0.0) - opex_by_agent.get(agent, 0.0) - fom
        simple, discounted = _payback_years(investment, cash_flow, discount_rate)
        discounted, at_margin = _apply_margin_snap(agent, cash_flow, discounted)
        coverage = _capital_coverage(agent, cash_flow)
        row_names.append(agent)
        rows.append({
            "investment (EUR)":            round(investment, 0),
            "annual FOM (EUR/y)":          round(fom, 0),
            "annual revenue (EUR/y)":      round(revenue_by_agent.get(agent, 0.0), 0),
            "annual running cost (EUR/y)": round(opex_by_agent.get(agent, 0.0), 0),
            "annual cash flow (EUR/y)":    round(cash_flow, 0),
            "capital cost coverage (%)":   round(coverage * 100, 1) if np.isfinite(coverage) else coverage,
            "simple payback (years)":      round(simple, 2) if np.isfinite(simple) else simple,
            "discounted payback (years)":  round(discounted, 2) if np.isfinite(discounted) else discounted,
            "priced at own margin":        at_margin,
            "technical lifetime, investment-weighted (years)":
                round(_weighted_lifetime(agent), 1),
        })

    total_investment = sum(investment_by_agent.values())
    total_fom = sum(fom_by_agent.values())
    total_revenue = sum(revenue_by_agent.values())
    total_opex = sum(opex_by_agent.values())
    total_cash_flow = total_revenue - total_opex - total_fom
    simple, discounted = _payback_years(total_investment, total_cash_flow, discount_rate)
    total_annuity_target = sum(pure_annuity_by_agent.values())
    total_coverage = total_cash_flow / total_annuity_target if total_annuity_target > 0 else float("nan")
    total_lifetime = (sum(lifetime_weighted_by_agent.values()) / sum(lifetime_investment_by_agent.values())
                       if sum(lifetime_investment_by_agent.values()) > 0 else float("nan"))
    total_eff_period = (sum(eff_period_weighted_by_agent.values()) / sum(eff_period_investment_by_agent.values())
                         if sum(eff_period_investment_by_agent.values()) > 0 else float("nan"))
    total_at_margin = False
    if np.isfinite(total_coverage) and abs(1.0 - total_coverage) <= MARGIN_TOLERANCE and np.isfinite(total_eff_period):
        discounted, total_at_margin = total_eff_period, True
    row_names.append("TOTAL")
    rows.append({
        "investment (EUR)":            round(total_investment, 0),
        "investment coverage (%)":     round(coverage_pct, 1) if np.isfinite(coverage_pct) else coverage_pct,
        "annual FOM (EUR/y)":          round(total_fom, 0),
        "annual revenue (EUR/y)":      round(total_revenue, 0),
        "annual running cost (EUR/y)": round(total_opex, 0),
        "annual cash flow (EUR/y)":    round(total_cash_flow, 0),
        "capital cost coverage (%)":   round(total_coverage * 100, 1) if np.isfinite(total_coverage) else total_coverage,
        "simple payback (years)":      round(simple, 2) if np.isfinite(simple) else simple,
        "discounted payback (years)":  round(discounted, 2) if np.isfinite(discounted) else discounted,
        "priced at own margin":        total_at_margin,
        "technical lifetime, investment-weighted (years)": round(total_lifetime, 1),
    })

    df_out = pd.DataFrame(rows, index=pd.Index(row_names, name="agent"))

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv)
    print(f"[{log_tag}] saved → {out_csv}")

    amortization_label = "tech lifetime" if amortization_period in (None, "", "null") \
        else f"{float(amortization_period):.0f} years"

    _plot_payback_bar(df_out, Path(out_plot), total_label="TOTAL",
                       title=title,
                       log_tag=log_tag,
                       lifetime_col="technical lifetime, investment-weighted (years)",
                       amortization_label=amortization_label, margin_tolerance=MARGIN_TOLERANCE)
    return df_out


# ---- VARIABLE COST BY TECHNOLOGY ----

def compute_srmc_by_technology(n, out_csv, out_plot):
    """Short-run marginal cost (SRMC) for each product-bus technology at every snapshot.

    For technology s at snapshot t, with main product on whichever bus slot
    is tagged as a product collection bus (not always bus1 — see
    _find_product_slot):

        SRMC_{s,t} = [ λ_{bus0,t}  −  Σ_{k≠main} η_k · λ_{bus_k,t}  +  VOM_{s,t} ]  /  η_main

    This is the instantaneous production cost per MWh of primary output — the cost of
    producing one more MWh right now given current input market prices. It drives the
    merit order. Note: VOM here is the model input (marginal_cost on links), not to be
    confused with this output metric.

    Saved outputs
    -------------
    CSV  : long-form table  (snapshot, link, product, SRMC_EUR_per_MWh, dispatch_MW,
                              π_product_bus, in_merit)
    Plot : one subplot per product — SRMC time series per technology + product shadow price.
    """
    import warnings as _warn

    if "is_product_bus" not in n.buses.columns:
        print("[SRMC] no 'is_product_bus' column — skipping")
        return pd.DataFrame()

    collection_buses = set(n.buses.index[n.buses["is_product_bus"].eq(True)])
    if not collection_buses:
        print("[SRMC] no collection buses tagged — skipping")
        return pd.DataFrame()

    links  = _slice_df_first_scenario(n.links) if not n.links.empty else pd.DataFrame()
    if links.empty or "bus1" not in links.columns:
        print("[SRMC] no links — skipping")
        return pd.DataFrame()

    product_slots = {}
    for lk in links.index:
        slot = _find_product_slot(links, lk, collection_buses)
        if slot is not None:
            product_slots[lk] = slot
    product_links = list(product_slots.keys())
    if not product_links:
        print("[SRMC] no links inject into collection buses — skipping")
        return pd.DataFrame()

    mp     = n.buses_t.marginal_price
    if isinstance(mp.columns, pd.MultiIndex):
        mp = mp[mp.columns.get_level_values(0)[0]]

    p0_raw = getattr(n.links_t, "p0", None)
    p0_df  = (
        (p0_raw[p0_raw.columns.get_level_values(0)[0]]
         if isinstance(p0_raw.columns, pd.MultiIndex) else p0_raw)
        if p0_raw is not None and not p0_raw.empty
        else pd.DataFrame(index=n.snapshots)
    )

    vom_t_raw = getattr(n.links_t, "marginal_cost", None)
    vom_df = (
        (vom_t_raw[vom_t_raw.columns.get_level_values(0)[0]]
         if isinstance(vom_t_raw.columns, pd.MultiIndex) else vom_t_raw)
        if vom_t_raw is not None and not vom_t_raw.empty
        else pd.DataFrame(index=n.snapshots)
    )

    def _price(bus):
        if pd.isna(bus) or str(bus) not in mp.columns:
            return pd.Series(0.0, index=n.snapshots)
        return mp[str(bus)].reindex(n.snapshots, fill_value=0.0)

    rows = []
    for lk in product_links:
        main_bus_col, main_eff_col = product_slots[lk]
        bus1  = links.at[lk, main_bus_col]
        eff1  = float(links.at[lk, main_eff_col]) if main_eff_col in links.columns else 1.0
        if eff1 == 0:
            continue

        # feedstock cost at bus0 (always consumed): positive = cost
        bus0  = links.at[lk, "bus0"] if "bus0" in links.columns else ""
        net_input_cost = _price(bus0)   # λ_{bus0,t}: cost of consuming bus0

        # every other bus slot: subtract by-product credits (eff>0) and add extra
        # input costs (eff<0) — i.e. every slot except bus0 and whichever slot is
        # this link's main product (found above, not always bus1).
        for bus_col, eff_col in _PRODUCT_BUS_SLOTS:
            if bus_col == main_bus_col:
                continue
            if bus_col not in links.columns or eff_col not in links.columns:
                continue
            eff_k = links.at[lk, eff_col]
            if pd.isna(eff_k) or float(eff_k) == 0.0:
                continue
            net_input_cost = net_input_cost - float(eff_k) * _price(links.at[lk, bus_col])

        # VOM (static scalar or time-varying series)
        vom_scalar = float(links.at[lk, "marginal_cost"]) if "marginal_cost" in links.columns else 0.0
        if lk in vom_df.columns:
            vom_series = vom_df[lk].reindex(n.snapshots, fill_value=vom_scalar)
        else:
            vom_series = pd.Series(vom_scalar, index=n.snapshots)

        # VC per MWh of primary output (bus1)
        vc = (net_input_cost + vom_series) / eff1

        # dispatch and product shadow price for reference
        p0 = (p0_df[lk].reindex(n.snapshots, fill_value=0.0).clip(lower=0)
              if lk in p0_df.columns else pd.Series(0.0, index=n.snapshots))
        pi1 = _price(bus1)

        product_name = n.buses.at[bus1, "product"] if (
            "product" in n.buses.columns and bus1 in n.buses.index) else bus1

        for t in n.snapshots:
            rows.append({
                "snapshot":          t,
                "link":              lk,
                "product":           product_name,
                "SRMC_EUR_per_MWh":  round(float(vc.at[t]), 4),
                "dispatch_MW":       round(float(p0.at[t] * eff1), 4),
                "π_product_bus":     round(float(pi1.at[t]), 4),
                "in_merit":          float(vc.at[t]) <= float(pi1.at[t]) + 1e-3,
            })

    if not rows:
        print("[SRMC] no data — skipping")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[SRMC] saved → {out_csv}")

    _plot_srmc(df, Path(out_plot))
    return df


def _plot_srmc(df, out_path):
    """One subplot per product: SRMC time series per technology + product shadow price."""
    products = df["product"].unique()
    n_prod   = len(products)
    if n_prod == 0:
        return

    carrier_colors_default = [
        "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
        "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
    ]

    fig, axes = plt.subplots(n_prod, 1,
                              figsize=(16, 4 * n_prod),
                              sharex=False,
                              squeeze=False)

    for ax, product in zip(axes[:, 0], products):
        sub = df[df["product"] == product].copy()
        sub = sub.sort_values("snapshot")

        technologies = sub["link"].unique()
        colors = {t: carrier_colors_default[i % len(carrier_colors_default)]
                  for i, t in enumerate(technologies)}

        for tech in technologies:
            ts = sub[sub["link"] == tech].set_index("snapshot")["SRMC_EUR_per_MWh"]
            ax.plot(ts.index, ts.values,
                    label=tech, linewidth=0.8,
                    color=colors[tech], alpha=0.85)

        # product bus shadow price — one series (same for all techs on this bus)
        pi_ts = sub.groupby("snapshot")["π_product_bus"].first()
        if pi_ts.std() > 0.01:
            ax.plot(pi_ts.index, pi_ts.values,
                    label=f"π ({product} bus)", color="black",
                    linewidth=1.2, linestyle="--", alpha=0.7)
        else:
            ax.axhline(pi_ts.mean(), color="black", linewidth=1.2,
                       linestyle="--", alpha=0.6,
                       label=f"π = {pi_ts.mean():.1f} €/MWh (flat)")

        ax.set_ylabel("SRMC  [€/MWh output]")
        ax.set_title(f"Short-run marginal cost (SRMC) — {product}")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

        # clip y-axis to ± 3× mean shadow price so outliers don't collapse the view
        pi_mean = float(pi_ts.mean())
        y_ceil  = max(pi_mean * 3, 50)
        y_floor = -pi_mean * 0.5
        ax.set_ylim(y_floor, y_ceil)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SRMC] plot saved → {out_path}")


# ---- OPTIMAL CAPACITIES ----
def _slice_df_first_scenario(df: pd.DataFrame):
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()

    if not isinstance(df.index, pd.MultiIndex):
        return df

    names = list(df.index.names)
    sc_level = "scenario" if "scenario" in names else names[0]

    # empty MI safety
    if len(df.index) == 0:
        return df

    # scenario level safety
    if sc_level not in df.index.names:
        return df

    sc_vals = pd.Index(df.index.get_level_values(sc_level)).drop_duplicates()
    if len(sc_vals) == 0:
        return df

    sc0 = sc_vals[0]
    return df.xs(sc0, level=sc_level)

def _expand_exi(names):
    out = []
    for nm in names:
        out.append(nm)
        if isinstance(nm, str) and not nm.startswith("EXI_"):
            out.append("EXI_" + nm)
    seen, out2 = set(), []
    for nm in out:
        if nm not in seen:
            seen.add(nm)
            out2.append(nm)
    return out2

def _as_list(sel):
    if sel is None:
        return []
    if isinstance(sel, str):
        return [sel]
    if isinstance(sel, (list, tuple, set)):
        return list(sel)
    return [sel]

def _cap_series_one(n, kind):
    """One capacity series from n (first scenario if scenario-indexed)."""
    if kind == "Link":
        links = _slice_df_first_scenario(n.links)
        col = "p_nom_opt" if "p_nom_opt" in links.columns else ("p_nom" if "p_nom" in links.columns else None)
        return links[col] if col else None
    if kind == "Generator":
        gens = _slice_df_first_scenario(n.generators)
        col = "p_nom_opt" if "p_nom_opt" in gens.columns else ("p_nom" if "p_nom" in gens.columns else None)
        return gens[col] if col else None
    if kind == "Store":
        stores = _slice_df_first_scenario(n.stores)
        col = "e_nom_opt" if "e_nom_opt" in stores.columns else ("e_nom" if "e_nom" in stores.columns else None)
        return stores[col] if col else None
    if kind == "StorageUnit":
        sus = _slice_df_first_scenario(n.storage_units)
        col = "p_nom_opt" if "p_nom_opt" in sus.columns else ("p_nom" if "p_nom" in sus.columns else None)
        return sus[col] if col else None
    if kind == "StorageUnit_E":
        sus = _slice_df_first_scenario(n.storage_units)
        col = "p_nom_opt" if "p_nom_opt" in sus.columns else ("p_nom" if "p_nom" in sus.columns else None)
        if col is None:
            return None
        p_nom = sus[col]
        mh = sus["max_hours"].reindex(p_nom.index, fill_value=1.0) if "max_hours" in sus.columns else 1.0
        return p_nom * mh
    raise ValueError(kind)

def _bus_unit_and_carrier(n, bus_name):
    buses = _slice_df_first_scenario(n.buses)
    if buses is None or bus_name not in buses.index:
        return ("", "")
    unit = str(buses.at[bus_name, "unit"]) if "unit" in buses.columns and pd.notna(buses.at[bus_name, "unit"]) else ""
    carrier = str(buses.at[bus_name, "carrier"]) if "carrier" in buses.columns and pd.notna(buses.at[bus_name, "carrier"]) else ""
    return carrier, unit

def _convert_store_unit_to_energy(unit: str) -> str:
    if unit is None:
        return ""
    u = str(unit).strip()
    if not u:
        return ""
    power_map = {"W": "Wh", "kW": "kWh", "MW": "MWh", "GW": "GWh", "TW": "TWh"}
    if u in power_map:
        return power_map[u]
    if u.endswith("/h"):
        return u[:-2].strip()
    if u.endswith("h"):
        return u
    return u + "h"

def _carrier_unit_for_item(n, kind, name):
    """Carrier from the relevant bus; unit rules per spec."""
    if kind == "Link":
        links = _slice_df_first_scenario(n.links)
        if links is None or name not in links.index or "bus0" not in links.columns:
            return ("", "")
        bus0 = links.at[name, "bus0"]
        return _bus_unit_and_carrier(n, bus0)

    if kind == "Generator":
        gens = _slice_df_first_scenario(n.generators)
        if gens is None or name not in gens.index or "bus" not in gens.columns:
            return ("", "")
        bus = gens.at[name, "bus"]
        return _bus_unit_and_carrier(n, bus)

    if kind == "Store":
        stores = _slice_df_first_scenario(n.stores)
        if stores is None or name not in stores.index or "bus" not in stores.columns:
            return ("", "")
        bus = stores.at[name, "bus"]
        carrier, unit = _bus_unit_and_carrier(n, bus)
        return carrier, _convert_store_unit_to_energy(unit)

    if kind in ("StorageUnit", "StorageUnit_E"):
        sus = _slice_df_first_scenario(n.storage_units)
        if sus is None or name not in sus.index or "bus" not in sus.columns:
            return ("", "")
        bus = sus.at[name, "bus"]
        carrier, unit = _bus_unit_and_carrier(n, bus)
        if kind == "StorageUnit_E":
            unit = _convert_store_unit_to_energy(unit)
        return carrier, unit

    return ("", "")


def build_capacity_compare_from_items(
    n_rp,                    # stochastic OR deterministic network
    items,                   # unified items list
    ws_networks=None,        # dict like {"WS-2023": n_ws23, ...} or None
    default_th=0.5,
    sp_col="SP",             # name for the SP column
):
    """
    Output index: MultiIndex (kind, name)
    Columns: label, carrier, unit, SP, WS-..., ...
    Applies per-item threshold 'th' (or default_th).
    Auto-includes EXI_<name> if present (exact match only).
    """
    ws_networks = ws_networks or {}

    # Build rows from selectors (exact + EXI), but only keep those that exist in RP or any WS
    rows = []
    meta = {}  # (kind,name) -> {label, th}

    for it in items:
        kind = it["kind"]
        label = it.get("label", "")
        th = float(it.get("th", default_th))
        wanted = _expand_exi(_as_list(it.get("selector")))

        rp_caps = _cap_series_one(n_rp, kind)
        ws_caps_list = {k: _cap_series_one(n_ws, kind) for k, n_ws in ws_networks.items()}

        for nm in wanted:
            exists = False
            if rp_caps is not None and nm in rp_caps.index:
                exists = True
            else:
                for _, caps in ws_caps_list.items():
                    if caps is not None and nm in caps.index:
                        exists = True
                        break
            if not exists:
                continue

            key = (kind, nm)
            if key not in meta:  # preserve order, avoid duplicates
                rows.append(key)
                meta[key] = {"label": label, "th": th}

            # Auto-add energy capacity row for every StorageUnit power row.
            # Use a small fixed energy threshold (1e-3 MWh) rather than the
            # power threshold: max_hours varies widely (e.g. 0.15h for TES DH
            # vs 2h for battery), so a power-derived threshold would filter out
            # short-duration stores whose power capacity already passed.
            if kind == "StorageUnit":
                e_key = ("StorageUnit_E", nm)
                if e_key not in meta:
                    rows.append(e_key)
                    meta[e_key] = {"label": label + " [Energy]", "th": 1e-3}

    idx = pd.MultiIndex.from_tuples(rows, names=["kind", "name"])
    out = pd.DataFrame(index=idx)

    # Add RP column
    out[sp_col] = pd.NA
    for (kind, nm) in out.index:
        caps = _cap_series_one(n_rp, kind)
        if caps is not None and nm in caps.index:
            out.at[(kind, nm), sp_col] = caps.loc[nm]

    # Add WS columns
    for ws_label, n_ws in ws_networks.items():
        out[ws_label] = pd.NA
        for (kind, nm) in out.index:
            caps = _cap_series_one(n_ws, kind)
            if caps is not None and nm in caps.index:
                out.at[(kind, nm), ws_label] = caps.loc[nm]

    # Add label, carrier, unit (derived from RP network; if missing, try first WS)
    labels, carriers, units = [], [], []
    for (kind, nm) in out.index:
        labels.append(meta[(kind, nm)]["label"])

        carrier, unit = _carrier_unit_for_item(n_rp, kind, nm)
        if (not carrier and not unit) and ws_networks:
            # fallback to first WS network where it exists
            for _, n_ws in ws_networks.items():
                c2, u2 = _carrier_unit_for_item(n_ws, kind, nm)
                if c2 or u2:
                    carrier, unit = c2, u2
                    break
        carriers.append(carrier)
        units.append(unit)

    out.insert(0, "label", labels)
    out.insert(1, "carrier", carriers)
    out.insert(2, "unit", units)

    # Apply per-row threshold to value columns
    value_cols = [c for c in out.columns if c not in ("label", "carrier", "unit")]
    for (kind, nm) in out.index:
        th = meta[(kind, nm)]["th"]
        for c in value_cols:
            v = pd.to_numeric(out.at[(kind, nm), c], errors="coerce")
            if pd.isna(v) or abs(v) < th:
                out.at[(kind, nm), c] = pd.NA
            else:
                out.at[(kind, nm), c] = float(v)

    # Drop rows empty everywhere
    out = out.dropna(subset=value_cols, how="all")

    # Round numeric
    out[value_cols] = out[value_cols].apply(pd.to_numeric, errors="coerce").round(2)

    return out


def plot_capacity_compare_from_items(
    df,
    outpath=None,
    title="Installed capacities (SP vs WS)",
    palette_name="Set2",
    max_items=None,
    legend_ncol=4,
    annotate_y_pad=0.02,  # fraction of y-range
):
    value_cols = [c for c in df.columns if c not in ("label", "carrier", "unit")]
    d = df.copy()
    d[value_cols] = d[value_cols].apply(pd.to_numeric, errors="coerce")
    d = d.dropna(subset=value_cols, how="all")

    if max_items is not None and len(d) > max_items:
        keep = d[value_cols].max(axis=1).sort_values(ascending=False).head(max_items).index
        d = d.loc[keep]

    comp_order = {"Link": 0, "Store": 1, "Generator": 2, "StorageUnit": 3, "StorageUnit_E": 4}
    d = d.sort_index(key=lambda idx: [comp_order.get(i[0], 99) for i in idx])

    xlabels = [d.at[i, "label"] if d.at[i, "label"] else i[1] for i in d.index]

    n_items = len(d)
    n_cols = len(value_cols)
    x = np.arange(n_items)
    width = 0.8 / max(n_cols, 1)

    cmap = mpl.colormaps.get_cmap(palette_name)
    colors = [cmap(i) for i in np.linspace(0.05, 0.95, n_cols)]

    fig, ax = plt.subplots(figsize=(max(11, 0.75 * n_items), 5.6))

    for k, (col, col_color) in enumerate(zip(value_cols, colors)):
        ax.bar(x + k * width, d[col].values, width=width, label=str(col), color=col_color)

    ax.set_title(title)
    ax.set_ylabel("Installed capacity (technology-specific units)")
    ax.set_xticks(x + width * (n_cols - 1) / 2)
    ax.set_xticklabels(xlabels, rotation=45, ha="right")
    ax.grid(True, axis="y", alpha=0.25)

    # separators between kind blocks
    kinds = [i[0] for i in d.index]
    for i in range(1, len(kinds)):
        if kinds[i] != kinds[i - 1]:
            ax.axvline(i - 0.5, linestyle="--", linewidth=0.8, alpha=0.6)

    # carrier \n unit annotations
    ymax = np.nanmax(d[value_cols].to_numpy(dtype=float)) if n_items else 1.0
    ypad = annotate_y_pad * (ymax if ymax > 0 else 1.0)

    for i_idx, idx in enumerate(d.index):
        carrier = d.at[idx, "carrier"] or ""
        unit = d.at[idx, "unit"] or ""
        if not (carrier or unit):
            continue

        y_top = np.nanmax(d.loc[idx, value_cols].to_numpy(dtype=float))
        if not np.isfinite(y_top):
            continue

        x_center = x[i_idx] + width * (n_cols - 1) / 2
        txt = f"{carrier}\n{unit}".strip()

        ax.text(x_center, y_top + ypad, txt, ha="center", va="bottom", fontsize=8)

    ax.legend(ncol=min(legend_ncol, len(value_cols)), frameon=False, fontsize=9)

    fig.tight_layout()
    if outpath:
        outpath = Path(outpath)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ---- shadow price distribution ----
def _pick_first_match(candidates, selector):
    if selector is None:
        return None

    if isinstance(selector, str):
        return selector if selector in candidates else None

    if isinstance(selector, dict):
        if "contains" in selector:
            token = selector["contains"]
            for c in candidates:
                if token in c:
                    return c
            return None
        if "regex" in selector:
            pat = re.compile(selector["regex"])
            for c in candidates:
                if pat.search(c):
                    return c
            return None

    if callable(selector):
        for c in candidates:
            try:
                if selector(c):
                    return c
            except Exception:
                continue
        return None

    return None


def _clip_series(s: pd.Series, handle_spikes="clip", quantile_hi=0.98, quantile_lo=None,
                 whisker=1.5, floor_zero=False):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return s

    if handle_spikes == "clip":
        q_lo = quantile_lo if quantile_lo is not None else (1 - quantile_hi)
        lo, hi = s.quantile(q_lo), s.quantile(quantile_hi)
        if floor_zero:
            lo = max(lo, 0.0)
        return s.clip(lower=lo, upper=hi)

    if handle_spikes == "iqr":
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - whisker * iqr, q3 + whisker * iqr
        if floor_zero:
            lo = max(lo, 0.0)
        return s.clip(lower=lo, upper=hi)

    # "none"
    if floor_zero:
        s = s.clip(lower=0.0)
    return s


def _weighted_resample(values, weights, n_draws=20000, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    v = np.asarray(values)
    w = np.asarray(weights, dtype=float)

    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v, w = v[mask], w[mask]
    if v.size == 0:
        return np.array([])

    p = w / w.sum()
    idx = rng.choice(v.size, size=n_draws, replace=True, p=p)
    return v[idx]


def _weighted_ldc(values, weights, n_points=1001):
    """
    Build a weighted duration curve:
      - sort values descending
      - compute weighted cumulative percentage (0..100)
      - interpolate to an evenly spaced percentile grid
    Returns (x_percent, y_value) where len(x)=len(y)=n_points
    """
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)

    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v, w = v[m], w[m]
    if v.size == 0:
        return None, None

    # sort descending by value
    order = np.argsort(v)[::-1]
    v = v[order]
    w = w[order]

    cw = np.cumsum(w)
    total = cw[-1]
    if total <= 0:
        return None, None

    x = 100.0 * (cw / total)  # weighted "percent of time"
    # ensure strictly increasing x for interp (handle ties safely)
    # if there are identical x due to zeros etc, take unique
    x_u, idx = np.unique(x, return_index=True)
    v_u = v[idx]

    xq = np.linspace(0, 100, n_points)
    # For xq below first x_u, use first value (max); above last, use last (min)
    yq = np.interp(xq, x_u, v_u, left=v_u[0], right=v_u[-1])
    return xq, yq


def shadow_prices_violinplot_stoch(
    n,
    bus_list,
    folder,
    link_mc_items=None,              # list of {"label":..., "selector":...}
    snapshot_weight_col="objective", # n.snapshot_weightings column
    scenario_weight_col="weight",    # n.scenario_weightings column
    n_draws=20000,
    seed=0,
    handle_spikes="clip",
    quantile_hi=0.98,
    quantile_lo=None,
    whisker=1.5,
    floor_zero=False,
    note_text="weighted by scenario; dunkelflaute spikes handled",
    mean_color="crimson",
    mean_linewidth=2.0,
    title="Shadow prices (violin w/ mean) – scenario weighted",
    fname="shd_prices_violin.png",
):

    rng = np.random.default_rng(seed)

    # ---- snapshot weights
    snap_w = n.snapshot_weightings[snapshot_weight_col].reindex(n.snapshots).fillna(0.0).to_numpy()

    # ---- scenario weights (deterministic fallback)
    scenarios = [None]
    scen_prob = {None: 1.0}
    scen_txt = "Deterministic"

    if hasattr(n, "scenario_weightings") and n.scenario_weightings is not None:
        try:
            # Works if scenario_weightings is a Series or DataFrame
            sw = n.scenario_weightings[scenario_weight_col]
            sw = sw.dropna()
            sw = sw[sw > 0]

            # Only use stochastic mode if there are actually scenarios
            if len(sw) > 0:
                scenarios = list(sw.index)
                scen_prob = sw.to_dict()
                scen_txt = "Scenario weights:\n" + "\n".join([f"{k}: {v:.2f}" for k, v in sw.items()])
        except Exception:
            # Fall back to deterministic if anything about scenario_weightings is weird
            pass

    # ---- tables
    mp = n.buses_t.marginal_price
    mc_links = n.links_t.marginal_cost

    def _series_from_mi_cols(df, scen, name):
        if isinstance(df.columns, pd.MultiIndex) and {"scenario", "name"}.issubset(df.columns.names):
            if scen is None:
                scen = df.columns.get_level_values("scenario").unique()[0]
            key = (scen, name)
            return df[key] if key in df.columns else None
        # deterministic columns
        return df[name] if name in df.columns else None

    def _available_link_names(scen):
        if isinstance(mc_links.columns, pd.MultiIndex) and "scenario" in mc_links.columns.names:
            if scen is None:
                scen = mc_links.columns.get_level_values("scenario").unique()[0]
            return list(mc_links.xs(scen, level="scenario", axis=1).columns)
        return list(mc_links.columns)

    items = []      # list of (label, sample_array)
    bus_flags = []  # parallel: True if the item is a bus (gets energy-weighted marker)

    # ---- buses
    for bus in bus_list:
        all_vals, all_wts = [], []
        for scen in scenarios:
            s = _series_from_mi_cols(mp, scen, bus)
            if s is None:
                continue
            v = pd.to_numeric(pd.Series(s, copy=False), errors="coerce").to_numpy()
            w = snap_w * float(scen_prob.get(scen, 1.0))
            all_vals.append(v)
            all_wts.append(w)

        if not all_vals:
            continue

        values = np.concatenate(all_vals)
        weights = np.concatenate(all_wts)

        clipped = _clip_series(pd.Series(values), handle_spikes, quantile_hi, quantile_lo, whisker, floor_zero)
        m = np.isfinite(clipped.to_numpy())
        sample = _weighted_resample(clipped.to_numpy()[m], weights[m], n_draws=n_draws, rng=rng)

        if sample.size:
            items.append((bus, sample))
            bus_flags.append(True)


    # ---- optional link marginal costs (selectors)
    link_mc_items = link_mc_items or []
    for it in link_mc_items:
        label = it["label"]
        selector = it.get("selector")

        all_vals, all_wts = [], []
        for scen in scenarios:
            chosen = _pick_first_match(_available_link_names(scen), selector)
            if chosen is None:
                continue
            s = _series_from_mi_cols(mc_links, scen, chosen)
            if s is None:
                continue
            v = pd.to_numeric(pd.Series(s, copy=False), errors="coerce").to_numpy()
            w = snap_w * float(scen_prob.get(scen, 1.0))
            all_vals.append(v)
            all_wts.append(w)

        if not all_vals:
            continue

        values = np.concatenate(all_vals)
        weights = np.concatenate(all_wts)

        clipped = _clip_series(pd.Series(values), handle_spikes, quantile_hi, quantile_lo, whisker, floor_zero)
        m = np.isfinite(clipped.to_numpy())
        sample = _weighted_resample(clipped.to_numpy()[m], weights[m], n_draws=n_draws, rng=rng)

        if sample.size:
            items.append((label, sample))
            bus_flags.append(False)

    if not items:
        raise ValueError("No data to plot: all requested buses/links were missing or empty.")

    labels = [lab for lab, _ in items]
    data = [arr for _, arr in items]

    fig, ax = plt.subplots(figsize=(max(9, 0.45 * len(labels)), 4.6))
    vp = ax.violinplot(data, showmeans=True, showmedians=False, showextrema=True)

    vp["cmeans"].set_color(mean_color)
    vp["cmeans"].set_linewidth(mean_linewidth)

    ax.set_xticks(range(1, len(labels) + 1), labels, rotation=90)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)

    ax.text(0.02, 0.95, scen_txt, transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

    if handle_spikes in ("clip", "iqr"):
        scope_note = note_text + ("\n(floored at 0)" if floor_zero else "")
        ax.text(0.98, 0.98, scope_note, transform=ax.transAxes,
                ha="right", va="top", fontsize=9,
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

    # ---- overlay the energy-weighted mean (buses only) -----------------------
    # The crimson violin mean is snapshot/scenario-weighted (the distribution
    # mean). Add the energy-weighted mean (same value as shadow_prices_mean.csv /
    # the bar chart) so the gap between "average over time" and "average weighted
    # by when energy flows" is visible. Computed from the unclipped data, so for
    # spiky carriers the marker may sit above the clipped violin body.
    ew_bus_labels = [lab for lab, isb in zip(labels, bus_flags) if isb]
    ew_means, _ = _energy_weighted_mean(n, ew_bus_labels) if ew_bus_labels else ({}, {})
    ew_x, ew_y = [], []
    for i, (lab, isb) in enumerate(zip(labels, bus_flags), start=1):
        if isb and lab in ew_means:
            ew_x.append(i)
            ew_y.append(ew_means[lab])
    if ew_x:
        ax.scatter(ew_x, ew_y, marker="D", s=34, color="royalblue",
                   edgecolor="white", linewidth=0.6, zorder=6)

    handles = [Line2D([0], [0], color=mean_color, lw=mean_linewidth,
                      label="snapshot-weighted mean")]
    if ew_x:
        handles.append(Line2D([0], [0], marker="D", color="royalblue", lw=0,
                              markeredgecolor="white",
                              label="energy-weighted mean (buses)"))
    ax.legend(handles=handles, loc="upper center", fontsize=8, framealpha=0.6)

    plt.tight_layout()

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / fname, dpi=300, bbox_inches="tight")
    plt.close(fig)

def shadow_prices_ldc_stoch(
    n,
    bus_list,
    folder,
    link_mc_items=None,              # list of {"label":..., "selector":...}
    snapshot_weight_col="objective",
    scenario_weight_col="weight",
    handle_spikes="clip",
    quantile_hi=0.98,
    quantile_lo=None,
    whisker=1.5,
    floor_zero=False,
    n_points=1001,
    title="Shadow prices (duration curves)",
    fname="shd_prices_ldc__subplots.png",
    lw=1.8,
    ncols=2,                         # subplot layout
    sharey=True,
):
    """
    Creates ONE figure with subplots:
      - one subplot per scenario (snapshot-weighted only)
      - one subplot for stochastic expected (scenario_prob × snapshot_weight)

    Saves PNG to folder/fname.
    """

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    # ---- snapshot weights
    snap_w = n.snapshot_weightings[snapshot_weight_col].reindex(n.snapshots).fillna(0.0).to_numpy()

    # ---- scenario weights (deterministic fallback)
    if hasattr(n, "scenario_weightings") and n.scenario_weightings is not None and len(n.scenario_weightings) > 0:
        sw = n.scenario_weightings[scenario_weight_col].copy()
        scenarios = list(sw.index.astype(str))
        scen_prob = sw.astype(float).to_dict()
        scen_txt = "Scenario weights:\n" + "\n".join([f"{k}: {v:.2f}" for k, v in sw.items()])
        is_stoch = True
    else:
        scenarios = ["deterministic"]
        scen_prob = {"deterministic": 1.0}
        scen_txt = "Deterministic"
        is_stoch = False

    mp = n.buses_t.marginal_price
    mc_links = getattr(n.links_t, "marginal_cost", None)

    def _series_from_mi_cols(df, scen, name):
        if df is None:
            return None
        if isinstance(df.columns, pd.MultiIndex) and {"scenario", "name"}.issubset(df.columns.names):
            key = (scen, name)
            return df[key] if key in df.columns else None
        return df[name] if name in df.columns else None

    def _available_link_names(scen):
        if mc_links is None:
            return []
        if isinstance(mc_links.columns, pd.MultiIndex) and "scenario" in mc_links.columns.names:
            try:
                return list(mc_links.xs(scen, level="scenario", axis=1).columns)
            except KeyError:
                return []
        return list(mc_links.columns)

    link_mc_items = link_mc_items or []

    # Labels (for consistent colors across subplots)
    label_list = list(bus_list) + [it["label"] for it in link_mc_items]
    label_list = [str(x) for x in label_list]

    def _compute_curves_for_mode(mode, scen=None):
        """
        mode:
          - "scenario": only that scenario, weights = snap_w
          - "combined": concat over scenarios, weights = snap_w * prob
        returns dict: label -> (xq, yq)
        """
        curves = {}

        # ---- buses
        for bus in bus_list:
            all_vals, all_wts = [], []

            scen_iter = scenarios if mode == "combined" else [scen]
            for sname in scen_iter:
                s = _series_from_mi_cols(mp, sname, bus)
                if s is None:
                    continue
                v = pd.to_numeric(pd.Series(s, copy=False), errors="coerce").to_numpy()
                w = snap_w * float(scen_prob.get(sname, 0.0)) if mode == "combined" else snap_w
                all_vals.append(v)
                all_wts.append(w)

            if not all_vals:
                continue

            values = np.concatenate(all_vals)
            weights = np.concatenate(all_wts)

            clipped = _clip_series(pd.Series(values), handle_spikes, quantile_hi, quantile_lo, whisker, floor_zero)
            m = np.isfinite(clipped.to_numpy())
            xq, yq = _weighted_ldc(clipped.to_numpy()[m], weights[m], n_points=n_points)
            if xq is not None:
                curves[str(bus)] = (xq, yq)

        # ---- link marginal costs (selectors)
        for it in link_mc_items:
            label = str(it["label"])
            selector = it.get("selector")

            all_vals, all_wts = [], []

            scen_iter = scenarios if mode == "combined" else [scen]
            for sname in scen_iter:
                chosen = _pick_first_match(_available_link_names(sname), selector)
                if chosen is None:
                    continue
                s = _series_from_mi_cols(mc_links, sname, chosen)
                if s is None:
                    continue
                v = pd.to_numeric(pd.Series(s, copy=False), errors="coerce").to_numpy()
                w = snap_w * float(scen_prob.get(sname, 0.0)) if mode == "combined" else snap_w
                all_vals.append(v)
                all_wts.append(w)

            if not all_vals:
                continue

            values = np.concatenate(all_vals)
            weights = np.concatenate(all_wts)

            clipped = _clip_series(pd.Series(values), handle_spikes, quantile_hi, quantile_lo, whisker, floor_zero)
            m = np.isfinite(clipped.to_numpy())
            xq, yq = _weighted_ldc(clipped.to_numpy()[m], weights[m], n_points=n_points)
            if xq is not None:
                curves[label] = (xq, yq)

        return curves

    # Build all subplot datasets
    panels = []
    if is_stoch:
        for scen in scenarios:
            panels.append(("scenario", scen))
        panels.append(("combined", "stochastic"))
    else:
        panels.append(("scenario", "deterministic"))

    # layout
    n_panels = len(panels)
    ncols = max(1, int(ncols))
    nrows = int(math.ceil(n_panels / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.2 * ncols, 4.2 * nrows), sharey=sharey)
    axes = np.atleast_1d(axes).ravel()

    # consistent colors per label
    cmap = plt.get_cmap("Dark2")
    color_map = {lab: cmap(i % cmap.N) for i, lab in enumerate(label_list)}

    any_data = False

    for ax_i, (mode, scen) in enumerate(panels):
        ax = axes[ax_i]

        if mode == "combined":
            curves = _compute_curves_for_mode("combined", scen=None)
            subtitle = "stochastic (scenario×snapshot weighted)"
        else:
            curves = _compute_curves_for_mode("scenario", scen=scen)
            subtitle = f"scenario {scen} (snapshot-weighted)"

        if curves:
            any_data = True
            for lab in label_list:
                if lab not in curves:
                    continue
                xq, yq = curves[lab]
                ax.plot(xq, yq, linewidth=lw, color=color_map.get(lab, None), label=lab)

        ax.set_title(subtitle)
        ax.set_xlabel("Percent of time (%)")
        if ax_i % ncols == 0:
            ax.set_ylabel("Price (€/MWh)")
        ax.grid(True, alpha=0.25)

        # annotate weights only in the stochastic panel (or deterministic overall)
        if mode == "combined":
            ax.text(
                0.02, 0.95, scen_txt,
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none")
            )

    # Hide unused axes
    for j in range(n_panels, len(axes)):
        axes[j].axis("off")

    if not any_data:
        plt.close(fig)
        raise ValueError("No data to plot: all requested buses/links were missing or empty.")

    # one legend for whole figure
    fig.suptitle(title, y=0.995)

    # ---- one legend for whole figure (collect from ALL axes, not just axes[0])
    handles, labels = [], []
    seen = set()
    for ax in axes[:n_panels]:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in seen:
                seen.add(ll)
                handles.append(hh)
                labels.append(ll)

    leg = None
    if handles:
        leg = fig.legend(
            handles, labels,
            frameon=False,
            fontsize=9,
            ncol=min(4, len(labels)),
            loc="lower center",
        )

    # ---- reserve space for legend dynamically (prevents overlap)
    fig.canvas.draw()
    bottom = 0.06  # default bottom margin if no legend

    if leg is not None:
        bbox = leg.get_window_extent(fig.canvas.get_renderer())
        bbox_fig = bbox.transformed(fig.transFigure.inverted())
        bottom = bbox_fig.height + 0.03  # legend height + padding

    fig.tight_layout(rect=[0, bottom, 1, 0.97])

    out = folder / fname
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ----  duration curves for operation ----
def _as_selector_exact_list(selector):
    """
    Normalize selector into:
      - str, OR
      - list of str

    Accepts:
      - "name"
      - ["a","b"]
      - {"a","b"}  (set)
      - ("a","b")
    """
    if selector is None:
        return None
    if isinstance(selector, str):
        return selector
    if isinstance(selector, (list, tuple, set)):
        return list(selector)
    return selector  # leave other types as-is (won't match)


def _expand_exi(names):
    out = []
    for nm in names:
        out.append(nm)
        if isinstance(nm, str) and not nm.startswith("EXI_"):
            out.append("EXI_" + nm)
    seen, out2 = set(), []
    for nm in out:
        if nm not in seen:
            seen.add(nm)
            out2.append(nm)
    return out2


def _match_names_exact_exi(candidates, selector):
    """
    ONLY exact names + EXI_ expansion.
    selector:
      - str
      - list/tuple/set of str
    """
    selector = _as_selector_exact_list(selector)

    if selector is None:
        return []

    if isinstance(selector, list):
        wanted = _expand_exi(selector)
        return [w for w in wanted if w in candidates]

    if isinstance(selector, str):
        wanted = _expand_exi([selector])
        return [w for w in wanted if w in candidates]

    return []


def _ldc(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    return np.sort(s.values)[::-1]


def _scenario_list_from_tcols(df: pd.DataFrame):
    cols = df.columns
    if isinstance(cols, pd.MultiIndex) and "scenario" in cols.names:
        return list(cols.get_level_values("scenario").unique())
    return []


def _series_from_mi_cols(df: pd.DataFrame, scen, name):
    if df is None:
        return None
    if isinstance(df.columns, pd.MultiIndex) and {"scenario", "name"}.issubset(df.columns.names):
        if scen is None:
            scen = df.columns.get_level_values("scenario").unique()[0]
        key = (scen, name)
        return df[key] if key in df.columns else None
    return df[name] if name in df.columns else None


def _available_names_from_tcols(df: pd.DataFrame, scen):
    if df is None:
        return []
    if isinstance(df.columns, pd.MultiIndex) and {"scenario", "name"}.issubset(df.columns.names):
        if scen is None:
            scen = df.columns.get_level_values("scenario").unique()[0]
        try:
            return list(df.xs(scen, level="scenario", axis=1).columns)
        except Exception:
            return []
    return list(df.columns)


def _nominal_from_component_table(comp_df: pd.DataFrame, scen, name, preferred_cols):
    if comp_df is None or comp_df.empty:
        return None

    if isinstance(comp_df.index, pd.MultiIndex) and {"scenario", "name"}.issubset(comp_df.index.names):
        if scen is None:
            scen = comp_df.index.get_level_values("scenario").unique()[0]
        key = (scen, name)
        if key not in comp_df.index:
            return None
        row = comp_df.loc[key]
    else:
        if name not in comp_df.index:
            return None
        row = comp_df.loc[name]

    for c in preferred_cols:
        if c in row.index:
            try:
                val = float(row[c])
            except Exception:
                return None
            return val if np.isfinite(val) and val > 0 else None
    return None

def plot_utilization_ldc_by_scenario(
    n,
    items,
    outpath=None,
    title="Capacity Factor duration curves",
    ncols=3,
    figsize_per_panel=(5.3, 4.3),
    abs_links=True,
    clip_01=True,
    legend_ncol=5,
    add_stochastic=True,
    snapshot_weight_col="objective",
    scenario_weight_col="weight",
    stochastic_label="stochastic (scenario×snapshot weighted)",
    n_points_stochastic=1001,  # resolution for the weighted LDC
    carrier_colors=None,
):
    """
    items = [{"label","kind","field","selector"}, ...]
    Adds an optional final panel: stochastic weighted LDC (scenario_prob × snapshot weights).
    Requires helper functions already in the codebase:
      - _scenario_list_from_tcols
      - _available_names_from_tcols
      - _match_names_exact_exi
      - _nominal_from_component_table
      - _series_from_mi_cols
      - _ldc (simple unweighted LDC)
      - _weighted_ldc (weighted LDC returning xq,yq)
    """

    # ---- snapshot weights (for stochastic)
    snap_w = (
        n.snapshot_weightings[snapshot_weight_col]
        .reindex(n.snapshots)
        .fillna(0.0)
        .to_numpy()
    )

    # ---- scenario weights
    has_sw = (
        hasattr(n, "scenario_weightings")
        and n.scenario_weightings is not None
        and len(n.scenario_weightings) > 0
    )
    if has_sw:
        sw = n.scenario_weightings[scenario_weight_col].copy()
        sw.index = sw.index.astype(str)
        scenarios = list(sw.index)
        scen_prob = sw.astype(float).to_dict()
        scen_txt = "Scenario weights:\n" + "\n".join([f"{k}: {v:.2f}" for k, v in sw.items()])
    else:
        # deterministic fallback
        scenarios = [None]
        scen_prob = {None: 1.0}
        scen_txt = "Deterministic"

    # -------- Panels
    panels = list(scenarios)
    if add_stochastic and has_sw:
        panels = panels + ["__stochastic__"]

    n_panels = len(panels)
    ncols = min(ncols, n_panels)
    nrows = int(np.ceil(n_panels / ncols))

    fig_w = figsize_per_panel[0] * ncols
    fig_h = figsize_per_panel[1] * nrows
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(fig_w, fig_h), sharex=False, sharey=True
    )
    axes = np.atleast_1d(axes).ravel()

    cmap = mpl.colormaps["Dark2"]

    def _normalize(series, denom: float):
        if series is None or denom is None or denom <= 0:
            return None
        y = pd.to_numeric(pd.Series(series, copy=False), errors="coerce") / float(denom)
        if clip_01:
            y = y.clip(lower=0.0, upper=1.0)
        return y

    # Stable patterns for StorageUnit modes (same for all SUs)
    SU_MODE_SPECS = {
        "soc": dict(label="SOC", linestyle="-", lw=2.0),                 # solid
        "discharge": dict(label="Discharge", linestyle=(0, (6, 2)), lw=1.8),  # dashed
        "charge": dict(label="Charge", linestyle=(0, (1, 2)), lw=1.8),        # dotted
    }

    def _su_mode_series(n, scen, name, mode):
        """Return the raw time series for a StorageUnit mode."""
        if mode == "soc":
            ts = getattr(n.storage_units_t, "state_of_charge", None)
            return _series_from_mi_cols(ts, scen, name) if ts is not None else None

        # charge/discharge use signed power p
        ts = getattr(n.storage_units_t, "p", None)
        s = _series_from_mi_cols(ts, scen, name) if ts is not None else None
        if s is None:
            return None
        s = pd.Series(s, copy=False)

        if mode == "discharge":
            return s.clip(lower=0.0)
        if mode == "charge":
            return (-s.clip(upper=0.0))  # positive charging magnitude
        return None

    def _storageunit_denom(n, scen, name, field):
        """
        Denominator for StorageUnit normalization.

        For state_of_charge:
          denom = p_nom * max_hours   (energy capacity)

        For power-like fields (fallback):
          denom = p_nom
        """
        if field in ("state_of_charge", "soc"):
            p_nom = _nominal_from_component_table(
                n.storage_units, scen, name, ["p_nom_opt", "p_nom"]
            )
            mh = _nominal_from_component_table(
                n.storage_units, scen, name, ["max_hours"]
            )
            if p_nom is None or mh is None:
                return None
            e_nom = float(p_nom) * float(mh)
            return e_nom if np.isfinite(e_nom) and e_nom > 0 else None

        return _nominal_from_component_table(
            n.storage_units, scen, name, ["p_nom_opt", "p_nom"]
        )

    # -------- GLOBAL curve specs (stable colors across scenarios)
    curve_specs = []  # expanded {kind, field, base_label, name, legend_label, linestyle, lw, mode?}

    def _candidates_for(kind, field):
        if kind == "Generator":
            ts = getattr(n.generators_t, field, None)
        elif kind == "Link":
            ts = getattr(n.links_t, field, None)
        elif kind == "Store":
            ts = getattr(n.stores_t, field, None)
        elif kind == "StorageUnit":
            # field might be "p" or "state_of_charge"; candidates are the same names either way
            ts = getattr(n.storage_units_t, field, None)
        else:
            return []
        if ts is None:
            return []
        union = set()
        for sc in scenarios:
            union.update(_available_names_from_tcols(ts, sc))
        return sorted(union)

    # Build curve specs
    for it in items:
        kind = it["kind"]
        field = it.get("field", "p")
        base_label = it["label"]
        selector = it.get("selector")
        lw0 = it.get("lw", 1.8)

        candidates = _candidates_for(kind, field)
        matches = _match_names_exact_exi(candidates, selector)

        for name in matches:
            suffix = " (EXI)" if isinstance(name, str) and name.startswith("EXI_") else ""

            if kind == "StorageUnit":
                # 3 curves per SU: SOC, discharge, charge (mode decides actual series + denom)
                for mode, ms in SU_MODE_SPECS.items():
                    leg = f"{base_label}{suffix} — {ms['label']}"
                    curve_specs.append({
                        "kind": kind,
                        "field": field,  # retained, but mode decides actual source
                        "mode": mode,
                        "base_label": base_label,
                        "name": name,
                        "legend_label": leg,
                        "linestyle": ms["linestyle"],
                        "lw": it.get("lw", ms["lw"]),
                    })
            else:
                ls = ":" if kind == "Store" else "-"
                leg = f"{base_label}{suffix}"
                # signed bidirectional links: LDC shows |p0|/p_nom (utilisation magnitude)
                leg_ldc = f"{leg} |net|" if (kind == "Link" and it.get("signed", False)) else leg
                curve_specs.append({
                    "kind": kind,
                    "field": field,
                    "base_label": base_label,
                    "name": name,
                    "legend_label": leg_ldc,
                    "signed": it.get("signed", False),
                    "linestyle": ls,
                    "lw": lw0,
                })

    # Stable colors per (kind,name) across scenarios and modes
    # Look up carrier for each component and use fixed carrier_colors when available.
    _carrier_colors = carrier_colors or {}
    _comp_tables = {
        "Generator":   getattr(n, "generators",    None),
        "Link":        getattr(n, "links",          None),
        "StorageUnit": getattr(n, "storage_units",  None),
        "Store":       getattr(n, "stores",         None),
    }

    def _get_carrier(kind, name):
        df = _comp_tables.get(kind)
        if df is None or df.empty or name not in df.index:
            return None
        return df.at[name, "carrier"] if "carrier" in df.columns else None

    uniq_keys, seen = [], set()
    for spec in curve_specs:
        key = (spec["kind"], spec["name"])
        if key not in seen:
            seen.add(key)
            uniq_keys.append(key)

    color_map = {}
    for i, k in enumerate(uniq_keys):
        kind, name = k
        carrier = _get_carrier(kind, name)
        color_map[k] = _carrier_colors.get(carrier, cmap(i % cmap.N))

    # -------- Legend handles (global)
    legend_map = {}  # label -> handle (first occurrence) for non-SU + SU names (colors)
    mode_map = {}  # "SOC"/"Discharge"/"Charge" -> handle (linestyle key)

    # -------- Compute & plot per panel
    for i, panel in enumerate(panels):
        ax = axes[i]
        any_plotted = False
        is_stoch_panel = (panel == "__stochastic__")

        for spec in curve_specs:
            kind, field, name = spec["kind"], spec["field"], spec["name"]

            h = None  # handle for this curve if plotted

            # ----- per-scenario
            if not is_stoch_panel:
                scen = panel  # actual scenario or None

                if kind == "Generator":
                    ts = getattr(n.generators_t, field, None)
                    denom = _nominal_from_component_table(
                        n.generators, scen, name, ["p_nom_opt", "p_nom"]
                    )
                    s = _series_from_mi_cols(ts, scen, name) if ts is not None else None

                elif kind == "Link":
                    ts = getattr(n.links_t, field, None)
                    denom = _nominal_from_component_table(
                        n.links, scen, name, ["p_nom_opt", "p_nom"]
                    )
                    s = _series_from_mi_cols(ts, scen, name) if ts is not None else None
                    # LDC always shows utilisation magnitude (abs); signed flag only
                    # affects heatmap coloring, not the capacity-factor curve
                    if s is not None and (abs_links or spec.get("signed", False)):
                        s = pd.Series(s, copy=False).abs()

                elif kind == "Store":
                    ts = getattr(n.stores_t, field, None)
                    denom = _nominal_from_component_table(
                        n.stores, scen, name, ["e_nom_opt", "e_nom"]
                    )
                    s = _series_from_mi_cols(ts, scen, name) if ts is not None else None

                elif kind == "StorageUnit":
                    mode = spec.get("mode", None)
                    s = _su_mode_series(n, scen, name, mode)
                    if s is None:
                        continue
                    if mode == "soc":
                        denom = _storageunit_denom(n, scen, name, "state_of_charge")  # e_nom
                    else:
                        denom = _storageunit_denom(n, scen, name, "p")                # p_nom
                else:
                    continue

                s = _normalize(s, denom)
                y = _ldc(s) if s is not None else None
                if y is None:
                    continue

                x = np.linspace(0, 100, len(y))
                col = color_map[(kind, name)]
                h, = ax.plot(x, y, color=col, linestyle=spec["linestyle"], linewidth=spec["lw"])
                any_plotted = True

            # ----- stochastic panel: scenario×snapshot weighted LDC
            else:
                all_vals, all_wts = [], []

                for scen in scenarios:
                    prob = float(scen_prob.get(scen, 0.0))
                    if prob == 0.0:
                        continue

                    if kind == "Generator":
                        ts = getattr(n.generators_t, field, None)
                        denom = _nominal_from_component_table(
                            n.generators, scen, name, ["p_nom_opt", "p_nom"]
                        )
                        s = _series_from_mi_cols(ts, scen, name) if ts is not None else None

                    elif kind == "Link":
                        ts = getattr(n.links_t, field, None)
                        denom = _nominal_from_component_table(
                            n.links, scen, name, ["p_nom_opt", "p_nom"]
                        )
                        s = _series_from_mi_cols(ts, scen, name) if ts is not None else None
                        if s is not None and (abs_links or spec.get("signed", False)):
                            s = pd.Series(s, copy=False).abs()

                    elif kind == "Store":
                        ts = getattr(n.stores_t, field, None)
                        denom = _nominal_from_component_table(
                            n.stores, scen, name, ["e_nom_opt", "e_nom"]
                        )
                        s = _series_from_mi_cols(ts, scen, name) if ts is not None else None

                    elif kind == "StorageUnit":
                        mode = spec.get("mode", None)
                        s = _su_mode_series(n, scen, name, mode)
                        if s is None:
                            continue
                        if mode == "soc":
                            denom = _storageunit_denom(n, scen, name, "state_of_charge")
                        else:
                            denom = _storageunit_denom(n, scen, name, "p")
                    else:
                        continue

                    s = _normalize(s, denom)
                    if s is None:
                        continue

                    v = pd.to_numeric(pd.Series(s, copy=False), errors="coerce").to_numpy()
                    w = snap_w * prob

                    m = np.isfinite(v) & np.isfinite(w)
                    if m.any():
                        all_vals.append(v[m])
                        all_wts.append(w[m])

                if not all_vals:
                    continue

                values = np.concatenate(all_vals)
                weights = np.concatenate(all_wts)

                xq, yq = _weighted_ldc(values, weights, n_points=n_points_stochastic)
                if xq is None:
                    continue

                col = color_map[(kind, name)]
                h, = ax.plot(xq, yq, color=col, linestyle=spec["linestyle"], linewidth=spec["lw"])
                any_plotted = True

            # ---- record legend entries
            if h is not None:
                if kind == "StorageUnit":
                    # one entry per StorageUnit (color), not per mode
                    suffix = " (EXI)" if isinstance(name, str) and name.startswith("EXI_") else ""
                    su_label = f"{spec['base_label']}{suffix}"

                    # proxy handle for SU color (solid line)
                    legend_map.setdefault(
                        su_label,
                        Line2D([0], [0], color=col, linestyle="-", linewidth=2.0),
                    )

                    # mode key (black lines showing linestyle)
                    mode_label = SU_MODE_SPECS[spec["mode"]]["label"]
                    mode_map.setdefault(
                        mode_label,
                        Line2D([0], [0], color="black", linestyle=spec["linestyle"], linewidth=2.0),
                    )
                else:
                    legend_map.setdefault(spec["legend_label"], h)

        # ---- titles/labels
        if is_stoch_panel:
            ax.set_title(stochastic_label)
            ax.text(
                0.02, 0.95, scen_txt,
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none")
            )
        else:
            scen_label = n.snapshots[0].year if panel is None else str(panel)
            ax.set_title(f"Scenario: {scen_label}")

        ax.set_xlabel("Percent of hours (%)")
        if any_plotted:
            ax.set_ylabel("Utilization / capacity factor (-)")
        ax.set_ylim(0, 1.05)
        ax.axhline(1.0, color="gray", lw=0.8, ls="--", alpha=0.45, zorder=0)
        ax.grid(True, alpha=0.25)

    # Hide unused axes
    for j in range(n_panels, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, y=1.02)
    # ---- One global legend below the subplots (safe positioning)
    fig.suptitle(title, y=1.02)

    if legend_map or mode_map:
        handles = list(legend_map.values())
        labels = list(legend_map.keys())

        # append mode key at the end (still one legend total)
        if mode_map:
            handles += list(mode_map.values())
            labels += [f"Mode: {k}" for k in mode_map.keys()]

        leg = fig.legend(
            handles, labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=min(legend_ncol, max(1, len(labels))),
            frameon=False,
            fontsize=9,
            handlelength=2.4,
            columnspacing=1.2,
        )

        # dynamic safe spacing (no overlap, no guessing)
        fig.canvas.draw()
        bbox = leg.get_window_extent(fig.canvas.get_renderer())
        bbox_fig = bbox.transformed(fig.transFigure.inverted())
        bottom = bbox_fig.height + 0.03

        fig.tight_layout(rect=[0, bottom, 1, 1])
    else:
        fig.tight_layout()

    if outpath:
        fig.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

# --- HEAT MAPS -----
def _expand_exi(names):
    """
    Given a list of exact names, also include EXI_<name> variants.
    If a name already starts with EXI_, keep it as-is.
    """
    out = []
    for nm in names:
        out.append(nm)
        if isinstance(nm, str) and not nm.startswith("EXI_"):
            out.append("EXI_" + nm)
    # preserve order, remove duplicates
    seen = set()
    out2 = []
    for nm in out:
        if nm not in seen:
            seen.add(nm)
            out2.append(nm)
    return out2

def _match_names(candidates, selector, auto_exi=True):
    """
    selector can be:
      - str               -> exact match (and EXI_ variant if auto_exi=True)
      - list/tuple/set    -> exact allow-list (and EXI_ variants if auto_exi=True)
      - {"contains": "..."}
      - {"regex": "..."}
      - callable(name)->bool
    """
    if selector is None:
        return []

    # exact allow-list
    if isinstance(selector, (list, tuple, set)):
        wanted = list(selector)
        if auto_exi:
            wanted = _expand_exi(wanted)
        return [w for w in wanted if w in candidates]

    # exact string
    if isinstance(selector, str):
        wanted = [selector]
        if auto_exi:
            wanted = _expand_exi(wanted)
        return [w for w in wanted if w in candidates]

    # pattern dict (no EXI auto-expansion here)
    if isinstance(selector, dict):
        if "contains" in selector:
            token = selector["contains"]
            return [c for c in candidates if token in c]
        if "regex" in selector:
            pat = re.compile(selector["regex"])
            return [c for c in candidates if pat.search(c)]

    # callable
    if callable(selector):
        out = []
        for c in candidates:
            try:
                if selector(c):
                    out.append(c)
            except Exception:
                pass
        return out

    return []

# ----------------------------
# PyPSA scenario slicing
# ----------------------------
def _scenarios_from_dfcols(df):
    if isinstance(df, pd.DataFrame) and isinstance(df.columns, pd.MultiIndex) and "scenario" in df.columns.names:
        return list(df.columns.get_level_values("scenario").unique())
    return []


def _series_from_mi_cols(df, scen, name):
    """df columns are either MultiIndex ('scenario','name') or flat."""
    if df is None:
        return None
    if isinstance(df.columns, pd.MultiIndex) and {"scenario", "name"}.issubset(df.columns.names):
        if scen is None:
            scen = df.columns.get_level_values("scenario").unique()[0]
        key = (scen, name)
        return df[key] if key in df.columns else None
    # deterministic
    return df[name] if name in df.columns else None


def _available_names_from_tcols(df, scen):
    """Available 'name' values for a given scenario from df columns."""
    if df is None:
        return []
    if isinstance(df.columns, pd.MultiIndex) and {"scenario", "name"}.issubset(df.columns.names):
        if scen is None:
            scen = df.columns.get_level_values("scenario").unique()[0]
        try:
            return list(df.xs(scen, level="scenario", axis=1).columns)
        except Exception:
            return []
    return list(df.columns)


def _cap_from_component_table(comp_df, scen, name, preferred_cols):
    """
    comp_df index may be MultiIndex ('scenario','name') (stochastic case) or flat.
    Returns float cap or None.
    """
    if comp_df is None or comp_df.empty:
        return None

    if isinstance(comp_df.index, pd.MultiIndex) and {"scenario", "name"}.issubset(comp_df.index.names):
        if scen is None:
            scen = comp_df.index.get_level_values("scenario").unique()[0]
        key = (scen, name)
        if key not in comp_df.index:
            return None
        row = comp_df.loc[key]
    else:
        if name not in comp_df.index:
            return None
        row = comp_df.loc[name]

    for c in preferred_cols:
        if c in row.index:
            try:
                v = float(row[c])
            except Exception:
                return None
            return v if np.isfinite(v) and v > 0 else None

    return None


# ----------------------------
# Heatmap utility - CF
def _upsample_to_hourly(s):
    """Forward-fill a sub-hourly or coarser series to 1-hour resolution for heatmap display."""
    if len(s) < 2:
        return s
    freq_ns = (s.index[1] - s.index[0]).value
    if freq_ns != 3_600_000_000_000:  # not already 1h
        s = s.resample("1h").ffill()
    return s


def heatmap_day_hour(series, ax, vmin=0, vmax=1, title="", cmap="viridis", show_months=True):
    """
    series: pd.Series with DatetimeIndex (any resolution)
    Creates a heatmap with y=hour(0..23), x=day-of-year.
    Coarser-than-hourly data is upsampled to 1 h by forward-fill for display.
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        ax.set_title(title + " (missing)", fontsize=9)
        ax.axis("off")
        return None

    s = s[~s.index.duplicated(keep="first")]
    s = _upsample_to_hourly(s)
    df = pd.DataFrame({"val": s.values}, index=pd.DatetimeIndex(s.index))
    df["doy"] = df.index.dayofyear
    df["hour"] = df.index.hour

    mat = df.pivot_table(index="hour", columns="doy", values="val", aggfunc="mean")
    mat = mat.reindex(index=range(24))

    im = ax.imshow(mat.values, aspect="auto", origin="lower", vmin=vmin, vmax=vmax, cmap=cmap)

    ax.set_yticks([0, 6, 12, 18, 23])
    ax.set_yticklabels(["0", "6", "12", "18", "23"])

    year = df.index[0].year
    month_starts = [pd.Timestamp(year, m, 1).dayofyear for m in range(1, 13)]
    ax.set_xticks([d - 1 for d in month_starts])

    if show_months:
        month_labels = [calendar.month_abbr[m] for m in range(1, 13)]
        ax.set_xticklabels(month_labels, rotation=25, ha="right", rotation_mode="anchor")
        ax.tick_params(axis="x", pad=2)
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", which="both", length=0)

    ax.set_title(title, fontsize=9)
    ax.set_xlabel("")
    ax.set_ylabel("Hour")
    return im

# ----------------------------
# Main: compare scenarios in one network - CF
def figure_heatmaps_compare_scenarios(
    n,
    items,
    outpath=None,
    title="Operational heatmaps by scenario (normalized 0–1)",
    cmap="viridis",
    abs_links=True,
    vmin=0,
    vmax=1,
    snapshot_weight_col="objective",
    scenario_weight_col="weight",
    add_stochastic_column=True,
    stochastic_col_label="stochastic",
):
    """
    Adds (optional) final column: stochastic (scenario×snapshot weighted).

    Stochastic column uses weighted averaging across scenarios:
      weights = snapshot_weight * scenario_probability
    """

    # Map kind -> (static table, time series dataframe getter, cap columns)
    kind_map = {
        "Generator":   (n.generators,    getattr(n.generators_t, "p", None),               ["p_nom_opt", "p_nom"]),
        "Link":        (n.links,         getattr(n.links_t, "p0", None),                  ["p_nom_opt", "p_nom"]),
        "Store":       (n.stores,        getattr(n.stores_t, "e", None),                  ["e_nom_opt", "e_nom"]),
        "StorageUnit": (n.storage_units, getattr(n.storage_units_t, "state_of_charge", None), ["p_nom_opt", "p_nom", "max_hours"]),
    }

    # Detect scenarios from any *_t table that has scenario columns
    scenarios = []
    for _, ts_df, _ in kind_map.values():
        scenarios = _scenarios_from_dfcols(ts_df)
        if scenarios:
            break
    if not scenarios:
        scenarios = [None]  # deterministic

    stochastic = (scenarios != [None])

    # Snapshot weights
    snap_w = (
        n.snapshot_weightings[snapshot_weight_col]
        .reindex(n.snapshots)
        .fillna(0.0)
    )

    # Scenario weights (probabilities)
    if stochastic and hasattr(n, "scenario_weightings") and n.scenario_weightings is not None:
        sw = n.scenario_weightings[scenario_weight_col].copy()
        sw.index = sw.index.astype(str)
        scen_prob = sw.astype(float).to_dict()
    else:
        scen_prob = {None: 1.0}

    # Build expanded rows (stable across scenarios)
    expanded = []
    for it in items:
        kind = it["kind"]
        field = it.get("field", None)
        selector = it.get("selector")
        label = it.get("label", kind)

        comp_df, ts_df, _ = kind_map[kind]
        if ts_df is None:
            continue

        cand_union = set()
        for scen in scenarios:
            cand_union.update(_available_names_from_tcols(ts_df, scen))
            # StorageUnit: state_of_charge may be absent/empty after some solvers;
            # also scan the dispatch (p) timeseries so candidates are never missed.
            if kind == "StorageUnit":
                _p_ts = getattr(n.storage_units_t, "p", None)
                if _p_ts is not None:
                    cand_union.update(_available_names_from_tcols(_p_ts, scen))
        cand_union = sorted(cand_union)

        matches = _match_names(cand_union, selector)
        for name in matches:
            expanded.append({
                "row_label": f"{label}\n({name})",
                "kind": kind,
                "field": field,
                "name": name,
                "signed": it.get("signed", False),  # bidirectional links: signed net flow
            })

    if not expanded:
        raise ValueError("No matching components found for the provided items/selectors.")

    # Function to get normalized series for a given (scenario, expanded row)
    def _get_norm_series(scen, row):
        kind = row["kind"]
        name = row["name"]
        field = row["field"]
        is_signed = row.get("signed", False)

        comp_df, _, _ = kind_map[kind]

        # pick time-series df by kind
        if kind == "Generator":
            ts_df = getattr(n.generators_t, field, None) if field else getattr(n.generators_t, "p", None)
            s = _series_from_mi_cols(ts_df, scen, name) if ts_df is not None else None
            cap = _cap_from_component_table(comp_df, scen, name, ["p_nom_opt", "p_nom"])

        elif kind == "Link":
            ts_df = getattr(n.links_t, field, None) if field else getattr(n.links_t, "p0", None)
            s = _series_from_mi_cols(ts_df, scen, name) if ts_df is not None else None
            cap = _cap_from_component_table(comp_df, scen, name, ["p_nom_opt", "p_nom"])
            # signed=True: keep sign (bidirectional net flow); otherwise take abs
            if s is not None and abs_links and not is_signed:
                s = pd.Series(s, copy=False).abs()

        elif kind == "Store":
            ts_df = getattr(n.stores_t, field, None) if field else getattr(n.stores_t, "e", None)
            s = _series_from_mi_cols(ts_df, scen, name) if ts_df is not None else None
            cap = _cap_from_component_table(comp_df, scen, name, ["e_nom_opt", "e_nom"])

        elif kind == "StorageUnit":
            if field in ("p", "dispatch"):
                ts_df = getattr(n.storage_units_t, "p", None)
                s = _series_from_mi_cols(ts_df, scen, name) if ts_df is not None else None
                s = pd.Series(s, copy=False).abs() if s is not None else None
                cap = _cap_from_component_table(comp_df, scen, name, ["p_nom_opt", "p_nom"])
                if cap is not None:
                    cap = float(cap)
            else:
                # state_of_charge (default)
                ts_df = getattr(n.storage_units_t, "state_of_charge", None) if not field else getattr(n.storage_units_t, field, None)
                s = _series_from_mi_cols(ts_df, scen, name) if ts_df is not None else None
                p_nom = _cap_from_component_table(comp_df, scen, name, ["p_nom_opt", "p_nom"])
                mh = _cap_from_component_table(comp_df, scen, name, ["max_hours"])
                cap = (float(p_nom) * float(mh)) if (p_nom is not None and mh is not None) else None

        else:
            return None

        if s is None:
            return None
        if cap is None or cap <= 0:
            return None

        s = pd.Series(s, copy=False)
        if is_signed:
            # Signed net flow: normalise to [-1, 1] then shift to [0, 1] so the
            # standard viridis-range colorbar still works; use RdBu_r cmap per-row.
            # 0 = max discharge, 0.5 = idle, 1 = max charge.
            cf = ((pd.to_numeric(s, errors="coerce") / cap) + 1.0) / 2.0
            cf = cf.clip(lower=0.0, upper=1.0)
        else:
            cf = (pd.to_numeric(s, errors="coerce") / cap).clip(lower=0.0, upper=1.0)
        return cf

    # compute stochastic "expected pattern" series as weighted day×hour matrix
    def _heatmap_day_hour_weighted(values, weights, ax, vmin=0, vmax=1, title="", cmap="viridis", show_months=True):
        """
        values, weights: pd.Series with DatetimeIndex aligned (already upsampled per-scenario).
        Produces day×hour heatmap using weighted mean per (doy,hour).
        """
        v = pd.to_numeric(values, errors="coerce")
        w = pd.to_numeric(weights, errors="coerce")
        v = v.dropna()
        w = w.dropna()
        m = np.isfinite(v) & np.isfinite(w) & (w > 0)
        v = v[m]
        w = w[m]

        if v.empty:
            ax.set_title(title + " (missing)", fontsize=9)
            ax.axis("off")
            return None

        idx = pd.DatetimeIndex(v.index)
        df = pd.DataFrame({"v": v.values, "w": w.values}, index=idx)
        df["doy"] = df.index.dayofyear
        df["hour"] = df.index.hour

        # weighted mean per (hour, doy): mean = sum(v*w)/sum(w)
        df["vw"] = df["v"] * df["w"]
        grp = df.groupby(["hour", "doy"], sort=False, observed=True)

        num = grp["vw"].sum()
        den = grp["w"].sum()
        mean = (num / den).rename("val").reset_index()

        mat = mean.pivot(index="hour", columns="doy", values="val")
        mat = mat.reindex(index=range(24))

        im = ax.imshow(mat.values, aspect="auto", origin="lower", vmin=vmin, vmax=vmax, cmap=cmap)

        ax.set_yticks([0, 6, 12, 18, 23])
        ax.set_yticklabels(["0", "6", "12", "18", "23"])

        year = idx[0].year
        month_starts = [pd.Timestamp(year, m, 1).dayofyear for m in range(1, 13)]
        ax.set_xticks([d - 1 for d in month_starts])

        if show_months:
            month_labels = [calendar.month_abbr[m] for m in range(1, 13)]
            ax.set_xticklabels(month_labels, rotation=25, ha="right", rotation_mode="anchor")
            ax.tick_params(axis="x", pad=2)
        else:
            ax.set_xticklabels([])
            ax.tick_params(axis="x", which="both", length=0)

        ax.set_title(title, fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("Hour")
        return im

    ims = []

    if stochastic:
        # Add stochastic column
        plot_cols = list(scenarios)
        if add_stochastic_column:
            plot_cols = plot_cols + ["__stochastic__"]

        n_rows = len(expanded)
        n_cols = len(plot_cols)

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(3.6 * n_cols, 2.0 * n_rows),
            sharey=True,
            constrained_layout=True
        )
        axes = np.atleast_2d(axes)

        for r, row in enumerate(expanded):
            for c, col in enumerate(plot_cols):
                ax = axes[r, c]
                show_months = (r == n_rows - 1)

                if col != "__stochastic__":
                    scen = col
                    s = _get_norm_series(scen, row)

                    scen_lab = "deterministic" if scen is None else str(scen)
                    col_title = scen_lab if r == 0 else ""
                    # per-row colormap: RdBu_r (0=discharge, 0.5=idle, 1=charge) for signed
                    row_cmap = "RdBu_r" if row.get("signed") else cmap
                    im = heatmap_day_hour(
                        s if s is not None else pd.Series(dtype=float),
                        ax=ax,
                        vmin=vmin, vmax=vmax,
                        title=col_title,
                        cmap=row_cmap,
                        show_months=show_months
                    )
                else:
                    # Build weighted expected pattern across scenarios
                    all_v = []
                    all_w = []
                    for scen in scenarios:
                        prob = float(scen_prob.get(str(scen), scen_prob.get(scen, 0.0)))
                        if prob == 0.0:
                            continue
                        s = _get_norm_series(scen, row)
                        if s is None or s.empty:
                            continue
                        # Align snapshot weights to this series' index (should match n.snapshots)
                        w = snap_w.reindex(s.index).fillna(0.0) * prob
                        # Upsample per-scenario before concat; concat would produce duplicate
                        # timestamps which break resample() inside the heatmap helper.
                        s = _upsample_to_hourly(s)
                        w = _upsample_to_hourly(w)
                        all_v.append(s)
                        all_w.append(w)

                    if all_v:
                        v_cat = pd.concat(all_v, axis=0)
                        w_cat = pd.concat(all_w, axis=0)
                    else:
                        v_cat = pd.Series(dtype=float)
                        w_cat = pd.Series(dtype=float)

                    col_title = stochastic_col_label if r == 0 else ""
                    row_cmap = "RdBu_r" if row.get("signed") else cmap
                    im = _heatmap_day_hour_weighted(
                        v_cat, w_cat,
                        ax=ax,
                        vmin=vmin, vmax=vmax,
                        title=col_title,
                        cmap=row_cmap,
                        show_months=show_months
                    )

                if im is not None:
                    ims.append(im)

                if c == 0:
                    ax.set_ylabel(f"{row['row_label']}\nHour")

        if ims:
            cbar = fig.colorbar(ims[0], ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
            cbar.set_label("Normalized (0–1)")

        fig.suptitle(title, y=1.02)

    else:
        # Deterministic: unchanged
        n_plots = len(expanded)
        n_cols = int(np.ceil(np.sqrt(n_plots)))
        n_rows = int(np.ceil(n_plots / n_cols))

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(3.6 * n_cols, 2.2 * n_rows),
            sharey=True,
            constrained_layout=True
        )
        axes = np.atleast_1d(axes).ravel()

        for i, row in enumerate(expanded):
            ax = axes[i]
            s = _get_norm_series(None, row)
            show_months = (i // n_cols == n_rows - 1)
            row_cmap = "RdBu_r" if row.get("signed") else cmap
            im = heatmap_day_hour(
                s if s is not None else pd.Series(dtype=float),
                ax=ax,
                vmin=vmin, vmax=vmax,
                title=row["row_label"],
                cmap=row_cmap,
                show_months=show_months
            )
            if im is not None:
                ims.append(im)

        for j in range(n_plots, len(axes)):
            axes[j].set_visible(False)

        if ims:
            cbar = fig.colorbar(ims[0], ax=[a for a in axes if a.get_visible()], fraction=0.02, pad=0.02)
            cbar.set_label("Normalized (0–1)")

        fig.suptitle(title, y=1.02)

    if outpath:
        outpath = Path(outpath)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

# ----------------------------
# Heatmap utility - values
def heatmap_day_hour_actual(series, ax, norm, title="", cmap="viridis", show_months=True):
    """
    series: pd.Series with DatetimeIndex (actual values, any resolution)
    norm: matplotlib Normalize/TwoSlopeNorm defining color scaling (capacity-based)
    Coarser-than-hourly data is upsampled to 1 h by forward-fill for display.
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        ax.set_title(title + " (missing)", fontsize=9)
        ax.axis("off")
        return None

    s = s[~s.index.duplicated(keep="first")]
    s = _upsample_to_hourly(s)
    df = pd.DataFrame({"val": s.values}, index=pd.DatetimeIndex(s.index))
    df["doy"] = df.index.dayofyear
    df["hour"] = df.index.hour

    mat = df.pivot_table(index="hour", columns="doy", values="val", aggfunc="mean")
    mat = mat.reindex(index=range(24))

    im = ax.imshow(mat.values, aspect="auto", origin="lower", cmap=cmap, norm=norm)

    ax.set_yticks([0, 6, 12, 18, 23])
    ax.set_yticklabels(["0", "6", "12", "18", "23"])

    year = df.index[0].year
    month_starts = [pd.Timestamp(year, m, 1).dayofyear for m in range(1, 13)]
    ax.set_xticks([d - 1 for d in month_starts])

    if show_months:
        month_labels = [calendar.month_abbr[m] for m in range(1, 13)]
        ax.set_xticklabels(month_labels, rotation=25, ha="right", rotation_mode="anchor")
        ax.tick_params(axis="x", pad=2)
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", which="both", length=0)

    ax.set_title(title, fontsize=9)
    ax.set_xlabel("")
    ax.set_ylabel("Hour")
    return im

def _slice_first_scenario_index(df: pd.DataFrame, scenario_level: str = "scenario"):
    if df is None or df.empty:
        return df
    if not isinstance(df.index, pd.MultiIndex):
        return df
    names = list(df.index.names)
    sc_level = scenario_level if scenario_level in names else (names[0] if names else None)
    if sc_level is None or sc_level not in df.index.names:
        return df
    sc_vals = pd.Index(df.index.get_level_values(sc_level)).drop_duplicates()
    if len(sc_vals) == 0:
        return df
    try:
        return df.xs(sc_vals[0], level=sc_level)
    except Exception:
        return df

def _get_bus_unit(n, bus_name):
    """Return the unit string from n.buses['unit'] for bus_name (handles scenario-indexed buses)."""
    buses = getattr(n, "buses", None)
    if buses is None or buses.empty:
        return None

    buses_static = _slice_first_scenario_index(buses)

    if "unit" not in buses_static.columns:
        return None
    if bus_name not in buses_static.index:
        return None

    u = buses_static.at[bus_name, "unit"]
    if u is None or (isinstance(u, float) and np.isnan(u)):
        return None
    u = str(u).strip()
    return u if u else None

def _normalize_unit(u: str | None):
    if u is None:
        return None
    s = str(u).strip()
    if not s:
        return None
    s_low = s.lower().replace(" ", "")
    if s_low == "mw":
        return "MW"
    if s_low == "mwh":
        return "MWh"
    if s_low in {"t/h", "tph", "tperh"}:
        return "t/h"
    if s_low in {"t", "ton", "tonne", "tonnes"}:
        return "t"
    return s

def _power_to_energy_unit(u: str | None):
    """Convert MW->MWh and t/h->t (otherwise leave unchanged)."""
    u = _normalize_unit(u)
    if u == "MW":
        return "MWh"
    if u == "t/h":
        return "t"
    return u

def _row_unit_from_bus(n, kind: str, comp_df: pd.DataFrame, scen, name: str, quantity: str, field: str | None):
    """
    kind: Generator/Link/Store/StorageUnit
    quantity: "Power" or "Energy"
    field: for StorageUnit: "p" or "state_of_charge"
    """
    # slice static comp table if it's scenario-indexed
    comp_static = comp_df
    if isinstance(comp_df.index, pd.MultiIndex):
        comp_static = _slice_first_scenario_index(comp_df)

    bus = None
    if comp_static is not None and not comp_static.empty and name in comp_static.index:
        if kind == "Generator":
            bus = comp_static.at[name, "bus"] if "bus" in comp_static.columns else None
        elif kind == "Link":
            # p0 uses bus0
            bus = comp_static.at[name, "bus0"] if "bus0" in comp_static.columns else None
        elif kind in {"Store", "StorageUnit"}:
            bus = comp_static.at[name, "bus"] if "bus" in comp_static.columns else None

    u = _get_bus_unit(n, bus) if bus is not None else None
    u = _normalize_unit(u)

    # Energy variables: stores.e, storage_units.state_of_charge
    if quantity == "Energy":
        return _power_to_energy_unit(u)

    # Power variables (including storage_units.p)
    return u
# ----------------------------
# Main: compare scenarios in one network - CF
def figure_heatmaps_compare_scenarios_actual(
    n,
    items,
    outpath=None,
    title="Operational heatmaps by scenario (actual values; capacity-normalized colors)",
    cmap_pos="viridis",              # sequential for >=0 series
    cmap_div="coolwarm",             # diverging for signed dispatch
    abs_links=True,
    snapshot_weight_col="objective",
    scenario_weight_col="weight",
    add_stochastic_column=True,
    stochastic_col_label="stochastic",
):
    """
    Same scenario logic as figure_heatmaps_compare_scenarios, but:
      - plots actual values (MW/MWh/...)
      - colors are normalized by each component's capacity (for visual comparability)
      - StorageUnit supports:
          * SOC via storage_units_t.state_of_charge (0..Emax)
          * Dispatch via storage_units_t.p (signed, -Pmax..Pmax)
    """

    # ---- Map kind -> (static table, default ts, cap columns)
    # Note: for StorageUnit we will decide based on field: "state_of_charge" vs "p"
    kind_map = {
        "Generator":   (n.generators,    getattr(n.generators_t, "p", None),                 ["p_nom_opt", "p_nom"]),
        "Link":        (n.links,         getattr(n.links_t, "p0", None),                    ["p_nom_opt", "p_nom"]),
        "Store":       (n.stores,        getattr(n.stores_t, "e", None),                    ["e_nom_opt", "e_nom"]),
        "StorageUnit": (n.storage_units, getattr(n.storage_units_t, "state_of_charge", None), ["p_nom_opt", "p_nom", "max_hours"]),
    }

    # ---- Detect scenarios from any *_t table that has scenario columns
    scenarios = []
    for _, ts_df, _ in kind_map.values():
        scenarios = _scenarios_from_dfcols(ts_df)
        if scenarios:
            break
    if not scenarios:
        scenarios = [None]  # deterministic

    stochastic = (scenarios != [None])

    # ---- Snapshot weights
    snap_w = (
        n.snapshot_weightings[snapshot_weight_col]
        .reindex(n.snapshots)
        .fillna(0.0)
    )

    # ---- Scenario weights (probabilities)
    if stochastic and hasattr(n, "scenario_weightings") and n.scenario_weightings is not None:
        sw = n.scenario_weightings[scenario_weight_col].copy()
        sw.index = sw.index.astype(str)
        scen_prob = sw.astype(float).to_dict()
    else:
        scen_prob = {None: 1.0}

    # ---- Build expanded rows (stable across scenarios) (same structure)
    expanded = []
    for it in items:
        kind = it["kind"]
        field = it.get("field", None)   # important for StorageUnit: "state_of_charge" or "p"
        selector = it.get("selector")
        label = it.get("label", kind)

        comp_df, ts_df_default, _ = kind_map[kind]
        if ts_df_default is None:
            continue

        # candidate names from ts table columns across scenarios
        cand_union = set()
        for scen in scenarios:
            cand_union.update(_available_names_from_tcols(ts_df_default, scen))
            # StorageUnit: state_of_charge may be absent/empty after some solvers;
            # also scan the dispatch (p) timeseries so candidates are never missed.
            if kind == "StorageUnit":
                _p_ts = getattr(n.storage_units_t, "p", None)
                if _p_ts is not None:
                    cand_union.update(_available_names_from_tcols(_p_ts, scen))
        cand_union = sorted(cand_union)

        th = it.get("th", 0)
        matches = _match_names(cand_union, selector)
        for name in matches:
            expanded.append({
                "row_label": f"{label}\n({name})",
                "kind": kind,
                "field": field,
                "name": name,
                "th": th,
                "signed": it.get("signed", False),  # bidirectional net flow
            })

    if not expanded:
        raise ValueError("No matching components found for the provided items/selectors.")

    # ---- Pre-filter: drop rows whose capacity is below threshold in every scenario
    # This prevents empty panels appearing in the figure for near-zero components.
    def _cap_passes_threshold(row):
        comp_df, _, cap_cols = kind_map[row["kind"]]
        for scen in scenarios:
            cap = _cap_from_component_table(comp_df, scen, row["name"], cap_cols)
            if cap is not None and float(cap) >= row["th"]:
                return True
        return False

    expanded = [row for row in expanded if _cap_passes_threshold(row)]

    if not expanded:
        print("Warning: all components filtered out by capacity threshold — nothing to plot.")
        return

    # ---- Get actual series + capacity-based norm (per row)
    def _get_series_and_norm(scen, row):

        NONE5 = (None, None, None, None, None)

        kind = row["kind"]
        name = row["name"]
        field = row["field"]

        comp_df, _, _ = kind_map[kind]

        if kind == "Generator":
            ts_df = getattr(n.generators_t, field, None) if field else getattr(n.generators_t, "p", None)
            s = _series_from_mi_cols(ts_df, scen, name) if ts_df is not None else None
            cap = _cap_from_component_table(comp_df, scen, name, ["p_nom_opt", "p_nom"])
            if s is None or cap is None or float(cap) < row["th"]:
                return NONE5
            cap = float(cap)
            norm = Normalize(vmin=0.0, vmax=cap)
            return pd.Series(s, copy=False), norm, cmap_pos, "Power", cap

        if kind == "Link":
            ts_df = getattr(n.links_t, field, None) if field else getattr(n.links_t, "p0", None)
            s = _series_from_mi_cols(ts_df, scen, name) if ts_df is not None else None
            cap = _cap_from_component_table(comp_df, scen, name, ["p_nom_opt", "p_nom"])
            if s is None or cap is None or float(cap) < row["th"]:
                return NONE5
            s = pd.Series(s, copy=False)
            cap = float(cap)
            if row.get("signed", False):
                # bidirectional net flow: keep sign, use diverging colormap
                norm = TwoSlopeNorm(vmin=-cap, vcenter=0.0, vmax=cap)
                return s, norm, cmap_div, "Power (net)", cap
            if abs_links:
                s = s.abs()
            norm = Normalize(vmin=0.0, vmax=cap)
            return s, norm, cmap_pos, "Power", cap

        if kind == "Store":
            ts_df = getattr(n.stores_t, field, None) if field else getattr(n.stores_t, "e", None)
            s = _series_from_mi_cols(ts_df, scen, name) if ts_df is not None else None
            cap = _cap_from_component_table(comp_df, scen, name, ["e_nom_opt", "e_nom"])
            if s is None or cap is None or float(cap) < row["th"]:
                return NONE5
            cap = float(cap)
            norm = Normalize(vmin=0.0, vmax=cap)
            return pd.Series(s, copy=False), norm, cmap_pos, "Energy", cap

        if kind == "StorageUnit":
            # SOC
            if field in (None, "state_of_charge", "soc", "SoC"):
                ts_df = getattr(n.storage_units_t, "state_of_charge", None)
                s = _series_from_mi_cols(ts_df, scen, name) if ts_df is not None else None
                p_nom = _cap_from_component_table(comp_df, scen, name, ["p_nom_opt", "p_nom"])
                mh = _cap_from_component_table(comp_df, scen, name, ["max_hours"])
                cap = (float(p_nom) * float(mh)) if (p_nom is not None and mh is not None) else None
                if s is None or cap is None or float(cap) < row["th"]:
                    return NONE5
                cap = float(cap)
                norm = Normalize(vmin=0.0, vmax=cap)
                return pd.Series(s, copy=False), norm, cmap_pos, "Energy", cap

            # Dispatch (signed)
            if field in ("p", "dispatch"):
                ts_df = getattr(n.storage_units_t, "p", None)
                s = _series_from_mi_cols(ts_df, scen, name) if ts_df is not None else None
                cap = _cap_from_component_table(comp_df, scen, name, ["p_nom_opt", "p_nom"])
                if s is None or cap is None or float(cap) < row["th"]:
                    return NONE5
                cap = float(cap)
                norm = TwoSlopeNorm(vmin=-cap, vcenter=0.0, vmax=cap)
                return pd.Series(s, copy=False), norm, cmap_div, "Power", cap

            return NONE5

        return NONE5

    # ---- weighted expected pattern (same logic, but uses actual values)
    def _heatmap_day_hour_weighted_actual(values, weights, ax, norm, title="", cmap="viridis", show_months=True):
        v = pd.to_numeric(values, errors="coerce")
        w = pd.to_numeric(weights, errors="coerce")
        v = v.dropna()
        w = w.dropna()
        m = np.isfinite(v) & np.isfinite(w) & (w > 0)
        v = v[m]
        w = w[m]

        if v.empty:
            ax.set_title(title + " (missing)", fontsize=9)
            ax.axis("off")
            return None

        idx = pd.DatetimeIndex(v.index)
        df = pd.DataFrame({"v": v.values, "w": w.values}, index=idx)
        df["doy"] = df.index.dayofyear
        df["hour"] = df.index.hour

        df["vw"] = df["v"] * df["w"]
        grp = df.groupby(["hour", "doy"], sort=False, observed=True)

        num = grp["vw"].sum()
        den = grp["w"].sum()
        mean = (num / den).rename("val").reset_index()

        mat = mean.pivot(index="hour", columns="doy", values="val")
        mat = mat.reindex(index=range(24))

        im = ax.imshow(mat.values, aspect="auto", origin="lower", cmap=cmap, norm=norm)

        ax.set_yticks([0, 6, 12, 18, 23])
        ax.set_yticklabels(["0", "6", "12", "18", "23"])

        year = idx[0].year
        month_starts = [pd.Timestamp(year, m, 1).dayofyear for m in range(1, 13)]
        ax.set_xticks([d - 1 for d in month_starts])

        if show_months:
            month_labels = [calendar.month_abbr[m] for m in range(1, 13)]
            ax.set_xticklabels(month_labels, rotation=25, ha="right", rotation_mode="anchor")
            ax.tick_params(axis="x", pad=2)
        else:
            ax.set_xticklabels([])
            ax.tick_params(axis="x", which="both", length=0)

        ax.set_title(title, fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("Hour")
        return im

    if stochastic:
        plot_cols = list(scenarios)
        if add_stochastic_column:
            plot_cols = plot_cols + ["__stochastic__"]

        n_rows = len(expanded)
        n_cols = len(plot_cols)

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(3.6 * n_cols, 2.0 * n_rows),
            sharey=True,
            constrained_layout=True
        )
        axes = np.atleast_2d(axes)

        for r, row in enumerate(expanded):
            # Compute row-specific norm once (based on first available scenario)
            row_norm = None
            row_cmap = cmap_pos
            row_quantity = None
            row_cap = None
            row_field = row.get("field", None)

            for scen in scenarios:
                s0, norm0, cmap0, qty0, cap0 = _get_series_and_norm(scen, row)
                if norm0 is not None:
                    row_norm = norm0
                    row_cmap = cmap0
                    row_quantity = qty0
                    row_cap = cap0
                    break

            for c, col in enumerate(plot_cols):
                ax = axes[r, c]
                show_months = (r == n_rows - 1)

                if col != "__stochastic__":
                    scen = col
                    s, norm, cmap_use, _, _ = _get_series_and_norm(scen, row)

                    # enforce consistent scaling across scenarios in a row
                    norm_use = row_norm if row_norm is not None else norm

                    scen_lab = "deterministic" if scen is None else str(scen)
                    col_title = scen_lab if r == 0 else ""

                    im = heatmap_day_hour_actual(
                        s if s is not None else pd.Series(dtype=float),
                        ax=ax,
                        norm=norm_use if norm_use is not None else Normalize(0, 1),
                        title=col_title,
                        cmap=row_cmap if row_norm is not None else (cmap_use or cmap_pos),
                        show_months=show_months
                    )
                else:
                    # Weighted expected pattern across scenarios
                    all_v, all_w = [], []
                    for scen in scenarios:
                        prob = float(scen_prob.get(str(scen), scen_prob.get(scen, 0.0)))
                        if prob == 0.0:
                            continue
                        s, _, _, _, _ = _get_series_and_norm(scen, row)
                        if s is None or s.empty:
                            continue
                        w = snap_w.reindex(s.index).fillna(0.0) * prob
                        # Upsample per-scenario before concat; concat produces duplicate
                        # timestamps which break resample() inside the heatmap helper.
                        s = _upsample_to_hourly(s)
                        w = _upsample_to_hourly(w)
                        all_v.append(s)
                        all_w.append(w)

                    if all_v:
                        v_cat = pd.concat(all_v, axis=0)
                        w_cat = pd.concat(all_w, axis=0)
                    else:
                        v_cat = pd.Series(dtype=float)
                        w_cat = pd.Series(dtype=float)

                    col_title = stochastic_col_label if r == 0 else ""
                    im = _heatmap_day_hour_weighted_actual(
                        v_cat, w_cat,
                        ax=ax,
                        norm=row_norm if row_norm is not None else Normalize(0, 1),
                        title=col_title,
                        cmap=row_cmap,
                        show_months=show_months
                    )


                if c == 0:
                    ax.set_ylabel(f"{row['row_label']}\nHour")

            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import TwoSlopeNorm

            # after plotting all columns for this row:
            if row_norm is not None:
                sm = ScalarMappable(norm=row_norm, cmap=row_cmap)
                sm.set_array([])

                cbar = fig.colorbar(
                    sm,
                    ax=axes[r, :].ravel().tolist(),
                    fraction=0.02,
                    pad=0.02,
                )

                # unit from bus + conversion for energy variables
                comp_df, _, _ = kind_map[row["kind"]]
                unit = _row_unit_from_bus(
                    n=n,
                    kind=row["kind"],
                    comp_df=comp_df,
                    scen=None,
                    name=row["name"],
                    quantity=row_quantity,
                    field=row_field,
                ) or ""

                if isinstance(row_norm, TwoSlopeNorm):
                    lbl = f"{unit} (±{row_cap:g})" if unit else f"(±{row_cap:g})"
                else:
                    lbl = f"{unit} (0–{row_cap:g})" if unit else f"(0–{row_cap:g})"

                cbar.set_label(lbl, fontsize=8)
                cbar.ax.tick_params(labelsize=8)

        fig.suptitle(title, y=1.02)

    else:
        n_plots = len(expanded)
        n_cols = int(np.ceil(np.sqrt(n_plots)))
        n_rows = int(np.ceil(n_plots / n_cols))

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(3.6 * n_cols, 2.2 * n_rows),
            sharey=True,
            constrained_layout=True
        )
        axes = np.atleast_1d(axes).ravel()

        for i, row in enumerate(expanded):
            ax = axes[i]
            s, norm, cmap_use, quantity, cap = _get_series_and_norm(None, row)

            im = heatmap_day_hour_actual(
                s if s is not None else pd.Series(dtype=float),
                ax=ax,
                norm=norm if norm is not None else Normalize(0, 1),
                title=row["row_label"],
                cmap=cmap_use if cmap_use is not None else cmap_pos,
                show_months=(i // n_cols == n_rows - 1)
            )

            if norm is not None and im is not None:
                from matplotlib.cm import ScalarMappable
                from matplotlib.colors import TwoSlopeNorm
                sm = ScalarMappable(norm=norm, cmap=cmap_use or cmap_pos)
                sm.set_array([])
                cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
                comp_df, _, _ = kind_map[row["kind"]]
                unit = _row_unit_from_bus(
                    n=n, kind=row["kind"], comp_df=comp_df,
                    scen=None, name=row["name"],
                    quantity=quantity, field=row.get("field"),
                ) or ""
                if isinstance(norm, TwoSlopeNorm):
                    lbl = f"{unit} (±{cap:g})" if unit else f"(±{cap:g})"
                else:
                    lbl = f"{unit} (0–{cap:g})" if unit else f"(0–{cap:g})"
                cbar.set_label(lbl, fontsize=8)
                cbar.ax.tick_params(labelsize=8)

        for j in range(n_plots, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(title, y=1.02)

    if outpath:
        outpath = Path(outpath)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

# ---- COST OPTIMAL SOLUTION ----
# carrier aggregation (Pypsa - default):
def make_global_summary_costs(
    n,
    scenario_weight_col="weight",
    carrier_map=None,
    include_expected=True,
    csv_path=None,  # optional: if not None, save ONE csv here
):
    cap = n.statistics.capex().rename("capex")
    op = n.statistics.opex().rename("opex")

    # --- align MultiIndex names so pandas can concat ---
    if isinstance(cap.index, pd.MultiIndex) and isinstance(op.index, pd.MultiIndex):
        # choose a common set of names (prefer op's, but make them identical)
        cap.index = cap.index.set_names(op.index.names)

    costs = pd.concat([cap, op], axis=1).fillna(0.0)
    costs["total"] = costs["capex"] + costs["opex"]

    df = costs.reset_index()

    # detect whether statistics output is scenario-resolved
    has_scenario_costs = ("scenario" in df.columns) or ("Scenario" in df.columns)

    # scenario (robust)
    if "scenario" not in df.columns:
        if "Scenario" in df.columns:
            df = df.rename(columns={"Scenario": "scenario"})
        else:
            df["scenario"] = "deterministic"

    # carrier (robust)
    if "carrier" not in df.columns:
        if "Carrier" in df.columns:
            df = df.rename(columns={"Carrier": "carrier"})
        else:
            df["carrier"] = "unknown"

    if carrier_map is None:
        df["group"] = df["carrier"]
    else:
        df["group"] = df["carrier"].map(carrier_map).fillna(df["carrier"])

    costs_long = df.groupby(["scenario", "group"], as_index=True)[["capex", "opex", "total"]].sum()
    total_by_scenario = costs_long["total"].groupby(level="scenario").sum()

    # ── LCOP recovery / market revenue split (generic, any revenue-bearing link) ──
    scenarios_for_split = list(df["scenario"].astype(str).unique()) or ["deterministic"]
    split_long = component_revenue_split_long_per_scenario(n, scenarios_for_split)
    if not split_long.empty:
        link_carrier = n.links["carrier"] if "carrier" in n.links.columns else pd.Series(dtype=object)
        split_long["carrier"] = split_long["name"].map(link_carrier).fillna("unknown")
        split_long["group"] = (
            split_long["carrier"].map(carrier_map).fillna(split_long["carrier"])
            if carrier_map is not None else split_long["carrier"]
        )
        split_agg = split_long.groupby(["scenario", "group"], as_index=True)[
            ["lcop_recovery", "market_revenue"]
        ].sum()
    else:
        split_agg = pd.DataFrame(columns=["lcop_recovery", "market_revenue"])

    costs_long = costs_long.join(split_agg, how="left").fillna(
        {"lcop_recovery": 0.0, "market_revenue": 0.0}
    )
    # Reconcile: the split only covers Links with an identifiable bus0 price;
    # anything opex captures beyond that (Stores, Generators, ...) is folded
    # into lcop_recovery so lcop_recovery + market_revenue == opex exactly,
    # by construction, for every (scenario, group).
    unexplained = costs_long["opex"] - (costs_long["lcop_recovery"] + costs_long["market_revenue"])
    costs_long["lcop_recovery"] = costs_long["lcop_recovery"] + unexplained

    # scenario weights
    scenario_weights = None
    if hasattr(n, "scenario_weightings") and n.scenario_weightings is not None:
        try:
            scenario_weights = n.scenario_weightings[scenario_weight_col].copy()
        except Exception:
            scenario_weights = None

    total_expected = None
    expected_long = None
    # Only compute expected values if costs are scenario-resolved
    if include_expected and has_scenario_costs and scenario_weights is not None and len(scenario_weights) > 0:
        w = scenario_weights.copy()
        w.index = w.index.astype(str)

        tmp = costs_long.reset_index()
        tmp["scenario"] = tmp["scenario"].astype(str)
        tmp["w"] = tmp["scenario"].map(w).fillna(0.0).astype(float)

        for c in ["capex", "opex", "total", "lcop_recovery", "market_revenue"]:
            tmp[c] = tmp[c] * tmp["w"]

        expected_long = tmp.groupby(["group"], as_index=True)[
            ["capex", "opex", "total", "lcop_recovery", "market_revenue"]
        ].sum()
        total_expected = float(expected_long["total"].sum())

    summary = {
        "costs_long": costs_long,
        "total_by_scenario": total_by_scenario,
        "expected_long": expected_long,
        "total_expected": total_expected,
        "scenario_weights": scenario_weights,
    }

    # -------------------------
    # Build ONE combined CSV df
    # -------------------------
    out = costs_long.reset_index().copy()
    out["scenario"] = out["scenario"].astype(str)

    # expected_long appended with scenario="stochastic"
    if expected_long is not None and not expected_long.empty:
        exp = expected_long.reset_index().copy()
        exp.insert(0, "scenario", "stochastic")
        out = pd.concat([out, exp], ignore_index=True, sort=False)

    out["unit"] = "€/y"

    # probability column: only meaningful if scenario-resolved
    out["probability"] = np.nan
    if has_scenario_costs and scenario_weights is not None and len(scenario_weights) > 0:
        w = scenario_weights.copy()
        w.index = w.index.astype(str)
        out.loc[out["scenario"] != "stochastic", "probability"] = (
            out.loc[out["scenario"] != "stochastic", "scenario"].map(w)
        )

    # totals row per scenario (including stochastic)
    totals = (
        out.groupby("scenario", as_index=False)[
            ["capex", "opex", "total", "lcop_recovery", "market_revenue"]
        ]
        .sum()
        .assign(group="total", unit="€/y")
    )

    totals["probability"] = np.nan
    if has_scenario_costs and scenario_weights is not None and len(scenario_weights) > 0:
        totals.loc[totals["scenario"] != "stochastic", "probability"] = (
            totals.loc[totals["scenario"] != "stochastic", "scenario"].map(w)
        )

    out = pd.concat([out, totals], ignore_index=True, sort=False)

    out = out[["scenario", "group", "capex", "opex", "total",
               "lcop_recovery", "market_revenue", "unit", "probability"]]

    out["__is_total"] = (out["group"] == "total").astype(int)
    out = out.sort_values(["scenario", "__is_total", "group"], ascending=[True, True, True]).drop(columns="__is_total")

    from pathlib import Path

    if csv_path is not None:
        p = Path(csv_path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(p.as_posix(), index=False)

    summary["csv_table"] = out
    return summary

def plot_total_system_cost_stacked(
    summary,
    outpath=None,
    title="Total system cost (stacked)",
    which="total",               # "total" or "capex" or "opex"
    add_expected=True,
    ncol_legend=4,
    figsize=(10, 5),
):
    """
    summary = output of make_global_summary_costs()
    Stacked bars: scenarios on x-axis, stacks are 'group' (carrier or mapped group).
    """
    costs_long = summary["costs_long"]
    expected_long = summary.get("expected_long", None)

    if costs_long is None or costs_long.empty:
        raise ValueError("No costs to plot (costs_long is empty).")

    def _pivot(col):
        """Pivot one costs_long column to wide (rows=scenario, cols=group),
        append the probability-weighted 'stochastic' row the same way `which`
        is handled, and drop the 'total' pseudo-group."""
        wide = costs_long[col].reset_index().pivot_table(
            index="scenario", columns="group", values=col, aggfunc="sum"
        ).fillna(0.0)
        if "total" in wide.columns:
            wide = wide.drop(columns=["total"])
        if add_expected and expected_long is not None and not expected_long.empty and col in expected_long.columns:
            exp = expected_long[col].copy()
            for g in wide.columns:
                if g not in exp.index:
                    exp.loc[g] = 0.0
            exp = exp.reindex(wide.columns).fillna(0.0)
            wide.loc["stochastic"] = exp.values
        return wide

    df = _pivot(which)

    # LCOP recovery / market revenue split of the negative (revenue) portion —
    # only meaningful for "opex"/"total" (capex is never split; it's always cost).
    has_split = (
        which in ("opex", "total")
        and "lcop_recovery" in costs_long.columns
        and "market_revenue" in costs_long.columns
    )
    if has_split:
        df_lcop = _pivot("lcop_recovery").reindex(index=df.index, columns=df.columns, fill_value=0.0)
        df_mkt  = _pivot("market_revenue").reindex(index=df.index, columns=df.columns, fill_value=0.0)

    # Sort groups by total contribution (so legend/order is stable)
    group_order = df.sum(axis=0).sort_values(ascending=False).index
    df = df[group_order]
    if has_split:
        df_lcop = df_lcop[group_order]
        df_mkt = df_mkt[group_order]
        # Reconcile against `which`: this is 0 when which="opex", and equals
        # capex when which="total" (capex is real cost, folded into LCOP
        # recovery rather than treated as "market revenue").
        leftover = df - (df_lcop + df_mkt)
        df_lcop = df_lcop + leftover

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(df.index))

    cmap = plt.get_cmap("tab20")  # good for many stacks
    colors = [cmap(i % cmap.N) for i in range(len(df.columns))]

    pos_bottoms = np.zeros(len(df.index))
    neg_bottoms = np.zeros(len(df.index))

    for g, col in zip(df.columns, colors):
        vals = df[g].to_numpy(dtype=float)

        pos = np.where(vals > 0, vals, 0.0)
        neg = np.where(vals < 0, vals, 0.0)

        labeled = False

        # Positive
        if np.any(pos):
            ax.bar(
                x, pos,
                bottom=pos_bottoms,
                label=str(g),
                color=col,
                edgecolor="black",
                linewidth=0.25,
            )
            pos_bottoms += pos
            labeled = True

        # Negative (give label ONLY if we didn't label via positive)
        if np.any(neg):
            if has_split:
                neg_lcop = np.where(vals < 0, df_lcop[g].to_numpy(dtype=float), 0.0)
                neg_mkt  = np.where(vals < 0, df_mkt[g].to_numpy(dtype=float), 0.0)
                if np.any(neg_lcop):
                    ax.bar(
                        x, neg_lcop,
                        bottom=neg_bottoms,
                        label=(str(g) if not labeled else None),
                        color=col,
                        alpha=0.35,
                        hatch="///",
                        edgecolor="black",
                        linewidth=0.25,
                    )
                    neg_bottoms += neg_lcop
                    labeled = True
                if np.any(neg_mkt):
                    ax.bar(
                        x, neg_mkt,
                        bottom=neg_bottoms,
                        label=(str(g) if not labeled else None),
                        color=col,
                        alpha=0.55,
                        hatch="\\\\\\",
                        edgecolor="black",
                        linewidth=0.25,
                    )
                    neg_bottoms += neg_mkt
            else:
                ax.bar(
                    x, neg,
                    bottom=neg_bottoms,
                    label=(str(g) if not labeled else None),
                    color=col,
                    alpha=0.35,
                    hatch="///",
                    edgecolor="black",
                    linewidth=0.25,
                )
                neg_bottoms += neg

    # zero line helps interpretation
    ax.axhline(0, linewidth=1.0, color="black", alpha=0.6)

    # --- net total per scenario (pos + neg)
    net = df.sum(axis=1).to_numpy(dtype=float)

    # overlay marker/line
    ax.plot(
        x, net,
        color="black",
        linewidth=2.2,
        marker="o",
        markersize=5,
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=1.2,
        label="Net total cost",
        zorder=5,  # draw on top of bars
    )

    # annotate net value
    y_range = (ax.get_ylim()[1] - ax.get_ylim()[0])
    dy = 0.02 * y_range
    scale = 1e6  # M€
    unit = "M€"

    for xi, yi in zip(x, net):
        va = "bottom" if yi >= 0 else "top"
        ax.text(
            xi, yi + (dy if yi >= 0 else -dy),
            f"{yi / scale:,.1f} {unit}",
            ha="center", va=va, fontsize=8,
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1.5),
            clip_on=True,
        )

    ax.set_title(title)
    ax.set_ylabel("System cost (€/year)")
    ax.set_xticks(x)
    ax.set_xticklabels(df.index.astype(str), rotation=0)
    ax.grid(True, axis="y", alpha=0.25)

    ymin, ymax = ax.get_ylim()
    padding_top = 0.10 * (ymax - ymin)
    padding_bottom = 0.05 * (ymax - ymin)

    ax.set_ylim(ymin - padding_bottom, ymax + padding_top)

    if has_split:
        sign_handles = [
            Patch(facecolor="white", edgecolor="black", label="Cost (+)"),
            Patch(facecolor="white", edgecolor="black", hatch="///", alpha=0.35, label="LCOP recovery (−)"),
            Patch(facecolor="white", edgecolor="black", hatch="\\\\\\", alpha=0.55, label="Market revenue (−)"),
        ]
    else:
        sign_handles = [
            Patch(facecolor="white", edgecolor="black", label="Cost (+)"),
            Patch(facecolor="white", edgecolor="black", hatch="///", alpha=0.35, label="Revenue (−)"),
        ]

    handles, labels = ax.get_legend_handles_labels()
    handles = handles + sign_handles
    labels = labels + [h.get_label() for h in sign_handles]

    leg = fig.legend(
        handles, labels,
        loc="lower center",
        ncol=min(ncol_legend, len(labels)),
        frameon=False,
        fontsize=9,
    )

    # --- draw once to get legend size in figure coordinates
    fig.canvas.draw()
    bbox = leg.get_window_extent(fig.canvas.get_renderer())
    bbox_fig = bbox.transformed(fig.transFigure.inverted())

    legend_height = bbox_fig.height

    # --- reserve space safely
    fig.tight_layout(rect=[0, legend_height + 0.02, 1, 1])

    if outpath:
        outpath = Path(outpath)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

# Agent aggregation (CGT)
def allocation_dict_to_df(n_allocation, kinds=("links","generators","stores","loads","buses","storage_units")):
    """
    Converts allocation dict:
      {agent: {"links":[...], "generators":[...], ...}}
    into a DataFrame with columns: kind, name, agent
    """
    rows = []
    for agent, block in n_allocation.items():
        if not isinstance(block, dict):
            continue
        for k in kinds:
            names = block.get(k, []) or []
            for name in names:
                rows.append((k, str(name), str(agent)))

    df = pd.DataFrame(rows, columns=["kind_raw", "name", "agent"])

    # Map allocation keys to PyPSA component names used in tables
    kind_map = {
        "links": "Link",
        "generators": "Generator",
        "stores": "Store",
        "storage_units": "StorageUnit",   # NEW
        "loads": "Load",
        "buses": "Bus",
    }
    df["kind"] = df["kind_raw"].map(kind_map).fillna(df["kind_raw"])
    df = df.drop(columns=["kind_raw"])

    df = df.drop_duplicates(subset=["kind", "name"], keep="first").reset_index(drop=True)
    return df


def _get_ts_scen_asset(df, scen, name):
    """
    df: DataFrame with columns MultiIndex ('scenario','name')
    Returns pd.Series or None
    """
    if df is None:
        return None
    if isinstance(df.columns, pd.MultiIndex) and ("scenario" in df.columns.names) and ("name" in df.columns.names):
        key = (scen, name)
        return df[key] if key in df.columns else None
    return df[name] if name in df.columns else None


def _scenario_slices_static(tbl, scenarios):
    """
    Return dict scen -> DataFrame slice for that scenario.
    If tbl is MultiIndex with level 'scenario', slice by scenario.
    Else return same table for every scenario.
    """
    out = {}
    if isinstance(tbl.index, pd.MultiIndex) and "scenario" in tbl.index.names:
        for scen in scenarios:
            if scen in tbl.index.get_level_values("scenario"):
                out[scen] = tbl.xs(scen, level="scenario")
            else:
                out[scen] = tbl.iloc[0:0].copy()
    else:
        for scen in scenarios:
            out[scen] = tbl
    return out


def build_allocation_lookup(n_allocation):
    kind_map = {
        "links": "Link",
        "generators": "Generator",
        "stores": "Store",
        "storage_units": "StorageUnit",   # NEW
        "loads": "Load",
        "buses": "Bus",
    }
    lookup = {}
    for agent, block in n_allocation.items():
        for k_raw, k in kind_map.items():
            for nm in (block.get(k_raw, []) or []):
                lookup[(k, str(nm))] = str(agent)
    return lookup


def _get_cols_level(df, preferred):
    if not isinstance(df.columns, pd.MultiIndex):
        return None
    if preferred in df.columns.names:
        return df.columns.names.index(preferred)
    return None


def _guess_name_level(df, valid_names):
    # pick the level with the largest overlap with valid_names
    if not isinstance(df.columns, pd.MultiIndex):
        return None
    valid = set(map(str, valid_names))
    best_i, best_hit = None, -1
    for i in range(df.columns.nlevels):
        vals = set(map(str, df.columns.get_level_values(i).unique()))
        hit = len(vals & valid)
        if hit > best_hit:
            best_hit = hit
            best_i = i
    return best_i


import numpy as np
import pandas as pd

def component_capex_long_per_scenario(
    n,
    scenarios,
    cap_cost_col="capital_cost",
):
    """
    Compute conditional (per-scenario) CAPEX by component *name*.

    Included components:
      - Generators:   p_nom_opt * capital_cost   (fallback p_nom)
      - Links:        p_nom_opt * capital_cost   (fallback p_nom)
      - Stores:       e_nom_opt * capital_cost   (fallback e_nom)
      - StorageUnits: p_nom_opt * capital_cost   (fallback p_nom)

    Notes:
    - No snapshot weights, no scenario probabilities (pure per-scenario static investment).
    - Reads columns from component tables (DataFrames), not attributes.
    - Keeps any negative capital_cost if present (rare, but not blocked).
    """

    rows = []

    def _col(df, name):
        """Safe column getter for DataFrame columns."""
        if df is None or df.empty:
            return None
        return df[name] if name in df.columns else None

    def _capex_for(df, asset_name, cap_cols):
        """
        df: scenario-sliced component table (index=asset name)
        cap_cols: list like ["p_nom_opt","p_nom"] or ["e_nom_opt","e_nom"]
        """
        if df is None or df.empty or asset_name not in df.index:
            return None

        cc_series = _col(df, cap_cost_col)
        if cc_series is None:
            return None

        cc = pd.to_numeric(cc_series.loc[asset_name], errors="coerce")
        if not np.isfinite(cc) or cc == 0.0:
            return None

        cap = None
        for c in cap_cols:
            s = _col(df, c)
            if s is None:
                continue
            v = pd.to_numeric(s.loc[asset_name], errors="coerce")
            if np.isfinite(v):
                cap = float(v)
                break

        if cap is None:
            return None

        val = float(cap) * float(cc)
        if not np.isfinite(val) or val == 0.0:
            return None
        return val

    # Scenario slices
    gens_by_s  = _scenario_slices_static(n.generators, scenarios)    if hasattr(n, "generators") else {}
    links_by_s = _scenario_slices_static(n.links, scenarios)         if hasattr(n, "links") else {}
    stores_by_s = _scenario_slices_static(n.stores, scenarios)       if hasattr(n, "stores") else {}
    sus_by_s   = _scenario_slices_static(n.storage_units, scenarios) if hasattr(n, "storage_units") else {}

    for scen in scenarios:
        scen_str = "deterministic" if scen is None else str(scen)

        # Generators
        gens = gens_by_s.get(scen, None)
        if gens is not None and not gens.empty:
            for name in gens.index:
                capex = _capex_for(gens, name, cap_cols=["p_nom_opt", "p_nom"])
                if capex is not None:
                    rows.append(("Generator", str(name), scen_str, capex))

        # Links
        links = links_by_s.get(scen, None)
        if links is not None and not links.empty:
            for name in links.index:
                capex = _capex_for(links, name, cap_cols=["p_nom_opt", "p_nom"])
                if capex is not None:
                    rows.append(("Link", str(name), scen_str, capex))

        # Stores
        stores = stores_by_s.get(scen, None)
        if stores is not None and not stores.empty:
            for name in stores.index:
                capex = _capex_for(stores, name, cap_cols=["e_nom_opt", "e_nom"])
                if capex is not None:
                    rows.append(("Store", str(name), scen_str, capex))

        # StorageUnits
        sus = sus_by_s.get(scen, None)
        if sus is not None and not sus.empty:
            for name in sus.index:
                capex = _capex_for(sus, name, cap_cols=["p_nom_opt", "p_nom"])
                if capex is not None:
                    rows.append(("StorageUnit", str(name), scen_str, capex))

    return pd.DataFrame(rows, columns=["kind", "name", "scenario", "capex"])

def component_opex_long_per_scenario(
    n,
    scenarios,
    snapshot_weight_col="objective",
    abs_link_p0=False,   # keep False to preserve sign (sales can be negative opex)
):
    """
    Compute conditional (per-scenario) OPEX by component *name* for Generators and Links.

    Key behavior:
    - Uses snapshot weights (objective) but NOT scenario probabilities.
    - Uses time-varying marginal_cost (generators_t.marginal_cost / links_t.marginal_cost) if available.
      Falls back to static marginal_cost otherwise.
    - Keeps sign of flows; abs_link_p0=True forces throughput-cost style (not recommended for revenue links).
    """
    snap_w = (
        n.snapshot_weightings[snapshot_weight_col]
        .reindex(n.snapshots)
        .fillna(0.0)
        .to_numpy()
    )

    gens_by_s = _scenario_slices_static(n.generators, scenarios)
    links_by_s = _scenario_slices_static(n.links, scenarios)

    def _get_ts_scen_asset(df, scen, name):
        """Return series for (scenario, name) from a *_t DataFrame, or None."""
        if df is None:
            return None
        if isinstance(df.columns, pd.MultiIndex) and {"scenario", "name"}.issubset(df.columns.names):
            key = (scen, name)
            return df[key] if key in df.columns else None
        return df[name] if name in df.columns else None

    rows = []

    # -----------------------
    # Generators: p * marginal_cost(t)
    # -----------------------
    if hasattr(n, "generators_t") and hasattr(n.generators_t, "p"):
        dfp = n.generators_t.p
        dfmc = getattr(n.generators_t, "marginal_cost", None)

        if isinstance(dfp.columns, pd.MultiIndex):
            scen_lvl = _get_cols_level(dfp, "scenario")
            if scen_lvl is None:
                scen_lvl = 0
            name_lvl = _get_cols_level(dfp, "name")
            if name_lvl is None:
                name_lvl = _guess_name_level(dfp, gens_by_s[scenarios[0]].index)

            for scen in scenarios:
                try:
                    sub_p = dfp.xs(scen, level=scen_lvl, axis=1)
                except KeyError:
                    continue

                # flatten columns to names if still MultiIndex
                if isinstance(sub_p.columns, pd.MultiIndex):
                    nl = _guess_name_level(sub_p, gens_by_s[scen].index)
                    sub_p = sub_p.copy()
                    sub_p.columns = sub_p.columns.get_level_values(nl)

                gens = gens_by_s[scen]

                for name in sub_p.columns:
                    if name not in gens.index:
                        continue

                    p = pd.to_numeric(sub_p[name], errors="coerce").fillna(0.0).to_numpy()

                    # time-varying marginal cost preferred
                    mc_ts = _get_ts_scen_asset(dfmc, scen, name) if dfmc is not None else None
                    if mc_ts is not None:
                        mc = pd.to_numeric(pd.Series(mc_ts, copy=False), errors="coerce").fillna(0.0).to_numpy()
                        opex = float(np.sum(p * mc * snap_w))
                    else:
                        mc0 = float(gens.at[name, "marginal_cost"]) if "marginal_cost" in gens.columns else 0.0
                        if mc0 == 0.0:
                            continue
                        opex = float(np.sum(p * mc0 * snap_w))

                    if opex != 0.0:
                        rows.append(("Generator", str(name), str(scen), opex))

        else:
            # deterministic flat columns
            gens = gens_by_s[scenarios[0]]
            dfmc = getattr(n.generators_t, "marginal_cost", None)

            for name in dfp.columns:
                if name not in gens.index:
                    continue

                p = pd.to_numeric(dfp[name], errors="coerce").fillna(0.0).to_numpy()

                mc_ts = _get_ts_scen_asset(dfmc, "deterministic", name) if dfmc is not None else None
                if mc_ts is not None:
                    mc = pd.to_numeric(pd.Series(mc_ts, copy=False), errors="coerce").fillna(0.0).to_numpy()
                    opex = float(np.sum(p * mc * snap_w))
                else:
                    mc0 = float(gens.at[name, "marginal_cost"]) if "marginal_cost" in gens.columns else 0.0
                    if mc0 == 0.0:
                        continue
                    opex = float(np.sum(p * mc0 * snap_w))

                if opex != 0.0:
                    rows.append(("Generator", str(name), "deterministic", opex))

    # -----------------------
    # Links: p0 * marginal_cost(t)   (SIGNED unless abs_link_p0=True)
    # -----------------------
    if hasattr(n, "links_t") and hasattr(n.links_t, "p0"):
        dfp0 = n.links_t.p0
        dfmc = getattr(n.links_t, "marginal_cost", None)

        if isinstance(dfp0.columns, pd.MultiIndex):
            scen_lvl = _get_cols_level(dfp0, "scenario")
            if scen_lvl is None:
                scen_lvl = 0

            for scen in scenarios:
                try:
                    sub_p0 = dfp0.xs(scen, level=scen_lvl, axis=1)
                except KeyError:
                    continue

                if isinstance(sub_p0.columns, pd.MultiIndex):
                    nl = _guess_name_level(sub_p0, links_by_s[scen].index)
                    sub_p0 = sub_p0.copy()
                    sub_p0.columns = sub_p0.columns.get_level_values(nl)

                links = links_by_s[scen]

                for name in sub_p0.columns:
                    if name not in links.index:
                        continue

                    p0 = pd.to_numeric(sub_p0[name], errors="coerce").fillna(0.0).to_numpy()
                    if abs_link_p0:
                        p0 = np.abs(p0)

                    mc_ts = _get_ts_scen_asset(dfmc, scen, name) if dfmc is not None else None
                    if mc_ts is not None:
                        mc = pd.to_numeric(pd.Series(mc_ts, copy=False), errors="coerce").fillna(0.0).to_numpy()
                        opex = float(np.sum(p0 * mc * snap_w))
                    else:
                        mc0 = float(links.at[name, "marginal_cost"]) if "marginal_cost" in links.columns else 0.0
                        if mc0 == 0.0:
                            continue
                        opex = float(np.sum(p0 * mc0 * snap_w))

                    if opex != 0.0:
                        rows.append(("Link", str(name), str(scen), opex))

        else:
            # deterministic flat columns
            links = links_by_s[scenarios[0]]
            dfmc = getattr(n.links_t, "marginal_cost", None)

            for name in dfp0.columns:
                if name not in links.index:
                    continue

                p0 = pd.to_numeric(dfp0[name], errors="coerce").fillna(0.0).to_numpy()
                if abs_link_p0:
                    p0 = np.abs(p0)

                mc_ts = _get_ts_scen_asset(dfmc, "deterministic", name) if dfmc is not None else None
                if mc_ts is not None:
                    mc = pd.to_numeric(pd.Series(mc_ts, copy=False), errors="coerce").fillna(0.0).to_numpy()
                    opex = float(np.sum(p0 * mc * snap_w))
                else:
                    mc0 = float(links.at[name, "marginal_cost"]) if "marginal_cost" in links.columns else 0.0
                    if mc0 == 0.0:
                        continue
                    opex = float(np.sum(p0 * mc0 * snap_w))

                if opex != 0.0:
                    rows.append(("Link", str(name), "deterministic", opex))

    return pd.DataFrame(rows, columns=["kind", "name", "scenario", "opex"])


def component_revenue_split_long_per_scenario(
    n,
    scenarios,
    snapshot_weight_col="objective",
):
    """
    Split each Link's opex into "LCOP recovery" and "market revenue":

        LCOP recovery  = -sum_t( p0_t * lambda_bus0_t * w_t )
        Market revenue = opex - LCOP recovery

    lambda_bus0 is the link's own bus0 marginal_price (n.buses_t.marginal_price)
    — the value internal production economics actually attributes to this
    flow. LCOP recovery is what a cost-based accounting would credit the
    system for; market revenue is everything beyond that — e.g. an exogenous
    flat sale price exceeding the internal shadow price because of a binding
    annual quota, a ratio constraint (max_RE_to_grid), etc. It's computed
    generically for every revenue-bearing link (opex < 0), not just tagged
    product-collection links, so it applies equally to bioCH4/H2/Methanol
    sales, electricity/DH exports, and CO2 credits — and correctly comes out
    as ~0 wherever there's no such decoupling (e.g. demand mode, or a sale
    link with no separate internal collection bus).

    Links with opex >= 0 (a real cost, not revenue) are not split: LCOP
    recovery == opex, market revenue == 0.

    Returns a long DataFrame: kind ("Link"), name, scenario, opex,
    lcop_recovery, market_revenue.
    """
    snap_w = (
        n.snapshot_weightings[snapshot_weight_col]
        .reindex(n.snapshots)
        .fillna(0.0)
        .to_numpy()
    )

    cols = ["kind", "name", "scenario", "opex", "lcop_recovery", "market_revenue"]
    if not (hasattr(n, "links_t") and hasattr(n.links_t, "p0")):
        return pd.DataFrame(columns=cols)

    links_by_s = _scenario_slices_static(n.links, scenarios)
    mp = getattr(n.buses_t, "marginal_price", None)
    dfp0 = n.links_t.p0
    dfmc = getattr(n.links_t, "marginal_cost", None)

    def _bus0_price_series(scen, bus0):
        if mp is None or pd.isna(bus0) or str(bus0) == "":
            return None
        bus0 = str(bus0)
        if isinstance(mp.columns, pd.MultiIndex) and {"scenario", "name"}.issubset(mp.columns.names):
            key = (scen, bus0)
            return mp[key] if key in mp.columns else None
        return mp[bus0] if bus0 in mp.columns else None

    def _split_one(name, links, scen, p0, opex):
        if opex >= 0.0:
            return opex, 0.0
        bus0 = links.at[name, "bus0"] if "bus0" in links.columns else ""
        pi0_ts = _bus0_price_series(scen, bus0)
        if pi0_ts is None:
            return opex, 0.0
        pi0 = pd.to_numeric(pd.Series(pi0_ts, copy=False), errors="coerce").fillna(0.0).to_numpy()
        lcop_recovery = -float(np.sum(p0 * pi0 * snap_w))
        return lcop_recovery, opex - lcop_recovery

    rows = []

    if isinstance(dfp0.columns, pd.MultiIndex):
        scen_lvl = _get_cols_level(dfp0, "scenario")
        if scen_lvl is None:
            scen_lvl = 0

        for scen in scenarios:
            try:
                sub_p0 = dfp0.xs(scen, level=scen_lvl, axis=1)
            except KeyError:
                continue

            if isinstance(sub_p0.columns, pd.MultiIndex):
                nl = _guess_name_level(sub_p0, links_by_s[scen].index)
                sub_p0 = sub_p0.copy()
                sub_p0.columns = sub_p0.columns.get_level_values(nl)

            links = links_by_s[scen]

            for name in sub_p0.columns:
                if name not in links.index:
                    continue

                p0 = pd.to_numeric(sub_p0[name], errors="coerce").fillna(0.0).to_numpy()

                mc_ts = _get_ts_scen_asset(dfmc, scen, name) if dfmc is not None else None
                if mc_ts is not None:
                    mc = pd.to_numeric(pd.Series(mc_ts, copy=False), errors="coerce").fillna(0.0).to_numpy()
                    opex = float(np.sum(p0 * mc * snap_w))
                else:
                    mc0 = float(links.at[name, "marginal_cost"]) if "marginal_cost" in links.columns else 0.0
                    if mc0 == 0.0:
                        continue
                    opex = float(np.sum(p0 * mc0 * snap_w))

                if opex == 0.0:
                    continue

                lcop_recovery, market_revenue = _split_one(name, links, scen, p0, opex)
                rows.append(("Link", str(name), str(scen), opex, lcop_recovery, market_revenue))

    else:
        links = links_by_s[scenarios[0]]

        for name in dfp0.columns:
            if name not in links.index:
                continue

            p0 = pd.to_numeric(dfp0[name], errors="coerce").fillna(0.0).to_numpy()

            mc_ts = _get_ts_scen_asset(dfmc, "deterministic", name) if dfmc is not None else None
            if mc_ts is not None:
                mc = pd.to_numeric(pd.Series(mc_ts, copy=False), errors="coerce").fillna(0.0).to_numpy()
                opex = float(np.sum(p0 * mc * snap_w))
            else:
                mc0 = float(links.at[name, "marginal_cost"]) if "marginal_cost" in links.columns else 0.0
                if mc0 == 0.0:
                    continue
                opex = float(np.sum(p0 * mc0 * snap_w))

            if opex == 0.0:
                continue

            lcop_recovery, market_revenue = _split_one(name, links, "deterministic", p0, opex)
            rows.append(("Link", str(name), "deterministic", opex, lcop_recovery, market_revenue))

    return pd.DataFrame(rows, columns=cols)


def make_global_summary_costs_by_agent(
    n,
    network_comp_allocation,
    snapshot_weight_col="objective",
    scenario_weight_col="weight",
    include_expected=True,
    csv_path=None,
    unit="€/y",
    abs_link_p0=False,
):

    # scenarios + weights
    if hasattr(n, "scenario_weightings") and n.scenario_weightings is not None and len(n.scenario_weightings) > 0:
        scenarios = list(n.scenario_weightings.index.astype(str))
        scen_w = n.scenario_weightings[scenario_weight_col].copy()
        scen_w.index = scen_w.index.astype(str)
    else:
        scenarios = ["deterministic"]
        scen_w = None

    lookup = build_allocation_lookup(network_comp_allocation)

    capex = component_capex_long_per_scenario(n, scenarios)
    opex  = component_opex_long_per_scenario(n, scenarios, snapshot_weight_col, abs_link_p0)
    split = component_revenue_split_long_per_scenario(n, scenarios, snapshot_weight_col)

    # map to agent
    capex["group"] = capex.apply(lambda r: lookup.get((r["kind"], r["name"]), "Unallocated"), axis=1)
    opex["group"]  = opex.apply(lambda r: lookup.get((r["kind"], r["name"]), "Unallocated"), axis=1) if not opex.empty else "Unallocated"
    split["group"] = split.apply(lambda r: lookup.get((r["kind"], r["name"]), "Unallocated"), axis=1) if not split.empty else "Unallocated"

    # aggregate (scenario, agent)
    capex_sa = capex.groupby(["scenario","group"], as_index=True)[["capex"]].sum()
    opex_sa  = opex.groupby(["scenario","group"],  as_index=True)[["opex"]].sum() if not opex.empty else None
    split_sa = (
        split.groupby(["scenario","group"], as_index=True)[["lcop_recovery","market_revenue"]].sum()
        if not split.empty else None
    )

    costs_long = capex_sa.join(opex_sa, how="outer").join(split_sa, how="outer").fillna(0.0)
    costs_long["total"] = costs_long["capex"] + costs_long["opex"]

    # Reconcile: fold anything the split doesn't cover (Stores, etc.) into
    # lcop_recovery so lcop_recovery + market_revenue == opex exactly.
    unexplained = costs_long["opex"] - (costs_long["lcop_recovery"] + costs_long["market_revenue"])
    costs_long["lcop_recovery"] = costs_long["lcop_recovery"] + unexplained

    total_by_scenario = costs_long["total"].groupby(level="scenario").sum()

    # stochastic expected (probability-weighted)
    expected_long = None
    total_expected = None
    if include_expected and scen_w is not None and len(scen_w) > 0:
        tmp = costs_long.reset_index()
        tmp["w"] = tmp["scenario"].map(scen_w).fillna(0.0).astype(float)
        for c in ["capex","opex","total","lcop_recovery","market_revenue"]:
            tmp[c] = tmp[c] * tmp["w"]
        expected_long = tmp.groupby("group", as_index=True)[
            ["capex","opex","total","lcop_recovery","market_revenue"]
        ].sum()
        total_expected = float(expected_long["total"].sum())

    summary = {
        "costs_long": costs_long,
        "total_by_scenario": total_by_scenario,
        "expected_long": expected_long,
        "total_expected": total_expected,
        "scenario_weights": scen_w,
    }

    # ---- single CSV spec ----
    out = costs_long.reset_index()

    # (1) append expected as scenario="stochastic"
    if expected_long is not None and not expected_long.empty:
        exp = expected_long.reset_index()
        exp.insert(0, "scenario", "stochastic")
        out = pd.concat([out, exp], ignore_index=True)

    # (2) unit
    out["unit"] = unit

    # (3) probability
    out["probability"] = np.nan
    if scen_w is not None and len(scen_w) > 0:
        mask = out["scenario"] != "stochastic"
        out.loc[mask, "probability"] = out.loc[mask, "scenario"].map(scen_w)

    # (4) totals row per scenario
    totals = (
        out.groupby("scenario", as_index=False)[
            ["capex","opex","total","lcop_recovery","market_revenue"]
        ]
           .sum()
           .assign(group="total", unit=unit)
    )
    totals["probability"] = np.nan
    if scen_w is not None and len(scen_w) > 0:
        mask = totals["scenario"] != "stochastic"
        totals.loc[mask, "probability"] = totals.loc[mask, "scenario"].map(scen_w)

    out = pd.concat([out, totals], ignore_index=True)
    out = out[["scenario","group","capex","opex","total",
               "lcop_recovery","market_revenue","unit","probability"]]

    # keep total last within each scenario
    out["__is_total"] = (out["group"] == "total").astype(int)
    out = out.sort_values(["scenario","__is_total","group"]).drop(columns="__is_total")

    summary["csv_table"] = out
    if csv_path is not None:
        p = Path(csv_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(p, index=False)

    # quick diagnostics (super useful)
    summary["diagnostics"] = {
        "capex_rows": int(len(capex)),
        "opex_rows": int(len(opex)),
        "unallocated_capex_total": float(capex.loc[capex["group"]=="Unallocated", "capex"].sum()) if len(capex) else 0.0,
        "unallocated_opex_total": float(opex.loc[opex["group"]=="Unallocated", "opex"].sum()) if len(opex) else 0.0,
    }

    return summary

####### Main

def run_plot_and_export(
    *,
    n,
    c,
    csv_folder: str | Path,
    plot_folder: str | Path,
    items: list[dict],
    bus_list_mp: list[str],
    network_comp_allocation: Optional[dict] = None,
    comp_tech_map: Optional[dict] = None,
    tech_costs_used=None,
    scenarios: Optional[Dict[Any, float]] = None,
    networks_dict: Optional[Dict[Any, Any]] = None,
) -> Dict[str, Exception]:
    """
    Run plotting + CSV exports. Any failing step raises a warning but does not stop execution.

    Parameters
    ----------
    items:
        List of dicts describing components to include (already has numeric 'th' per item).
    bus_list_mp:
        List of buses used for shadow price plots.
    network_comp_allocation:
        Needed for agent-cost summary and capacity export. If None, those steps are skipped (warned).
    scenarios/networks_dict:
        Only needed if c.stochastic['EVPI'] is True and you want WS comparisons.

    Returns
    -------
    failures : dict
        Mapping {step_name: exception} for steps that failed.
    """
    if not c.n_flags_opt.get("plot", False):
        return {}

    csv_folder = Path(csv_folder)
    plot_folder = Path(plot_folder)

    csv_folder.mkdir(parents=True, exist_ok=True)
    plot_folder.mkdir(parents=True, exist_ok=True)

    failures: Dict[str, Exception] = {}

    def _safe_step(name: str, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception as e:
            failures[name] = e
            warnings.warn(
                f"[plot/export] Step failed: {name}\n"
                f"  {type(e).__name__}: {e}",
                category=RuntimeWarning,
                stacklevel=2,
            )

    def _require_allocation(step: str) -> dict:
        if network_comp_allocation is None:
            raise ValueError(
                f"{step} requires network_comp_allocation, but it was None. "
                "Pass it from main (loaded from pkl or computed upstream)."
            )
        return network_comp_allocation

    # ---------------- Steps ----------------

    def step_cost_by_carrier() -> None:
        summary = make_global_summary_costs(
            n,
            carrier_map=None,
            csv_path=csv_folder / "TSC_by_carrier.csv",
        )
        plot_total_system_cost_stacked(
            summary,
            outpath=str(plot_folder / "TSC_by_carrier.png"),
            title="Total system cost by carrier (stacked)",
            which="total",
            add_expected=True,
        )

    def step_cost_by_agent() -> None:
        alloc = _require_allocation("cost_by_agent")
        summary_agents = make_global_summary_costs_by_agent(
            n,
            alloc,
            csv_path=csv_folder / "TSC_by_agent.csv",
            abs_link_p0=False,
        )
        plot_total_system_cost_stacked(
            summary_agents,
            outpath=str(plot_folder / "TSC_by_agents.png"),
            title="Total system cost by agents (stacked)",
            which="total",
            add_expected=True,
        )

    opt_cap_holder: Dict[str, Any] = {}
    items_f_holder: Dict[str, Any] = {}

    def step_save_opt_caps() -> None:
        alloc = _require_allocation("save_optimal_capacities")

        file_path = csv_folder / "optimal_capacities"
        opt_cap = save_opt_capacity_components(
            n,
            network_comp_allocation,
            csv_folder / "optimal_capacities",
        )

        opt_cap_holder["obj"] = opt_cap

    def step_filter_items() -> None:
        items_f = filter_items_by_capacity_threshold(
            n,
            items,
            include_exi=True,
            verbose=True,
        )
        items_f_holder["items_f"] = items_f

    def step_capacity_compare_sp_vs_ws() -> None:
        ws = {}
        if getattr(c, "stochastic", {}).get("EVPI", False):
            if scenarios is None or networks_dict is None:
                raise ValueError(
                    "c.stochastic['EVPI'] is True but scenarios/networks_dict were not provided."
                )
            for year in scenarios.keys():
                ws[year] = networks_dict[year]

        df_caps = build_capacity_compare_from_items(
            n,
            items,
            ws_networks=ws,
            default_th=0.5,
            sp_col="SP",
        )
        df_caps.to_csv(csv_folder / "opt_capacities_SP_vs_WP.csv")

        plot_capacity_compare_from_items(
            df_caps,
            outpath=str(plot_folder / "Opt_capacities_SP_vs_WS.png"),
            title="Installed capacities (SP vs WS)",
        )

    def step_inputs_ldc() -> None:
        plot_ldc_inputs_by_scenario(
            n,
            outpath=str(plot_folder / "inputs_LDC_by_scenario.png"),
            ncols=3,
            price_links=[
                {"label": "El pruchase price", "selector": {"contains": "DK1_to_El_"}, "ls": "-", "lw": 1.8},
                {"label": "El selling price", "name": "El3 bus_to_DK1", "ls": "-", "lw": 1.8},
                {"label": "NG price", "selector": {"regex": r"_NG boiler$"}, "ls": "-", "lw": 1.8},
                {"label": "NG selling price", "name": "bioCH4_to_delivery", "ls": "-", "lw": 1.8},
                {"label": "DH selling price", "name": "DH_GL_to_DH_grid", "ls": "-", "lw": 1.8},
                {"label": "Biochar selling price", "name": "biochar sequestration", "ls": "-", "lw": 1.8},
                {"label": "CO2 (L) selling price", "name": "CO2 Liq seq", "ls": "-", "lw": 1.8},
            ],
            price_gens=[
                {"label": "Pellets price", "selector": "pellets market", "ls": "-.", "lw": 1.8},
                {"label": "Biomass chips", "selector": "moist biomass market", "ls": "-.", "lw": 1.8},
            ],
            cf_gens=[
                {"label": "Wind CF", "name": "onshorewind", "ls": "--", "lw": 1.8},
                {"label": "Solar CF", "name": "solar", "ls": "--", "lw": 1.8},
            ],
        )

    def step_shadow_prices() -> None:
        # Build bus list: demand mode → delivery buses only; price mode → all buses
        driver = c.targets_dict.get("driver", "demand")
        if driver == "price":
            bus_list_bar = list(bus_list_mp)
        else:
            bus_list_bar = [b for b in bus_list_mp if "collection" not in b]

        # CSV and bar chart use the same bus list
        e_means, e_throughputs = export_shadow_prices_mean_csv(n, bus_list_bar, csv_folder / "shadow_prices_mean.csv")
        plot_shadow_prices_mean_bar(
            e_means, plot_folder / "shd_prices_mean_bar.png",
            bus_filter=bus_list_bar,
            throughput=e_throughputs if driver == "price" else None,
        )

        # Violin + LDC: same list, further filtered by capacity threshold
        bus_list_f = filter_bus_list_mp(n, bus_list_bar, link_th=1e-3)
        shadow_prices_violinplot_stoch(
            n,
            bus_list=bus_list_f,
            folder=str(plot_folder),
            link_mc_items=[
                {"label": "Electricity price", "selector": {"contains": "DK1_to_El_"}},
                {"label": "NG price", "selector": {"regex": r"_NG boiler$"}},
            ],
            handle_spikes="clip",
            quantile_hi=0.98,
            n_draws=25000,
            title="Shadow prices — snapshot distribution over time (mean: scenario-weighted)",
        )

        shadow_prices_ldc_stoch(
            n,
            bus_list=bus_list_f,
            folder=str(plot_folder),
            link_mc_items=[
                {"label": "Electricity price", "selector": {"contains": "DK1_to_El_"}},
                {"label": "NG price", "selector": {"regex": r"_NG boiler$"}},
            ],
            handle_spikes="clip",
            quantile_hi=0.98,
            n_points=1001,
            fname="shd_prices_ldc.png",
            title="Shadow prices — load-duration curves over snapshots",
        )

    def step_operation_plots() -> None:
        items_f = items_f_holder.get("items_f")
        if items_f is None:
            raise RuntimeError("items_f not available; filter_items step likely failed.")

        plot_utilization_ldc_by_scenario(
            n,
            items=items_f,
            outpath=plot_folder / "CF_operation_by_scenario.png",
            title="Utilization LDCs by scenario (exact + EXI only)",
            ncols=3,
            carrier_colors=c.carrier_colors,
        )

        figure_heatmaps_compare_scenarios(
            n,
            items_f,
            outpath=plot_folder / "CF_operation_heat_maps_by_scenario.png",
            title="Optimal CF patterns by scenario (normalized 0–1)",
            cmap="viridis",
            abs_links=True,
        )

        figure_heatmaps_compare_scenarios_actual(
            n,
            items,
            outpath=plot_folder / "Operation_heat_maps_by_scenario.png",
            title="Operational heatmaps by scenario (actual values; capacity-normalized colors)",
            cmap_pos="viridis",  # sequential for >=0 series
            cmap_div="coolwarm",  # diverging for signed dispatch
            abs_links=True,
            snapshot_weight_col="objective",
            scenario_weight_col="weight",
            add_stochastic_column=True,
            stochastic_col_label="stochastic",
        )

    def step_full_component_table() -> None:
        alloc = _require_allocation("full_component_table")
        save_full_component_csv(
            n,
            alloc,
            csv_folder / "full_component_table.csv",
            comp_tech_map=comp_tech_map,
        )

    def step_cost_assumptions() -> None:
        save_cost_assumptions_csv(tech_costs_used, csv_folder / "cost_assumptions.csv")

    def step_pypsa_statistics() -> None:
        save_pypsa_statistics(n, csv_folder / "pypsa_statistics.csv")

    def step_lcop() -> None:
        compute_lcop_by_technology(
            n,
            out_csv=csv_folder / "lcop_by_technology.csv",
            out_plot=plot_folder / "lcop_by_technology.png",
        )

    def step_lcop_kkt() -> None:
        compute_lcop_kkt_by_technology(
            n,
            out_csv=csv_folder / "lcop_kkt_by_technology.csv",
        )

    def step_srmc() -> None:
        compute_srmc_by_technology(
            n,
            out_csv=csv_folder / "srmc_by_technology.csv",
            out_plot=plot_folder / "srmc_by_technology.png",
        )

    tech_costs_full_holder: Dict[str, Any] = {}

    def _tech_costs_full():
        # tech_costs_used is deliberately a trimmed subset (only technologies
        # comp_tech_map could already resolve — see save_cost_assumptions_csv);
        # several composite technologies' underlying catalogue rows (e.g.
        # "industrial heat pump medium temperature", "Concrete-charger") are
        # NOT in that subset. Re-derive the full cost table for the payback
        # override lookups; falls back to the trimmed table (reduced coverage,
        # reported via the coverage-% diagnostic) if that fails for any reason.
        if "obj" not in tech_costs_full_holder:
            try:
                from scripts.helpers import read_costs
                from scripts.technology_inputs import tech_inputs as _tech_inputs
                from scripts import parameters as _p
                tech_costs_full_holder["obj"] = read_costs(_p.cost_path, _tech_inputs, c.USD_to_EUR, c.discount_rate)
            except Exception as e:
                warnings.warn(f"[plot/export] Could not load full tech_costs for payback "
                               f"(falling back to trimmed subset): {e}", category=RuntimeWarning)
                tech_costs_full_holder["obj"] = tech_costs_used
        return tech_costs_full_holder["obj"]

    def step_payback_agent() -> None:
        alloc = _require_allocation("payback_agent")
        compute_payback_by_agent(
            n,
            network_comp_allocation=alloc,
            tech_costs=_tech_costs_full(),
            comp_tech_map=comp_tech_map,
            n_config=c.n_config,
            discount_rate=c.discount_rate,
            out_csv=csv_folder / "payback_by_agent.csv",
            out_plot=plot_folder / "payback_by_agent.png",
            amortization_period=c.amortization_period,
        )

    # ---------------- Run in order ----------------

    _safe_step("cost_by_carrier", step_cost_by_carrier)
    _safe_step("cost_by_agent", step_cost_by_agent)

    _safe_step("save_optimal_capacities", step_save_opt_caps)
    _safe_step("full_component_table", step_full_component_table)
    _safe_step("cost_assumptions", step_cost_assumptions)
    _safe_step("pypsa_statistics", step_pypsa_statistics)
    _safe_step("filter_items", step_filter_items)
    _safe_step("capacity_compare_sp_vs_ws", step_capacity_compare_sp_vs_ws)

    _safe_step("inputs_ldc_by_scenario", step_inputs_ldc)
    _safe_step("shadow_prices", step_shadow_prices)
    _safe_step("lcop", step_lcop)
    _safe_step("lcop_kkt", step_lcop_kkt)
    _safe_step("srmc", step_srmc)
    if c.targets_dict.get("driver") == "price":
        _safe_step("payback_agent", step_payback_agent)
    _safe_step("operation_plots", step_operation_plots)

    if failures:
        print(f"[plot/export] Finished with {len(failures)} failing step(s): {list(failures.keys())}")
    else:
        print("[plot/export] Finished successfully.")

    return failures


def run_plot_operational(
    *,
    n,
    c,
    plot_folder: str | Path,
    csv_folder: str | Path,
    items: list[dict],
    bus_list_mp: list[str],
) -> Dict[str, Exception]:
    """Run only the operational (dispatch) plots.

    Produces the same three operational figures as ``run_plot_and_export``
    (utilisation LDCs, CF heatmaps, actual-value heatmaps) plus the input
    LDC and shadow-price plots.  Capacity and cost steps are skipped because
    they are not meaningful for a dispatch-only RH result.

    Intended to be called from the rolling-horizon plotting script.
    """
    if not c.n_flags_opt.get("plot", False):
        return {}

    plot_folder = Path(plot_folder)
    plot_folder.mkdir(parents=True, exist_ok=True)
    csv_folder = Path(csv_folder)
    csv_folder.mkdir(parents=True, exist_ok=True)

    failures: Dict[str, Exception] = {}

    def _safe(name: str, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception as e:
            failures[name] = e
            warnings.warn(
                f"[plot/rh] Step failed: {name}\n  {type(e).__name__}: {e}",
                category=RuntimeWarning,
                stacklevel=2,
            )

    items_f_holder: Dict[str, Any] = {}

    def step_filter_items() -> None:
        items_f_holder["items_f"] = filter_items_by_capacity_threshold(
            n, items, include_exi=True, verbose=True,
        )

    def step_inputs_ldc() -> None:
        plot_ldc_inputs_by_scenario(
            n,
            outpath=plot_folder / "inputs_LDC_by_scenario.png",
            ncols=3,
            price_links=[
                {"label": "El purchase price", "selector": {"contains": "DK1_to_El_"}, "ls": "-", "lw": 1.8},
                {"label": "El selling price",  "name": "El3 bus_to_DK1",               "ls": "-", "lw": 1.8},
                {"label": "NG price",          "selector": {"regex": r"_NG boiler$"},   "ls": "-", "lw": 1.8},
                {"label": "NG selling price",  "name": "bioCH4_to_delivery",            "ls": "-", "lw": 1.8},
                {"label": "DH selling price",  "name": "DH_GL_to_DH_grid",             "ls": "-", "lw": 1.8},
                {"label": "Biochar selling price", "name": "biochar sequestration",     "ls": "-", "lw": 1.8},
                {"label": "CO2 (L) selling price", "name": "CO2 Liq seq",              "ls": "-", "lw": 1.8},
            ],
            price_gens=[
                {"label": "Pellets price",  "selector": "pellets market",       "ls": "-.", "lw": 1.8},
                {"label": "Biomass chips",  "selector": "moist biomass market", "ls": "-.", "lw": 1.8},
            ],
            cf_gens=[
                {"label": "Wind CF",  "name": "onshorewind", "ls": "--", "lw": 1.8},
                {"label": "Solar CF", "name": "solar",       "ls": "--", "lw": 1.8},
            ],
        )

    def step_shadow_prices() -> None:
        # Build bus list: demand mode → delivery buses only; price mode → all buses
        driver = c.targets_dict.get("driver", "demand")
        if driver == "price":
            bus_list_bar = list(bus_list_mp)
        else:
            bus_list_bar = [b for b in bus_list_mp if "collection" not in b]

        # CSV and bar chart use the same bus list
        e_means, e_throughputs = export_shadow_prices_mean_csv(n, bus_list_bar, csv_folder / "shadow_prices_mean.csv")
        plot_shadow_prices_mean_bar(
            e_means, plot_folder / "shd_prices_mean_bar.png",
            bus_filter=bus_list_bar,
            throughput=e_throughputs if driver == "price" else None,
        )

        # Violin + LDC: same list, further filtered by capacity threshold
        bus_list_f = filter_bus_list_mp(n, bus_list_bar, link_th=1e-3)
        shadow_prices_violinplot_stoch(
            n,
            bus_list=bus_list_f,
            folder=str(plot_folder),
            link_mc_items=[
                {"label": "Electricity price", "selector": {"contains": "DK1_to_El_"}},
                {"label": "NG price",          "selector": {"regex": r"_NG boiler$"}},
            ],
            handle_spikes="clip",
            quantile_hi=0.98,
            n_draws=25000,
            title="Shadow prices — snapshot distribution over time (mean: scenario-weighted)",
        )
        shadow_prices_ldc_stoch(
            n,
            bus_list=bus_list_f,
            folder=str(plot_folder),
            link_mc_items=[
                {"label": "Electricity price", "selector": {"contains": "DK1_to_El_"}},
                {"label": "NG price",          "selector": {"regex": r"_NG boiler$"}},
            ],
            handle_spikes="clip",
            quantile_hi=0.98,
            n_points=1001,
            fname="shd_prices_ldc.png",
            title="Shadow prices — load-duration curves over snapshots",
        )

    def step_operation_plots() -> None:
        items_f = items_f_holder.get("items_f")
        if items_f is None:
            raise RuntimeError("items_f not available; filter_items step likely failed.")

        plot_utilization_ldc_by_scenario(
            n,
            items=items_f,
            outpath=plot_folder / "CF_operation_by_scenario.png",
            title="Utilization LDCs (rolling horizon)",
            ncols=3,
            carrier_colors=c.carrier_colors,
        )

        figure_heatmaps_compare_scenarios(
            n,
            items_f,
            outpath=plot_folder / "CF_operation_heat_maps_by_scenario.png",
            title="CF patterns — rolling horizon (normalized 0–1)",
            cmap="viridis",
            abs_links=True,
        )

        figure_heatmaps_compare_scenarios_actual(
            n,
            items,
            outpath=plot_folder / "Operation_heat_maps_by_scenario.png",
            title="Operational heatmaps — rolling horizon (actual values)",
            cmap_pos="viridis",
            cmap_div="coolwarm",
            abs_links=True,
            snapshot_weight_col="objective",
            scenario_weight_col="weight",
            add_stochastic_column=False,
        )

    def step_lcop() -> None:
        compute_lcop_by_technology(
            n,
            out_csv=csv_folder / "lcop_by_technology.csv",
            out_plot=plot_folder / "lcop_by_technology.png",
        )

    def step_lcop_kkt() -> None:
        compute_lcop_kkt_by_technology(
            n,
            out_csv=csv_folder / "lcop_kkt_by_technology.csv",
        )

    def step_srmc() -> None:
        compute_srmc_by_technology(
            n,
            out_csv=csv_folder / "srmc_by_technology.csv",
            out_plot=plot_folder / "srmc_by_technology.png",
        )

    _safe("filter_items",       step_filter_items)
    _safe("inputs_ldc",         step_inputs_ldc)
    _safe("shadow_prices",      step_shadow_prices)
    _safe("lcop",               step_lcop)
    _safe("lcop_kkt",           step_lcop_kkt)
    _safe("srmc",               step_srmc)
    _safe("operation_plots",    step_operation_plots)

    if failures:
        print(f"[plot/rh] Finished with {len(failures)} failing step(s): {list(failures.keys())}")
    else:
        print("[plot/rh] Finished successfully.")

    return failures


def run_plot_rh_comparison(
    *,
    n_pf,
    n_rh,
    plot_folder: str | Path,
    csv_folder:  str | Path,
    c,
) -> Dict[str, Exception]:
    """Generate side-by-side PF vs RH cost comparison plots and CSV.

    Produces three outputs:

    * ``PF_vs_RH_total_cost.png``  — stacked-bar total system cost (PF | RH)
      broken down by carrier, reusing :func:`plot_total_system_cost_stacked`.
    * ``PF_vs_RH_opex_delta.png``  — per-carrier OPEX difference (RH − PF).
      Red = RH costs more, green = RH costs less.
    * ``PF_vs_RH_costs.csv``       — wide comparison table with capex/opex/total
      for both networks and the delta per carrier.

    Both networks must have correct ``statistics.capex()`` / ``statistics.opex()``
    — for the RH network this requires that the extendability flags were restored
    after the dispatch solve (done in ``snakemake_rolling_horizon.py``).
    """
    if not c.n_flags_opt.get("plot", False):
        return {}

    plot_folder = Path(plot_folder)
    csv_folder  = Path(csv_folder)
    plot_folder.mkdir(parents=True, exist_ok=True)
    csv_folder.mkdir(parents=True, exist_ok=True)

    failures: Dict[str, Exception] = {}

    def _safe(name: str, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception as e:
            failures[name] = e
            warnings.warn(
                f"[plot/rh_compare] Step failed: {name}\n  {type(e).__name__}: {e}",
                category=RuntimeWarning, stacklevel=2,
            )

    # ── Shared data ────────────────────────────────────────────────────────────
    summary_pf = make_global_summary_costs(n_pf)
    summary_rh = make_global_summary_costs(n_rh)

    def _rename_scenario(costs_long: pd.DataFrame, new_name: str) -> pd.DataFrame:
        """Rename all scenario-level values in a (scenario, group) MultiIndex."""
        if not isinstance(costs_long.index, pd.MultiIndex):
            return costs_long
        mapper = {v: new_name for v in costs_long.index.get_level_values("scenario").unique()}
        return costs_long.rename(index=mapper, level="scenario")

    costs_pf = _rename_scenario(summary_pf["costs_long"].copy(), "PF")
    costs_rh = _rename_scenario(summary_rh["costs_long"].copy(), "RH")
    combined_long = pd.concat([costs_pf, costs_rh])

    combined_summary = {
        "costs_long":       combined_long,
        "expected_long":    None,
        "total_by_scenario": combined_long["total"].groupby(level="scenario").sum(),
        "total_expected":   None,
        "scenario_weights": None,
    }

    # ── Steps ─────────────────────────────────────────────────────────────────

    def step_comparison_csv() -> None:
        pf_df = costs_pf.droplevel("scenario").rename(
            columns={"capex": "capex_pf", "opex": "opex_pf", "total": "total_pf"}
        )
        rh_df = costs_rh.droplevel("scenario").rename(
            columns={"capex": "capex_rh", "opex": "opex_rh", "total": "total_rh"}
        )
        merged = pf_df.join(rh_df, how="outer").fillna(0.0)
        merged["delta_opex"]  = merged["opex_rh"]  - merged["opex_pf"]
        merged["delta_total"] = merged["total_rh"] - merged["total_pf"]

        totals = merged.sum(numeric_only=True)
        totals.name = "TOTAL"
        merged = pd.concat([merged, totals.to_frame().T])
        merged.index.name = "carrier"
        merged["unit"] = "€/y"
        merged.to_csv(csv_folder / "PF_vs_RH_costs.csv")

    def step_total_cost_bar() -> None:
        plot_total_system_cost_stacked(
            combined_summary,
            outpath=str(plot_folder / "PF_vs_RH_total_cost.png"),
            title="Total system cost: Perfect Foresight vs Rolling Horizon",
            which="total",
            add_expected=False,
            figsize=(10, 5),
        )

    def step_opex_delta() -> None:
        pf_opex = costs_pf.droplevel("scenario")["opex"]
        rh_opex = costs_rh.droplevel("scenario")["opex"]
        all_idx = pf_opex.index.union(rh_opex.index)
        delta   = (rh_opex.reindex(all_idx, fill_value=0.0)
                   - pf_opex.reindex(all_idx, fill_value=0.0)).sort_values()

        total_pf_opex  = pf_opex.sum()
        total_rh_opex  = rh_opex.sum()
        total_pf_total = n_pf.statistics.capex().sum() + total_pf_opex
        total_rh_total = n_rh.statistics.capex().sum() + total_rh_opex

        fig, ax = plt.subplots(figsize=(max(8, len(delta) * 0.65 + 2), 5))
        colors = ["#d73027" if v > 0 else "#1a9850" for v in delta.values]
        ax.bar(range(len(delta)), delta.values / 1e3, color=colors,
               edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(delta)))
        ax.set_xticklabels(delta.index, rotation=45, ha="right", fontsize=9)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("k€/y   (positive = RH costs more)")
        ax.set_title(
            "OPEX delta per carrier: RH − PF\n"
            f"OPEX  PF={total_pf_opex/1e6:.2f} M€   RH={total_rh_opex/1e6:.2f} M€   "
            f"Δ={(total_rh_opex - total_pf_opex)/1e6:+.3f} M€\n"
            f"Total PF={total_pf_total/1e6:.2f} M€   RH={total_rh_total/1e6:.2f} M€   "
            f"Δ={(total_rh_total - total_pf_total)/1e6:+.3f} M€  "
            f"({(total_rh_total / total_pf_total - 1) * 100:+.2f} %)",
            fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(str(plot_folder / "PF_vs_RH_opex_delta.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    _safe("comparison_csv",  step_comparison_csv)
    _safe("total_cost_bar",  step_total_cost_bar)
    _safe("opex_delta",      step_opex_delta)

    if failures:
        print(f"[plot/rh_compare] Finished with {len(failures)} failing step(s): {list(failures.keys())}")
    else:
        print("[plot/rh_compare] Finished successfully.")

    return failures
