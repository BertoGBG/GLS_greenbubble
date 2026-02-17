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

def print_network(n, n_flags, nc_path, network_name, suffix, plot_folder, is_stochastic):
    # function that prints .svg of network topology with pypsatopo

    if not n_flags.get("print", False):
        return None

    if is_stochastic:
        # Reload ONLY to safely collapse scenarios for plotting
        if nc_path is None:
            print("[WARN] No nc_path provided; skipping network plot.")
            return None
        n_plot = pypsa.Network(nc_path)
        n_plot = extract_deterministic_from_stochastic(
            n_plot, scenario=None, slice_timeseries=False
        )
    else:
        # Deterministic network can be plotted directly (no scenario slicing needed)
        n_plot = n

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
    thresholds: dict | None = None,
    # defaults (still here for backward compatibility)
    GEN_TH=0.5,          # MW
    LINK_TH=0.5,         # MW
    LINK_MASS_TH=0.5,    # t/h
    STORE_TH=1.0,        # MWh
    STORE_MASS_TH=0.5,   # t
    SU_TH=0.5,           # MW
    SU_MASS_TH=0.5,      # t/h
):
    """
    Saves optimal capacities + annualized capex for allocated assets.

    thresholds:
      dict with keys: GEN_TH, LINK_TH, LINK_MASS_TH, STORE_TH, STORE_MASS_TH, SU_TH, SU_MASS_TH
      If provided, overrides the default keyword args above.
    """

    # --- override defaults from YAML thresholds if given ---
    if thresholds is not None:
        GEN_TH = float(thresholds.get("GEN_TH", GEN_TH))
        LINK_TH = float(thresholds.get("LINK_TH", LINK_TH))
        LINK_MASS_TH = float(thresholds.get("LINK_MASS_TH", LINK_MASS_TH))
        STORE_TH = float(thresholds.get("STORE_TH", STORE_TH))
        STORE_MASS_TH = float(thresholds.get("STORE_MASS_TH", STORE_MASS_TH))
        SU_TH = float(thresholds.get("SU_TH", SU_TH))
        SU_MASS_TH = float(thresholds.get("SU_MASS_TH", SU_MASS_TH))

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

    def threshold(component, unit):
        u = norm_unit(unit)
        if component == "generator":
            return GEN_TH
        if component == "link":
            return LINK_MASS_TH if u == "t/h" else LINK_TH
        if component == "store":
            return STORE_MASS_TH if u == "t" else STORE_TH
        # storage_unit is power-rated (like generator), but allow t/h buses too
        if component == "storage_unit":
            return SU_MASS_TH if u == "t/h" else SU_TH
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
                th = threshold("generator", u)
                cap = float(gens.at[g, gen_opt])
                cc = float(gens.at[g, "capital_cost"]) if "capital_cost" in gens.columns else np.nan
                cost_out = cap * cc
                cap_out = f"< {th}" if (th is not None and cap < th) else cap

                rows.append({
                    "plant": plant,
                    "component": "generator",
                    "asset": str(g),
                    "capacity": cap_out,
                    "Fixed cost (€/y)": cost_out,
                    "reference inlet": bus,
                    "unit": norm_unit(u),
                    "threshold": th,
                })

        # Links
        if link_opt:
            for l in alloc.get("links", []) or []:
                if l not in links.index:
                    continue
                bus0 = links.at[l, "bus0"]
                u = unit_of_bus(bus0)
                th = threshold("link", u)
                cap = float(links.at[l, link_opt])
                cc = float(links.at[l, "capital_cost"]) if "capital_cost" in links.columns else np.nan
                cost_out = cap * cc
                cap_out = f"< {th}" if (th is not None and cap < th) else cap

                rows.append({
                    "plant": plant,
                    "component": "link",
                    "asset": str(l),
                    "capacity": cap_out,
                    "Fixed cost (€/y)": cost_out,
                    "reference inlet": bus0,
                    "unit": norm_unit(u),
                    "threshold": th,
                })

        # Stores
        if store_opt:
            for s in alloc.get("stores", []) or []:
                if s not in stores.index:
                    continue
                bus = stores.at[s, "bus"]
                u = unit_of_bus(bus)
                th = threshold("store", u)
                cap = float(stores.at[s, store_opt])
                cc = float(stores.at[s, "capital_cost"]) if "capital_cost" in stores.columns else np.nan
                cost_out = cap * cc
                cap_out = f"< {th}" if (th is not None and cap < th) else cap

                rows.append({
                    "plant": plant,
                    "component": "store",
                    "asset": str(s),
                    "capacity": cap_out,
                    "Fixed cost (€/y)": cost_out,
                    "reference inlet": bus,
                    "unit": norm_unit(u),
                    "threshold": th,
                })

        # NEW: Storage Units
        if su_opt:
            for su in alloc.get("storage_units", []) or []:
                if su not in sus.index:
                    continue
                bus = sus.at[su, "bus"]
                u = unit_of_bus(bus)
                th = threshold("storage_unit", u)
                cap = float(sus.at[su, su_opt])

                cc = float(sus.at[su, "capital_cost"]) if "capital_cost" in sus.columns else np.nan
                cost_out = cap * cc

                cap_out = f"< {th}" if (th is not None and cap < th) else cap

                # Optional: implicit energy capacity (MWh) via max_hours if available
                e_cap = np.nan
                if "max_hours" in sus.columns:
                    mh = sus.at[su, "max_hours"]
                    if mh is not None and not pd.isna(mh):
                        e_cap = float(cap) * float(mh)

                rows.append({
                    "plant": plant,
                    "component": "storage_unit",
                    "asset": str(su),
                    "capacity": cap_out,
                    "energy_capacity": e_cap,
                    "Fixed cost (€/y)": cost_out,
                    "reference inlet": bus,
                    "unit": norm_unit(u),              # this is power unit
                    "threshold": th,
                })

    df = pd.DataFrame(rows)

    out = Path(file_path)
    if out.suffix.lower() != ".csv":
        out = out.with_suffix(".csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    df.attrs["chosen_scenario"] = str(chosen_scenario) if chosen_scenario is not None else None
    return df

# Filter very small capacities:
def filter_items_by_capacity_threshold(
    n,
    items,
    default_th=0.0,
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

    # NEW
    if kind == "StorageUnit":
        sus = _slice_df_first_scenario(n.storage_units)
        if sus is None or name not in sus.index or "bus" not in sus.columns:
            return ("", "")
        bus = sus.at[name, "bus"]
        return _bus_unit_and_carrier(n, bus)

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

    comp_order = {"Link": 0, "Store": 1, "Generator": 2, "StorageUnit" :3}
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

    items = []  # list of (label, sample_array)

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

    if not items:
        raise ValueError("No data to plot: all requested buses/links were missing or empty.")

    labels = [lab for lab, _ in items]
    data = [arr for _, arr in items]

    fig, ax = plt.subplots(figsize=(max(9, 0.45 * len(labels)), 4.6))
    vp = ax.violinplot(data, showmeans=True, showmedians=False, showextrema=True)

    ax.set_xticks(range(1, len(labels) + 1), labels, rotation=90)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)

    vp["cmeans"].set_color(mean_color)
    vp["cmeans"].set_linewidth(mean_linewidth)

    #vp["cmedians"].set_color(median_color)
    #vp["cmedians"].set_linewidth(median_linewidth)


    ax.text(0.02, 0.95, scen_txt, transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

    if handle_spikes in ("clip", "iqr"):
        scope_note = note_text + ("\n(floored at 0)" if floor_zero else "")
        ax.text(0.98, 0.98, scope_note, transform=ax.transAxes,
                ha="right", va="top", fontsize=9,
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

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
        field = it["field"]
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
                curve_specs.append({
                    "kind": kind,
                    "field": field,
                    "base_label": base_label,
                    "name": name,
                    "legend_label": leg,
                    "linestyle": ls,
                    "lw": lw0,
                })

    # Stable colors per (kind,name) across scenarios and modes
    uniq_keys, seen = [], set()
    for spec in curve_specs:
        key = (spec["kind"], spec["name"])
        if key not in seen:
            seen.add(key)
            uniq_keys.append(key)
    color_map = {k: cmap(i % cmap.N) for i, k in enumerate(uniq_keys)}

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
                    if s is not None and abs_links:
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
                        if s is not None and abs_links:
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
        ax.set_ylim(0, 1)
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
def heatmap_day_hour(series, ax, vmin=0, vmax=1, title="", cmap="viridis", show_months=True):
    """
    series: hourly pd.Series with DatetimeIndex
    Creates a heatmap with y=hour(0..23), x=day-of-year.
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        ax.set_title(title + " (missing)", fontsize=9)
        ax.axis("off")
        return None

    s = s[~s.index.duplicated(keep="first")]
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
        cand_union = sorted(cand_union)

        matches = _match_names(cand_union, selector)
        for name in matches:
            expanded.append({
                "row_label": f"{label}\n({name})",
                "kind": kind,
                "field": field,
                "name": name,
            })

    if not expanded:
        raise ValueError("No matching components found for the provided items/selectors.")

    # Function to get normalized series for a given (scenario, expanded row)
    def _get_norm_series(scen, row):
        kind = row["kind"]
        name = row["name"]
        field = row["field"]

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
            if s is not None and abs_links:
                s = pd.Series(s, copy=False).abs()

        elif kind == "Store":
            ts_df = getattr(n.stores_t, field, None) if field else getattr(n.stores_t, "e", None)
            s = _series_from_mi_cols(ts_df, scen, name) if ts_df is not None else None
            cap = _cap_from_component_table(comp_df, scen, name, ["e_nom_opt", "e_nom"])

        elif kind == "StorageUnit":
            # SOC time series
            ts_df = getattr(n.storage_units_t, field, None) if field else getattr(n.storage_units_t, "state_of_charge", None)
            s = _series_from_mi_cols(ts_df, scen, name) if ts_df is not None else None

            # energy capacity = p_nom * max_hours
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
        cf = (pd.to_numeric(s, errors="coerce") / cap).clip(lower=0.0, upper=1.0)
        return cf

    # compute stochastic "expected pattern" series as weighted day×hour matrix
    def _heatmap_day_hour_weighted(values, weights, ax, vmin=0, vmax=1, title="", cmap="viridis", show_months=True):
        """
        values, weights: pd.Series with DatetimeIndex aligned (hourly).
        Produces day×hour heatmap using weighted mean per (doy,hour).
        """
        v = pd.to_numeric(values, errors="coerce")
        w = pd.to_numeric(weights, errors="coerce")
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
                    im = heatmap_day_hour(
                        s if s is not None else pd.Series(dtype=float),
                        ax=ax,
                        vmin=vmin, vmax=vmax,
                        title=col_title,
                        cmap=cmap,
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
                        all_v.append(s)
                        all_w.append(w)

                    if all_v:
                        v_cat = pd.concat(all_v, axis=0)
                        w_cat = pd.concat(all_w, axis=0)
                    else:
                        v_cat = pd.Series(dtype=float)
                        w_cat = pd.Series(dtype=float)

                    col_title = stochastic_col_label if r == 0 else ""
                    im = _heatmap_day_hour_weighted(
                        v_cat, w_cat,
                        ax=ax,
                        vmin=vmin, vmax=vmax,
                        title=col_title,
                        cmap=cmap,
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
            im = heatmap_day_hour(
                s if s is not None else pd.Series(dtype=float),
                ax=ax,
                vmin=vmin, vmax=vmax,
                title=row["row_label"],
                cmap=cmap,
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

        # Give a little extra bottom margin for rotated month labels
        fig.subplots_adjust(bottom=0.12)

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
    series: hourly pd.Series with DatetimeIndex (actual values)
    norm: matplotlib Normalize/TwoSlopeNorm defining color scaling (capacity-based)
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        ax.set_title(title + " (missing)", fontsize=9)
        ax.axis("off")
        return None

    s = s[~s.index.duplicated(keep="first")]
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
        cand_union = sorted(cand_union)

        matches = _match_names(cand_union, selector)
        for name in matches:
            expanded.append({
                "row_label": f"{label}\n({name})",
                "kind": kind,
                "field": field,
                "name": name,
            })

    if not expanded:
        raise ValueError("No matching components found for the provided items/selectors.")

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
            if s is None or cap is None or cap <= 0:
                return NONE5
            cap = float(cap)
            norm = Normalize(vmin=0.0, vmax=cap)
            return pd.Series(s, copy=False), norm, cmap_pos, "Power", cap

        if kind == "Link":
            ts_df = getattr(n.links_t, field, None) if field else getattr(n.links_t, "p0", None)
            s = _series_from_mi_cols(ts_df, scen, name) if ts_df is not None else None
            cap = _cap_from_component_table(comp_df, scen, name, ["p_nom_opt", "p_nom"])
            if s is None or cap is None or cap <= 0:
                return NONE5
            s = pd.Series(s, copy=False)
            if abs_links:
                s = s.abs()
            cap = float(cap)
            norm = Normalize(vmin=0.0, vmax=cap)
            return s, norm, cmap_pos, "Power", cap

        if kind == "Store":
            ts_df = getattr(n.stores_t, field, None) if field else getattr(n.stores_t, "e", None)
            s = _series_from_mi_cols(ts_df, scen, name) if ts_df is not None else None
            cap = _cap_from_component_table(comp_df, scen, name, ["e_nom_opt", "e_nom"])
            if s is None or cap is None or cap <= 0:
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
                if s is None or cap is None or cap <= 0:
                    return NONE5
                cap = float(cap)
                norm = Normalize(vmin=0.0, vmax=cap)
                return pd.Series(s, copy=False), norm, cmap_pos, "Energy", cap

            # Dispatch (signed)
            if field in ("p", "dispatch"):
                ts_df = getattr(n.storage_units_t, "p", None)
                s = _series_from_mi_cols(ts_df, scen, name) if ts_df is not None else None
                cap = _cap_from_component_table(comp_df, scen, name, ["p_nom_opt", "p_nom"])
                if s is None or cap is None or cap <= 0:
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
            s, norm, cmap_use, _, _ = _get_series_and_norm(None, row)

            im = heatmap_day_hour_actual(
                s if s is not None else pd.Series(dtype=float),
                ax=ax,
                norm=norm if norm is not None else Normalize(0, 1),
                title=row["row_label"],
                cmap=cmap_use if cmap_use is not None else cmap_pos,
                show_months=(i // n_cols == n_rows - 1)
            )

        for j in range(n_plots, len(axes)):
            axes[j].set_visible(False)


        fig.suptitle(title, y=1.02)
        fig.subplots_adjust(bottom=0.12)

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

        for c in ["capex", "opex", "total"]:
            tmp[c] = tmp[c] * tmp["w"]

        expected_long = tmp.groupby(["group"], as_index=True)[["capex", "opex", "total"]].sum()
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
        out.groupby("scenario", as_index=False)[["capex", "opex", "total"]]
        .sum()
        .assign(group="total", unit="€/y")
    )

    totals["probability"] = np.nan
    if has_scenario_costs and scenario_weights is not None and len(scenario_weights) > 0:
        totals.loc[totals["scenario"] != "stochastic", "probability"] = (
            totals.loc[totals["scenario"] != "stochastic", "scenario"].map(w)
        )

    out = pd.concat([out, totals], ignore_index=True, sort=False)

    out = out[["scenario", "group", "capex", "opex", "total", "unit", "probability"]]

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

    # Pivot to wide: rows=scenario, cols=group, values=which
    df = costs_long[which].reset_index().pivot_table(
        index="scenario", columns="group", values=which, aggfunc="sum"
    ).fillna(0.0)

    if "total" in df.columns:
        df = df.drop(columns=["total"])

    # Add expected as an extra row (optional)
    if add_expected and expected_long is not None and not expected_long.empty:
        exp = expected_long[which].copy()
        # ensure all columns exist
        for g in df.columns:
            if g not in exp.index:
                exp.loc[g] = 0.0
        exp = exp.reindex(df.columns).fillna(0.0)

        # keep consistent with make_global_summary_costs() csv "stochastic"
        df.loc["stochastic"] = exp.values

    # Sort groups by total contribution (so legend/order is stable)
    group_order = df.sum(axis=0).sort_values(ascending=False).index
    df = df[group_order]

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

    # map to agent
    capex["group"] = capex.apply(lambda r: lookup.get((r["kind"], r["name"]), "Unallocated"), axis=1)
    opex["group"]  = opex.apply(lambda r: lookup.get((r["kind"], r["name"]), "Unallocated"), axis=1) if not opex.empty else "Unallocated"

    # aggregate (scenario, agent)
    capex_sa = capex.groupby(["scenario","group"], as_index=True)[["capex"]].sum()
    opex_sa  = opex.groupby(["scenario","group"],  as_index=True)[["opex"]].sum() if not opex.empty else None

    costs_long = capex_sa.join(opex_sa, how="outer").fillna(0.0)
    costs_long["total"] = costs_long["capex"] + costs_long["opex"]

    total_by_scenario = costs_long["total"].groupby(level="scenario").sum()

    # stochastic expected (probability-weighted)
    expected_long = None
    total_expected = None
    if include_expected and scen_w is not None and len(scen_w) > 0:
        tmp = costs_long.reset_index()
        tmp["w"] = tmp["scenario"].map(scen_w).fillna(0.0).astype(float)
        for c in ["capex","opex","total"]:
            tmp[c] = tmp[c] * tmp["w"]
        expected_long = tmp.groupby("group", as_index=True)[["capex","opex","total"]].sum()
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
        out.groupby("scenario", as_index=False)[["capex","opex","total"]]
           .sum()
           .assign(group="total", unit=unit)
    )
    totals["probability"] = np.nan
    if scen_w is not None and len(scen_w) > 0:
        mask = totals["scenario"] != "stochastic"
        totals.loc[mask, "probability"] = totals.loc[mask, "scenario"].map(scen_w)

    out = pd.concat([out, totals], ignore_index=True)
    out = out[["scenario","group","capex","opex","total","unit","probability"]]

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
    thresholds: dict,
    bus_list_mp: list[str],
    network_comp_allocation: Optional[dict] = None,
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
            thresholds=thresholds,
        )

        opt_cap_holder["obj"] = opt_cap

    def step_filter_items() -> None:
        items_f = filter_items_by_capacity_threshold(
            n,
            items,
            default_th=0.0,
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
        shadow_prices_violinplot_stoch(
            n,
            bus_list=bus_list_mp,
            folder=str(plot_folder),
            link_mc_items=[
                {"label": "Electricity price", "selector": {"contains": "DK1_to_El_"}},
                {"label": "NG price", "selector": {"regex": r"_NG boiler$"}},
            ],
            handle_spikes="clip",
            quantile_hi=0.98,
            n_draws=25000,
        )

        shadow_prices_ldc_stoch(
            n,
            bus_list=bus_list_mp,
            folder=str(plot_folder),
            link_mc_items=[
                {"label": "Electricity price", "selector": {"contains": "DK1_to_El_"}},
                {"label": "NG price", "selector": {"regex": r"_NG boiler$"}},
            ],
            handle_spikes="clip",
            quantile_hi=0.98,
            n_points=1001,
            fname="shd_prices_ldc.png",
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

    # ---------------- Run in order ----------------

    _safe_step("cost_by_carrier", step_cost_by_carrier)
    _safe_step("cost_by_agent", step_cost_by_agent)

    _safe_step("save_optimal_capacities", step_save_opt_caps)
    _safe_step("filter_items", step_filter_items)
    _safe_step("capacity_compare_sp_vs_ws", step_capacity_compare_sp_vs_ws)

    _safe_step("inputs_ldc_by_scenario", step_inputs_ldc)
    _safe_step("shadow_prices", step_shadow_prices)
    _safe_step("operation_plots", step_operation_plots)

    if failures:
        print(f"[plot/export] Finished with {len(failures)} failing step(s): {list(failures.keys())}")
    else:
        print("[plot/export] Finished successfully.")

    return failures
