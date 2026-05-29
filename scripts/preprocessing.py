# SPDX-License-Identifier: MIT
"""Energy-market data preprocessing and network input assembly.

This module contains two groups of functions:

**Data download and preprocessing** (called by :mod:`scripts.snakemake_preprocess`)

* :func:`pre_processing_energy_data` — entry point that downloads and stores
  all CSV inputs for a given energy-price year (electricity prices, CO₂
  emission intensities, natural gas prices, district-heating demand, and
  renewable capacity factors).

**Network input assembly** (called by :mod:`scripts.snakemake_prepare_inputs`)

* :func:`prepare_all_inputs` — loads the preprocessed CSVs for all scenario
  years and assembles the ``inputs_dict`` consumed by
  :func:`scripts.prepare_network.build_network`.

All CSV outputs are written to ``data/Inputs_{year}/`` (EU locations) or
``data/California/Inputs_{year}/`` (US locations), as determined by the
project coordinates in ``config.yaml``.

.. note::
   Leap-year days (Feb 29) are dropped to keep all years on the same
   8 760-hour snapshot index.
"""

import pandas as pd
import numpy as np
import requests
from pathlib import Path
from scripts import parameters as p
import os
from io import StringIO
import json
import time
from timezonefinder import TimezoneFinder
from entsoe import EntsoePandasClient
from datetime import datetime, timedelta
from scripts.config import (En_price_year,
                            EUR_to_DKK,
                            latitude,
                            longitude,
                            )

# ── Demand profiles ─────────────────────────────────────────────────────────
# Year-agnostic seasonal demand profiles live here (not in Inputs_{year}/).
_COMMON_DIR = Path("data/common")
_DEFAULT_PROFILE_PATH = _COMMON_DIR / "NG_demand_DK_profile.csv"


# ------ INPUTS PRE-PROCESSING ----

def GL_inputs_to_eff(GL_inputs: "pandas.DataFrame") -> "pandas.DataFrame":
    """Convert GreenLab energy/material flow table to PyPSA MultiLink efficiencies.

    Parameters
    ----------
    GL_inputs : pandas.DataFrame
        Raw GreenLab flow table read from ``GreenLab_Input_file.xlsx``
        (sheet ``"Overview_2"``).  Rows are bus names; columns are plant
        names.  Row ``"bus0"`` indicates the reference bus for each plant.

    Returns
    -------
    pandas.DataFrame
        Efficiency table with the same shape as *GL_inputs* (minus the
        ``"bus0"`` row and ``"Bus Unit"`` column).  Each value is the
        bus-to-bus efficiency relative to ``bus0``.  Zero entries are
        replaced with ``NaN`` to signal unused ports to PyPSA.

    Notes
    -----
    * ``(-)`` flow values are energy/material **consumed** by the plant.
    * ``(+)`` flow values are energy/material **produced** by the plant.
    """

    # NOTE: (-) refers to energy or material flow CONSUMED by the plant
    #      (+) refers to energy or material flow PRODUCED by the plant
    # Calculates Efficiencies for MultiLinks
    GL_eff = GL_inputs
    GL_eff = GL_eff.drop(columns='Bus Unit')  # drops not relevant columns
    GL_eff = GL_eff.drop(index='bus0')
    # bus-to-bus efficiency set with bus0 as reference (normalized)
    for j in list(GL_eff.columns.values):
        bus0_prc = GL_inputs.loc['bus0', j]
        bus0_val = GL_inputs.loc[bus0_prc, j]
        GL_eff.loc[:, j] = GL_eff.loc[:, j] / -bus0_val
        GL_eff[GL_eff == 0] = np.nan

    return GL_eff


def _load_profile_ts(
    path: "str | None",
    snapshots: "pd.DatetimeIndex | None" = None,
) -> "pd.Series":
    """Load a seasonal demand profile CSV and return an hourly Series.

    Parameters
    ----------
    path : str or None
        Path to a semicolon-separated CSV with a datetime index (daily or
        sub-daily) and one numeric column.  ``None`` falls back to the
        built-in NG DK profile at ``data/common/NG_demand_DK_profile.csv``.
    snapshots : pd.DatetimeIndex, optional
        Optimization snapshot index.  When provided the profile's year is
        remapped to the optimization year so that year-agnostic profiles
        (e.g. from a different historical year) align correctly.

    Returns
    -------
    pd.Series
        Hourly values (daily originals divided by 24).  Index is tz-naive.
    """
    src = Path(path) if path else _DEFAULT_PROFILE_PATH
    if not src.exists():
        raise FileNotFoundError(
            f"Demand profile not found: {src}\n"
            "Run `snakemake preprocess_inputs` first, or place a custom\n"
            "semicolon-separated CSV at that path."
        )
    df = pd.read_csv(src, sep=";", index_col=0)
    _s = df.iloc[:, 0].astype(float)
    _idx = pd.DatetimeIndex(_s.index)
    if _idx.tz is not None:
        _idx = _idx.tz_convert("UTC").tz_localize(None)
    _s.index = _idx
    _hi = pd.date_range(
        _s.index.min(),
        _s.index.max() + pd.Timedelta(days=1),
        freq="h",
        inclusive="left",
    )
    result = _s.reindex(_hi, method="ffill") / 24.0

    if snapshots is not None:
        opt_year = pd.DatetimeIndex(snapshots).year[0]
        profile_year = result.index.year[0]
        if profile_year != opt_year:
            result.index = result.index.map(lambda t: t.replace(year=opt_year))

    return result


def build_product_demand_ts(
    annual_demand: float,
    mode: str,
    snapshots: "pandas.DatetimeIndex",
    profile_ts: "pandas.Series | None" = None,
    n_bins: int = 1,
    flexibility_fraction: float = 0.0,
    store_buffer: float = 0.0,
    col_name: str = "demand MWh",
) -> "tuple[pandas.DataFrame, float | None]":
    """Build an hourly demand time series and the delivery-store ``e_nom_max``.

    Parameters
    ----------
    annual_demand : float
        Total demand over the full snapshot period (MWh or equivalent).
    mode : str
        ``"flat"``         — constant MW every hour.
        ``"profile"``      — continuous TS scaled to *annual_demand* via *profile_ts*.
        ``"bins_flat"``    — zero except at *n_bins* endpoints; equal delivery per bin.
        ``"bins_profile"`` — zero except at endpoints; delivery ∝ *profile_ts* integral
                             within each bin.
    snapshots : pandas.DatetimeIndex
        Network snapshot index (typically 8 760 hourly steps).
    profile_ts : pandas.Series, optional
        Reference time series for ``"profile"`` and ``"bins_profile"`` modes.
        Must cover the same period as *snapshots* (reindexed + forward-filled).
    n_bins : int
        Number of equal delivery intervals (only for ``"bins_*"`` modes).
        1 = single year-end delivery; 12 ≈ monthly; 52 ≈ weekly.
    flexibility_fraction : float
        For ``"flat"``/``"profile"`` modes: fraction of *annual_demand* used as
        ``e_nom_max`` for the delivery store.  0 → no store (rigid demand).
    store_buffer : float
        For ``"bins_profile"`` mode: extra headroom added on top of the largest
        bin delivery (e.g. 0.05 → 5 % capacity margin).
    col_name : str
        Column label in the returned DataFrame.

    Returns
    -------
    demand_df : pandas.DataFrame
        Single-column DataFrame indexed by *snapshots*.
    e_nom_max : float or None
        Delivery-store ``e_nom_max``.  ``None`` → no store (rigid demand).
    """
    n_snap = len(snapshots)

    if annual_demand <= 0:
        return pd.DataFrame({col_name: 0.0}, index=snapshots), None

    # ── Continuous modes ──────────────────────────────────────────────────────
    if mode == "flat":
        ts = pd.Series(annual_demand / n_snap, index=snapshots)
        e_nom_max = flexibility_fraction * annual_demand if flexibility_fraction > 0 else None
        return ts.to_frame(col_name), e_nom_max

    if mode == "profile":
        if profile_ts is None:
            raise ValueError("profile_ts is required for mode='profile'")
        prof = profile_ts.reindex(snapshots).ffill().bfill()
        total = prof.sum()
        if total <= 0:
            raise ValueError("profile_ts sums to zero — cannot scale to annual_demand")
        ts = prof / total * annual_demand
        e_nom_max = flexibility_fraction * annual_demand if flexibility_fraction > 0 else None
        return ts.to_frame(col_name), e_nom_max

    # ── Bin modes ─────────────────────────────────────────────────────────────
    if mode not in ("bins_flat", "bins_profile"):
        raise ValueError(
            f"Unknown demand mode '{mode}'. Choose: flat | profile | bins_flat | bins_profile"
        )

    ts = pd.Series(0.0, index=snapshots)

    # Divide snapshot index into n_bins equal parts and snap to actual timestamps
    endpoints = []
    for i in range(n_bins):
        end_idx = min(int(round((i + 1) * n_snap / n_bins)) - 1, n_snap - 1)
        endpoints.append(snapshots[end_idx])
    endpoints[-1] = snapshots[-1]  # guarantee last bin ends at final snapshot

    if mode == "bins_flat":
        delivery = annual_demand / n_bins
        for ep in endpoints:
            ts[ep] += delivery
        e_nom_max = float(delivery)  # store sized to one bin's delivery

    else:  # bins_profile
        if profile_ts is None:
            raise ValueError("profile_ts is required for mode='bins_profile'")
        prof = profile_ts.reindex(snapshots).ffill().bfill()
        total_prof = prof.sum()
        if total_prof <= 0:
            raise ValueError("profile_ts sums to zero — cannot compute bin weights")

        bin_deliveries = []
        prev_idx = 0
        for ep in endpoints:
            ep_idx = snapshots.get_loc(ep)
            bin_prof = prof.iloc[prev_idx : ep_idx + 1].sum()
            delivery = bin_prof / total_prof * annual_demand
            bin_deliveries.append(delivery)
            ts[ep] += delivery
            prev_idx = ep_idx + 1

        max_bin = max(bin_deliveries) if bin_deliveries else annual_demand
        e_nom_max = max_bin * (1.0 + store_buffer)

    return ts.to_frame(col_name), e_nom_max


def build_demands_TS(targets_dict: dict) -> dict:
    """Build hourly demand time series for bioCH₄, H₂ and methanol.

    Demand shape (flat, profile, or periodic bins) and delivery-store sizing are
    controlled by ``targets_dict`` keys set in ``config.yaml``.  No intermediate
    CSV files are written — all values are computed in-memory and returned.

    Seasonal profiles are loaded from CSV files under ``data/common/``.  Each
    product can point to its own profile via the ``*_profile`` key in
    ``targets``; ``null`` uses ``data/common/NG_demand_DK_profile.csv``.

    Parameters
    ----------
    targets_dict : dict
        Demand targets from ``config.yaml`` (``targets`` block).  Required keys:
        ``"demand_CH4"``, ``"demand_H2"``, ``"demand_meoh"``.  Optional shape
        keys: ``"CH4_demand_mode"``, ``"CH4_bins"``, ``"CH4_flexibility"``,
        ``"CH4_profile"``, and their ``H2_*`` / ``MeOH_*`` equivalents, plus
        ``"demand_store_buffer"``.

    Returns
    -------
    dict
        Keys: ``"bioCH4"``, ``"bioCH4_e_nom_max"``, ``"H2"``,
        ``"H2_e_nom_max"``, ``"meoh"``, ``"meoh_e_nom_max"``, ``"NG_DK"``.
    """
    snapshots     = p.hours_in_period
    store_buffer  = float(targets_dict.get("demand_store_buffer", 0.0))

    # ── bioCH4 ────────────────────────────────────────────────────────────────
    CH4_mode = targets_dict.get("CH4_demand_mode", "bins_flat")
    CH4_bins = int(targets_dict.get("CH4_bins", 1))
    CH4_flex = float(targets_dict.get("CH4_flexibility", 0.0))
    bioCH4_demand, bioCH4_e_nom_max = build_product_demand_ts(
        annual_demand        = targets_dict["demand_CH4"],
        mode                 = CH4_mode,
        snapshots            = snapshots,
        profile_ts           = _load_profile_ts(targets_dict.get("CH4_profile"), snapshots) if "profile" in CH4_mode else None,
        n_bins               = CH4_bins,
        flexibility_fraction = CH4_flex,
        store_buffer         = store_buffer,
        col_name             = "bioCH4 demand MWh",
    )

    # ── Methanol ──────────────────────────────────────────────────────────────
    MeOH_mode = targets_dict.get("MeOH_demand_mode", "bins_flat")
    MeOH_bins = int(targets_dict.get("MeOH_bins", 1))
    MeOH_flex = float(targets_dict.get("MeOH_flexibility", 0.0))
    Methanol_demand, Methanol_e_nom_max = build_product_demand_ts(
        annual_demand        = targets_dict["demand_meoh"],
        mode                 = MeOH_mode,
        snapshots            = snapshots,
        profile_ts           = _load_profile_ts(targets_dict.get("MeOH_profile"), snapshots) if "profile" in MeOH_mode else None,
        n_bins               = MeOH_bins,
        flexibility_fraction = MeOH_flex,
        store_buffer         = store_buffer,
        col_name             = "Methanol demand MWh",
    )

    # ── H2 ────────────────────────────────────────────────────────────────────
    H2_mode = targets_dict.get("H2_demand_mode", "bins_flat")
    H2_bins = int(targets_dict.get("H2_bins", 1))
    H2_flex = float(targets_dict.get("H2_flexibility", 0.0))

    H2_input_demand, H2_e_nom_max = build_product_demand_ts(
        annual_demand        = targets_dict["demand_H2"],
        mode                 = H2_mode,
        snapshots            = snapshots,
        profile_ts           = _load_profile_ts(targets_dict.get("H2_profile"), snapshots) if "profile" in H2_mode else None,
        n_bins               = H2_bins,
        flexibility_fraction = H2_flex,
        store_buffer         = store_buffer,
        col_name             = "H2_demand_MWh",
    )

    # ── Default NG_DK profile (kept in inputs_dict for downstream reference) ──
    try:
        NG_DK_h = _load_profile_ts(None, snapshots)
    except FileNotFoundError:
        NG_DK_h = None

    return {
        "bioCH4":           bioCH4_demand,
        "bioCH4_e_nom_max": bioCH4_e_nom_max,
        "H2":               H2_input_demand,
        "H2_e_nom_max":     H2_e_nom_max,
        "meoh":             Methanol_demand,
        "meoh_e_nom_max":   Methanol_e_nom_max,
        "NG_DK":            NG_DK_h,
    }


def load_input_data():
    """Load csv files and prepare Input Data to GL network"""

    GL_inputs = pd.read_excel(p.GL_input_file, sheet_name='Overview_2', index_col=0)
    GL_eff = GL_inputs_to_eff(GL_inputs)
    Elspotprices = pd.read_csv(p.El_price_input_file, sep=';', index_col=0)  # currency/MWh
    Elspotprices = Elspotprices.set_axis(p.hours_in_period)
    CO2_emiss_El = pd.read_csv(p.CO2emis_input_file, sep=';', index_col=0)  # kg/MWh CO2
    CO2_emiss_El = CO2_emiss_El.set_axis(p.hours_in_period)
    CF_wind = pd.read_csv(p.CF_wind_input_file, sep=';', index_col=0)  # MWh/h y
    CF_wind = CF_wind.set_axis(p.hours_in_period)
    CF_solar = pd.read_csv(p.CF_solar_input_file, sep=';', index_col=0)  # MWh/h y
    CF_solar = CF_solar.set_axis(p.hours_in_period)
    NG_price_year = pd.read_csv(p.NG_price_year_input_file, sep=';', index_col=0)  # MWh/h y
    NG_price_year = NG_price_year.set_axis(p.hours_in_period)
    DH_external_demand = pd.read_csv(p.DH_external_demand_input_file, sep=';', index_col=0)  # currency/MWh
    DH_external_demand = DH_external_demand.set_axis(p.hours_in_period)

    return GL_inputs, GL_eff, Elspotprices, CO2_emiss_El, CF_wind, CF_solar, NG_price_year, DH_external_demand


# ---- DEMANDS for H2, MeOH and El_DK1_GLS

# ----- EXTERNAL ENERGY MARKETS

def remove_feb_29(df):
    # Function to remove February 29 if it's a leap year, works on df and series
    # Check if the year is a leap year
    if any((df.index.month == 2) & (df.index.day == 29)):
        # Remove rows where the date is February 29
        df = df[~((df.index.month == 2) & (df.index.day == 29))]
    return df


BASE = "https://api.energidataservice.dk/dataset"

def download_energidata(dataset_name, start_date, end_date, sort_val=None, price_area=None,
                        limit=None, offset=None, timeout=60):
    """
    Robust download from energidataservice.dk returning a DataFrame.
    start_date/end_date formats allowed: yyyy, yyyy-MM, yyyy-MM-dd, yyyy-MM-ddTHH:mm  (DK local time) :contentReference[oaicite:2]{index=2}
    """
    url = f"{BASE}/{dataset_name}"
    params = {"start": start_date, "end": end_date}

    if sort_val:
        # pass without "sort=" prefix
        params["sort"] = sort_val

    if price_area:
        # API expects JSON object, values as arrays is the documented format :contentReference[oaicite:3]{index=3}
        params["filter"] = json.dumps({"PriceArea": [price_area]})

    if limit is not None:
        params["limit"] = int(limit)
    if offset is not None:
        params["offset"] = int(offset)

    r = requests.get(url, params=params, headers={"Accept": "application/json"}, timeout=timeout)

    # If this raises, you'll see the HTTP code (e.g. 429, 502, 504, 400, ...)
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        # Print a short snippet so you can see what the server actually returned (often HTML)
        snippet = (r.text or "")[:500]
        raise RuntimeError(f"HTTP {r.status_code} from Energi Data Service.\nURL: {r.url}\nBody snippet:\n{snippet}") from e

    # Only parse JSON if it looks like JSON
    ctype = r.headers.get("Content-Type", "")
    if "json" not in ctype.lower():
        snippet = (r.text or "")[:500]
        raise RuntimeError(f"Expected JSON but got Content-Type={ctype}\nURL: {r.url}\nBody snippet:\n{snippet}")

    payload = r.json()
    records = payload.get("records", [])
    return pd.json_normalize(records)


def download_dk_day_ahead_prices(
    start_date,
    end_date,
    price_area="DK1",
    timeout=60,
    resolution="1h",   # "native" or "1h"
    how="mean",            # for 1h: "mean" (default) or "last"
):
    """
    Downloads day-ahead prices across the dataset switch:
    - Elspotprices up to 2025-09-30
    - DayAheadPrices from 2025-10-01 onwards

    Returns a single normalized DF with:
      TimeUTC, TimeDK, PriceArea, SpotPriceEUR, SpotPriceDKK

    If resolution="1h", resamples everything to hourly in UTC (DST-safe).
    """
    cutoff = "2025-10-01T00:00"
    dfs = []

    # part A: old dataset (hourly)
    if start_date < cutoff:
        end_a = min(end_date, cutoff)
        df_a = download_energidata(
            dataset_name="Elspotprices",
            start_date=start_date,
            end_date=end_a,
            sort_val="HourDK asc",
            price_area=price_area,
            limit=0,
            timeout=timeout,
        )
        if not df_a.empty:
            df_a = df_a.rename(columns={
                "HourUTC": "TimeUTC",
                "HourDK": "TimeDK",
                "SpotPriceEUR": "SpotPriceEUR",
                "SpotPriceDKK": "SpotPriceDKK",
            })
            df_a = df_a[["TimeUTC", "TimeDK", "PriceArea", "SpotPriceEUR", "SpotPriceDKK"]]
            dfs.append(df_a)

    # part B: new dataset (15-min from Oct 2025)
    if end_date > cutoff:
        start_b = max(start_date, cutoff)
        df_b = download_energidata(
            dataset_name="DayAheadPrices",
            start_date=start_b,
            end_date=end_date,
            sort_val="TimeUTC asc",
            price_area=price_area,
            limit=0,
            timeout=timeout,
        )
        if not df_b.empty:
            df_b = df_b.rename(columns={
                "DayAheadPriceEUR": "SpotPriceEUR",
                "DayAheadPriceDKK": "SpotPriceDKK",
            })
            df_b = df_b[["TimeUTC", "TimeDK", "PriceArea", "SpotPriceEUR", "SpotPriceDKK"]]
            dfs.append(df_b)

    out = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    if out.empty:
        return out

    # ---- Normalize datetimes ----
    # Parse to datetime; keep "naive" timestamps as provided by API
    out["TimeUTC"] = pd.to_datetime(out["TimeUTC"])
    out["TimeDK"]  = pd.to_datetime(out["TimeDK"])

    # ---- Optional: resample to hourly ----
    if resolution.lower() in ("1h", "hour", "hourly"):
        out = out.sort_values(["PriceArea", "TimeUTC"]).set_index("TimeUTC")

        if how == "last":
            agg = "last"
        else:
            agg = "mean"  # default

        # Resample per area
        resampled = (
            out.groupby("PriceArea")[["SpotPriceEUR", "SpotPriceDKK"]]
               .resample("1h")
               .agg(agg)
               .reset_index()
        )

        # Recreate TimeDK from TimeUTC in a DST-safe way:
        # interpret TimeUTC as UTC, convert to Europe/Copenhagen, then drop tz.
        resampled["TimeDK"] = (
            resampled["TimeUTC"]
            .dt.tz_localize("UTC")
            .dt.tz_convert("Europe/Copenhagen")
            .dt.tz_localize(None)
        )

        # Keep column order
        out = resampled[["TimeUTC", "TimeDK", "PriceArea", "SpotPriceEUR", "SpotPriceDKK"]]

    else:
        out = out[["TimeUTC", "TimeDK", "PriceArea", "SpotPriceEUR", "SpotPriceDKK"]].sort_values(["PriceArea", "TimeUTC"])

    return out



tf = TimezoneFinder()

RN_MAX_DATE = pd.Timestamp("2024-12-31").date()
RN_LAST_END_EXCL_UTC = pd.Timestamp("2025-01-01 00:00", tz="UTC")  # end-exclusive


def _rn_get(session, url, params, max_retries=5):
    """GET with automatic retry/backoff on RN 429 rate-limit responses."""
    for attempt in range(max_retries):
        r = session.get(url, params=params)
        if r.status_code == 429:
            wait = 2 ** attempt  # 1, 2, 4, 8, 16 s
            time.sleep(wait)
            continue
        return r
    return r  # return last response if all retries exhausted


def retrieve_renewable_capacity_factors(
    token,
    start_date,
    end_date,
    latitude,
    longitude,
    dataset="merra2",
    return_tz="UTC",  # "UTC" or "local"
):
    """
    Call RN for PV and wind.
    Inputs start_date/end_date are LOCAL time for the location timezone.

    RN API only supports date_to <= 2024-12-31, and date_from must also be <= 2024-12-31.
    We clamp date_to; if date_from is beyond RN_MAX_DATE we raise (wrapper should avoid calling).
    """
    api_base = "https://www.renewables.ninja/api/"
    s = requests.session()
    s.headers = {"Authorization": "Token " + token}

    tzname = tf.timezone_at(lat=latitude, lng=longitude)
    if tzname is None:
        raise ValueError(f"Could not determine timezone for lat={latitude}, lon={longitude}")

    # Inputs are LOCAL time (safe parsing)
    start_local = pd.Timestamp(start_date)
    end_local = pd.Timestamp(end_date)

    if start_local.tzinfo is None:
        start_local = start_local.tz_localize(tzname)
    else:
        start_local = start_local.tz_convert(tzname)

    if end_local.tzinfo is None:
        end_local = end_local.tz_localize(tzname)
    else:
        end_local = end_local.tz_convert(tzname)

    # RN date window (date-based). Clamp to RN coverage.
    date_from_d = start_local.date()
    date_to_d = end_local.date()

    if date_from_d > RN_MAX_DATE:
        raise ValueError(f"RN cannot serve date_from after {RN_MAX_DATE} (got {date_from_d})")
    if date_to_d > RN_MAX_DATE:
        date_to_d = RN_MAX_DATE

    date_from = pd.Timestamp(date_from_d).strftime("%Y-%m-%d")
    date_to = pd.Timestamp(date_to_d).strftime("%Y-%m-%d")

    # --- PV ---
    optimal_tilt = latitude * 0.87 + 3.1
    r = _rn_get(s, api_base + "data/pv", params={
        "lat": latitude,
        "lon": longitude,
        "date_from": date_from,
        "date_to": date_to,
        "dataset": dataset,
        "capacity": 1.0,
        "system_loss": 0.1,
        "tracking": 0,
        "tilt": optimal_tilt,
        "azim": 180,
        "format": "json",
    })
    if r.status_code != 200:
        raise RuntimeError(f"RN pv failed {r.status_code}: {r.text[:2000]}")
    parsed = r.json()
    CF_solar = pd.read_json(StringIO(json.dumps(parsed["data"])), orient="index")
    CF_solar.rename(columns={CF_solar.columns[0]: "CF solar"}, inplace=True)

    # --- Wind ---
    r = _rn_get(s, api_base + "data/wind", params={
        "lat": latitude,
        "lon": longitude,
        "date_from": date_from,
        "date_to": date_to,
        "capacity": 1.0,
        "height": 100,
        "turbine": "Vestas V80 2000",
        "format": "json",
    })
    if r.status_code != 200:
        raise RuntimeError(f"RN wind failed {r.status_code}: {r.text[:2000]}")
    parsed = r.json()
    CF_wind = pd.read_json(StringIO(json.dumps(parsed["data"])), orient="index")
    CF_wind.rename(columns={CF_wind.columns[0]: "CF wind"}, inplace=True)

    # RN timestamps are UTC
    CF_solar.index = pd.to_datetime(CF_solar.index).tz_localize("UTC")
    CF_wind.index = pd.to_datetime(CF_wind.index).tz_localize("UTC")

    # Clip in UTC using local intent converted to UTC
    start_utc = start_local.tz_convert("UTC")
    end_utc = end_local.tz_convert("UTC")
    CF_solar = CF_solar.loc[(CF_solar.index >= start_utc) & (CF_solar.index < end_utc)]
    CF_wind = CF_wind.loc[(CF_wind.index >= start_utc) & (CF_wind.index < end_utc)]

    if return_tz == "local":
        CF_solar.index = CF_solar.index.tz_convert(tzname)
        CF_wind.index = CF_wind.index.tz_convert(tzname)

    return CF_solar, CF_wind


def retrieve_renewable_capacity_factors_with_fallback(
    RN_token,
    start_date,
    end_date,
    latitude,
    longitude,
    dataset="merra2",
    out_timezone="Europe/Copenhagen",
):
    """
    Wrapper when inputs are ALWAYS LOCAL time for the location.

    Rule:
    - If start_local.date() >= 2025-01-01, do NOT call RN at all (RN would reject).
      Instead replay all hours from 2024 and shift forward.
    - If start_local.date() <= 2024-12-31, fetch that portion from RN (clamped),
      and replay the rest if needed.

    Output:
    - Complete hourly series on an expected UTC grid, then converted to out_timezone.
    """
    tzname = tf.timezone_at(lat=latitude, lng=longitude)
    if tzname is None:
        raise ValueError(f"Could not determine timezone for lat={latitude}, lon={longitude}")

    # Inputs are LOCAL time
    start_local = pd.Timestamp(start_date)
    end_local = pd.Timestamp(end_date)

    if start_local.tzinfo is None:
        start_local = start_local.tz_localize(tzname)
    else:
        start_local = start_local.tz_convert(tzname)

    if end_local.tzinfo is None:
        end_local = end_local.tz_localize(tzname)
    else:
        end_local = end_local.tz_convert(tzname)

    # Expected hourly UTC index for [start, end)
    start_utc = start_local.tz_convert("UTC")
    end_utc = end_local.tz_convert("UTC")
    expected_utc = pd.date_range(start_utc, end_utc, freq="h", inclusive="left", tz="UTC")

    parts_solar, parts_wind = [], []

    # ---------- Supported part: only if local start date is within RN coverage ----------
    if start_local.date() <= RN_MAX_DATE:
        # cap the RN portion to end of RN coverage in local time
        rn_end_local = RN_LAST_END_EXCL_UTC.tz_convert(tzname)
        end_ok_local = min(end_local, rn_end_local)

        if start_local < end_ok_local:
            CF_s, CF_w = retrieve_renewable_capacity_factors(
                RN_token,
                start_local.strftime("%Y-%m-%d %H:%M"),
                end_ok_local.strftime("%Y-%m-%d %H:%M"),
                latitude,
                longitude,
                dataset=dataset,
                return_tz="UTC",
            )
            parts_solar.append(CF_s)
            parts_wind.append(CF_w)

    # ---------- Overflow: anything beyond RN coverage (or all of it if start is 2025+) ----------
    # Determine overflow in UTC
    overflow_start_utc = max(start_utc, RN_LAST_END_EXCL_UTC)
    if end_utc > overflow_start_utc:
        overflow_idx = pd.date_range(overflow_start_utc, end_utc, freq="h", inclusive="left", tz="UTC")
        n = len(overflow_idx)

        mapped_end_utc = RN_LAST_END_EXCL_UTC
        mapped_start_utc = mapped_end_utc - pd.Timedelta(hours=n)

        mapped_start_local = mapped_start_utc.tz_convert(tzname).strftime("%Y-%m-%d %H:%M")
        mapped_end_local = mapped_end_utc.tz_convert(tzname).strftime("%Y-%m-%d %H:%M")

        CF_s_24, CF_w_24 = retrieve_renewable_capacity_factors(
            RN_token,
            mapped_start_local,
            mapped_end_local,
            latitude,
            longitude,
            dataset=dataset,
            return_tz="UTC",
        )

        shift = overflow_start_utc - mapped_start_utc
        CF_s_24 = CF_s_24.copy()
        CF_w_24 = CF_w_24.copy()
        CF_s_24.index = CF_s_24.index + shift
        CF_w_24.index = CF_w_24.index + shift

        parts_solar.append(CF_s_24)
        parts_wind.append(CF_w_24)

    # Combine + force complete grid
    CF_solar_utc = pd.concat(parts_solar).sort_index().reindex(expected_utc)
    CF_wind_utc = pd.concat(parts_wind).sort_index().reindex(expected_utc)

    # Fix edge NaNs from date-based RN windows
    CF_solar_utc = CF_solar_utc.ffill().bfill()
    CF_wind_utc = CF_wind_utc.ffill().bfill()

    # Convert to requested output timezone
    CF_solar = CF_solar_utc.copy()
    CF_wind = CF_wind_utc.copy()
    CF_solar.index = CF_solar.index.tz_convert(out_timezone)
    CF_wind.index = CF_wind.index.tz_convert(out_timezone)

    return CF_solar, CF_wind


def retrive_entsoe_el_demand(API_KEY, start_day, end_day, country_code):
    """function that retrives historical el demand with hourly resolution from a specific bidding zone"""
    # NOTE: list of country codes available here: https://github.com/EnergieID/entsoe-py/blob/master/entsoe/mappings.py

    client = EntsoePandasClient(api_key= API_KEY)

    start = pd.Timestamp(start_day, tz='Europe/Brussels')
    end = pd.Timestamp(end_day, tz='Europe/Brussels')

    ts = client.query_load(country_code, start=start, end=end)

    return ts


def pre_processing_energy_data(year: int = None) -> None:
    """Download and preprocess all energy-market inputs for one price year.

    Fetches data from the Energi Data Service API (electricity spot prices,
    CO₂ emission intensities, natural gas prices, district-heating demand)
    and from the Renewables.ninja API (wind and solar capacity factors), then
    writes all results as semicolon-delimited CSV files to
    ``data/Inputs_{year}/`` (EU) or ``data/California/Inputs_{year}/`` (US).

    A ``"HourDK"`` sorted, leap-year-stripped hourly index is enforced on all
    outputs so they align with the model snapshot index from
    :func:`scripts.helpers.build_snapshots`.

    Parameters
    ----------
    year : int, optional
        Energy-price year to preprocess (e.g. ``2023``).  Defaults to
        ``En_price_year`` from ``config.yaml`` when *None*.

    Returns
    -------
    None
        All outputs are written to CSV files; nothing is returned.

    Raises
    ------
    requests.HTTPError
        If an API request fails after retries.
    RuntimeError
        If the downloaded time series cannot be aligned to the expected
        8 760-hour snapshot index.

    Notes
    -----
    * DK electricity prices are downloaded in DKK and converted to EUR
      using ``EUR_to_DKK`` from ``config.yaml``.
    * Leap days (Feb 29) are dropped to maintain a uniform 8 760-hour year.
    * Some CSV outputs are not used by every network configuration but are
      always written to avoid downstream ``FileNotFoundError`` exceptions.
    """
    # --- Year-specific setup (supports multi-year stochastic preprocessing) ---
    from scripts.helpers import build_snapshots, is_eu_or_us
    _year = int(year) if year is not None else En_price_year
    hours_in_period, _, _ = build_snapshots(_year)
    if is_eu_or_us(p.latitude, p.longitude) == 'EU':
        _folder = f'data/Inputs_{_year}'
    else:
        _folder = f'data/California/Inputs_{_year}'
    os.makedirs(_folder, exist_ok=True)
    El_price_input_file           = f'{_folder}/Elspotprices_input.csv'
    CO2emis_input_file            = f'{_folder}/CO2emis_input.csv'
    NG_price_year_input_file      = f'{_folder}/NG_price_year_input.csv'
    DH_external_demand_input_file = str(_COMMON_DIR / 'DH_external_demand_input.csv')
    CF_wind_input_file            = f'{_folder}/CF_wind.csv'
    CF_solar_input_file           = f'{_folder}/CF_solar.csv'
    # -------------------------------------------------------------------------
    """ Dates"""
    dates = hours_in_period.date
    start_date = dates[0].strftime("%Y-%m-%d")
    end_date = (dates[-1] + timedelta(days=1)).strftime("%Y-%m-%d")

    '''El spot prices DK1 - input DKK/MWh or EUR/MWh'''
    if not Path(El_price_input_file).exists():
        Elspotprices_data = download_dk_day_ahead_prices(
            start_date=start_date,
            end_date=end_date,
            price_area=p.price_area,
            timeout=60,
            resolution="1h",  # "native" or "1h"
            how="mean")

        #Elspotprices_data = download_energidata(dataset_name, p.start_date, p.end_date, sort_val, p.filter_area)
        Elspotprices = Elspotprices_data[['TimeDK', 'SpotPrice' + 'EUR']].copy()
        Elspotprices.rename(columns={'SpotPrice' + 'EUR': 'SpotPrice'}, inplace=True)
        Elspotprices['TimeDK'] = pd.to_datetime(Elspotprices['TimeDK'])
        Elspotprices.set_index('TimeDK', inplace=True)
        Elspotprices = remove_feb_29(Elspotprices)
        Elspotprices.index.name = None
        Elspotprices.to_csv(El_price_input_file, sep=';')  # currency/MWh
    else:
        print(f"[preprocess] Skipping Elspotprices download — {El_price_input_file} already exists.")

    '''CO2 emission from El Grid DK1'''
    # DeclarationEmissionHour was removed from the API; DeclarationGridEmission covers all years
    if not Path(CO2emis_input_file).exists():
        CO2emis_data = download_energidata(
            dataset_name='DeclarationGridEmission',
            start_date=start_date,
            end_date=end_date,
            sort_val="HourDK asc",
            price_area=p.price_area,
            limit=0
        )
        CO2_emiss_El = CO2emis_data.query("FuelAllocationMethod == '125%'")[['HourDK', 'CO2PerkWh']].copy()

        CO2_emiss_El['CO2PerkWh'] = CO2_emiss_El['CO2PerkWh'] / 1000  # t/MWh
        CO2_emiss_El.rename(columns={'CO2PerkWh': 'CO2PerMWh'}, inplace=True)
        CO2_emiss_El['HourDK'] = pd.to_datetime(CO2_emiss_El['HourDK'])
        CO2_emiss_El.set_index('HourDK', inplace=True)
        CO2_emiss_El = remove_feb_29(CO2_emiss_El)
        CO2_emiss_El.to_csv(CO2emis_input_file, sep=';')
    else:
        print(f"[preprocess] Skipping CO2 emissions download — {CO2emis_input_file} already exists.")

    # NG prices depending on the year
    ''' NG prices prices in DKK/kWh or EUR/kWH'''
    if not Path(NG_price_year_input_file).exists():
        if _year <= 2022:
            # due to different structure of Energinet dataset for the year 2019 and 2022
            dataset_name = 'GasMonthlyNeutralPrice'
            #sort_val = 'sort=Month%20ASC'
            filter_area = ''
            sort_val = "Month ASC"  # 'sort=HourDK%20asc'
            NG_price_year = download_energidata(
                dataset_name=dataset_name,
                start_date=start_date,  # "2025-01-01",
                end_date=end_date,  # "2026-01-01",
                sort_val=sort_val,
                price_area='',
                limit=0
            )
            #NG_price_year = download_energidata(dataset_name, p.start_date, p.end_date, sort_val, filter_area)
            NG_price_col_name = 'Neutral gas price ' + 'EUR' + '/MWh'
            NG_price_year.rename(columns={'MonthlyNeutralGasPriceDKK_kWh': NG_price_col_name}, inplace=True)
            NG_price_year.rename(columns={'Month': 'HourDK'}, inplace=True)
            NG_price_year['HourDK'] = pd.to_datetime(NG_price_year['HourDK']).dt.tz_localize(None)
            NG_price_year.set_index('HourDK', inplace=True)
            NG_price_year[NG_price_col_name] = NG_price_year[NG_price_col_name] * 1000 / EUR_to_DKK  # coversion to €/MWh
            last_rows3 = pd.DataFrame(
                {'HourDK': hours_in_period[-1:len(hours_in_period)], NG_price_col_name: NG_price_year.iloc[-1, 0]})
            last_rows3.set_index('HourDK', inplace=True)
            NG_price_year = pd.concat([NG_price_year, last_rows3])
            NG_price_year = NG_price_year.asfreq('h', method='ffill')

        elif _year > 2022:
            # due to different structure of Energinet dataset for the year 2019 and 2022
            dataset_name = 'GasDailyBalancingPrice'
            #sort_val = 'sort=GasDay%20ASC'
            #filter_area = ''
            sort_val = "GasDay ASC"  # 'sort=HourDK%20asc'
            THE_daily_NG_prices = download_energidata(
                dataset_name=dataset_name,
                start_date=start_date,  # "2025-01-01",
                end_date=end_date,  # "2026-01-01",
                sort_val=sort_val,
                price_area='',
                limit=0
            )

            # --- Compute EUR/MWh
            THE_daily_NG_prices["THE_NG_pricesEUR_MWh"] = (
                    THE_daily_NG_prices["THEPriceDKK_kWh"] * 1000
                    / THE_daily_NG_prices["ExchangeRateEUR_DKK"] * 100
            )

            # --- Rename GasDay -> HourDK and parse datetime once
            THE_daily_NG_prices = THE_daily_NG_prices.rename(columns={"GasDay": "HourDK"})
            THE_daily_NG_prices["HourDK"] = pd.to_datetime(THE_daily_NG_prices["HourDK"], errors="coerce")

            # Optional: if GasDay is a date (00:00), ensure it is normalized
            # (doesn't hurt if it's already at midnight)
            THE_daily_NG_prices["HourDK"] = THE_daily_NG_prices["HourDK"].dt.floor("D")

            # --- Make index and keep it timezone-naive to match p.hours_in_period (also tz-naive)
            THE_daily_NG_prices = THE_daily_NG_prices.set_index("HourDK").sort_index()
            THE_daily_NG_prices.index = THE_daily_NG_prices.index.tz_localize(None)

            # --- Reindex to your full hourly index and forward fill
            hours = pd.DatetimeIndex(hours_in_period)  # ensures it's a DatetimeIndex
            THE_daily_NG_prices = THE_daily_NG_prices.reindex(hours).ffill()

            # --- Final series
            NG_price_year = THE_daily_NG_prices[["THE_NG_pricesEUR_MWh"]].copy()

        NG_price_year = remove_feb_29(NG_price_year)
        NG_price_year = NG_price_year.interpolate(method='linear')
        NG_price_year.to_csv(NG_price_year_input_file, sep=';')  # €/MWh
    else:
        print(f"[preprocess] Skipping NG price download — {NG_price_year_input_file} already exists.")

    '''  Estimated NG Demand DK '''
    # source: https://www.energidataservice.dk/tso-gas/Gasflow
    # used to create a profile for H2 demand - if required.
    _COMMON_DIR.mkdir(parents=True, exist_ok=True)
    if not _DEFAULT_PROFILE_PATH.exists():
        dataset_name = 'Gasflow'
        sort_val = "GasDay"  # 'sort=HourDK%20asc'
        NG_demand_DK_data = download_energidata(
            dataset_name=dataset_name,
            start_date=start_date,  # "2025-01-01",
            end_date=end_date,  # "2026-01-01",
            sort_val=sort_val,
            price_area='',
            limit=0
        )
        #NG_demand_DK_data = download_energidata(dataset_name, start_date, end_date, sort_val, filter_area)
        NG_demand_DK = NG_demand_DK_data[['GasDay', 'KWhToDenmark']].copy()
        NG_demand_DK['KWhToDenmark'] = NG_demand_DK['KWhToDenmark'] / -1000  # kWh-> MWh
        NG_demand_DK.rename(columns={'KWhToDenmark': 'NG Demand DK MWh'}, inplace=True)
        NG_demand_DK['GasDay'] = pd.to_datetime(NG_demand_DK['GasDay'])
        NG_demand_DK['GasDay'] = pd.to_datetime(NG_demand_DK['GasDay']).dt.tz_localize(None)
        NG_demand_DK.set_index('GasDay', inplace=True)
        NG_demand_DK = remove_feb_29(NG_demand_DK)
        # Save to data/common/ as the year-agnostic seasonal demand profile
        NG_demand_DK.to_csv(_DEFAULT_PROFILE_PATH, sep=';')
    else:
        print(f"[preprocess] Skipping NG demand download — {_DEFAULT_PROFILE_PATH} already exists.")

    '''District heating data'''
    # Profile is derived from fixed 2019 Skive weather data — year-agnostic.
    # Skip if already present in data/common/ to avoid re-reading local DMI files
    # on non-EU (e.g. California) machines where p.DH_data_folder may be absent.
    if not Path(DH_external_demand_input_file).exists():
        # https://www.dmi.dk/friedata/observationer/
        data_folder = p.DH_data_folder
        name_files = os.listdir(data_folder)
        DH_Skive = pd.DataFrame()

        for name in name_files:
            df_temp_2 = pd.read_csv(os.path.join(data_folder, name), sep=';', usecols=['DateTime', 'Middeltemperatur'])
            DH_Skive = pd.concat([DH_Skive, df_temp_2])

        DH_Skive = DH_Skive.drop_duplicates(subset='DateTime', keep='first')
        DH_Skive = DH_Skive.sort_values(by=['DateTime'], ascending=True)
        DH_Skive['DateTime'] = pd.to_datetime(DH_Skive['DateTime'])
        DH_Skive['DateTime'] = pd.to_datetime(DH_Skive['DateTime'].dt.strftime("%Y-%m-%d %H:%M:%S+00:00"))
        hours_in_2019 = pd.date_range('2019-01-01T00:00' + 'Z', '2020-01-01T00:00' + 'Z', freq='h')
        hours_in_2019 = hours_in_2019.drop(hours_in_2019[-1])
        DH_Skive = DH_Skive.set_index("DateTime").reindex(hours_in_2019)

        DH_max_capacity = p.DH_Skive_Capacity  # MW
        # source: https://ens.dk/sites/ens.dk/files/Statistik/denmarks_heat_supply_2020_eng.pdf
        DH_Tamb_min = p.DH_Tamb_min  # minimum outdoor temp --> maximum Capacity Factor
        DH_Tamb_max = p.DH_Tamb_max  # maximum outdoor temp--> capacity Factor = 0
        CF_DH = (DH_Tamb_max - DH_Skive['Middeltemperatur'].values) / (DH_Tamb_max - DH_Tamb_min)
        CF_DH[CF_DH < 0] = 0
        DH_Skive['Capacity Factor DH'] = CF_DH
        # adjust for base load in summer months due to sanitary water
        # assumption: mean heat load in January/July = 6 (from Aarhus data).
        DH_CFmean_Jan = np.mean(DH_Skive.loc['2019-01', 'Capacity Factor DH'])
        DH_CFbase_load = DH_CFmean_Jan / 4
        DH_Skive['Capacity Factor DH'] = DH_Skive['Capacity Factor DH'] + DH_CFbase_load
        DH_Skive['DH demand MWh'] = DH_Skive[
                                        'Capacity Factor DH'] * DH_max_capacity
        DH_Skive = remove_feb_29(DH_Skive)
        DH_Skive = DH_Skive.set_axis(hours_in_period)
        DH_Skive = DH_Skive.interpolate(method='linear')
        DH_Skive.to_csv(DH_external_demand_input_file, sep=';')  # MWh/h

    '''Onshore Wind and Solar Capacity Factors'''
    # Download CF for wind and solar corresponding to the energy year
    if not (Path(CF_wind_input_file).exists() and Path(CF_solar_input_file).exists()):
        if not p.RN_token:
            raise RuntimeError(
                f"CF_wind.csv / CF_solar.csv are missing for year {_year} and RN_TOKEN is not set.\n"
                f"Either copy .env.example → .env and add your Renewables.ninja token,\n"
                f"or place pre-downloaded CF files in {_folder}/"
            )
        # TODO remove fallback function: when RN data for 2025 are available
        #CF_solar, CF_wind = retrieve_renewable_capacity_factors(p.RN_token, start_date, end_date, latitude, longitude)
        CF_solar, CF_wind = retrieve_renewable_capacity_factors_with_fallback(
            p.RN_token,
            start_date,
            end_date,
            latitude,
            longitude,
        )
        CF_wind = remove_feb_29(CF_wind)
        CF_solar = remove_feb_29(CF_solar)
        CF_wind.to_csv(CF_wind_input_file, sep=';')  # kg/MWh
        CF_solar.to_csv(CF_solar_input_file, sep=';')  # kg/MWh
    else:
        print(f"[preprocess] Skipping CF download — {CF_wind_input_file} and {CF_solar_input_file} already exist.")

    return


# ---- Pre-processing for PyPSA network

def prepare_all_inputs(targets_dict: dict, CO2_cost: float,
                       CO2_cost_ref_year: float, max_RE_to_grid: float) -> dict:
    """Load preprocessed CSVs and assemble the network input dictionary.

    Calls all lower-level loaders and demand builders and packages their
    outputs into the single ``inputs_dict`` consumed by
    :func:`scripts.prepare_network.build_network`.

    Parameters
    ----------
    targets_dict : dict
        Demand targets keyed by carrier name (``"demand_H2"``,
        ``"demand_CH4"``, ``"demand_meoh"``) and demand driver
        (``"driver"``).  Corresponds to the ``targets`` block in
        ``config.yaml``.
    CO2_cost : float
        CO₂ price in EUR/t used for the current optimisation scenario.
    CO2_cost_ref_year : float
        CO₂ price in the reference (historical) year, used to scale
        time-series cost signals.
    max_RE_to_grid : float
        Maximum fraction of renewable electricity that may be exported to
        the grid (0–1).

    Returns
    -------
    dict
        ``inputs_dict`` with the following keys (each value is a
        :class:`pandas.DataFrame` or scalar):

        * ``"GL_inputs"`` — GreenLab energy/material flow table
        * ``"GL_eff"`` — bus-to-bus efficiency DataFrame
        * ``"Elspotprices"`` — hourly electricity spot prices (EUR/MWh)
        * ``"CO2_emiss_El"`` — hourly CO₂ emission intensity (kg/MWh)
        * ``"CF_wind"`` — hourly wind capacity factors
        * ``"CF_solar"`` — hourly solar capacity factors
        * ``"NG_price_year"`` — hourly natural gas price (EUR/MWh)
        * ``"DH_external_demand"`` — hourly district-heating demand (MWh)
        * ``"demands"`` — sub-dict with H₂, bioCH₄ and MeOH demand series
        * ``"CO2_cost"`` — CO₂ price scalar (EUR/t)
        * ``"max_RE_to_grid"`` — grid export fraction scalar
    """

    # load the inputs form CSV files
    GL_inputs, GL_eff, Elspotprices, CO2_emiss_El, CF_wind, CF_solar, NG_price_year, DH_external_demand = load_input_data()

    '''Build all demands'''
    # build demands TS (considers targets_dict['driver'])
    demands = build_demands_TS(targets_dict)
    NG_DK = demands['NG_DK']

    H2_input_demand   = demands['H2']
    bioCH4_demand     = demands['bioCH4']
    Methanol_demand   = demands['meoh']

    inputs_dict = {
        'GL_inputs':               GL_inputs,
        'GL_eff':                  GL_eff,
        'Elspotprices':            Elspotprices,
        'CO2_emiss_El':            CO2_emiss_El,
        'bioCH4_demand':           bioCH4_demand,
        'bioCH4_store_e_nom_max':  demands['bioCH4_e_nom_max'],
        'CF_wind':                 CF_wind,
        'CF_solar':                CF_solar,
        'NG_price_year':           NG_price_year,
        'Methanol_input_demand':   Methanol_demand,
        'Methanol_store_e_nom_max': demands['meoh_e_nom_max'],
        'NG_demand_DK':            NG_DK,
        'DH_external_demand':      DH_external_demand,
        'H2_input_demand':         H2_input_demand,
        'H2_store_e_nom_max':      demands['H2_e_nom_max'],
        'CO2 cost':                CO2_cost,
        'CO2 cost ref year':       CO2_cost_ref_year,
        'max_RE_to_grid':          max_RE_to_grid,
    }

    if targets_dict["driver"] == "price":
        idx = Elspotprices.index  # <- align with scenario/year data

        prices = {
            "price_H2": targets_dict["price_H2"],
            "price_meoh": targets_dict["price_meoh"],
            "price_bioCH4": targets_dict["price_bioCH4"],
        }

        price_ts = {
            k: pd.Series(-float(v), index=idx).to_frame(k)
            for k, v in prices.items()
            if isinstance(v, (int, float))
        }
        inputs_dict.update(price_ts)

    return inputs_dict

