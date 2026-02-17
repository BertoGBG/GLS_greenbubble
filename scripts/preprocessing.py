import pandas as pd
import numpy as np
import requests
from scripts import parameters as p
import os
from io import StringIO
import json
import time
from timezonefinder import TimezoneFinder
from entsoe import EntsoePandasClient
from datetime import datetime, timedelta
from scripts.config import (En_price_year,
                            DKK_Euro,
                            latitude,
                            longitude,
                            H2_delivery_frequency,
                            H2_profile_flag,
                            )


# ------ INPUTS PRE-PROCESSING ----

def GL_inputs_to_eff(GL_inputs):
    ''' function that reads csv file with GreenLab energy and material flows for each plant and calculates
     efficiencies for multilinks in the network'''

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


def build_demands_TS(targets_dict, NG_demand_DK):

    '''Load GreenLab inputs'''
    demand_H2 = targets_dict['demand_H2']
    demand_CH4 = targets_dict['demand_CH4']
    demand_meoh = targets_dict['demand_meoh']

    '''bioCH4 demand '''
    bioCH4_demand = p.ref_df.copy()
    bioCH4_demand = bioCH4_demand.rename(columns={p.ref_col_name: 'bioCH4 demand MWh'})
    bioCH4_demand.at[bioCH4_demand.index[-1], 'bioCH4 demand MWh'] = demand_CH4
    bioCH4_demand.to_csv(p.bioCH4_prod_input_file, sep=';')  # MWh/h

    '''Methanol demand'''
    Methanol_demand = p.ref_df.copy()
    Methanol_demand.rename(columns={p.ref_col_name: 'Methanol demand MWh'}, inplace=True)
    Methanol_demand.at[Methanol_demand.index[-1], 'Methanol demand MWh'] = demand_meoh
    Methanol_demand.to_csv(p.Methanol_demand_input_file, sep=';')  # t/h

    '''H2 demand with annual profile'''
    H2_input_demand, NG_demand_DK_h = build_H2_grid_demand(targets_dict, NG_demand_DK, profile_flag=H2_profile_flag, n=H2_delivery_frequency)

    demands = {'bioCH4': bioCH4_demand,
               'H2' : H2_input_demand,
               'meoh' : Methanol_demand,
               'NG_DK' : NG_demand_DK_h}

    return demands


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
    NG_demand_DK = pd.read_csv(p.NG_demand_input_file, sep=';', index_col=0)  # currency/MWh
    DH_external_demand = pd.read_csv(p.DH_external_demand_input_file, sep=';', index_col=0)  # currency/MWh
    DH_external_demand = DH_external_demand.set_axis(p.hours_in_period)


    return GL_inputs, GL_eff, Elspotprices, CO2_emiss_El, CF_wind, CF_solar, NG_price_year, NG_demand_DK, DH_external_demand


# ---- DEMANDS for H2, MeOH and El_DK1_GLS

def build_H2_grid_demand(targets_dict, NG_demand_DK, profile_flag, n):
    """
    Calculate H2 demand distribution over a given number of intervals (n),
    ensuring deliveries align with the last hour of each interval.

    Parameters:
    - H2_size: Hydrogen capacity size
    - flh_H2: Full load hours of H2 system
    - NG_demand_DK: DataFrame containing natural gas demand data
    - col_name: Column name for storing H2 demand
    - profile_flag: Boolean flag for profile-based allocation
    - n: Number of intervals (default: 12 for months, 52 for weeks, 1 for single year-end delivery)

    Returns:
    - H2_demand_y: DataFrame aligned with p.ref_df, with deliveries at correct timestamps
    """
    demand_H2 = targets_dict['demand_H2']

    # Initialize output DataFrame with the same structure and index as p.ref_df
    H2_demand_y = p.ref_df.copy()
    col_name= 'H2_demand_MWh'
    H2_demand_y.rename(columns={'ref col': col_name}, inplace=True)
    H2_demand_y[col_name] = 0.0

    # NG_demand_DK align timestamp
    s = NG_demand_DK.iloc[:, 0].astype(float)
    idx = pd.DatetimeIndex(s.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    s.index = idx
    start = s.index.min()
    end = s.index.max() + pd.Timedelta(days=1)  # include the entire last day
    hourly_index = pd.date_range(start, end, freq="h", inclusive="left")
    NG_demand_DK_2 = (s.reindex(hourly_index, method="ffill") / 24.0)

    # Convert start_date and end_date from ISO 8601 format
    start_date = datetime.strptime(p.start_date, "%Y-%m-%d %H:%M")
    end_date = datetime.strptime(p.end_date, "%Y-%m-%d %H:%M")

    # Determine the time step based on n (monthly or weekly)
    if n == 12:
        step = timedelta(days=30)  # Approximate monthly step

    elif n == 52:
        step = timedelta(weeks=1)  # Weekly step
    elif n == 1:
        step = end_date - start_date  # Single delivery at the end of the year
    else:
        raise ValueError("Invalid value for n. Use 1 (yearly), 12 (monthly), or 52 (weekly).")

    # Generate delivery timestamps
    delivery_dates = []
    current_time = start_date

    for i in range(n):
        # Calculate next delivery time
        if n == 1:
            next_time = end_date  # One delivery at year-end
        else:
            next_time = (current_time + step).replace(hour=23, minute=0, second=0)  # Last hour of the interval

        if next_time > end_date or i == n - 1:  # Ensure last delivery is exactly at year-end
            next_time = end_date.replace(hour=23, minute=0, second=0)

        # Find the last available hour within the reference DataFrame index
        valid_times = H2_demand_y.index[H2_demand_y.index <= next_time]
        if valid_times.empty:
            continue
        last_hour = valid_times[-1]  # Ensures delivery at the last available hour

        delivery_dates.append(last_hour)
        #current_time = next_time  # Move to next interval start

    if profile_flag :
        # Assign H2 demand values at the correct timestamps
        def slice_by_month_day_hour(df, start, end):
            mask = (
                           (df.index.month > start.month) |
                           ((df.index.month == start.month) &
                            ((df.index.day > start.day) |
                             ((df.index.day == start.day) & (df.index.hour >= start.hour))))
                   ) & (
                           (df.index.month < end.month) |
                           ((df.index.month == end.month) &
                            ((df.index.day < end.day) |
                             ((df.index.day == end.day) & (df.index.hour <= end.hour))))
                   )
            return df.loc[mask]


        for i in range(len(delivery_dates)):
            end_time = delivery_dates[i]
            st_time = delivery_dates[i - 1] if i > 0 else start_date  # Ensure first interval starts from start_date

            # Compute H2_val based only on NG demand within the current interval
            period_data = slice_by_month_day_hour(NG_demand_DK_2, st_time, end_time)
            total_demand = NG_demand_DK_2.sum() # Total demand for normalization

            if total_demand > 0:  # Avoid division by zero
                H2_val = period_data.sum() / total_demand * demand_H2 # H2_size * flh_H2
            else:
                H2_val = 0  # If there's no demand data, keep it zero

            H2_demand_y.loc[delivery_dates[i], :] = H2_val

    else:
        H2_val = demand_H2 / n
        H2_demand_y.loc[delivery_dates, :] = H2_val

    H2_demand_y.to_csv(p.H2_demand_input_file, sep=';')

    return H2_demand_y, NG_demand_DK_2

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
    r = s.get(
        api_base + "data/pv",
        params={
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
        },
    )
    if r.status_code != 200:
        raise RuntimeError(f"RN pv failed {r.status_code}: {r.text[:2000]}")
    parsed = r.json()
    CF_solar = pd.read_json(StringIO(json.dumps(parsed["data"])), orient="index")
    CF_solar.rename(columns={CF_solar.columns[0]: "CF solar"}, inplace=True)

    # --- Wind ---
    r = s.get(
        api_base + "data/wind",
        params={
            "lat": latitude,
            "lon": longitude,
            "date_from": date_from,
            "date_to": date_to,
            "capacity": 1.0,
            "height": 100,
            "turbine": "Vestas V80 2000",
            "format": "json",
        },
    )
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


def pre_processing_energy_data():
    """ function that preprocess all the energy input data and saves in
    NOTE:Some data are not always used depending on the network configuration
    Prices from DK are downlaoded in DKK"""
    """ Dates"""
    dates= p.hours_in_period.date
    start_date = dates[0].strftime("%Y-%m-%d")
    end_date = (dates[-1] + timedelta(days=1)).strftime("%Y-%m-%d")

    '''El spot prices DK1 - input DKK/MWh or EUR/MWh'''
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
    Elspotprices.to_csv(p.El_price_input_file, sep=';')  # currency/MWh

    '''CO2 emission from El Grid DK1'''
    sort_val = 'sort=HourDK%20asc'
    # filter_area = r'filter={"PriceArea":"DK1"}' # defined in parameters
    if En_price_year <= 2022:
        dataset_name = 'DeclarationEmissionHour'
        CO2emis_data = download_energidata(dataset_name, p.start_date, p.end_date, sort_val,
                                           p.filter_area)  # g/kWh = kg/MWh
        CO2_emiss_El = CO2emis_data[['HourDK', 'CO2PerkWh']].copy()

    elif En_price_year > 2022:
        dataset_name = 'DeclarationGridEmission'
        sort_val = "HourDK asc"  # 'sort=HourDK%20asc'
        CO2emis_data = download_energidata(
            dataset_name=dataset_name,
            start_date=start_date,  # "2025-01-01",
            end_date=end_date,  # "2026-01-01",
            sort_val=sort_val,
            price_area=p.price_area,
            limit=0
        )

        CO2_emiss_El = CO2emis_data.query("FuelAllocationMethod == '125%'")[['HourDK', 'CO2PerkWh']].copy()

    CO2_emiss_El['CO2PerkWh'] = CO2_emiss_El['CO2PerkWh'] / 1000  # t/MWh
    CO2_emiss_El.rename(columns={'CO2PerkWh': 'CO2PerMWh'}, inplace=True)
    CO2_emiss_El['HourDK'] = pd.to_datetime(CO2_emiss_El['HourDK'])
    CO2_emiss_El.set_index('HourDK', inplace=True)
    CO2_emiss_El = remove_feb_29(CO2_emiss_El)
    CO2_emiss_El.to_csv(p.CO2emis_input_file, sep=';')

    # NG prices depending on the year
    ''' NG prices prices in DKK/kWh or EUR/kWH'''
    if En_price_year <= 2022:
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
        NG_price_year['HourDK'] = pd.to_datetime(NG_price_year['HourDK'])
        NG_price_year['HourDK'] = pd.to_datetime(NG_price_year['HourDK'].dt.strftime("%Y-%m-%d %H:%M:%S+00:00"))
        NG_price_year.set_index('HourDK', inplace=True)
        NG_price_year[NG_price_col_name] = NG_price_year[NG_price_col_name] * 1000 / DKK_Euro  # coversion to €/MWh
        last_rows3 = pd.DataFrame(
            {'HourDK': p.hours_in_period[-1:len(p.hours_in_period)], NG_price_col_name: NG_price_year.iloc[-1, 0]})
        last_rows3.set_index('HourDK', inplace=True)
        NG_price_year = pd.concat([NG_price_year, last_rows3])
        NG_price_year = NG_price_year.asfreq('h', method='ffill')

    elif En_price_year > 2022:
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
        hours = pd.DatetimeIndex(p.hours_in_period)  # ensures it's a DatetimeIndex
        THE_daily_NG_prices = THE_daily_NG_prices.reindex(hours).ffill()

        # --- Final series
        NG_price_year = THE_daily_NG_prices[["THE_NG_pricesEUR_MWh"]].copy()

    NG_price_year = remove_feb_29(NG_price_year)
    NG_price_year = NG_price_year.interpolate(method='linear')
    NG_price_year.to_csv(p.NG_price_year_input_file, sep=';')  # €/MWh

    '''  Estimated NG Demand DK '''
    # source: https://www.energidataservice.dk/tso-gas/Gasflow
    # used to create a profile for H2 demand - if required.
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
    NG_demand_DK['GasDay'] = pd.to_datetime(NG_demand_DK['GasDay'].dt.strftime("%Y-%m-%d %H:%M:%S+00:00"))
    NG_demand_DK.set_index('GasDay', inplace=True)
    NG_demand_DK = remove_feb_29(NG_demand_DK)
    NG_demand_DK.to_csv(p.NG_demand_input_file, sep=';')  # €/MWh

    '''District heating data'''
    # Download weather data near Skive (Mejrup)
    # https://www.dmi.dk/friedata/observationer/
    data_folder = p.DH_data_folder  # prices in currency/kWh
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
                                    'Capacity Factor DH'] * DH_max_capacity  # estimated demand for DH in Skive municipality
    DH_Skive = remove_feb_29(DH_Skive)
    DH_Skive = DH_Skive.set_axis(p.hours_in_period)
    DH_Skive = DH_Skive.interpolate (method='linear')
    DH_Skive.to_csv(p.DH_external_demand_input_file, sep=';')  # MWh/h

    '''Onshore Wind and Solar Capacity Factors'''
    # Download CF for wind and solar corresponding to the energy year
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
    CF_wind.to_csv(p.CF_wind_input_file, sep=';')  # kg/MWh
    CF_solar.to_csv(p.CF_solar_input_file, sep=';')  # kg/MWh

    return


# ---- Pre-processing for PyPSA network

def prepare_all_inputs(targets_dict, CO2_cost, CO2_cost_ref_year, max_RE_to_grid, preprocess_flag):
    # functions calling all other functions and build inputs dictionary to the model
    # returns: inputs_dict which contains DataFrames with all inputs for the pypsa network

    if preprocess_flag:
        pre_processing_energy_data()  # download + preprocessing + save to CSV

    # load the inputs form CSV files
    GL_inputs, GL_eff, Elspotprices, CO2_emiss_El, CF_wind, CF_solar, NG_price_year, NG_demand_DK, DH_external_demand = load_input_data()

    '''Build all demands'''
    # build demands TS (considers targets_dict['driver'])
    demands = build_demands_TS (targets_dict, NG_demand_DK)
    NG_DK = demands['NG_DK']

    H2_input_demand = demands['H2']
    bioCH4_demand = demands['bioCH4']
    Methanol_demand = demands['meoh']


    inputs_dict = {'GL_inputs': GL_inputs,
                   'GL_eff': GL_eff,
                   'Elspotprices': Elspotprices,
                   'CO2_emiss_El': CO2_emiss_El,
                   'bioCH4_demand': bioCH4_demand,
                   'CF_wind': CF_wind,
                   'CF_solar': CF_solar,
                   'NG_price_year': NG_price_year,
                   'Methanol_input_demand': Methanol_demand,
                   'NG_demand_DK': NG_DK,
                   'DH_external_demand': DH_external_demand,
                   'H2_input_demand': H2_input_demand,
                   'CO2 cost': CO2_cost,
                   'CO2 cost ref year' : CO2_cost_ref_year,
                   'max_RE_to_grid': max_RE_to_grid,
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

