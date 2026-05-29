# SPDX-License-Identifier: MIT
"""Data retrieval utilities for external energy-market APIs.

This module provides functions to download time-series data from three
external sources used by the GreenBubble preprocessing pipeline:

* **Energi Data Service** (``api.energidataservice.dk``) — Danish electricity
  spot prices, CO₂ emission intensities and natural gas prices.
* **Renewables.ninja** — hourly wind and solar capacity-factor profiles for
  any geographic location (MERRA-2 reanalysis data).
* **ENTSO-E Transparency Platform** — historical electricity demand for
  European bidding zones.
* **EIA / CAISO** — US-side electricity data (California runs).

All functions return :class:`pandas.DataFrame` objects indexed on UTC-naive
hourly timestamps matching the model snapshot index built by
:func:`scripts.helpers.build_snapshots`.

.. note::
   API tokens for Renewables.ninja and ENTSO-E are stored in
   :mod:`scripts.parameters`.  Obtain your own tokens before running the
   pipeline on a new machine.
"""

import pandas as pd
import numpy as np
import requests
from scripts import parameters as p
import os
from io import StringIO
import json
from entsoe import EntsoePandasClient
from datetime import datetime, timedelta
import pytz
import hashlib
import urllib.parse


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


# ----- EXTERNAL ENERGY MARKETS

def remove_feb_29(df):
    # Function to remove February 29 if it's a leap year, works on df and series
    # Check if the year is a leap year
    if any((df.index.month == 2) & (df.index.day == 29)):
        # Remove rows where the date is February 29
        df = df[~((df.index.month == 2) & (df.index.day == 29))]
    return df


def download_energidata(dataset_name, start_date, end_date, sort_val, filter_area):
    """Download a dataset from the Energi Data Service REST API.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset endpoint, e.g. ``"Elspotprices"`` or
        ``"DeclarationGridEmission"``.
    start_date : str
        Start of the requested period in ``"YYYY-MM-DD HH:MM"`` format.
        Spaces are replaced with ``T`` before sending the request.
    end_date : str
        End of the requested period, same format as *start_date*.
    sort_val : str
        Pre-encoded sort parameter string, e.g. ``"sort=HourDK%20asc"``.
        The function URL-decodes this before passing it to *requests* to
        avoid double-encoding.
    filter_area : str
        Filter expression string, e.g. ``r'filter={"PriceArea":"DK1"}'``.
        String values are automatically wrapped in arrays as required by
        the API (``{"PriceArea": ["DK1"]}``).

    Returns
    -------
    pandas.DataFrame
        Flat DataFrame of all records returned by the API (``result["records"]``
        normalised with :func:`pandas.json_normalize`).

    Raises
    ------
    requests.HTTPError
        If the API returns a non-2xx status code.
    """
    url = f"https://api.energidataservice.dk/dataset/{dataset_name}"
    # API expects T separator: "2023-01-01T00:00" not "2023-01-01 00:00"
    params = {
        "start": str(start_date).replace(" ", "T"),
        "end": str(end_date).replace(" ", "T"),
    }

    if sort_val:
        # sort_val may already be %-encoded (e.g. "sort=HourDK%20asc") — decode first
        # so requests doesn't double-encode it
        params["sort"] = urllib.parse.unquote(sort_val.replace("sort=", "", 1))

    if filter_area:
        # filter_area is e.g. r'filter={"PriceArea":"DK1"}' — extract the JSON part
        filter_json_str = filter_area.split("filter=", 1)[1]
        filter_dict = json.loads(filter_json_str)
        # API expects array values: {"PriceArea": ["DK1"]}
        params["filter"] = json.dumps({k: [v] if isinstance(v, str) else v
                                       for k, v in filter_dict.items()})

    response = requests.get(url=url, params=params, headers={"Accept": "application/json"})
    response.raise_for_status()
    result = response.json()
    records = result.get('records', [])
    return pd.json_normalize(records)


def retrieve_renewable_capacity_factors(token, start_date, end_date, latitude, longitude):
    """Retrieve hourly wind and solar capacity factors from the Renewables.ninja API.

    Uses MERRA-2 reanalysis data.  Solar tilt is estimated from latitude via
    ``tilt = 0.87 * lat + 3.1``; wind is modelled as a Vestas V80 2000 kW
    turbine at 100 m hub height.

    Parameters
    ----------
    token : str
        Personal API token for Renewables.ninja (see
        ``https://www.renewables.ninja/documentation/api``).
    start_date : str
        First day of the period in ``"YYYY-MM-DD"`` format.
    end_date : str
        Last day of the period in ``"YYYY-MM-DD"`` format.
    latitude : float
        Site latitude in decimal degrees.
    longitude : float
        Site longitude in decimal degrees.

    Returns
    -------
    CF_solar : pandas.DataFrame
        Hourly solar capacity factors with column ``"CF solar"``.
    CF_wind : pandas.DataFrame
        Hourly wind capacity factors with column ``"CF wind"``.

    Raises
    ------
    requests.HTTPError
        If either API request fails (e.g. 429 rate-limit after retries).
    """
    api_base = 'https://www.renewables.ninja/api/'
    s = requests.session()
    s.headers = {'Authorization': 'Token ' + token}

    # Solar PV request
    url = api_base + 'data/pv'
    optimal_tilt = latitude * 0.87 + 3.1  #  simple optimal tilt expression

    args = {
        'lat': latitude,
        'lon': longitude,
        'date_from': start_date,
        'date_to': end_date,
        'dataset': 'merra2',
        'capacity': 1.0,
        'system_loss': 0.1,
        'tracking': 0,
        'tilt': optimal_tilt,
        'azim': 180,
        'format': 'json'
    }

    r = s.get(url, params=args)
    r.raise_for_status()  # Raise an error if request fails
    parsed_response = json.loads(r.text)
    CF_solar = pd.read_json(StringIO(json.dumps(parsed_response['data'])), orient='index')
    CF_solar.rename(columns={CF_solar.columns.values[0] : 'CF solar'}, inplace=True)

    # Wind power request
    url = api_base + 'data/wind'
    args = {
        'lat': latitude,
        'lon': longitude,
        'date_from': start_date,
        'date_to': end_date,
        'capacity': 1.0,
        'height': 100,
        'turbine': 'Vestas V80 2000',
        'format': 'json'
    }

    r = s.get(url, params=args)
    r.raise_for_status()
    parsed_response = json.loads(r.text)
    CF_wind = pd.read_json(StringIO(json.dumps(parsed_response['data'])), orient='index')
    CF_wind.rename(columns={CF_wind.columns.values[0] : 'CF wind'}, inplace=True)

    return CF_solar, CF_wind


def retrive_entsoe_el_demand(API_KEY, start_day, end_day, country_code):
    """Retrieve historical electricity demand from the ENTSO-E Transparency Platform.

    .. note::
       Country codes are listed at
       ``https://github.com/EnergieID/entsoe-py/blob/master/entsoe/mappings.py``.

    Parameters
    ----------
    API_KEY : str
        ENTSO-E API key (stored in :mod:`scripts.parameters` as ``entsoe_api``).
    start_day : str
        Start date in ``"YYYY-MM-DD"`` format.
    end_day : str
        End date in ``"YYYY-MM-DD"`` format.
    country_code : str
        ENTSO-E bidding-zone code, e.g. ``"DK_1"`` for Western Denmark.

    Returns
    -------
    pandas.Series
        Hourly electricity load in MW, UTC-localised index.
    """
    # NOTE: list of country codes available here: https://github.com/EnergieID/entsoe-py/blob/master/entsoe/mappings.py

    client = EntsoePandasClient(api_key= API_KEY)

    start = pd.Timestamp(start_day, tz='Europe/Brussels')
    end = pd.Timestamp(end_day, tz='Europe/Brussels')

    ts = client.query_load(country_code, start=start, end=end)

    return ts


# ---- Technology data

def _raw_url_to_api_url(base_url, file_name):
    """Convert a raw.githubusercontent.com URL to a GitHub Contents API URL."""
    raw_prefix = "https://raw.githubusercontent.com/"
    if not base_url.startswith(raw_prefix):
        return None
    parts = base_url[len(raw_prefix):].split("/", 3)  # owner, repo, branch, path_prefix
    owner, repo, branch = parts[0], parts[1], parts[2]
    path_prefix = parts[3] if len(parts) > 3 else ""
    return f"https://api.github.com/repos/{owner}/{repo}/contents/{path_prefix}{file_name}?ref={branch}"


def fetch_remote_sha(base_url, file_name):
    """Return the git blob SHA for *file_name* in the remote GitHub repo.

    Parameters
    ----------
    base_url : str
        Raw GitHub URL prefix (``https://raw.githubusercontent.com/...``).
    file_name : str
        Bare file name, e.g. ``"costs_2030.csv"``.

    Returns
    -------
    str or None
        The blob SHA string, or ``None`` if the API is unreachable.
    """
    api_url = _raw_url_to_api_url(base_url, file_name)
    if api_url is None:
        return None
    try:
        resp = requests.get(api_url, headers={"Accept": "application/vnd.github+json"}, timeout=10)
        resp.raise_for_status()
        return resp.json().get("sha")
    except requests.exceptions.RequestException as e:
        print(f"[fetch_remote_sha] Could not reach GitHub API: {e}")
        return None


def get_cached_sha(local_file_path):
    """Return the SHA stored in ``<local_file_path>.sha``, or ``None``."""
    sha_path = str(local_file_path) + ".sha"
    if os.path.exists(sha_path):
        with open(sha_path) as f:
            return f.read().strip()
    return None


def _git_blob_sha(content: bytes) -> str:
    """Compute the git blob SHA1 for raw file bytes (matches GitHub's blob SHA)."""
    header = f"blob {len(content)}\0".encode()
    return hashlib.sha1(header + content).hexdigest()


def retrieve_technology_data(local_file_path, base_url, max_retries: int = 3):
    """Download a cost CSV from the remote technology-data repo.

    Downloads and verifies the git blob SHA of the content against the GitHub
    Contents API.  Retries up to *max_retries* times if the CDN serves stale
    content (blob SHA mismatch).  Writes the verified SHA to
    ``<local_file_path>.sha`` so the Snakefile ``onstart`` block can detect
    future changes without downloading.

    Parameters
    ----------
    local_file_path : str
        Destination path for the downloaded CSV.
    base_url : str
        Raw GitHub URL prefix, e.g.
        ``"https://raw.githubusercontent.com/BertoGBG/technology-data/pypsa-eur_AA/outputs/"``.
    max_retries : int
        Number of download attempts before giving up on SHA verification.

    Returns
    -------
    str or None
        Path to the downloaded file, or ``None`` on network error.
    """
    local_file_path = str(local_file_path)
    local_folder = os.path.dirname(local_file_path)
    file_name = os.path.basename(local_file_path)
    sha_cache_path = local_file_path + ".sha"

    os.makedirs(local_folder, exist_ok=True)

    remote_sha = fetch_remote_sha(base_url, file_name)
    file_url = base_url + file_name

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(
                file_url,
                stream=True,
                timeout=30,
                headers={"Cache-Control": "no-cache", "Pragma": "no-cache"},
            )
            response.raise_for_status()
            content = response.content

            # Verify content against the git blob SHA from the API.
            # raw.githubusercontent.com can serve CDN-cached stale content
            # right after a push, causing us to store old data with a new SHA.
            if remote_sha:
                actual_sha = _git_blob_sha(content)
                if actual_sha != remote_sha:
                    print(
                        f"[retrieve] Attempt {attempt}/{max_retries}: "
                        f"SHA mismatch for {file_name} "
                        f"(expected {remote_sha[:8]}..., got {actual_sha[:8]}...). "
                        f"CDN likely served stale content — retrying."
                    )
                    if attempt < max_retries:
                        continue
                    print(f"[retrieve] Giving up SHA verification after {max_retries} attempts. Saving anyway.")

            with open(local_file_path, "wb") as fh:
                fh.write(content)
            if remote_sha:
                with open(sha_cache_path, "w") as fh:
                    fh.write(remote_sha)
            print(f"Technology-data downloaded: {file_name} (SHA: {remote_sha or 'unknown'})")
            return local_file_path

        except requests.exceptions.RequestException as e:
            print(f"Error downloading {file_name} (attempt {attempt}/{max_retries}): {e}")
            if attempt == max_retries:
                return None
