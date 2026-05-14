"""Quick check: are the custom RE-sales constraint duals captured in the duals export file?

Usage:
    python check_duals.py                      # auto-finds the most recent duals file
    python check_duals.py path/to/duals.nc     # explicit file
"""
import sys
from pathlib import Path
import xarray as xr
import numpy as np

KNOWN_CONSTRAINTS = [
    "El3_export_fraction_of_total_RE",  # mode: sales
    "RE_grid_connection_cap",           # mode: connection
]

# ── Find file ────────────────────────────────────────────────────────────────
if len(sys.argv) > 1:
    duals_path = Path(sys.argv[1])
else:
    candidates = sorted(
        Path("outputs").rglob("duals_export_*.nc"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        print("No duals_export_*.nc found under outputs/. Pass the path explicitly.")
        sys.exit(1)
    duals_path = candidates[0]

print(f"Duals file : {duals_path}")
print(f"Modified   : {duals_path.stat().st_mtime:.0f}")
print()

ds = xr.open_dataset(duals_path)

# ── All blocks ────────────────────────────────────────────────────────────────
print(f"All captured dual blocks ({len(ds.data_vars)}):")
for v in ds.data_vars:
    arr = ds[v].values
    finite = arr[np.isfinite(arr)]
    tag = ""
    for kc in KNOWN_CONSTRAINTS:
        if v.startswith(kc):
            tag = "  ← custom RE-sales constraint"
    print(f"  {v:60s}  shape={arr.shape}  nonzero={np.count_nonzero(finite)}/{finite.size}{tag}")

print()

# ── Custom constraint check ───────────────────────────────────────────────────
print("Custom RE-sales constraints:")
found_any = False
for kc in KNOWN_CONSTRAINTS:
    matches = [v for v in ds.data_vars if v.startswith(kc)]
    if matches:
        for v in matches:
            arr = ds[v].values
            finite = arr[np.isfinite(arr)]
            print(f"  ✓ {v}")
            print(f"      shape={arr.shape}  min={finite.min():.4f}  max={finite.max():.4f}"
                  f"  mean={finite.mean():.4f}  nonzero={np.count_nonzero(finite)}/{finite.size}")
        found_any = True
    else:
        print(f"  ✗ '{kc}*' — NOT FOUND")

if not found_any:
    print()
    print("Neither custom constraint dual was found.")
    print("Possible reasons:")
    print("  • The network was solved without collect_all_duals=True")
    print("  • The constraint was not added (check re_sales_mode in config)")
    print("  • The solver did not return duals (check solver settings)")
