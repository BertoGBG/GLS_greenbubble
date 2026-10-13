# -*- coding: utf-8 -*-
"""Generate one connection figure per agent for the Technologies pages.

The figures are drawn from the network itself, not by hand. The script builds the
network with every agent and every technology switched on, then draws each agent's
links with all their bus ports and efficiencies. It builds a second network with the
methanol split enabled and adds those links to the ``meoh`` figure as an alternative.

Run from the repository root:

    python docs/_static/model/generate_tech_figures.py
"""
import io
import logging
import os
import pickle
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

OUT = ROOT / "docs/_static/model"
AGENTS = ["renewables", "electrolysis", "biogas", "methanation", "meoh",
          "central_heat", "storage", "symbiosis"]
TITLES = {"renewables": "Renewables", "electrolysis": "Electrolysis", "biogas": "Biogas and biomass",
          "methanation": "Methanation", "meoh": "Methanol", "central_heat": "Heat system",
          "storage": "Storage", "symbiosis": "Symbiosis (shared buses)", "external_grids": "External grids"}

# carrier colours follow generate_figures.py
COL = dict(el="#c9820f", h2="#1668c4", co2="#7048e8", gas="#1c7ed6", heat="#d4552c",
           meoh="#e8590c", bio="#2f9e44", other="#7d8fa1")
INK, INK2, INK3 = "#14202c", "#44586b", "#7d8fa1"
LINE, LINE2, SURF, SURF2 = "#d3dbe4", "#bcc8d4", "#ffffff", "#f5f7fa"
FONT = 'font-family="IBM Plex Sans, -apple-system, Segoe UI, Helvetica, Arial, sans-serif"'
MONO = 'font-family="IBM Plex Mono, ui-monospace, SFMono-Regular, Menlo, monospace"'


def carrier(bus):
    b = bus.lower()
    if b.startswith(("heat", "tes", "dh grid")):
        return "heat"
    if b.startswith("el") or "eldk1" in b:
        return "el"
    if "h2" in b:
        return "h2"
    if "co2" in b:
        return "co2"
    if "methanol" in b or "meoh" in b:
        return "meoh"
    if any(k in b for k in ("biogas", "ch4", "ng")) and "biomass" not in b:
        return "gas"
    if any(k in b for k in ("biomass", "digestate", "pellets", "biochar", "moist")):
        return "bio"
    return "other"


# ───────────────────────────── network ─────────────────────────────
def build(split):
    """Build the network with everything switched on; return (network, allocation)."""
    from scripts import config as c, parameters as p
    c.n_config.loc[:, "expansion"] = True
    for k in ["moist biomass market", "pellets market", "biochar credits", "CO2 Liq credits", "DH"]:
        c.n_options.at[k, "enable"] = True
    c.n_options.at["meoh split", "enable"] = split
    for k in ["biogas", "central_heat", "renewables", "electrolysis", "meoh", "methanation",
              "symbiosis", "storage"]:
        c.n_flags[k] = True
    import scripts.prepare_network as pn
    from scripts.helpers import prepare_costs
    from scripts.technology_inputs import tech_inputs
    pn.define_total_supply_constraints = lambda *a, **k: None    # needs a solver model; topology only
    tc = prepare_costs(latitude=c.latitude, longitude=c.longitude, tech_inputs=tech_inputs,
                       USD_to_EUR=c.USD_to_EUR, discount_rate=c.discount_rate,
                       cost_path_EU=p.cost_path, cost_path_US=p.cost_path_US,
                       dict_tech_US_EU=p.dict_tech_US_EU)
    inputs = pickle.load(open(ROOT / f"resources/inputs_{c.En_price_year}.pkl", "rb"))
    n = pn.build_network(tc, inputs, pn.network_dependencies(c.n_flags), c.n_options, p,
                         costs_dir=p.cost_folder, USD_to_EUR=c.USD_to_EUR, tech_inputs=tech_inputs)
    return n, n.network_comp_allocation, set(c.n_config.index)


def ports(n, link):
    """[(port, bus, efficiency)] of a link; bus0 has efficiency 1 by definition."""
    L = n.links
    out = []
    for col in sorted([k for k in L.columns if k.startswith("bus") and k[3:].isdigit()], key=lambda k: int(k[3:])):
        bus = L.at[link, col]
        if not isinstance(bus, str) or not bus:
            continue
        i = int(col[3:])
        eff = 1.0 if i == 0 else float(L.at[link, "efficiency" if i == 1 else f"efficiency{i}"])
        out.append((i, bus, eff))
    return out


AUX = ("_to_symb", "_from_symb", "El3_to_El", "DK1_to_El", "compressor", "storage send", "storage return",
       "_to_demand", "_to_amb", " return", "_to_DH")


def is_aux(name):
    return any(k in name for k in AUX)


def agent_graph(n, alloc, agent, extra=None):
    """Technologies of one agent with their inputs and outputs.

    Connector, compressor and storage links are not drawn as boxes. They become notes on
    the buses they join, so each technology box shows only its own ports.
    """
    names = [l for l in alloc[agent].get("links", []) if not l.startswith("EXI_") and l in n.links.index]
    aux = [l for l in names if is_aux(l)]
    main = [l for l in names if l not in aux] or names          # an agent of connectors only shows them all
    aux = [l for l in aux if l not in main]
    items = []
    for l in main:
        ps = ports(n, l)
        items.append(dict(name=l, ins=[x for x in ps if x[0] == 0 or x[2] < 0],
                          outs=[x for x in ps if x[0] > 0 and x[2] >= 0], alt=False))
    items += extra or []
    notes = {}
    pairs = {(n.links.at[l, "bus0"], n.links.at[l, "bus1"]): l for l in aux}
    for (a, b), l in pairs.items():
        kind = " (compressor)" if "comp" in l else ""
        if (b, a) in pairs:
            notes.setdefault(a, []).append(f"↔ {b}")
        else:
            eff = float(n.links.at[l, "efficiency"])
            notes.setdefault(b, []).append(f"← {a}{kind}")
            notes.setdefault(a, []).append(f"→ {b}" + (f" ×{eff:.2f}" if abs(eff - 1) > 1e-6 else "") + kind)
    for comp, kind in (("generators", "generator"), ("loads", "load"), ("stores", "store"),
                       ("storage_units", "storage unit")):
        df = getattr(n, comp)
        for g in alloc[agent].get(comp, []):
            if not g.startswith("EXI_") and g in df.index:
                notes.setdefault(df.at[g, "bus"], []).append(f"{kind}: {g}")
    made = {}
    for d in items:
        for _, b, _ in d["outs"]:
            made.setdefault(b, []).append(d["name"])
    for b, v in made.items():
        if len(v) > 1 and not b.lower().startswith(("heat", "el")):
            notes.setdefault(b, []).insert(0, f"competing: {len(v)} technologies")
    notes = {b: list(dict.fromkeys(v)) for b, v in notes.items()}
    return items, notes


# ───────────────────────────── drawing ─────────────────────────────
def esc(t):
    return str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, t, size=12, fill=INK2, mono=False, weight="400", anchor="start"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" {MONO if mono else FONT} font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{esc(t)}</text>\n')


def fmt(e):
    a = abs(e)
    return f"{e:.3g}" if a >= 0.01 or a == 0 else f"{e:.2e}"


def layout_column(order, want, heights, gap, top):
    """Place boxes in the given order near their wanted centre without overlap."""
    y, pos = top, {}
    for k in order:
        y0 = max(y, want[k] - heights[k] / 2)
        pos[k] = y0
        y = y0 + heights[k] + gap
    return pos


def draw(agent, items, attach):
    if not items and not attach:
        return None
    # main technologies first, then alternatives, then auxiliary links (compressors, connectors)
    items = sorted(items, key=lambda d: (d["alt"], d["name"].lower()))
    ins = list(dict.fromkeys(b for d in items for _, b, _ in d["ins"]))
    outs = list(dict.fromkeys(b for d in items for _, b, _ in d["outs"]))
    lone = [b for b in attach if b not in ins and b not in outs
            and any(a.startswith(("generator", "storage unit")) for a in attach[b])]   # e.g. wind and solar
    outs += lone
    attach = {b: v for b, v in attach.items() if b in ins or b in outs}

    W, XI, XL, XO = 1180, 16, 450, 880
    BWI, BWL, BWO = 290, 280, 284
    PH = 17                                                                # height per port row
    hL = {d["name"]: max(46, 30 + PH * max(len(d["ins"]), len(d["outs"]))) for d in items}
    hB = lambda b: 32 + 15 * len(attach.get(b, []))
    hI, hO = {b: hB(b) for b in ins}, {b: hB(b) for b in outs}

    # order links and buses by barycentre, two sweeps
    pL = layout_column([d["name"] for d in items], {d["name"]: 0 for d in items}, hL, 22, 70)
    cL = {k: pL[k] + hL[k] / 2 for k in pL}
    for _ in range(2):
        wi = {b: sum(cL[d["name"]] for d in items if b in [x for _, x, _ in d["ins"]]) /
              max(1, sum(1 for d in items if b in [x for _, x, _ in d["ins"]])) for b in ins}
        wo = {b: (sum(cL[d["name"]] for d in items if b in [x for _, x, _ in d["outs"]]) /
                  max(1, sum(1 for d in items if b in [x for _, x, _ in d["outs"]]))) if b not in lone else 1e9
              for b in outs}
        pI = layout_column(sorted(ins, key=wi.get), wi, hI, 12, 70)
        pO = layout_column(sorted(outs, key=wo.get), {b: (v if v < 1e8 else 0) for b, v in wo.items()}, hO, 12, 70)
        cI = {b: pI[b] + hI[b] / 2 for b in ins}
        cO = {b: pO[b] + hO[b] / 2 for b in outs}
    H = 30 + int(max([pL[k] + hL[k] for k in pL] + [pI[b] + hI[b] for b in ins] +
                [pO[b] + hO[b] for b in outs] + [100])) + 60

    s = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" role="img" '
         f'aria-label="{esc(TITLES[agent])}: links and their bus connections">\n'
         f'<title>{esc(TITLES[agent])}: links and their bus connections</title>\n'
         f'<rect width="{W}" height="{H}" fill="{SURF}"/>\n')
    s += text(XI, 30, "INPUT BUSES", 11, INK3, True, "600")
    s += text(XL, 30, "LINKS  ·  port: efficiency per unit of bus0", 11, INK3, True, "600")
    s += text(XO + BWO, 30, "OUTPUT BUSES", 11, INK3, True, "600", "end")
    s += f'<line x1="{XI}" y1="44" x2="{XO + BWO}" y2="44" stroke="{LINE}"/>\n'

    edges, boxes, labels = "", "", ""
    for d in items:
        y0, h, nm = pL[d["name"]], hL[d["name"]], d["name"]
        for side, lst in (("in", d["ins"]), ("out", d["outs"])):
            for j, (i, b, e) in enumerate(lst):
                yl = y0 + 30 + PH * j + PH / 2
                col = COL[carrier(b)]
                if side == "in":
                    xb, yb, xl = XI + BWI, cI[b], XL
                    lab, ax, anc = (f"bus{i}" if i == 0 else f"bus{i}  {fmt(e)}"), XL + 10, "start"
                else:
                    xb, yb, xl = XO, cO[b], XL + BWL
                    lab, ax, anc = f"{fmt(e)}  bus{i}", XL + BWL - 10, "end"
                mx = (xb + xl) / 2
                edges += (f'<path d="M{xb},{yb:.1f} C{mx},{yb:.1f} {mx},{yl:.1f} {xl},{yl:.1f}" fill="none" '
                          f'stroke="{col}" stroke-width="2" stroke-opacity=".75"/>\n')
                edges += f'<circle cx="{xl}" cy="{yl:.1f}" r="3" fill="{col}"/>\n'
                labels += text(ax, yl + 4, lab, 11, col, True, "600", anc)
        st = (f'stroke="{COL["meoh"]}" stroke-width="1.6" stroke-dasharray="5 4"' if d["alt"]
              else f'stroke="{INK3}" stroke-width="1.4"')
        boxes += f'<rect x="{XL}" y="{y0:.1f}" width="{BWL}" height="{h}" rx="6" fill="{SURF2}" {st}/>\n'
        title = nm if len(nm) <= 34 else nm[:33] + "…"
        boxes += text(XL + BWL / 2, y0 + 19, title, 13.5, INK, False, "600", "middle")
    for side, lst, pos, hh, x, w in (("in", ins, pI, hI, XI, BWI), ("out", outs, pO, hO, XO, BWO)):
        for b in lst:
            col = COL[carrier(b)]
            y0 = pos[b]
            boxes += (f'<rect x="{x}" y="{y0:.1f}" width="{w}" height="{hh[b]}" rx="5" fill="{SURF}" '
                      f'stroke="{col}" stroke-width="1.3"/>\n'
                      f'<rect x="{x if side == "in" else x + w - 5}" y="{y0:.1f}" width="5" height="{hh[b]}" rx="2" fill="{col}"/>\n')
            boxes += text(x + 14, y0 + 20, b, 12, INK, True, "600")
            for j, a in enumerate(attach.get(b, [])):
                hot = a.startswith("competing")
                boxes += text(x + 14, y0 + 35 + 15 * j, a if len(a) <= 38 else a[:37] + "…", 10.5,
                              COL["meoh"] if hot else INK2, True, "600" if hot else "400")
    s += edges + boxes + labels

    s += text(XI, H - 22, "Bus notes: ← fed from, → feeds, ↔ store or two-way link (connectors, compressors "
              "and storage links of this agent). Efficiencies are per unit of bus0; negative means consumed.",
              11, INK3)
    if any(d["alt"] for d in items):
        s += (f'<rect x="{XI}" y="{H - 52}" width="26" height="16" rx="4" fill="{SURF2}" stroke="{COL["meoh"]}" '
              f'stroke-width="1.6" stroke-dasharray="5 4"/>\n')
        s += text(XI + 34, H - 40, "Alternative to methanolisation, built when options['meoh split'].enable is true.",
                  11.5, INK2)
    return s + "</svg>\n"


def main():
    n, alloc, _ = build(split=False)
    ns, alloc_s, _ = build(split=True)
    base = {l for l in alloc["meoh"].get("links", [])}
    alt = []
    for l in alloc_s["meoh"].get("links", []):
        if l in base or l.startswith("EXI_") or l not in ns.links.index:
            continue
        ps = ports(ns, l)
        if is_aux(l):
            continue
        alt.append(dict(name=l, ins=[x for x in ps if x[0] == 0 or x[2] < 0],
                        outs=[x for x in ps if x[0] > 0 and x[2] >= 0], alt=True))
    for agent in AGENTS:
        items, attach = agent_graph(n, alloc, agent, alt if agent == "meoh" else None)
        if agent == "meoh":
            for g in alloc_s["meoh"].get("stores", []):
                if g in ns.stores.index and g not in n.stores.index:
                    attach.setdefault(ns.stores.at[g, "bus"], []).append(f"store: {g} (split)")
        svg = draw(agent, items, attach)
        if svg is None:
            continue
        p = OUT / f"tech_{agent}.svg"
        io.open(p, "w", encoding="utf-8").write(svg)
        print(f"  wrote {p.relative_to(ROOT)}  ({len(items)} links)")


if __name__ == "__main__":
    main()
