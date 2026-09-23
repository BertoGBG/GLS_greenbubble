# -*- coding: utf-8 -*-
"""Generate the three static model-structure SVGs for the docs."""
import io, os

OUT = "docs/_static/model"
# carrier colours follow config/plots_config.default.yaml families
C = dict(el="#c9820f", h2="#1668c4", co2="#6b7a87", gas="#8c610a",
         heat="#d4552c", meoh="#d0352a", bio="#4d7c2f")
INK, INK2, INK3 = "#14202c", "#44586b", "#7d8fa1"
LINE, LINE2, SURF, SURF2 = "#d3dbe4", "#bcc8d4", "#ffffff", "#f5f7fa"
GREEN = "#2f7d4f"

FONT = ('font-family="IBM Plex Sans, -apple-system, Segoe UI, Helvetica, Arial, sans-serif"')
MONO = ('font-family="IBM Plex Mono, ui-monospace, SFMono-Regular, Menlo, monospace"')

def esc(t):
    return (str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

def head(w, h, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'width="100%" role="img" aria-label="{title}">\n'
            f'<title>{esc(title)}</title>\n'
            f'<rect width="{w}" height="{h}" fill="{SURF}"/>\n')

def box(x, y, w, h, label, sub=None, accent=None, dashed=False, fill=SURF2):
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" '
         f'stroke="{LINE2}" stroke-width="1.2"'
         + (' stroke-dasharray="5 4"' if dashed else '') + '/>\n')
    if accent:
        s += f'<rect x="{x}" y="{y}" width="3.5" height="{h}" rx="2" fill="{accent}"/>\n'
    s += (f'<text x="{x+12}" y="{y+ (20 if sub else h/2+4)}" {FONT} font-size="12.5" '
          f'font-weight="600" fill="{INK}">{esc(label)}</text>\n')
    if sub:
        s += (f'<text x="{x+12}" y="{y+35}" {MONO} font-size="10" fill="{INK3}">{esc(sub)}</text>\n')
    return s

def txt(x, y, t, size=12, fill=INK2, mono=False, weight="400", anchor="start", style=""):
    f = MONO if mono else FONT
    st = f' font-style="{style}"' if style else ""
    return (f'<text x="{x}" y="{y}" {f} font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}" text-anchor="{anchor}"{st}>{esc(t)}</text>\n')

# ───────────────────────── FIG 1 : the green bubble ─────────────────────────
def fig_bubble():
    W, H = 900, 560
    s = head(W, H, "The GreenBubble system boundary and its external interfaces")
    # boundary
    s += (f'<rect x="210" y="70" width="480" height="420" rx="26" fill="none" '
          f'stroke="{GREEN}" stroke-width="2.4" stroke-dasharray="9 7"/>\n')
    s += txt(450, 58, "SYSTEM BOUNDARY — everything priced in the objective", 11,
             GREEN, mono=True, weight="600", anchor="middle")
    # inside
    s += txt(450, 108, "GreenBubble", 22, INK, weight="600", anchor="middle")
    s += txt(450, 130, "plants, shared buses, storage, conversion", 12, INK2, anchor="middle")
    inner = [("Biogas", 250), ("Renewables", 385), ("Electrolysis", 520),
             ("Methanol", 250), ("Methanation", 385), ("Heat & storage", 520)]
    for i, (lab, x) in enumerate(inner):
        y = 175 if i < 3 else 245
        s += box(x - 55, y, 125, 52, lab, fill=SURF)
    s += (f'<rect x="250" y="330" width="390" height="120" rx="8" fill="{SURF2}" '
          f'stroke="{LINE2}" stroke-dasharray="3 3"/>\n')
    s += txt(445, 354, "shared carrier buses  (n_flags.symbiosis)", 11.5, INK2,
             mono=True, weight="600", anchor="middle")
    rails = [("El3", C["el"]), ("H2", C["h2"]), ("CO2", C["co2"]),
             ("biogas / CH4", C["gas"]), ("Heat MT · DH · LT", C["heat"])]
    for i, (lab, col) in enumerate(rails):
        y = 374 + i * 15
        s += f'<line x1="268" y1="{y}" x2="470" y2="{y}" stroke="{col}" stroke-width="2.4"/>\n'
        s += txt(480, y + 4, lab, 10, col, mono=True, weight="600")
    # external interfaces
    ext = [("Electricity", "DK1 spot · buy/sell", C["el"], 60),
           ("Natural gas", "NG grid · buy/sell", C["gas"], 132),
           ("Hydrogen", "delivery / sale", C["h2"], 204),
           ("District heating", "heat off-take", C["heat"], 276),
           ("CO2 liquid", "sequestration", C["co2"], 348),
           ("Biochar", "sequestration credit", "#5d4037", 420)]
    for lab, sub, col, y in ext:
        s += box(20, y, 168, 54, lab, sub, accent=col)
        s += (f'<line x1="188" y1="{y+27}" x2="210" y2="{y+27}" stroke="{col}" '
              f'stroke-width="1.8" stroke-dasharray="4 3"/>\n')
    amb = [("Ambient heat", "sink", C["heat"], 132), ("Biomass markets", "pellets · chips", "#795548", 204),
           ("Methanol", "delivery / sale", C["meoh"], 276), ("Methane", "bioCH4 · eCH4", C["gas"], 348)]
    for lab, sub, col, y in amb:
        s += box(712, y, 168, 54, lab, sub, accent=col)
        s += (f'<line x1="690" y1="{y+27}" x2="712" y2="{y+27}" stroke="{col}" '
              f'stroke-width="1.8" stroke-dasharray="4 3"/>\n')
    s += txt(104, 46, "EXTERNAL INTERFACES", 10.5, INK3, mono=True, weight="600", anchor="middle")
    s += txt(796, 46, "EXTERNAL INTERFACES", 10.5, INK3, mono=True, weight="600", anchor="middle")
    s += txt(450, 520, "Interfaces carry no capital cost. Only purchase and sale prices of the",
             12, INK2, anchor="middle")
    s += txt(450, 538, "carriers crossing the boundary enter the objective.", 12, INK2, anchor="middle")
    return s + "</svg>\n"

# ──────────────────── FIG 2 : with and without symbiosis ────────────────────
def fig_symbiosis():
    W, H = 900, 400
    s = head(W, H, "The same four agents with and without the symbiosis network")
    agents = ["Biogas", "Electrolysis", "Methanol", "Heat"]

    def panel(ox, title, subtitle, connected):
        t = txt(ox + 200, 36, title, 14, INK, weight="600", anchor="middle")
        t += txt(ox + 200, 55, subtitle, 11, INK3, mono=True, anchor="middle")
        for i, a in enumerate(agents):
            x = ox + 18 + i * 96
            t2 = SURF2 if connected else SURF
            t += (f'<rect x="{x}" y="80" width="84" height="54" rx="6" fill="{t2}" '
                  f'stroke="{LINE2}"' + ('' if connected else ' stroke-dasharray="4 4"') + '/>\n')
            t += txt(x + 42, 112, a, 11, INK, weight="600", anchor="middle")
            if connected:
                t += f'<line x1="{x+42}" y1="134" x2="{x+42}" y2="196" stroke="{LINE2}" stroke-width="1.4"/>\n'
                t += f'<circle cx="{x+42}" cy="196" r="3.4" fill="{C["h2"]}"/>\n'
            else:
                t += f'<line x1="{x+42}" y1="134" x2="{x+42}" y2="158" stroke="{LINE2}" stroke-width="1.4" stroke-dasharray="3 3"/>\n'
                t += txt(x + 42, 176, "external", 9.5, INK3, mono=True, anchor="middle")
                t += txt(x + 42, 188, "grid only", 9.5, INK3, mono=True, anchor="middle")
        if connected:
            for i, (lab, col) in enumerate([("El3", C["el"]), ("H2", C["h2"]), ("CO2", C["co2"]), ("Heat", C["heat"])]):
                y = 196 + i * 17
                t += f'<line x1="{ox+18}" y1="{y}" x2="{ox+378}" y2="{y}" stroke="{col}" stroke-width="2.6"/>\n'
                t += txt(ox + 384, y + 4, lab, 9.5, col, mono=True, weight="600")
            t += txt(ox + 200, 292, "one hub — agents trade directly", 11.5, INK2, anchor="middle")
        else:
            for i in range(4):
                y = 196 + i * 17
                t += (f'<line x1="{ox+18}" y1="{y}" x2="{ox+56}" y2="{y}" stroke="{LINE}" '
                      f'stroke-width="2.6" stroke-dasharray="1 5"/>\n')
                t += (f'<line x1="{ox+340}" y1="{y}" x2="{ox+378}" y2="{y}" stroke="{LINE}" '
                      f'stroke-width="2.6" stroke-dasharray="1 5"/>\n')
            t += txt(ox + 200, 240, "no shared buses", 12, C["meoh"], weight="600", anchor="middle")
            t += txt(ox + 200, 262, "meoh and methanation cannot build at all", 11, INK2, anchor="middle")
            t += txt(ox + 200, 292, "four standalone plants", 11.5, INK2, anchor="middle")
        return t

    s += f'<rect x="14" y="16" width="416" height="310" rx="10" fill="none" stroke="{LINE}"/>\n'
    s += f'<rect x="470" y="16" width="416" height="310" rx="10" fill="none" stroke="{LINE}"/>\n'
    s += panel(14, "symbiosis: true", "the default", True)
    s += panel(470, "symbiosis: false", "plants isolated", False)
    s += txt(450, 360, "The symbiosis flag builds the shared distribution buses. It is the difference", 12, INK2, anchor="middle")
    s += txt(450, 378, "between an industrial cluster and four plants that happen to share a map pin.", 12, INK2, anchor="middle")
    return s + "</svg>\n"

# ─────────────────────── FIG 3 : agents and their techs ───────────────────────
def fig_agents():
    W, H = 980, 520
    s = head(W, H, "Agents switched on by n_flags, and the competing technologies inside each")
    agents = [
        ("Renewables", "renewables", ["onwind", "solar", "grid connection"], C["el"]),
        ("Electrolysis", "electrolysis", ["AEC", "PEMEC", "SOEC"], C["h2"]),
        ("Central heat", "central_heat", ["biomass boiler", "NG boiler", "El boiler", "heat pump", "pyrolysis"], C["heat"]),
        ("Biogas", "biogas", ["biogas", "biogas upgrading", "biogas storage", "biogas engine", "dewatering"], C["gas"]),
        ("Methanation", "methanation", ["biomethanation", "biomethanation CO2", "methanation biogas", "methanation CO2"], C["bio"]),
        ("Methanol", "meoh", ["methanolisation", "methanol from biogas", "— or the split —", "methanol synthesis", "methanol distillation"], C["meoh"]),
        ("Storage", "storage", ["battery", "H2 HP storage", "CO2 HP / Liq", "TES DH", "TES concrete"], C["co2"]),
    ]
    cols, gap, x0 = 4, 16, 16
    bw = (W - 2 * x0 - gap * (cols - 1)) / cols
    for i, (lab, flag, techs, col) in enumerate(agents):
        cx = x0 + (i % cols) * (bw + gap)
        cy = 56 + (i // cols) * 232
        s += (f'<rect x="{cx}" y="{cy}" width="{bw}" height="212" rx="8" fill="{SURF}" '
              f'stroke="{LINE2}" stroke-width="1.2"/>\n')
        s += f'<rect x="{cx}" y="{cy}" width="{bw}" height="4" rx="2" fill="{col}"/>\n'
        s += txt(cx + 13, cy + 26, lab, 13, INK, weight="600")
        s += txt(cx + 13, cy + 42, "n_flags." + flag, 10, col, mono=True, weight="600")
        s += f'<line x1="{cx}" y1="{cy+52}" x2="{cx+bw}" y2="{cy+52}" stroke="{LINE}"/>\n'
        for j, t in enumerate(techs):
            ty = cy + 74 + j * 21
            if t.startswith("—"):
                s += txt(cx + 13, ty, t, 9.5, INK3, mono=True)
            else:
                s += f'<circle cx="{cx+18}" cy="{ty-4}" r="2.4" fill="{INK3}"/>\n'
                s += txt(cx + 28, ty, t, 10.5, INK2, mono=True)
        if len(techs) > 1:
            s += txt(cx + 13, cy + 198, f"{len([t for t in techs if not t.startswith('—')])} technologies compete",
                     9.5, INK3, style="italic")
    s += txt(16, 32, "AGENTS — each is one builder in prepare_network.py", 11, INK3, mono=True, weight="600")
    s += txt(490, 504, "The optimiser sizes any subset of the technologies inside an active agent, including none.",
             11.5, INK2, anchor="middle")
    return s + "</svg>\n"

os.makedirs(OUT, exist_ok=True)
for name, fn in [("system_boundary", fig_bubble), ("symbiosis_on_off", fig_symbiosis), ("agents_technologies", fig_agents)]:
    p = os.path.join(OUT, name + ".svg")
    io.open(p, "w", encoding="utf-8").write(fn())
    print(f"  wrote {p}  ({os.path.getsize(p)} bytes)")
