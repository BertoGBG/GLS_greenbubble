# GreenBubble

[![Documentation](https://readthedocs.org/projects/gls-greenbubble/badge/?version=latest)](https://gls-greenbubble.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Built on PyPSA](https://img.shields.io/badge/built%20on-PyPSA-orange)](https://pypsa.readthedocs.io)

**GreenBubble** is an open-source techno-economic optimisation model for **Power-to-X industrial clusters** — co-located plants that share electricity, hydrogen, CO₂, biomethane, methanol and heat infrastructure. Built on [PyPSA](https://pypsa.readthedocs.io), it co-optimises **capacity expansion** and **hourly dispatch** across a multi-energy hub over a full year at 1-hour resolution.

<p align="center">
<img src="https://github.com/user-attachments/assets/5f6ee063-35cb-4a9e-b6d0-26efd2ed2069" width="90%" alt="GreenBubble network diagram" />
</p>

The model was developed based on the [GreenLab Skive](https://www.greenlab.dk) industrial park — an agricultural-industrial hub in Denmark that integrates biogas, electrolysis, methanation and methanol synthesis. The methodology is described in:

> *Optimizing hydrogen and e-methanol production through Power-to-X integration in biogas plants*, Energy Conversion and Management, 2024.  
> DOI: [10.1016/j.enconman.2024.119175](https://doi.org/10.1016/j.enconman.2024.119175)

📖 **Full documentation:** [gls-greenbubble.readthedocs.io](https://gls-greenbubble.readthedocs.io)

---

## Key features

| | |
|---|---|
| **Multi-energy LP** | Electricity, H₂, CO₂, biomethane, methanol and heat (3 temperature levels) in a single solve |
| **Capacity + dispatch** | No decomposition — capacity expansion and hourly operation co-optimised |
| **Greenfield & brownfield** | Existing assets parameterised by construction year and remaining investment fraction |
| **Stochastic scenarios** | Multi-scenario LP with expected value of perfect information (EVPI) |
| **Rolling horizon** | Dispatch-only mode on a fixed network for operational studies |
| **Industrial symbiosis** | Shapley-value cost allocation across co-located partners |
| **RFNBO compliance** | Additionality and emission constraints for renewable hydrogen certification |
| **Economic post-processing** | LCOP, short-run marginal cost (SRMC), KKT shadow prices, annual profit per technology |

---

## Technologies (examples)

GreenBubble is designed to be extensible. The following are examples of technologies currently implemented — the list is not exhaustive:

**Hydrogen production** — Alkaline electrolysis

**Biomethane production** — Biogas upgrading · Biomethanation of biogas or CO₂ · Catalytic methanation of biogas or CO₂

**Methanol production** — CO₂ hydrogenation

**Renewable electricity** — Onshore wind · Solar PV

**Storage** — Li-ion batteries · H₂ in steel vessels · CO₂ liquefaction · Pressurised CO₂ cylinders · Hot-water thermal storage · Concrete-based thermal energy storage

**Biomass handling** — Belt dryer · Digestate dewatering

**Shared infrastructure** — H₂, CO₂ and heat distribution networks · Gas compressors · Grid connection

---

## Quick start

```bash
git clone https://github.com/BertoGBG/GLS_greenbubble.git
cd GLS_greenbubble

# Create environment — shown for macOS Apple Silicon; see docs for other platforms
conda config --add channels conda-forge && conda config --set channel_priority strict
conda install -n base -c conda-forge conda-lock
conda-lock install -n greenbubble-pypsa107 --platform osx-arm64 envs/locks/conda-lock-osx-arm64.yml
conda activate greenbubble-pypsa107

# Copy and fill in API tokens (required for data retrieval)
cp .env.example .env

# Preview the execution plan, then run
snakemake -n
snakemake -j4
```

See the [installation guide](https://gls-greenbubble.readthedocs.io/en/latest/installation.html) for platform-specific instructions and solver setup (Gurobi / HiGHS).
New to Snakemake? See the [Snakemake documentation](https://snakemake.readthedocs.io/en/stable/).

---

## Licence and citation

- Code: MIT  
- Documentation: CC-BY-4.0

If you use GreenBubble in your research, please cite:

```bibtex
@article{greenbubble2024,
  title   = {Optimizing hydrogen and e-methanol production through
             Power-to-X integration in biogas plants},
  journal = {Energy Conversion and Management},
  year    = {2024},
  doi     = {10.1016/j.enconman.2024.119175},
}
```

---

## Contributors

Developed at:

- [DTU Wind and Energy Systems — Power and Energy Systems Division](https://wind.dtu.dk/research/research-divisions/power-and-energy-systems), Technical University of Denmark
- [Department of Mechanical and Production Engineering](https://mpe.au.dk/en/), Aarhus University
- [Department of Biological and Chemical Engineering](https://bce.au.dk/en/), Aarhus University
