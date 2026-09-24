Guide: Methanol synthesis / distillation split
==============================================

Splitting ``methanolisation`` into two links with a crude-methanol tank between
them, so the reactor and the column can run at different times.

Why
---

The aggregate ``methanolisation`` link forces synthesis and distillation to
operate in lockstep: one link, one dispatch variable. A real plant can hold crude
methanol in a tank and distil it later, which lets the reactor follow cheap
electricity while the column runs on its own schedule.

That flexibility is worth something, and the point of the split is to find out how
much — as a **difference between two solves**, not as an assumption.

Enabling it
-----------

.. code-block:: yaml

   # config/n_config.yaml
   options:
     meoh split:
       enable: true

   methanol synthesis:    {expansion: true}
   methanol distillation: {expansion: true}
   crude MeOH storage:    {expansion: true}

``enable: false`` (the default) leaves the monolithic ``methanolisation`` link
untouched. The two modes are mutually exclusive — ``prepare_network.py`` selects
one set of technologies or the other:

.. code-block:: python

   techs = ["methanol synthesis", "methanol distillation"] if meoh_split else ["methanolisation"]

so there is no need to disable ``methanolisation`` by hand.

Network structure
-----------------

.. list-table::
   :header-rows: 1
   :widths: 26 20 54

   * - Component
     - Type
     - Role
   * - ``methanol synthesis``
     - Link
     - H₂ + CO₂ + el → crude MeOH, rejecting reactor heat to **Heat MT**.
       ``bus0`` is H₂, so capacity is in MW\_H2, as for ``methanolisation``.
   * - ``crude MeOH``
     - Bus
     - Intermediate methanol/water mixture, 64.0 wt% MeOH
   * - ``crude MeOH store``
     - Store
     - The buffer. ``e_cyclic=True``. Costed from ``methanol storage``
       in technology-data.
   * - ``methanol distillation``
     - Link
     - crude MeOH → product, consuming reboiler heat from **Heat MT** and
       rejecting condenser heat to **Heat LT**. ``bus0`` is crude MeOH, so
       capacity is in MW\_MeOH contained.

The collapse invariant
----------------------

The split must be able to reproduce the monolithic solution exactly, or a
comparison between them is meaningless. Two properties guarantee it.

**Capital cost.** The 90/10 CAPEX split is applied to the same total, so the two
halves sum to the aggregate figure to the last digit — investment *and* annualised
``fixed``:

.. code-block:: text

   methanol synthesis      1228.2706 EUR/kW      fixed 133,373.4870
   methanol distillation    136.4745 EUR/kW      fixed  14,819.2751
                           ----------                  ------------
   methanolisation         1364.7451 EUR/kW      fixed 148,192.7621

**Heat.** DEA reports only two plant-level numbers (net heat in 0.1047, heat out
0.2562); a split needs three. The missing degree of freedom is **X**, the heat the
synthesis block exports. ``prepare_network.py`` *derives* the reboiler rather than
transcribing it:

.. code-block:: text

   synthesis    heat-output =  X
   distillation heat-input  =  X + 0.1047     <- derived
   distillation heat-output =      0.2562

so the coupled net is ``(X + 0.1047) - X = 0.1047`` identically, **for any X**.
With the store sized at zero the two links must run in lockstep, and the pair is
equivalent to ``methanolisation``. The split's feasible set therefore *contains*
the monolithic one, and split cost ≤ monolithic cost necessarily.

Where the numbers come from
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Quantity
     - Value
     - Source
   * - CAPEX split
     - 90 / 10
     - [OLI] Table S17, summed by unit. A CO₂ + H₂ plant fed at H₂/CO₂ = 3.000,
       i.e. our exact stoichiometry. Excludes feed compression (modelled
       separately here) and their purge burner / Rankine cycle.
   * - X (synthesis heat-out)
     - 0.1288
     - [OLI] Table S14, HE5, the duty that measurably crosses the
       synthesis/distillation boundary
   * - Condenser heat-out
     - 0.2562
     - [DEA] sheet 98 district-heat row, rebased per MWh\_MeOH
   * - FOM
     - 2.8 %/year
     - [DEA] sheet 98. A percentage of investment, so it is scale-free and the
       90/10 split carries it automatically
   * - Tank cost
     - 851.98 EUR/MWh\_MeOH
     - technology-data ``methanol storage``, derived from [MAG] Table A12 with
       Chilton installation factors
   * - Water out
     - 0.5622 t/t
     - Derived, not data: ``M_H2O / M_MeOH``. CO₂ carries two oxygens and
       methanol one, so the spare must leave with hydrogen

**[MAG] is deliberately absent from the heat and CAPEX basis.** It is an SMR
syngas plant: its crude is 10.6 wt% water against our 36.0 wt%, its columns strip
reformer inerts we do not have, and it targets Grade AA at reflux > 7. Useful only
as a documented example of the decoupled architecture.

Heat bus wiring
---------------

* **Reboiler → Heat MT.** [OLI] put the reboiler at 99.6 °C, fed by 1.43 bar steam
  at 110 °C, so it needs a source *above* the district-heating temperature. Heat DH's
  ``T_min`` of 90 °C is the DH **supply** temperature, not the saturation
  temperature of its 6 bar — DH delivers at 90 °C and cannot drive a 110 °C
  reboiler. Heat MT (140–180 °C) can.
* **Condenser → Heat LT.** [OLI] measure it at 53 °C, which is LT-grade. Surplus
  leaves through the tier's ambient dump, so this cannot make the network
  infeasible. It matters only when DH off-take is switched on: with
  ``options['DH']['enable']`` false (the default) the sole sink on Heat DH is the
  ambient dump, so DH and LT are economically identical, a demand-mode pair
  before and after the change returned bit-identical objectives. With DH sales
  enabled, 53 °C heat can no longer be sold straight to the grid and reaching the
  90 °C supply needs the ``heat pump`` technology.

Reading the results
-------------------

**Total value of decoupling** = the objective difference between a split and a
monolithic solve of the same case. This is the number to quote.

**Marginal value** = the dual on the store's *capacity*, which at an interior
optimum equals its capital cost (85.70 EUR/MWh/yr). A consistency check, not a
finding.

**The dual on the ``crude MeOH`` bus is NOT the value of decoupling.** It is the
shadow price of crude methanol, and reads as the product price net of the
distillation step:

.. code-block:: text

   lambda Methanol collection  -  lambda crude MeOH  =  marginal cost of distilling
                      144.15   -            137.90  =  6.25 EUR/MWh   (demand mode)
                      600.00   -            593.41  =  6.59 EUR/MWh   (price mode)

That agreement across two price regimes is the useful part: the marginal cost of
the distillation step is a property of the technology, and the split is what makes
it observable at all. The *spread* of the crude dual across hours is what the store
arbitrages.

Worked example
--------------

GreenLab brownfield (biogas 30 MW CH₄, wind 52 MW, solar 30 MW fixed), methanation
off, RFNBO ``limit: price`` at 20 EUR/MWh, 4 h resolution. ``biogas upgrading`` is
pinned at 262,800 MWh/yr in every case, so CO₂ availability binds.

.. code-block:: text

   price driver
     300 EUR/MWh   obj -4.30e7 vs -4.18e7   MeOH  81,378 vs 80,564   store 151.6 MWh
     400 EUR/MWh   obj -5.19e7 vs -5.04e7   MeOH 100,335 vs 97,004   store 320.1 MWh

   demand driver, 60,000 MWh/y
     objective                 8,838,907 -> 8,036,080   -9.08%
     lambda Methanol delivery     166.40 ->    144.15   -13.37%

The split is cheaper *and* produces more methanol from the same CO₂. In demand mode
the shadow price falls further than total cost, i.e. the store relieves the binding
constraint rather than only buying infra-marginal savings.

Known approximations
--------------------

* **X is a modelling choice, not data.** Any value in 0.0772–0.1288 reproduces DEA
  equally well, and X *is* the cost of decoupling. 0.0772 is the reaction enthalpy
  alone ([MBA] / [OLI] reactor); 0.1288 is [OLI]'s measured HE5, used here because
  it is the conservative end. Sweep it before quoting a flexibility result.
* **Only 0.0772 of the 0.1288 synthesis export is genuinely MT-grade**; the balance
  is [OLI]'s HE5 stream at roughly 150 → 60 °C, so wiring it all to Heat MT is
  generous by about 40 % of that export.
* **Electricity is split 50/50** between the two halves, an admission of
  ignorance, not an estimate. Needs a flowsheet.
* The reactor at 247.5 °C is above every heat band the model has.

References
----------

``scripts/technology_inputs.py`` carries the full citations in its header:
[DEA] Danish Energy Agency sheet 98; [ALA] doi:10.1016/j.enconman.2024.119175;
[OLI] doi:10.3390/pr10081535; [MBA] doi:10.1039/D1SE00635E;
[MUC] arXiv:2305.18338; [MAG] Methanol Magic LLC (ChE473k, UT Austin, 2015).
