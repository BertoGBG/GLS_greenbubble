Guide for Process Streams (``p_config``)
========================================

Where physical stream state lives, how a process declares its ports, and the one
invariant you must not break.

Overview
--------

GreenBubble separates three kinds of input on purpose:

.. list-table::
   :header-rows: 1
   :widths: 24 34 42

   * - File
     - Holds
     - Example
   * - ``technology-data`` / ``tech_inputs``
     - **Magnitudes** — how much
     - ``heat-input: 0.1047 MWh_th/MWh_MeOH``
   * - ``config/p_config.default.yaml``
     - **State** — at what condition
     - ``{fluid: Water, T: 180, P: 10}``
   * - ``config/n_config.default.yaml``
     - **Network component config** — capacity, expansion, ramps
     - ``expansion: true``

The split between the first two is not cosmetic. ``technology-data`` reports *net*
duties and never temperatures: the DEA sheet gives 0.58 MWh/t of reboiler steam
without saying at what temperature it is required. Heat integration is impossible
without the second half, and no amount of upstream data will supply it — so
temperature attribution has to live here.

**The rule that follows: never copy a magnitude into ``p_config``, and never write a
temperature into ``tech_inputs``.** Duplicated numbers drift.

File structure
--------------

``p_config.default.yaml`` has three sections. A gitignored ``config/p_config.yaml``
is deep-merged on top, exactly like ``config.yaml`` and ``n_config.yaml``.

``globals``
~~~~~~~~~~~

Primitives referenced elsewhere as ``${...}``::

    globals:
      T_max_comp: 160        # max compressor discharge temperature [C]
      T_ambient:  20         # [C]
      lhv: {ch4: 13.9, h2: 33.33, meoh: 5.54}
      mixtures:
        biogas: {Methane: 0.65, CarbonDioxide: 0.35}

Anything **derived** stays in Python. ``lhv.biogas`` is computed at load time from
``mixtures.biogas`` and the CoolProp molar masses, then exposed as ``${lhv.biogas}``
— it is never typed into the YAML, where it could drift from the composition it is
supposed to describe.

``circuits``
~~~~~~~~~~~~

The pressurised heat-transfer loops. **Pressure is the input**; the temperatures are
declared against it and validated::

    circuits:
      "Heat MT":
        fluid: "Water"
        P: 12          # T_sat = 188.0 C, so 180 C operates with 8 K margin
        T_min: 140     # FLOOR of the band -- what the compressor split reads
        T_max: 180     # design operating top; or the literal `saturation`
        buses: ["Heat MT", "Heat MT storage"]

Each circuit expands into the flat ``"<name> min"`` / ``"<name> max"`` streams used
throughout the model. Two checks run at load and fail the build:

* **liquid with margin** — ``T_max <= T_sat(P) - liquid_margin_K``. Water sitting
  exactly at ``P_sat`` is at its boiling point, which is not how a pumped loop runs.
  Before this block existed, MT declared 180 °C at 10 bar (``P_sat`` = 10.03) and
  140 °C at 3 bar (``P_sat`` = 3.62) — both at or below saturation, i.e. flashing.
* **cascade separation** — consecutive floors at least ``globals.dT_min`` apart.

``T_max: saturation`` derives ``T_sat(P)`` instead of validating against it. That is
how a **steam** circuit will declare itself: steam does run at saturation, so pressure
alone fixes its temperature. Hot-water loops below 100 °C are the opposite case — their
pressure comes from pump and static head, not boiling (real DH distribution runs
6–10 bar, transmission 16–25), and their top temperature is a design choice.

``shared``
~~~~~~~~~~

States belonging to the **network**, not to any one process: distribution headers,
heat tiers, storage conditions, external markets, and the normal-condition reference
states used for density calculations::

    shared:
      "H2 production":
        state:  {fluid: "H2", T: 50, P: 30}
        energy: {LHV: "${lhv.h2}"}
        model:  {carrier: "H2", buses: ["H2", "H2 distribution", "H2 delivery"]}

``processes``
~~~~~~~~~~~~~

Ports of each unit operation, keyed by the **port role** the code uses, so this
mirrors the ``*_buses`` frames in ``prepare_network.py``::

    processes:
      methanation:
        "H2 in":
          stream: "H2 to methanation"          # legacy flat name = symbiosis_n index
          from:   "shared:H2 production"       # inherit fluid / LHV / carrier
          state:  {T: "${T_max_comp}", P: 20}  # override what the compressor changes
          model:  {buses: ["H2 to methanation"]}

A port inherits from one shared state and overrides only what its unit operation
changes — in practice just ``T`` and ``P`` downstream of a compressor.

.. important::

   ``from:`` never inherits ``buses``. A bus carries exactly one state, so inheriting
   bus lists is precisely how you would create the conflict described below. Every
   port declares its own buses.

The invariant: one bus, one state
---------------------------------

**A bus carries exactly one thermodynamic state.** The loader enforces it: all ports
targeting a bus must declare the same ``(fluid, T, P, carrier)``, or the build fails
naming both offenders.

This has a physical reading, and it is the useful part:

.. list-table::
   :header-rows: 1
   :widths: 32 34 34

   * - Two processes → same bus
     - Means
     - What happens
   * - Same declared state
     - They genuinely mix
     - Allowed — the normal case
   * - Different state
     - A unit operation is missing
     - Build error
   * - Different buses
     - Already separated by a compressor/cooler
     - Nothing to check

If two processes need a stream at different conditions, that is **not** a config
conflict to be resolved by ordering — it means a compressor or cooler belongs between
them, and the model already represents those as separate buses joined by a component.
``H2 production`` at 30 bar and ``H2 to methanation`` at 20 bar are two buses with a
compressor between them, not one bus with two opinions.

Inheriting from a shared state makes agreement *automatic* rather than coincidental:
two ports that both write ``from: "shared:bioCH4"`` cannot disagree, because there is
only one copy of the number.

Adding a stream for a new technology
------------------------------------

Follow :doc:`guide_new_technology` for the component itself. For its streams:

1. **Does the state already exist?** If the port consumes something from an existing
   header (``H2 distribution``, ``CO2 distribution``, ``biogas``), inherit it —
   do not restate it.

2. **Add a ``processes:`` entry** keyed by your process name, with one entry per port
   role. Use the same role names the code uses (``H2 in``, ``CO2 in``, ``product bus``),
   because that is what makes the config readable next to ``prepare_network.py``.

3. **Only add to ``shared:``** if the state genuinely belongs to the network — a new
   distribution header, storage condition or market. A state used by one process is
   a port, not a shared state.

4. **Declare ``buses:`` on the port**, listing the bus names the model will create.
   If two ports legitimately feed one bus, have both inherit the same shared state.

5. **Run the tests**::

       pytest tests/test_p_config.py

   They pin: every stream has a bus carrier, no pressure resolves through a
   temperature, ports do not inherit buses, and the one-state-per-bus validator both
   catches conflicts and permits agreement.

Gotchas
-------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Gotcha
     - Why it bites
   * - Never write ``P: "${T_ambient}"``
     - A pressure bound to a temperature. It happened: three methanation feed
       pressures were silently tied to ``T_ambient`` because both were 20. The
       numbers matched, so nothing failed — until someone changed the ambient
       temperature. ``test_no_pressure_references_a_temperature_global`` now blocks it
   * - Do not add derived values to ``globals``
     - ``lhv.biogas`` follows from the mixture. Typing it in creates a second source
       of truth that will disagree with the first
   * - Do not put duties or costs here
     - They belong in ``technology-data`` / ``tech_inputs``. ``p_config`` answers
       *at what condition*, never *how much*
   * - ``symbiosis_n`` is a derived view
     - It is built from this file. Do not edit the frame at runtime; change the YAML

Who owns the bus mapping
------------------------

``p_config`` catalogues **states**. It does not decide which bus carries which state
— ``prepare_network.py`` does, because that is where bus names are invented.

``stream_for_bus(bus_name)`` is the single place that answers "what state is on this
bus": exact name first (from a ``buses:`` list), then a declared ``bus_suffix``.

The many-to-one case is normal and expected. A plant-local heat bus is the same
physical state as the shared tier it hangs off, so ``add_local_heat_connections``
stamps the tier's stream onto every ``Heat MT_<plant>`` it creates —
``Heat MT``, ``Heat MT_methanolisation``, ``Heat MT_methanation`` and
``Heat MT_biogas`` all resolve to ``Heat MT min``. The config never lists them; it
could not, since they are generated per plant at runtime.

Plant-prefixed buses
--------------------

Some buses are created with a runtime prefix — ``meoh H2 HP storage``,
``methanation H2 HP storage`` — so ``p_config`` cannot enumerate them in ``buses:``.
The stream declares a suffix instead::

    "H2 HP storage":
      model: {carrier: "H2", buses: ["H2 HP storage"], bus_suffix: "H2 HP storage"}

``add_requirements_buses`` tries the exact ``buses:`` list first, then falls back to
suffix matching. Before this, the rule was an ``if bus_name.endswith(...)`` naming
those two streams in source; the config now owns it.

Declared but not consumed
-------------------------

Some fields exist for work that has not landed yet. **No code reads them and they change
no results.** They are listed here because an unread field is not free — ``Heat MT max``
sat in the table unread and was twice mistaken for a binding constraint when reasoning
about what could feed the MT bus.

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Field
     - Status
   * - ``Heat MT max`` (180 °C)
     - The ceiling of the MT interval. Never read: sources injecting into MT (combustion
       and similar) are far hotter than the floor, so the roof does not bind. A tier is
       physically an interval — MT is 140→180 °C — whose fluid picks up heat progressively
       along the loop, so a source contributes over whatever span it can cover rather than
       having to deliver at the top. The model cannot express that yet: each tier is one
       bus, i.e. a single node with no internal temperature resolution
   * - ``globals.dT_min`` (10 K)
     - Minimum approach temperature. Pinch convention is to allocate on *shifted*
       temperatures — hot streams down by ``dT_min/2``, cold up by ``dT_min/2``. The full
       value is stored and the half derived; storing the half invites halving it twice.
       Exposed as ``scripts.config.p_dT_min``
   * - ``process_streams``
     - Temperature attribution of duties that live in technology-data. Empty; see below

Heat integration hooks
----------------------

``process_streams`` is declared, documented and **not yet consumed**. It is the home
for temperature attribution of duties that live in ``technology-data``::

    process_streams:
      methanol distillation:
        reboiler:
          duty_ref: "methanol distillation:heat-input"   # magnitude, by reference
          T_supply: 180
          T_target: 179
          type: sink

``duty_ref`` points at a technology-data parameter rather than copying it. Supply and
target temperatures plus a duty are exactly the input a pinch analysis or
heat-exchanger-network synthesis needs, and grouping by process means the streams
arrive already sorted by unit operation.

Exposed as ``scripts.config.p_process_streams``. Populate it as each process is
characterised.
