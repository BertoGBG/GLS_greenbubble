.. SPDX-FileCopyrightText: Contributors to GreenBubble
.. SPDX-License-Identifier: CC-BY-4.0

.. _carbon-balance:

Carbon Balance (interactive)
============================

This page shows every route that carbon can take through the model. Carbon enters
with the digester feedstock, bought biomass and natural gas. It leaves as biomethane,
methanol, liquid CO₂, biochar, digestate returned to the fields, or CO₂ released to air.

The diagram shows which routes exist before optimisation, not how much carbon flows.
The optimiser chooses among them. Switch agents and technologies as in ``n_flags`` and
``n_config`` to see how the routes change. The starting point is
``config.default.yaml`` with ``n_config.default.yaml``.

Methanol has two carbon routes: CO₂ from the CO₂ distribution bus, and biogas.
The ``meoh split`` switch replaces ``methanolisation`` with methanol synthesis, a
crude methanol bus and distillation. The biogas route stays available in both
cases.

.. raw:: html

   <div style="border:1px solid var(--pst-color-border,#d3dbe4);border-radius:10px;
               overflow:hidden;margin:1.4rem 0;">
     <iframe id="gb-carbon" src="_static/model/carbon_balance.html"
             title="Carbon routes in the GreenBubble model"
             loading="lazy"
             style="width:100%;height:1600px;border:0;display:block;"></iframe>
   </div>
   <script>
   window.addEventListener("message", function(e){
     var d = e && e.data;
     if (!d || typeof d.gbCarbonHeight !== "number") return;
     var f = document.getElementById("gb-carbon");
     if (f && d.gbCarbonHeight > 400 && d.gbCarbonHeight < 8000)
       f.style.height = d.gbCarbonHeight + "px";
   });
   </script>
   <p style="font-size:0.9em;margin-top:-0.6rem;">
     <a href="_static/model/carbon_balance.html" target="_blank" rel="noopener">
       Open the diagram in its own tab ↗</a>
   </p>

Upstream emissions are outside the model boundary. These include field emissions,
manure storage, transport and methane slip.
