.. SPDX-FileCopyrightText: Contributors to GreenBubble
.. SPDX-License-Identifier: CC-BY-4.0

.. _model-anatomy:

Model Anatomy (interactive)
===========================

The same structure described in :doc:`model_approach`, but as something you can
operate. Toggle an ``n_flags`` entry and the diagram redraws: the shared carrier
rails appear or are severed, agents grey out, and any agent blocked by a
dependency says which one it is missing.

The dependency rules are the real ones, the page runs the same logic as
``network_dependencies()`` in ``scripts/prepare_network.py``. Three things are
worth trying:

* Turn off **symbiosis**. The rails break, and ``meoh`` and ``methanation`` go
  dark: neither can build without the shared hydrogen and CO₂ buses.
* Turn off **biogas**. ``meoh`` and ``methanation`` fail again, for a different
  reason — no CO₂ source.
* Turn off **electrolysis**, then everything else in turn. ``renewables``
  eventually blocks itself, because it needs at least one on-site consumer
  before it may be built.

.. raw:: html

   <div style="border:1px solid var(--pst-color-border,#d3dbe4);border-radius:10px;
               overflow:hidden;margin:1.4rem 0;">
     <iframe id="gb-anatomy" src="_static/model/anatomy.html"
             title="Interactive diagram of the GreenBubble model structure"
             loading="lazy"
             style="width:100%;height:1560px;border:0;display:block;"></iframe>
   </div>
   <script>
   window.addEventListener("message", function(e){
     var d = e && e.data;
     if (!d || typeof d.gbAnatomyHeight !== "number") return;
     var f = document.getElementById("gb-anatomy");
     if (f && d.gbAnatomyHeight > 400 && d.gbAnatomyHeight < 6000)
       f.style.height = d.gbAnatomyHeight + "px";
   });
   </script>
   <p style="font-size:0.9em;margin-top:-0.6rem;">
     <a href="_static/model/anatomy.html" target="_blank" rel="noopener">
       Open the diagram in its own tab ↗</a>
   </p>

.. note::

   The diagram is a faithful picture of the **structure**, with two deliberate
   simplifications. ``electrolysis`` is drawn as always permitted, whereas its
   real gate also depends on ``rfnbos_dict.limit`` and on whether the run is in
   demand or price mode — too conditional to draw honestly in one box. And
   ``central_heat`` and ``storage`` have no hard dependencies in the code, so
   they never block.

For the static versions of these figures, and for everything the diagram does not
show, the system boundary, the pressure ladder, the heat circuits — see
:doc:`model_approach`.
