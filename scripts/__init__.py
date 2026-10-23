"""GreenBubble model scripts."""
import pypsa

# pandas 3 infers the `str` dtype for text. Keep PyPSA's object dtype on import
# until the code has been checked against `str` columns.
pypsa.options.api.legacy_string_dtype = True

# Strict linopy arithmetic: align by label, raise on NaN and ambiguity.
# Default in linopy 1.0; opt-in in 0.10.
import linopy
linopy.options["semantics"] = "v1"
