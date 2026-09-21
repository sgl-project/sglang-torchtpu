# TPU Raiden runtime verification

`formulas.ltl` contains one LTL property per line. It is passed through the C
preprocessor before Spot parses it, so `#define` macros and `//` comments are
supported. All formulas are conjoined into one monitor.

Generate the runtime monitor and optional inspection artifacts with:

```bash
python3 generate.py
```

Spot and its `buddy` bindings, `cpp`, and Graphviz `dot` are needed only for
generation. The generated Python monitor has no runtime dependency on Spot.
Run `python3 generate.py --help` to override or disable individual outputs.
