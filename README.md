# Margo

**Margin analysis and inverse design for superconducting (SFQ) logic cells — a JoSIM-verified oracle paired with a fast machine-learning surrogate for cell-level design-space exploration.**

From [Cogitan](https://cogitan.ai) — superconducting design tools. Website: [cogitan.ai](https://cogitan.ai).

Margo is a Python toolkit for working with single-flux-quantum (SFQ) logic cells at the
netlist level. It pairs a JoSIM-based oracle — functional truth tables plus bias- and
critical-current margin sweeps — with a learned netlist surrogate that estimates cell
margins quickly, enabling surrogate-accelerated design-space search.

> **Scope / honesty note.** Margins are computed against idealized junction models, so
> they are **simulator-level and relative** — useful for ranking, optimization, and
> design-space exploration, but **not fab-calibrated**. Absolute, process-accurate
> margins require a real device model (PDK extraction), which is out of scope here.

## What's inside

- **`margo.netlist`** — a lossless `.cir` parser/emitter, a pre-simulation DRC/validity
  gate, and a JoSIM verifier with Ic/bias margin sweeps. Round-trips netlists to and
  from JoSIM.
- **`margo.builders`** — clean-room SFQ cell builders (JTL, splitter, feeder, storage
  loop) authored from RSFQ device physics.
- **`margo.comparator`, `margo.and2`** — a clocked comparator D flip-flop and a clocked
  AND2/OR2 coincidence gate.
- **`margo.gensearch`, `margo.cell_problems`** — a cell-agnostic inverse-design engine
  that co-tunes cell parameters to widen JoSIM-verified margins, plus ready-made cell
  problems (JTL, splitter, DFF, AND2, OR2).
- **`margo.corpus`** — builds a JoSIM-labeled cell corpus (functional + margin labels).
- **`margo.graph_features`, `margo.train_netlist_surrogate`, `margo.surrogate`** — a
  netlist-to-graph converter and a small message-passing GNN that predicts cell margins.
- **`margo.search_accel`** — "surrogate proposes, JoSIM disposes": screen many candidate
  cells with the surrogate in milliseconds, then verify only the finalists in JoSIM.

A baseline trained surrogate (`checkpoints/netlist_surrogate_medium.pt`) and the corpus
it was trained on (`data/corpus_medium_n250.jsonl`, 750 cells) are included.

## Install

```bash
pip install -e .          # core (numpy)
pip install -e ".[ml]"    # + torch, for the surrogate
```

**JoSIM** is required for the verification/oracle features (margin sweeps, truth-table
checks). Install `josim-cli` and make it reachable. *Note:* the verifier currently
invokes `josim-cli` through WSL (`wsl -e bash -lc ...`), which suits a Windows + WSL
setup; adapt `margo/netlist/verifier.py` if you run `josim-cli` natively. The surrogate
(margin prediction) works without JoSIM.

## Quickstart

```python
from margo.netlist import parse_cir, emit_cir
from margo.builders import JTLSpec, build_jtl_testbench

# build a cell and round-trip it through the netlist layer
deck = build_jtl_testbench(JTLSpec(), pulse_ps=[200.0])
cir_text = emit_cir(deck)

# fast surrogate margin prediction (no JoSIM needed)
from margo.surrogate import NetlistSurrogate
surrogate = NetlistSurrogate("checkpoints/netlist_surrogate_medium.pt")
print(surrogate.predict_cir(cir_text))   # predicted bias-margin width

# JoSIM-verified inverse design (requires josim-cli)
from margo.cell_problems import comp_dff_problem
from margo import gensearch
problem = comp_dff_problem()
# gensearch.search_margin(problem, ...) widens the cell's margin, JoSIM-verified
```

## License

MIT — see `LICENSE`. Copyright (c) 2026 [Cogitan](https://cogitan.ai).
