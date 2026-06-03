"""
surrogate.py — the trained netlist surrogate as a fast oracle.

The trainer fits a small GNN to predict a JoSIM label (bias-margin width) directly from
a netlist's element graph, and saves a checkpoint. This module loads that
checkpoint and exposes a microsecond-scale ``predict`` for use INSIDE the search inner
loop. The whole point of the surrogate is speed: where ``gensearch.margin_fitness``
spends a full JoSIM margin-walk (many simulations) per candidate, this predicts the
same width from the candidate's graph in ~microseconds — so the search can screen a
large pool cheaply and pay JoSIM only on the finalists.

Honesty rule preserved end-to-end: the surrogate only PROPOSES (ranks candidates);
JoSIM still DISPOSES (verifies the finalists). Nothing here changes who the judge is.

Clean-IP: loads only our own trained head, operates only on our own emitted netlists.
"""
from __future__ import annotations

import torch

from margo.netlist import emit_cir

from .graph_features import graph_from_cir
from .train_netlist_surrogate import NetlistGNN, to_tensors


class NetlistSurrogate:
    """Loads a trained checkpoint and predicts the trained target (default
    bias_margin_width) from either a ``.cir`` string or a (CellProblem, spec) pair."""

    def __init__(self, ckpt_path: str, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.target = ckpt.get("target", "bias_margin_width")
        self.val_mae = ckpt.get("val_mae")
        self.model = NetlistGNN(hidden=ckpt["hidden"]).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    @torch.no_grad()
    def predict_cir(self, cir_text: str) -> float:
        """Predicted target value for a runnable netlist deck (the honest input)."""
        g = graph_from_cir(cir_text)
        x, src, dst = to_tensors(g, self.device)
        return float(self.model(x, src, dst).item())

    def predict_spec(self, problem, spec, exp_name: str | None = None) -> float:
        """Predicted target for a CellProblem spec. Builds the canonical experiment
        deck (the same representation the corpus stored), emits its ``.cir``, and
        predicts — exactly the train-time input pipeline, so no train/serve skew."""
        exp = (problem.experiments[0] if exp_name is None
               else next(e for e in problem.experiments if e.name == exp_name))
        cir = emit_cir(problem.build_fn(spec, exp.stimulus))
        return self.predict_cir(cir)
