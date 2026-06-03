"""
train_netlist_surrogate.py — train the netlist-graph margin surrogate.

A small graph network is trained DIRECTLY on:

    input  = the element graph (graph_features) of a real netlist
    target = a JoSIM label (bias-margin width) from the gensearch oracle

We use a small, purpose-built netlist GNN (plain PyTorch, no PyG dependency): the
netlist graph is the natural input, and a small fast model is what drops into the
search inner loop.

Usage:
    python -m margo.train_netlist_surrogate data/<corpus>.jsonl \
        [--target bias_margin_width] [--epochs 400]
"""
from __future__ import annotations

import argparse
import json
import random

import torch
import torch.nn as nn

from .graph_features import FEAT_DIM, graph_from_cir


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


def load_corpus(path: str, target: str):
    """Return a list of (graph_dict, target_float, cell_type) for functional cells.
    Non-functional cells have undefined margins (width 0) and would bias the head; we
    keep them only if their target is a real number."""
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            y = r.get(target)
            if y is None:
                continue
            g = graph_from_cir(r["cir"])
            rows.append((g, float(y), r["cell_type"]))
    return rows


def to_tensors(g: dict, device):
    x = torch.tensor(g["x"], dtype=torch.float32, device=device)
    src = torch.tensor(g["edge_index"][0], dtype=torch.long, device=device)
    dst = torch.tensor(g["edge_index"][1], dtype=torch.long, device=device)
    return x, src, dst


# --------------------------------------------------------------------------- #
# Model — a tiny message-passing GNN (no PyG)
# --------------------------------------------------------------------------- #


class NetlistGNN(nn.Module):
    def __init__(self, in_dim: int = FEAT_DIM, hidden: int = 32, layers: int = 2):
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden)
        self.self_w = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        self.neigh_w = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(self, x, src, dst):
        h = torch.relu(self.embed(x))
        n = h.shape[0]
        for sw, nw in zip(self.self_w, self.neigh_w):
            if src.numel() > 0:
                agg = torch.zeros_like(h)
                agg.index_add_(0, dst, h[src])
                deg = torch.zeros(n, device=h.device).index_add_(
                    0, dst, torch.ones_like(dst, dtype=torch.float32))
                deg = deg.clamp(min=1.0).unsqueeze(1)
                agg = agg / deg
            else:
                agg = torch.zeros_like(h)
            h = torch.relu(sw(h) + nw(agg))
        pooled = h.mean(dim=0, keepdim=True)        # global mean pool
        return self.head(pooled).squeeze()


# --------------------------------------------------------------------------- #
# Train / eval
# --------------------------------------------------------------------------- #


def split(rows, val_frac=0.25, seed=0):
    """Stratified-ish split by cell type so val has both cell types."""
    rng = random.Random(seed)
    by_type: dict[str, list] = {}
    for r in rows:
        by_type.setdefault(r[2], []).append(r)
    train, val = [], []
    for _, group in by_type.items():
        rng.shuffle(group)
        k = max(1, int(round(len(group) * val_frac)))
        val += group[:k]
        train += group[k:]
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def mae(model, rows, device):
    model.eval()
    errs = []
    with torch.no_grad():
        for g, y, _ in rows:
            x, src, dst = to_tensors(g, device)
            pred = model(x, src, dst).item()
            errs.append(abs(pred - y))
    return sum(errs) / len(errs) if errs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--target", default="bias_margin_width")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default=None,
                    help="path to write the best checkpoint (.pt) for inference")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = load_corpus(args.corpus, args.target)
    train, val = split(rows, seed=args.seed)
    print(f"corpus={len(rows)} (train={len(train)} val={len(val)}) target={args.target} "
          f"device={device}")

    # Baseline: predict the train mean.
    ymean = sum(y for _, y, _ in train) / len(train)
    base_val_mae = sum(abs(y - ymean) for _, y, _ in val) / len(val)
    yspread = (max(y for _, y, _ in rows) - min(y for _, y, _ in rows))
    print(f"target range spread={yspread:.3f}  train_mean={ymean:.3f}  "
          f"baseline(val,predict-mean) MAE={base_val_mae:.4f}")

    model = NetlistGNN(hidden=args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lossf = nn.SmoothL1Loss()

    best_val = float("inf")
    best_state = None
    for ep in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train)
        tot = 0.0
        opt.zero_grad()
        for g, y, _ in train:
            x, src, dst = to_tensors(g, device)
            pred = model(x, src, dst)
            loss = lossf(pred, torch.tensor(y, dtype=torch.float32, device=device))
            loss.backward()
            tot += loss.item()
        opt.step()
        if ep % 50 == 0 or ep == 1:
            vmae = mae(model, val, device)
            tmae = mae(model, train, device)
            if vmae < best_val:
                best_val = vmae
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"  ep{ep:4d} loss={tot/len(train):.4f} train_MAE={tmae:.4f} "
                  f"val_MAE={vmae:.4f}")

    final_val = mae(model, val, device)
    if final_val < best_val:
        best_val = final_val
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    print(f"== final val MAE={final_val:.4f}  (best={best_val:.4f}, "
          f"baseline={base_val_mae:.4f}) ==")
    # honest read: did the graph model beat predicting the mean?
    verdict = "BEATS baseline" if best_val < base_val_mae else "NOT better than baseline"
    print(f"== {verdict} ==")

    if args.save:
        torch.save({
            "state_dict": best_state,
            "hidden": args.hidden,
            "feat_dim": FEAT_DIM,
            "target": args.target,
            "val_mae": best_val,
            "baseline_mae": base_val_mae,
        }, args.save)
        print(f"== saved checkpoint -> {args.save} (val_MAE={best_val:.4f}) ==")


if __name__ == "__main__":
    main()
