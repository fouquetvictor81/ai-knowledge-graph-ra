"""
Knowledge Graph Embedding Training — TransE & RotatE (pure PyTorch)
====================================================================
Trains TransE and RotatE models on the prepared KGE dataset.
Evaluates with MRR, Hits@1, Hits@3, Hits@10.
Generates t-SNE visualization and comparison plots.

Usage:
    python src/kge/kge_train.py
    python src/kge/kge_train.py --epochs 100 --embedding-dim 64
"""

import argparse
import logging
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR    = Path("data")
MODELS_DIR  = Path("models")
RESULTS_DIR = Path("results")

DEFAULT_EPOCHS    = 200
DEFAULT_EMB_DIM   = 64
DEFAULT_BATCH     = 64
DEFAULT_LR        = 0.01
DEFAULT_NEG_RATIO = 5

MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_triples(path: Path):
    triples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                triples.append(tuple(parts))
    return triples


def build_mappings(train, valid, test):
    entities  = set()
    relations = set()
    for h, r, t in train + valid + test:
        entities.update([h, t])
        relations.add(r)
    ent2id = {e: i for i, e in enumerate(sorted(entities))}
    rel2id = {r: i for i, r in enumerate(sorted(relations))}
    return ent2id, rel2id


def encode(triples, ent2id, rel2id):
    return [(ent2id[h], rel2id[r], ent2id[t])
            for h, r, t in triples
            if h in ent2id and r in rel2id and t in ent2id]


# ---------------------------------------------------------------------------
# TransE model
# ---------------------------------------------------------------------------

class TransE(nn.Module):
    def __init__(self, n_entities, n_relations, dim, margin=1.0, norm=1):
        super().__init__()
        self.dim    = dim
        self.margin = margin
        self.norm   = norm
        self.ent_emb = nn.Embedding(n_entities, dim)
        self.rel_emb = nn.Embedding(n_relations, dim)
        nn.init.uniform_(self.ent_emb.weight, -6/math.sqrt(dim), 6/math.sqrt(dim))
        nn.init.uniform_(self.rel_emb.weight, -6/math.sqrt(dim), 6/math.sqrt(dim))
        # Normalize relation embeddings
        with torch.no_grad():
            self.rel_emb.weight.data = nn.functional.normalize(self.rel_emb.weight.data)

    def score(self, h, r, t):
        h_e = nn.functional.normalize(self.ent_emb(h))
        r_e = self.rel_emb(r)
        t_e = nn.functional.normalize(self.ent_emb(t))
        return -torch.norm(h_e + r_e - t_e, p=self.norm, dim=-1)

    def forward(self, pos_h, pos_r, pos_t, neg_h, neg_r, neg_t):
        pos = self.score(pos_h, pos_r, pos_t)
        neg = self.score(neg_h, neg_r, neg_t)
        # neg may be neg_ratio times larger — repeat pos accordingly
        neg_ratio = neg.shape[0] // pos.shape[0]
        pos_rep = pos.repeat_interleave(neg_ratio)
        loss = torch.relu(self.margin - pos_rep + neg).mean()
        return loss


# ---------------------------------------------------------------------------
# RotatE model
# ---------------------------------------------------------------------------

class RotatE(nn.Module):
    def __init__(self, n_entities, n_relations, dim, margin=6.0, epsilon=2.0):
        super().__init__()
        self.dim     = dim
        self.margin  = margin
        self.epsilon = epsilon
        # Entities: complex (dim/2 real + dim/2 imag stored as 2*dim)
        emb_range = (margin + epsilon) / dim
        self.ent_emb = nn.Embedding(n_entities, dim * 2)
        self.rel_emb = nn.Embedding(n_relations, dim)
        nn.init.uniform_(self.ent_emb.weight, -emb_range, emb_range)
        nn.init.uniform_(self.rel_emb.weight, -emb_range, emb_range)

    def score(self, h, r, t):
        pi = math.pi
        h_e = self.ent_emb(h)
        r_e = self.rel_emb(r)
        t_e = self.ent_emb(t)

        h_re, h_im = h_e[..., :self.dim], h_e[..., self.dim:]
        t_re, t_im = t_e[..., :self.dim], t_e[..., self.dim:]

        # Phase of relation
        phase = r_e / ((self.margin + self.epsilon) / pi)
        r_re  = torch.cos(phase)
        r_im  = torch.sin(phase)

        # Complex multiplication: (h_re + i*h_im) * (r_re + i*r_im)
        rot_re = h_re * r_re - h_im * r_im
        rot_im = h_re * r_im + h_im * r_re

        diff_re = rot_re - t_re
        diff_im = rot_im - t_im
        dist = torch.sqrt(diff_re**2 + diff_im**2 + 1e-9).sum(dim=-1)
        return -dist

    def forward(self, pos_h, pos_r, pos_t, neg_h, neg_r, neg_t):
        pos = self.score(pos_h, pos_r, pos_t)
        neg = self.score(neg_h, neg_r, neg_t)
        neg_ratio = neg.shape[0] // pos.shape[0]
        pos_rep = pos.repeat_interleave(neg_ratio)
        loss = torch.relu(self.margin - pos_rep + neg).mean()
        return loss


# ---------------------------------------------------------------------------
# Negative sampling
# ---------------------------------------------------------------------------

def corrupt(batch, n_entities, neg_ratio=1):
    heads, rels, tails = zip(*batch)
    neg_h, neg_r, neg_t = [], [], []
    for h, r, t in zip(heads, rels, tails):
        for _ in range(neg_ratio):
            if random.random() < 0.5:
                neg_h.append(random.randint(0, n_entities - 1))
                neg_r.append(r)
                neg_t.append(t)
            else:
                neg_h.append(h)
                neg_r.append(r)
                neg_t.append(random.randint(0, n_entities - 1))
    return neg_h, neg_r, neg_t


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(model, triples, n_entities, epochs, batch_size, lr, neg_ratio, device):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    n = len(triples)

    for epoch in range(1, epochs + 1):
        random.shuffle(triples)
        total_loss = 0.0
        steps = 0

        for i in range(0, n, batch_size):
            batch = triples[i:i + batch_size]
            if not batch:
                continue
            pos_h, pos_r, pos_t = zip(*batch)
            neg_h, neg_r, neg_t = corrupt(batch, n_entities, neg_ratio)

            pos_h = torch.tensor(pos_h, dtype=torch.long, device=device)
            pos_r = torch.tensor(pos_r, dtype=torch.long, device=device)
            pos_t = torch.tensor(pos_t, dtype=torch.long, device=device)
            neg_h = torch.tensor(neg_h, dtype=torch.long, device=device)
            neg_r = torch.tensor(neg_r, dtype=torch.long, device=device)
            neg_t = torch.tensor(neg_t, dtype=torch.long, device=device)

            optimizer.zero_grad()
            loss = model(pos_h, pos_r, pos_t, neg_h, neg_r, neg_t)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1

        if epoch % 50 == 0 or epoch == 1:
            logger.info(f"  Epoch {epoch:4d}/{epochs} — loss: {total_loss/max(steps,1):.4f}")

    return model


# ---------------------------------------------------------------------------
# Evaluation: MRR, Hits@k (filtered)
# ---------------------------------------------------------------------------

def evaluate(model, test_triples, all_triples_set, n_entities, device, batch_size=128):
    model.eval()
    all_entities = torch.arange(n_entities, device=device)

    mrr_sum = 0.0
    hits = {1: 0, 3: 0, 10: 0}
    count = 0

    with torch.no_grad():
        for h, r, t in test_triples:
            # --- Tail prediction: (h, r, ?) ---
            h_t = torch.tensor([h], device=device).expand(n_entities)
            r_t = torch.tensor([r], device=device).expand(n_entities)
            t_t = all_entities

            scores = model.score(h_t, r_t, t_t).cpu().numpy()

            # Filtered: mask out known true triples (except the test one)
            for e in range(n_entities):
                if e != t and (h, r, e) in all_triples_set:
                    scores[e] = -1e9

            rank = int((scores > scores[t]).sum()) + 1
            mrr_sum += 1.0 / rank
            for k in hits:
                if rank <= k:
                    hits[k] += 1
            count += 1

    if count == 0:
        return {"MRR": 0, "Hits@1": 0, "Hits@3": 0, "Hits@10": 0}

    return {
        "MRR":    round(mrr_sum / count, 4),
        "Hits@1": round(hits[1]  / count, 4),
        "Hits@3": round(hits[3]  / count, 4),
        "Hits@10":round(hits[10] / count, 4),
    }


# ---------------------------------------------------------------------------
# t-SNE visualization
# ---------------------------------------------------------------------------

def tsne_plot(model, ent2id, output_path: Path):
    try:
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
    except ImportError:
        logger.warning("matplotlib/sklearn not available — skipping t-SNE plot")
        return

    logger.info("Generating t-SNE visualization…")
    embeddings = model.ent_emb.weight.detach().cpu().numpy()
    if embeddings.shape[1] > embeddings.shape[0]:
        embeddings = embeddings[:, :embeddings.shape[0]]

    n_components = min(2, embeddings.shape[0] - 1)
    perplexity   = min(30, max(5, embeddings.shape[0] // 3))

    tsne   = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
    coords = tsne.fit_transform(embeddings)

    labels = list(ent2id.keys())

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.scatter(coords[:, 0], coords[:, 1], alpha=0.6, s=40, c="steelblue", edgecolors="white", linewidths=0.5)

    # Annotate a sample of points
    sample_indices = random.sample(range(len(labels)), min(40, len(labels)))
    for idx in sample_indices:
        lbl = labels[idx][:25]  # truncate long URIs
        ax.annotate(lbl, (coords[idx, 0], coords[idx, 1]),
                    fontsize=6, alpha=0.8, ha="center",
                    xytext=(0, 4), textcoords="offset points")

    ax.set_title("t-SNE of Entity Embeddings", fontsize=14, fontweight="bold")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"t-SNE plot saved → {output_path}")


# ---------------------------------------------------------------------------
# Comparison bar chart
# ---------------------------------------------------------------------------

def comparison_plot(results: dict, output_path: Path):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    models  = list(results.keys())
    metrics = ["MRR", "Hits@1", "Hits@3", "Hits@10"]
    x       = np.arange(len(metrics))
    width   = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    colors  = ["#7c3aed", "#06b6d4"]

    for i, (model_name, metric_dict) in enumerate(results.items()):
        vals = [metric_dict.get(m, 0) for m in metrics]
        bars = ax.bar(x + i * width - width / 2, vals, width,
                      label=model_name, color=colors[i % len(colors)], alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title("KGE Model Comparison — TransE vs RotatE", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Comparison plot saved → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train KGE models (TransE & RotatE)")
    parser.add_argument("--epochs",        type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--embedding-dim", type=int,   default=DEFAULT_EMB_DIM)
    parser.add_argument("--batch-size",    type=int,   default=DEFAULT_BATCH)
    parser.add_argument("--lr",            type=float, default=DEFAULT_LR)
    parser.add_argument("--neg-ratio",     type=int,   default=DEFAULT_NEG_RATIO)
    parser.add_argument("--data-dir",      type=Path,  default=DATA_DIR)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load data
    train_path = args.data_dir / "train.txt"
    valid_path = args.data_dir / "valid.txt"
    test_path  = args.data_dir / "test.txt"

    if not train_path.exists():
        logger.error(f"Missing {train_path} — run kge_prep.py first")
        return

    train_triples = load_triples(train_path)
    valid_triples = load_triples(valid_path)
    test_triples  = load_triples(test_path)

    logger.info(f"Loaded: {len(train_triples)} train / {len(valid_triples)} valid / {len(test_triples)} test")

    ent2id, rel2id = build_mappings(train_triples, valid_triples, test_triples)
    n_ent = len(ent2id)
    n_rel = len(rel2id)
    logger.info(f"Entities: {n_ent} | Relations: {n_rel}")

    train_enc = encode(train_triples, ent2id, rel2id)
    valid_enc = encode(valid_triples, ent2id, rel2id)
    test_enc  = encode(test_triples,  ent2id, rel2id)
    all_set   = set(train_enc) | set(valid_enc) | set(test_enc)

    all_results = {}
    best_model_name = None
    best_mrr = -1
    best_model = None

    for model_name, ModelClass, extra in [
        ("TransE", TransE,  {}),
        ("RotatE", RotatE,  {}),
    ]:
        logger.info(f"\n{'='*50}")
        logger.info(f"  Training {model_name}")
        logger.info(f"{'='*50}")

        model = ModelClass(n_ent, n_rel, args.embedding_dim, **extra)
        t0 = time.time()
        model = train(model, train_enc, n_ent,
                      args.epochs, args.batch_size, args.lr,
                      args.neg_ratio, device)
        elapsed = time.time() - t0
        logger.info(f"Training done in {elapsed:.1f}s")

        metrics = evaluate(model, test_enc, all_set, n_ent, device)
        all_results[model_name] = metrics

        logger.info(f"\n  {model_name} Results:")
        for k, v in metrics.items():
            logger.info(f"    {k:10s}: {v:.4f}")

        # Save model
        save_path = MODELS_DIR / f"{model_name.lower()}.pt"
        torch.save(model.state_dict(), save_path)
        logger.info(f"Model saved → {save_path}")

        if metrics["MRR"] > best_mrr:
            best_mrr = metrics["MRR"]
            best_model_name = model_name
            best_model = model
            best_ent2id = ent2id

    # Summary table
    logger.info("\n" + "="*55)
    logger.info("  FINAL COMPARISON")
    logger.info("="*55)
    header = f"{'Model':<10} {'MRR':>8} {'Hits@1':>8} {'Hits@3':>8} {'Hits@10':>8}"
    logger.info(header)
    logger.info("-" * 55)
    for mname, metrics in all_results.items():
        row = (f"{mname:<10} {metrics['MRR']:>8.4f} {metrics['Hits@1']:>8.4f}"
               f" {metrics['Hits@3']:>8.4f} {metrics['Hits@10']:>8.4f}")
        logger.info(row)
    logger.info("="*55)
    logger.info(f"Best model: {best_model_name} (MRR={best_mrr:.4f})")

    # Plots
    if best_model is not None:
        tsne_plot(best_model, best_ent2id, RESULTS_DIR / "tsne_embeddings.png")
    comparison_plot(all_results, RESULTS_DIR / "model_comparison.png")

    logger.info("\nDone! Results saved in results/")


if __name__ == "__main__":
    main()
