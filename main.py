import os
import zipfile
import argparse
import random
from dataclasses import dataclass
from typing import Dict, Set, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ----------------------------
# Reproducibility
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------
# Data utils
# ----------------------------
def maybe_extract_zip(zip_path: str, out_dir: str) -> str:
    if not zip_path or not os.path.isfile(zip_path):
        return out_dir

    if os.path.isdir(os.path.join(out_dir, "ml-100k")):
        return os.path.join(out_dir, "ml-100k")

    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)

    ml_dir = os.path.join(out_dir, "ml-100k")
    if not os.path.isdir(ml_dir):
        raise FileNotFoundError("Expected 'ml-100k/' folder after extraction, but didn't find it.")
    return ml_dir


def read_movielens_split(ml_dir: str, split: str = "u1") -> Tuple[np.ndarray, np.ndarray]:
    base_path = os.path.join(ml_dir, f"{split}.base")
    test_path = os.path.join(ml_dir, f"{split}.test")
    if not os.path.isfile(base_path) or not os.path.isfile(test_path):
        raise FileNotFoundError(f"Missing split files: {base_path} / {test_path}")

    def load(path):
        rows = []
        with open(path, "r", encoding="latin-1") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                u = int(parts[0])
                i = int(parts[1])
                r = int(parts[2])
                rows.append((u, i, r))
        return np.array(rows, dtype=np.int64)

    return load(base_path), load(test_path)


def read_u_item(ml_dir: str) -> Dict[int, str]:
    """
    u.item format: movieId|title|release date|... (pipe-separated)
    """
    path = os.path.join(ml_dir, "u.item")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing metadata file: {path}")

    movieid_to_title: Dict[int, str] = {}
    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) < 2:
                continue
            movie_id = int(parts[0])
            title = parts[1]
            movieid_to_title[movie_id] = title
    return movieid_to_title


def build_id_maps(train: np.ndarray, test: np.ndarray) -> Tuple[Dict[int, int], Dict[int, int]]:
    users = np.unique(np.concatenate([train[:, 0], test[:, 0]]))
    items = np.unique(np.concatenate([train[:, 1], test[:, 1]]))
    user2idx = {u: idx for idx, u in enumerate(users.tolist())}
    item2idx = {i: idx for idx, i in enumerate(items.tolist())}
    return user2idx, item2idx


def invert_map(m: Dict[int, int]) -> Dict[int, int]:
    return {v: k for k, v in m.items()}


def to_implicit_sets(
    train: np.ndarray,
    test: np.ndarray,
    user2idx: Dict[int, int],
    item2idx: Dict[int, int],
    pos_threshold: int = 4
) -> Tuple[Dict[int, Set[int]], Dict[int, Set[int]]]:
    train_pos: Dict[int, Set[int]] = {}
    test_pos: Dict[int, Set[int]] = {}

    for (u_raw, i_raw, r) in train:
        if r >= pos_threshold:
            u = user2idx[int(u_raw)]
            i = item2idx[int(i_raw)]
            train_pos.setdefault(u, set()).add(i)

    for (u_raw, i_raw, r) in test:
        if r >= pos_threshold:
            u = user2idx[int(u_raw)]
            i = item2idx[int(i_raw)]
            test_pos.setdefault(u, set()).add(i)

    return train_pos, test_pos


def filter_users_with_pos(
    train_pos: Dict[int, Set[int]],
    test_pos: Dict[int, Set[int]]
) -> List[int]:
    users = sorted(set(train_pos.keys()) & set(test_pos.keys()))
    users = [u for u in users if len(train_pos[u]) > 0 and len(test_pos[u]) > 0]
    return users


# ----------------------------
# Dataset for real positives
# ----------------------------
class PositivePairDataset(Dataset):
    def __init__(self, train_pos: Dict[int, Set[int]], users: List[int]):
        self.pairs = []
        for u in users:
            for i in train_pos.get(u, set()):
                self.pairs.append((u, i))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        u, i = self.pairs[idx]
        return torch.tensor(u, dtype=torch.long), torch.tensor(i, dtype=torch.long)


# ----------------------------
# Models (CGAN-style)
# ----------------------------
class Generator(nn.Module):
    def __init__(self, num_users: int, num_items: int, embed_dim=64, noise_dim=32, hidden=256):
        super().__init__()
        self.num_items = num_items
        self.noise_dim = noise_dim
        self.embed_dim = embed_dim
        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(embed_dim + noise_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_items)
        )

    def forward(self, user_ids: torch.Tensor) -> torch.Tensor:
        b = user_ids.size(0)
        u = self.user_emb(user_ids)  # (B, embed_dim)
        z = torch.randn(b, self.noise_dim, device=user_ids.device)  # (B, noise_dim)
        x = torch.cat([u, z], dim=1)
        logits = self.net(x)  # (B, num_items)
        return logits


class Discriminator(nn.Module):
    def __init__(self, num_users: int, num_items: int, embed_dim=64, hidden=256):
        super().__init__()
        self.embed_dim = embed_dim
        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.item_emb = nn.Embedding(num_items, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        u = self.user_emb(user_ids)
        i = self.item_emb(item_ids)
        x = torch.cat([u, i], dim=1)
        return self.net(x)  # (B, 1)


def sample_items_from_logits(logits: torch.Tensor, strategy="multinomial") -> torch.Tensor:
    if strategy == "argmax":
        return torch.argmax(logits, dim=1)
    probs = torch.softmax(logits, dim=1)
    return torch.multinomial(probs, num_samples=1).squeeze(1)


def sample_random_negatives(
    user_ids: torch.Tensor,
    train_pos: Dict[int, Set[int]],
    num_items: int,
    device: torch.device
) -> torch.Tensor:
    negs = []
    user_ids_cpu = user_ids.detach().cpu().numpy().tolist()
    for u in user_ids_cpu:
        seen = train_pos.get(u, set())
        while True:
            j = random.randrange(num_items)
            if j not in seen:
                negs.append(j)
                break
    return torch.tensor(negs, dtype=torch.long, device=device)


# ----------------------------
# Evaluation
# ----------------------------
def recall_at_k(recs: List[int], gt: Set[int], k: int) -> float:
    topk = recs[:k]
    hits = sum(1 for x in topk if x in gt)
    return hits / max(1, len(gt))


def ndcg_at_k(recs: List[int], gt: Set[int], k: int) -> float:
    dcg = 0.0
    for rank, item in enumerate(recs[:k], start=1):
        if item in gt:
            dcg += 1.0 / np.log2(rank + 1)
    ideal_hits = min(len(gt), k)
    idcg = sum(1.0 / np.log2(r + 1) for r in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def hitrate_at_k(recs: List[int], gt: Set[int], k: int) -> float:
    return 1.0 if any(x in gt for x in recs[:k]) else 0.0


@torch.no_grad()
def recommend_for_user(
    G: Generator,
    u: int,
    num_items: int,
    seen: Set[int],
    k: int,
    device: torch.device,
    noise_samples: int = 5
) -> List[int]:
    G.eval()
    u_tensor = torch.tensor([u], dtype=torch.long, device=device)

    logits_accum = None
    for _ in range(noise_samples):
        logits = G(u_tensor).squeeze(0)  # (num_items,)
        logits_accum = logits if logits_accum is None else (logits_accum + logits)
    logits_mean = logits_accum / noise_samples

    scores = logits_mean.detach().cpu().numpy()
    for i in seen:
        scores[i] = -1e9

    topk = np.argsort(-scores)[:k].tolist()
    return topk


def evaluate(
    G: Generator,
    train_pos: Dict[int, Set[int]],
    test_pos: Dict[int, Set[int]],
    users_eval: List[int],
    num_items: int,
    device: torch.device,
    k: int
):
    recalls, ndcgs, hits = [], [], []
    for u in users_eval:
        seen = train_pos.get(u, set())
        gt = test_pos.get(u, set())
        recs = recommend_for_user(G, u, num_items, seen=seen, k=k, device=device, noise_samples=5)
        recalls.append(recall_at_k(recs, gt, k))
        ndcgs.append(ndcg_at_k(recs, gt, k))
        hits.append(hitrate_at_k(recs, gt, k))

    print(f"\nEvaluation @K={k}")
    print(f"Users evaluated: {len(users_eval)}")
    print(f"Recall@{k}:  {float(np.mean(recalls)):.4f}")
    print(f"NDCG@{k}:    {float(np.mean(ndcgs)):.4f}")
    print(f"HitRate@{k}: {float(np.mean(hits)):.4f}")


# ----------------------------
# Training
# ----------------------------
@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 1024
    lr: float = 2e-4
    embed_dim: int = 64
    noise_dim: int = 32
    hidden: int = 256
    d_steps: int = 1
    g_steps: int = 1
    neg_mix: float = 0.5
    fake_strategy: str = "multinomial"


def train_cgan(
    train_pos: Dict[int, Set[int]],
    users_train: List[int],
    num_users: int,
    num_items: int,
    device: torch.device,
    cfg: TrainConfig
) -> Tuple[Generator, Discriminator]:
    ds = PositivePairDataset(train_pos, users_train)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    G = Generator(num_users, num_items, cfg.embed_dim, cfg.noise_dim, cfg.hidden).to(device)
    D = Discriminator(num_users, num_items, cfg.embed_dim, cfg.hidden).to(device)

    bce = nn.BCELoss()
    opt_G = torch.optim.Adam(G.parameters(), lr=cfg.lr)
    opt_D = torch.optim.Adam(D.parameters(), lr=cfg.lr)

    for epoch in range(1, cfg.epochs + 1):
        G.train()
        D.train()
        lossD_sum, lossG_sum, steps = 0.0, 0.0, 0

        for u_real, i_real in dl:
            u_real = u_real.to(device)
            i_real = i_real.to(device)
            b = u_real.size(0)

            # --- Train D ---
            for _ in range(cfg.d_steps):
                opt_D.zero_grad()

                pred_real = D(u_real, i_real)
                loss_real = bce(pred_real, torch.ones_like(pred_real))

                fake_logits = G(u_real)
                i_fake_g = sample_items_from_logits(fake_logits, strategy=cfg.fake_strategy)
                i_fake_r = sample_random_negatives(u_real, train_pos, num_items, device)

                mask = (torch.rand(b, device=device) < cfg.neg_mix)
                i_fake = torch.where(mask, i_fake_r, i_fake_g)

                pred_fake = D(u_real, i_fake.detach())
                loss_fake = bce(pred_fake, torch.zeros_like(pred_fake))

                loss_D = loss_real + loss_fake
                loss_D.backward()
                opt_D.step()

            # --- Train G ---
            for _ in range(cfg.g_steps):
                opt_G.zero_grad()
                fake_logits = G(u_real)
                i_fake = sample_items_from_logits(fake_logits, strategy=cfg.fake_strategy)
                pred = D(u_real, i_fake)
                loss_G = bce(pred, torch.ones_like(pred))
                loss_G.backward()
                opt_G.step()

            lossD_sum += loss_D.item()
            lossG_sum += loss_G.item()
            steps += 1

        print(f"Epoch {epoch:02d} | loss_D={lossD_sum/steps:.4f} | loss_G={lossG_sum/steps:.4f}")

    return G, D


# ----------------------------
# Human-readable printing
# ----------------------------
def format_recs(
    rec_item_indices: List[int],
    idx2movieid: Dict[int, int],
    movieid_to_title: Dict[int, str]
) -> List[str]:
    out = []
    for idx in rec_item_indices:
        mid = idx2movieid.get(idx, None)
        title = movieid_to_title.get(mid, "UNKNOWN_TITLE") if mid is not None else "UNKNOWN_TITLE"
        out.append(f"{mid} | {title}")
    return out


def search_titles(movieid_to_title: Dict[int, str], query: str, limit: int = 10) -> List[Tuple[int, str]]:
    q = query.lower().strip()
    matches = []
    for mid, title in movieid_to_title.items():
        if q in title.lower():
            matches.append((mid, title))
    matches.sort(key=lambda x: x[1])
    return matches[:limit]


# ----------------------------
# New user: append embedding row + adapt only that row
# ----------------------------
def expand_user_embedding(emb: nn.Embedding, new_num_users: int) -> nn.Embedding:
    old_w = emb.weight.data
    old_num, dim = old_w.shape
    assert new_num_users == old_num + 1
    new_emb = nn.Embedding(new_num_users, dim)
    new_emb.weight.data[:old_num] = old_w
    # init new row
    new_emb.weight.data[old_num] = torch.randn(dim) * 0.01
    return new_emb


def adapt_new_user_embedding(
    G: Generator,
    D: Discriminator,
    new_user_idx: int,
    new_user_pos: Set[int],
    train_pos: Dict[int, Set[int]],
    num_items: int,
    device: torch.device,
    steps: int = 200,
    lr: float = 5e-2,
    fake_strategy: str = "multinomial",
):
    """
    Freeze everything except:
      - G.user_emb[new_user_idx]
      - D.user_emb[new_user_idx]
    Then do a small optimization loop using the new user's positives.
    """
    # Freeze all params
    for p in G.parameters():
        p.requires_grad = False
    for p in D.parameters():
        p.requires_grad = False

    # Unfreeze only new user rows by optimizing their weight vectors directly
    g_row = G.user_emb.weight[new_user_idx:new_user_idx+1]
    d_row = D.user_emb.weight[new_user_idx:new_user_idx+1]
    g_row.requires_grad_(True)
    d_row.requires_grad_(True)

    bce = nn.BCELoss()
    opt = torch.optim.SGD([g_row, d_row], lr=lr)

    pos_list = list(new_user_pos)
    if len(pos_list) == 0:
        return

    u_tensor = torch.tensor([new_user_idx], dtype=torch.long, device=device)

    for t in range(1, steps + 1):
        opt.zero_grad()

        # sample one positive item
        i_pos = random.choice(pos_list)
        i_pos_t = torch.tensor([i_pos], dtype=torch.long, device=device)

        # D on real
        pred_real = D(u_tensor, i_pos_t)
        loss_real = bce(pred_real, torch.ones_like(pred_real))

        # D on random negative (not in new_user_pos)
        while True:
            j = random.randrange(num_items)
            if j not in new_user_pos:
                break
        i_neg_t = torch.tensor([j], dtype=torch.long, device=device)
        pred_neg = D(u_tensor, i_neg_t)
        loss_neg = bce(pred_neg, torch.zeros_like(pred_neg))

        # G tries to fool D
        fake_logits = G(u_tensor)
        i_fake = sample_items_from_logits(fake_logits, strategy=fake_strategy)
        pred_fake = D(u_tensor, i_fake)
        loss_g = bce(pred_fake, torch.ones_like(pred_fake))

        loss = loss_real + loss_neg + loss_g
        loss.backward()
        opt.step()

        if t % 50 == 0:
            print(f"  [new-user adapt] step {t}/{steps} loss={loss.item():.4f}")

    # Unfreeze back for safety (optional)
    for p in G.parameters():
        p.requires_grad = True
    for p in D.parameters():
        p.requires_grad = True


def interactive_new_user_ratings(
    movieid_to_title: Dict[int, str],
    item2idx: Dict[int, int],
    idx2movieid: Dict[int, int],
    pos_threshold: int
) -> Set[int]:
    """
    Collect ratings from terminal. User can search titles.
    Return set of positive item indices (implicit).
    """
    print("\n=== New User Setup ===")
    print("You will enter a few ratings. We'll treat rating >= threshold as 'liked'.")
    print(f"Positive threshold = {pos_threshold}")
    print("Tip: type 'search <words>' to find movieIds.")
    print("Tip: type 'done' when finished.\n")

    new_pos: Set[int] = set()
    seen_any = set()

    while True:
        s = input("Enter: <movieId> <rating>  OR  search <query>  OR  done\n> ").strip()
        if not s:
            continue
        if s.lower() == "done":
            break
        if s.lower().startswith("search "):
            q = s[7:].strip()
            matches = search_titles(movieid_to_title, q, limit=10)
            if not matches:
                print("  No matches.")
            else:
                for mid, title in matches:
                    print(f"  {mid} | {title}")
            continue

        parts = s.split()
        if len(parts) != 2:
            print("  Invalid input. Example: 50 5   OR   search star wars   OR done")
            continue

        try:
            movie_id = int(parts[0])
            rating = int(parts[1])
        except ValueError:
            print("  movieId and rating must be integers.")
            continue

        if movie_id not in movieid_to_title:
            print("  Unknown movieId. Use: search <query>")
            continue

        # Convert movieId -> internal item index if exists in our mapping
        if movie_id not in item2idx:
            # MovieLens 100K split mapping includes all movieIds used in train/test union,
            # but in rare cases your split may exclude a movie. We'll warn.
            print("  This movieId is not in the current split's item mapping. Try another.")
            continue

        item_idx = item2idx[movie_id]
        seen_any.add(item_idx)

        if rating >= pos_threshold:
            new_pos.add(item_idx)
            print(f"  Liked: {movie_id} | {movieid_to_title[movie_id]}")
        else:
            print(f"  Not liked (ignored): {movie_id} | {movieid_to_title[movie_id]}")

    print(f"\nNew user provided {len(seen_any)} rated movies; positives (liked) = {len(new_pos)}")
    return new_pos


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip_path", type=str, default="a69190f7-2dfc-4313-a968-1222b827bbb9.zip")
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--split", type=str, default="u1", choices=["u1","u2","u3","u4","u5"])
    parser.add_argument("--pos_threshold", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--noise_dim", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--neg_mix", type=float, default=0.5)
    parser.add_argument("--fake_strategy", type=str, default="multinomial", choices=["multinomial","argmax"])
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--interactive_new_user",
        action="store_true",
        help="Prompt a new user to rate movies BEFORE training; then train and recommend for that user."
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print("Device:", device)

    # 1) Locate/extract dataset
    zip_path = args.zip_path if os.path.isfile(args.zip_path) else ""
    ml_dir = maybe_extract_zip(zip_path, args.data_dir)

    # 2) Load item titles
    movieid_to_title = read_u_item(ml_dir)

    # 3) Load split files
    train_raw, test_raw = read_movielens_split(ml_dir, split=args.split)
    print(f"Loaded split {args.split}: train={len(train_raw)} rows, test={len(test_raw)} rows")

    # 4) Build ID mappings from union(train,test)
    user2idx, item2idx = build_id_maps(train_raw, test_raw)
    idx2user = invert_map(user2idx)
    idx2movieid = invert_map(item2idx)

    num_users = len(user2idx)
    num_items = len(item2idx)
    print(f"Users: {num_users} | Items: {num_items}")

    # 5) Convert to implicit positives
    train_pos, test_pos = to_implicit_sets(
        train_raw, test_raw, user2idx, item2idx, pos_threshold=args.pos_threshold
    )
    users_eval = filter_users_with_pos(train_pos, test_pos)
    print(f"Users with >=1 train+test positive: {len(users_eval)}")

    # 6) OPTIONAL: get NEW USER ratings BEFORE training and add to training positives
    new_user_idx = None
    new_user_pos = set()

    if args.interactive_new_user:
        new_user_pos = interactive_new_user_ratings(
            movieid_to_title=movieid_to_title,
            item2idx=item2idx,
            idx2movieid=idx2movieid,
            pos_threshold=args.pos_threshold
        )

        if len(new_user_pos) == 0:
            print("New user has 0 positive interactions; skipping adding new user.")
        else:
            new_user_idx = num_users  # append as the last user index
            train_pos[new_user_idx] = set(new_user_pos)
            num_users += 1  # IMPORTANT: expand user count for embeddings
            print(f"Added NEW USER as user_idx={new_user_idx} with {len(new_user_pos)} positives.")
            # Include this user in the training dataset
            users_train = users_eval + [new_user_idx]
    else:
        users_train = users_eval

    # 7) Train CGAN
    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        embed_dim=args.embed_dim,
        noise_dim=args.noise_dim,
        hidden=args.hidden,
        neg_mix=args.neg_mix,
        fake_strategy=args.fake_strategy
    )

    G, D = train_cgan(train_pos, users_train, num_users, num_items, device, cfg)

    # 8) Evaluate on original users only (standard evaluation split)
    evaluate(G, train_pos, test_pos, users_eval, num_items, device, k=args.k)

    # 9) Human-readable sample recommendations for existing users
    print("\nSample recommendations (human-readable; excludes TRAIN seen):")
    for u in users_eval[:5]:
        rec_idxs = recommend_for_user(
            G, u, num_items,
            seen=train_pos.get(u, set()),
            k=args.k, device=device,
            noise_samples=5
        )
        rec_lines = format_recs(rec_idxs, idx2movieid, movieid_to_title)
        print(f"\nUser idx={u} (raw userId={idx2user[u]}):")
        for line in rec_lines:
            print(" ", line)

    # 10) Recommendations for NEW USER (also human-readable)
    if new_user_idx is not None:
        rec_idxs = recommend_for_user(
            G, new_user_idx, num_items,
            seen=train_pos.get(new_user_idx, set()),   # exclude what they liked/rated
            k=args.k, device=device,
            noise_samples=10
        )
        rec_lines = format_recs(rec_idxs, idx2movieid, movieid_to_title)

        print("\n=== Recommendations for NEW USER (trained from scratch) ===")
        for line in rec_lines:
            print(" ", line)



if __name__ == "__main__":
    main()
