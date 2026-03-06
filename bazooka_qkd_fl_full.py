#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bazooka to Kill a Mosquito?
Right-Sized Quantum Security for Federated Learning

ONE-FILE COMPLETE EXPERIMENT FRAMEWORK:
- Baselines: CLASSICAL, ALL_QKD, STATIC_SPLIT, RANDOM
- Proposed: RIGHT_SIZED (dynamic + selective + proportional)
- FL: FedAvg, FedProx
- Datasets: MNIST, CIFAR-10
- Topologies: STAR, RING, TWO_HOP (client->gateway->server)
- Network variation: bandwidth, RTT
- QKD scarcity: key-rate, pool
- Multiple seeds, full grid
- Attacks: sign-flip poisoning (simple, defensible)
- Outputs: raw per-round CSV, final CSV, aggregated summary CSV, LaTeX tables,
          figures PNG+PDF, CDF, boxplots, ranking tables.

Deps:
  pip install torch torchvision numpy pandas matplotlib

Usage:
  python bazooka_qkd_fl_full.py run-grid --out results --dataset mnist --n-seeds 10 --jobs 4
  python bazooka_qkd_fl_full.py run-grid --out results --dataset cifar10 --n-seeds 10 --jobs 4
  python bazooka_qkd_fl_full.py summarize --in results --out results_summary

Notes:
- This is simulation-level networking + QKD resource abstraction; no quantum physics.
- Compute can be heavy for CIFAR-10 + 200 clients + many seeds; use --jobs and adjust grid if needed.
"""

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import uuid

from multiprocessing import Pool, cpu_count


# -----------------------------
# Utilities / Repro / Stats
# -----------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def now_ms() -> int:
    return int(time.time() * 1000)

def safe_json_dump(obj: Any, path: str):
    """
    Atomic-ish write safe for multiprocessing:
    - tmp unique per process/call
    - ensures directory exists
    """
    ensure_dir(os.path.dirname(path) or ".")
    tmp = f"{path}.{os.getpid()}.{time.time_ns()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)

def jain_fairness(x: List[float]) -> float:
    x = np.array(x, dtype=np.float64)
    if np.all(x == 0):
        return 1.0
    return float((x.sum() ** 2) / (len(x) * (x**2).sum() + 1e-12))

def mean_ci95(x: np.ndarray) -> Tuple[float, float, float]:
    # return mean, std, ci_halfwidth
    x = np.array(x, dtype=np.float64)
    m = float(x.mean()) if len(x) else 0.0
    s = float(x.std(ddof=1)) if len(x) > 1 else 0.0
    half = 1.96 * s / math.sqrt(max(len(x), 1))
    return m, s, half

def auc_trapz(y: np.ndarray, x: np.ndarray) -> float:
    # Simple AUC
    if len(y) < 2:
        return float(y[0]) if len(y) == 1 else 0.0
    return float(np.trapezoid(y, x))

def time_to_target(acc_curve: np.ndarray, rounds: np.ndarray, target: float) -> float:
    idx = np.where(acc_curve >= target)[0]
    if len(idx) == 0:
        return float("inf")
    return float(rounds[int(idx[0])])

def pick_device(device: str):
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def model_num_bytes(model: nn.Module) -> int:
    total_params = sum(p.numel() for p in model.parameters())
    return int(total_params * 4)  # float32 payload

def get_params(model: nn.Module) -> List[torch.Tensor]:
    return [p.detach().cpu().clone() for p in model.parameters()]

def set_params(model: nn.Module, params: List[torch.Tensor]):
    with torch.no_grad():
        for p, w in zip(model.parameters(), params):
            p.copy_(w.to(p.device))

def avg_params(weighted_params: List[Tuple[List[torch.Tensor], int]]) -> List[torch.Tensor]:
    total = sum(n for _, n in weighted_params)
    out = [torch.zeros_like(w) for w in weighted_params[0][0]]
    for params, n in weighted_params:
        for i, w in enumerate(params):
            out[i] += (n / total) * w
    return out

def delta_norm(after: List[torch.Tensor], before: List[torch.Tensor]) -> float:
    s = 0.0
    for a, b in zip(after, before):
        d = (a.float() - b.float())
        s += float((d ** 2).sum().item())
    return math.sqrt(max(s, 0.0))


# -----------------------------
# Models
# -----------------------------
class MLP_MNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 10)
        )
    def forward(self, x): return self.net(x)

class CNN_CIFAR10(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256), nn.ReLU(),
            nn.Linear(256, 10),
        )
    def forward(self, x):
        return self.classifier(self.features(x))


# -----------------------------
# Data loading + partition (Dirichlet non-IID)
# -----------------------------
def load_data(dataset: str):
    if dataset == "mnist":
        transform = transforms.Compose([transforms.ToTensor()])
        train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
        test_ds = datasets.MNIST("./data", train=False, download=True, transform=transform)
        model_fn = MLP_MNIST
        y = np.array(train_ds.targets)
    elif dataset == "cifar10":
        transform = transforms.Compose([transforms.ToTensor()])
        train_ds = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
        test_ds = datasets.CIFAR10("./data", train=False, download=True, transform=transform)
        model_fn = CNN_CIFAR10
        y = np.array(train_ds.targets)
    else:
        raise ValueError("dataset must be mnist or cifar10")
    return train_ds, test_ds, model_fn, y


def _fix_empty_clients(client_indices: List[List[int]], rng: np.random.Generator) -> List[List[int]]:
    """
    Ensure no client is empty by moving samples from the largest partitions.
    """
    if not client_indices:
        return client_indices

    while True:
        empties = [i for i, idxs in enumerate(client_indices) if len(idxs) == 0]
        if not empties:
            break

        donor = int(np.argmax([len(idxs) for idxs in client_indices]))
        if len(client_indices[donor]) <= 1:
            # cannot donate without creating another empty
            break

        for e in empties:
            if len(client_indices[donor]) <= 1:
                break
            j = int(rng.integers(0, len(client_indices[donor])))
            client_indices[e].append(client_indices[donor].pop(j))

    return client_indices


def _ensure_min_per_client(client_indices: List[List[int]], rng: np.random.Generator, m: int) -> List[List[int]]:
    """
    Ensure each client has at least m samples (best effort).
    """
    n = len(client_indices)
    if m <= 1:
        return _fix_empty_clients(client_indices, rng)

    while True:
        small = [i for i in range(n) if len(client_indices[i]) < m]
        if not small:
            break
        donor = int(np.argmax([len(idxs) for idxs in client_indices]))
        if len(client_indices[donor]) <= m:
            # not enough to satisfy everyone; stop
            break
        for s in small:
            if len(client_indices[s]) >= m:
                continue
            if len(client_indices[donor]) <= m:
                break
            j = int(rng.integers(0, len(client_indices[donor])))
            client_indices[s].append(client_indices[donor].pop(j))

    return _fix_empty_clients(client_indices, rng)


def dirichlet_partition(labels: np.ndarray, n_clients: int, alpha: float, seed: int,
                        min_per_client: int = 1) -> List[np.ndarray]:
    """
    Dirichlet partition with robust guarantees:
    - Fixes empty clients (prevents DataLoader num_samples=0)
    - Optionally enforces a minimum number of samples per client (best effort)
    """
    rng = np.random.default_rng(seed)
    n_classes = int(labels.max() + 1)
    idx_by_class = [np.where(labels == c)[0] for c in range(n_classes)]
    for c in range(n_classes):
        rng.shuffle(idx_by_class[c])

    client_indices: List[List[int]] = [[] for _ in range(n_clients)]
    for c in range(n_classes):
        proportions = rng.dirichlet(alpha * np.ones(n_clients))
        splits = (proportions / proportions.sum() * len(idx_by_class[c])).astype(int)

        while splits.sum() < len(idx_by_class[c]):
            splits[int(rng.integers(0, n_clients))] += 1
        while splits.sum() > len(idx_by_class[c]):
            i = int(rng.integers(0, n_clients))
            if splits[i] > 0:
                splits[i] -= 1

        start = 0
        for i in range(n_clients):
            take = int(splits[i])
            if take > 0:
                client_indices[i].extend(idx_by_class[c][start:start + take].tolist())
            start += take

    # critical fixes
    client_indices = _fix_empty_clients(client_indices, rng)
    client_indices = _ensure_min_per_client(client_indices, rng, int(min_per_client))

    return [np.array(idxs, dtype=np.int64) for idxs in client_indices]


# -----------------------------
# Networking + QKD abstractions with topologies
# -----------------------------
@dataclass
class Link:
    bandwidth_mbps: float
    rtt_ms: float

    def tx_time_s(self, payload_bytes: int) -> float:
        tx = (payload_bytes * 8) / (self.bandwidth_mbps * 1e6)
        return (self.rtt_ms / 1000.0) + tx

@dataclass
class QKDLinkConfig:
    key_rate_bps: float
    pool_capacity_bits: float

class QKDKeyPool:
    def __init__(self, cfg: QKDLinkConfig, init_fill: float = 0.5):
        self.cfg = cfg
        self.pool_bits = cfg.pool_capacity_bits * init_fill
        self.total_generated = 0.0
        self.total_consumed = 0.0
        self.starvation_events = 0

    def tick(self, seconds: float):
        gen = self.cfg.key_rate_bps * seconds
        self.total_generated += gen
        self.pool_bits = min(self.cfg.pool_capacity_bits, self.pool_bits + gen)

    def try_consume(self, bits_needed: float) -> bool:
        if self.pool_bits >= bits_needed:
            self.pool_bits -= bits_needed
            self.total_consumed += bits_needed
            return True
        self.starvation_events += 1
        return False

@dataclass
class TopologyConfig:
    name: str  # STAR, RING, TWO_HOP
    n_gateways: int = 10

class NetworkSimulator:
    """
    Provides per-client communication time per round under different topologies.
    Also holds per-client QKD key pools (client<->server security resource).
    """
    def __init__(self,
                 n_clients: int,
                 topo: TopologyConfig,
                 base_link: Link,
                 gateway_link: Optional[Link],
                 qkd_rate_mean: float,
                 qkd_rate_hetero: float,
                 qkd_pool_bits: float,
                 seed: int):
        self.n_clients = n_clients
        self.topo = topo
        self.base_link = base_link
        self.gateway_link = gateway_link
        rng = np.random.default_rng(seed)

        # QKD pools per client (client <-> server abstraction)
        self.pools: Dict[int, QKDKeyPool] = {}
        for cid in range(n_clients):
            mult = float(1.0 + qkd_rate_hetero * (rng.random() * 2.0 - 1.0))
            rate = float(max(1.0, qkd_rate_mean * mult))
            self.pools[cid] = QKDKeyPool(QKDLinkConfig(rate, float(qkd_pool_bits)), init_fill=0.5)

        # For TWO_HOP: assign clients to gateways deterministically by id
        if topo.name.upper() == "TWO_HOP":
            self.n_gateways = max(1, topo.n_gateways)
            self.client_gw = {cid: (cid % self.n_gateways) for cid in range(n_clients)}
        else:
            self.n_gateways = 0
            self.client_gw = {}

    def tick_qkd(self, seconds: float):
        for cid in range(self.n_clients):
            self.pools[cid].tick(seconds)

    def comm_time_s(self, cid: int, payload_bytes: int) -> float:
        t = 0.0
        name = self.topo.name.upper()
        if name == "STAR":
            t += self.base_link.tx_time_s(payload_bytes)
        elif name == "RING":
            hops = max(1, int(self.n_clients / 4))
            t += hops * self.base_link.tx_time_s(payload_bytes)
        elif name == "TWO_HOP":
            if self.gateway_link is None:
                raise ValueError("TWO_HOP requires gateway_link")
            t += self.base_link.tx_time_s(payload_bytes)      # client->gw
            t += self.gateway_link.tx_time_s(payload_bytes)   # gw->server
        else:
            raise ValueError(f"Unknown topology: {self.topo.name}")
        return t


# -----------------------------
# FL training
# -----------------------------
def train_local_fedavg(model, loader, epochs, lr, device):
    model.train()
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()

def train_local_fedprox(model, global_params, loader, epochs, lr, mu, device):
    model.train()
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            prox = 0.0
            for p, g in zip(model.parameters(), global_params):
                prox += ((p - g.to(device)) ** 2).sum()
            loss = loss + (mu / 2.0) * prox
            loss.backward()
            opt.step()

def eval_model(model, loader, device) -> Tuple[float, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += loss_fn(logits, y).item()
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return (total_loss / max(total, 1)), (correct / max(total, 1))


# -----------------------------
# Attack model: simple sign-flip poisoning
# -----------------------------
@dataclass
class AttackConfig:
    enabled: bool = False
    adv_frac: float = 0.0
    strength: float = 1.0
    seed: int = 0

class Attacker:
    def __init__(self, cfg: AttackConfig):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)

    def pick_adversaries(self, selected: List[int]) -> set:
        if not self.cfg.enabled or self.cfg.adv_frac <= 0:
            return set()
        k = max(0, int(round(self.cfg.adv_frac * len(selected))))
        if k == 0:
            return set()
        return set(self.rng.sample(selected, k))

    def poison_update(self, update_params: List[torch.Tensor], global_before: List[torch.Tensor]) -> List[torch.Tensor]:
        strength = float(self.cfg.strength)
        out = []
        for w_local, w_global in zip(update_params, global_before):
            delta = (w_local - w_global)
            out.append(w_global + (-strength) * delta)
        return out


# -----------------------------
# Security policies
# -----------------------------
class SecurityPolicy:
    name: str = "BASE"
    def allocate(self, round_id: int, selected: List[int], payload_bytes: int, ctx: Dict[str, Any]) -> Dict[int, float]:
        raise NotImplementedError

class ClassicalOnly(SecurityPolicy):
    name = "CLASSICAL"
    def allocate(self, round_id, selected, payload_bytes, ctx):
        return {cid: 0.0 for cid in selected}

class AllQKD(SecurityPolicy):
    name = "ALL_QKD"
    def allocate(self, round_id, selected, payload_bytes, ctx):
        return {cid: 1.0 for cid in selected}

class StaticSplit(SecurityPolicy):
    name = "STATIC_SPLIT"
    def __init__(self, fraction_clients: float):
        self.f = float(fraction_clients)
    def allocate(self, round_id, selected, payload_bytes, ctx):
        k = max(1, int(math.ceil(self.f * len(selected))))
        chosen = set(sorted(selected)[:k])
        return {cid: (1.0 if cid in chosen else 0.0) for cid in selected}

class RandomQKD(SecurityPolicy):
    name = "RANDOM"
    def __init__(self, p: float, seed: int):
        self.p = float(p)
        self.rng = random.Random(seed)
    def allocate(self, round_id, selected, payload_bytes, ctx):
        return {cid: (1.0 if self.rng.random() < self.p else 0.0) for cid in selected}

class RightSized(SecurityPolicy):
    """
    Right-Sized Quantum Security:
      - Dynamic: depends on per-round scarcity state
      - Selective: prioritizes high-impact clients (impact proxy)
      - Proportional: assigns continuous fraction in [0,1] subject to pool
    """
    name = "RIGHT_SIZED"
    def __init__(self,
                 base_fraction: float = 0.9,
                 prioritize_topk: float = 0.6,
                 scarcity_beta: float = 0.7,
                 fairness_gamma: float = 0.15,
                 min_fraction: float = 0.0,
                 max_fraction: float = 1.0):
        self.base_fraction = float(base_fraction)
        self.prioritize_topk = float(prioritize_topk)
        self.scarcity_beta = float(scarcity_beta)
        self.fairness_gamma = float(fairness_gamma)
        self.min_fraction = float(min_fraction)
        self.max_fraction = float(max_fraction)

    def allocate(self, round_id, selected, payload_bytes, ctx):
        pools: Dict[int, QKDKeyPool] = ctx["pools"]
        impact: Dict[int, float] = ctx["impact_score"]
        used_bits: Dict[int, float] = ctx["used_bits_so_far"]

        payload_bits = float(payload_bytes * 8)

        fills = [pools[cid].pool_bits / max(pools[cid].cfg.pool_capacity_bits, 1.0) for cid in selected]
        avg_fill = float(np.mean(fills)) if fills else 0.0
        scarcity_factor = max(0.0, min(1.0, avg_fill ** self.scarcity_beta))

        k = max(1, int(math.ceil(self.prioritize_topk * len(selected))))
        ranked = sorted(selected, key=lambda c: impact.get(c, 0.0), reverse=True)
        eligible = set(ranked[:k])

        imp_vals = np.array([max(impact.get(cid, 1e-12), 1e-12) for cid in selected], dtype=np.float64)
        imp_max = float(imp_vals.max()) if len(imp_vals) else 1.0

        used_arr = np.array([used_bits.get(cid, 0.0) for cid in selected], dtype=np.float64)
        used_norm = used_arr / (used_arr.max() + 1e-12) if len(used_arr) else used_arr

        out = {cid: 0.0 for cid in selected}
        for i, cid in enumerate(selected):
            if cid not in eligible:
                out[cid] = 0.0
                continue
            imp_n = max(impact.get(cid, 0.0), 0.0) / (imp_max + 1e-12)
            fairness_penalty = (1.0 - self.fairness_gamma * float(used_norm[i]))
            q = self.base_fraction * scarcity_factor * imp_n * fairness_penalty
            q = max(self.min_fraction, min(self.max_fraction, q))
            out[cid] = q

        # Cap by pool feasibility (per-client)
        for cid in selected:
            q = out[cid]
            if q <= 0:
                continue
            avail = pools[cid].pool_bits
            max_q = max(0.0, min(1.0, avail / (payload_bits + 1e-12)))
            if q > max_q:
                out[cid] = max_q

        return out


def make_policy(name: str, seed: int,
                static_frac: float, random_p: float,
                rs_base: float, rs_topk: float, rs_beta: float, rs_gamma: float) -> SecurityPolicy:
    n = name.lower()
    if n == "classical":
        return ClassicalOnly()
    if n == "all_qkd":
        return AllQKD()
    if n == "static_split":
        return StaticSplit(static_frac)
    if n == "random":
        return RandomQKD(random_p, seed)
    if n == "right_sized":
        return RightSized(base_fraction=rs_base, prioritize_topk=rs_topk, scarcity_beta=rs_beta, fairness_gamma=rs_gamma)
    raise ValueError(f"Unknown policy: {name}")


# -----------------------------
# Experiment config
# -----------------------------
@dataclass
class ExpConfig:
    # Identity
    dataset: str = "mnist"
    fl_algo: str = "fedavg"
    policy: str = "right_sized"
    topology: str = "STAR"

    # FL setup
    n_clients: int = 50
    clients_per_round: float = 0.2
    rounds: int = 200
    local_epochs: int = 1
    batch_size: int = 32
    lr: float = 0.01
    fedprox_mu: float = 0.01
    dirichlet_alpha: float = 0.3

    # NEW (robustness): avoid empty clients
    min_samples_per_client: int = 5  # set to 2/5 if you want stronger guarantees

    # Network setup
    bandwidth_mbps: float = 10.0
    rtt_ms: float = 20.0
    gw_bandwidth_mbps: float = 50.0
    gw_rtt_ms: float = 10.0
    n_gateways: int = 10

    # QKD
    key_rate_bps_mean: float = 2e5
    key_rate_hetero: float = 0.5
    pool_capacity_bits: float = 2e7
    round_gap_s: float = 1.0

    # Baseline knobs
    static_fraction_clients: float = 0.3
    random_p: float = 0.3

    # Right-Sized knobs
    rs_base_fraction: float = 0.9
    rs_prioritize_topk: float = 0.6
    rs_scarcity_beta: float = 0.7
    rs_fairness_gamma: float = 0.15

    # Attack
    attack_enabled: bool = False
    adv_frac: float = 0.0
    attack_strength: float = 1.0

    # Run
    seed: int = 1
    device: str = "auto"
    log_every: int = 25


# -----------------------------
# One run (full logging)
# -----------------------------
def run_one(cfg: ExpConfig, out_root: str) -> str:
    ensure_dir(out_root)
    run_id = f"run_{time.time_ns()}_{os.getpid()}_{uuid.uuid4().hex[:8]}_{cfg.dataset}_{cfg.topology}_{cfg.fl_algo}_{cfg.policy}_seed{cfg.seed}"
    run_dir = os.path.join(out_root, run_id)
    ensure_dir(run_dir)

    # Save config (atomic)
    safe_json_dump(asdict(cfg), os.path.join(run_dir, "config.json"))

    set_seed(cfg.seed)
    device = pick_device(cfg.device)

    # Data
    train_ds, test_ds, model_fn, y = load_data(cfg.dataset)
    parts = dirichlet_partition(y, cfg.n_clients, cfg.dirichlet_alpha, cfg.seed,
                               min_per_client=int(cfg.min_samples_per_client))

    data_sizes = [len(p) for p in parts]
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False)

    # Network
    topo = TopologyConfig(cfg.topology, n_gateways=cfg.n_gateways)
    base_link = Link(cfg.bandwidth_mbps, cfg.rtt_ms)
    gw_link = Link(cfg.gw_bandwidth_mbps, cfg.gw_rtt_ms) if cfg.topology.upper() == "TWO_HOP" else None
    net = NetworkSimulator(
        n_clients=cfg.n_clients,
        topo=topo,
        base_link=base_link,
        gateway_link=gw_link,
        qkd_rate_mean=cfg.key_rate_bps_mean,
        qkd_rate_hetero=cfg.key_rate_hetero,
        qkd_pool_bits=cfg.pool_capacity_bits,
        seed=cfg.seed,
    )

    # Policy + attack
    policy = make_policy(
        cfg.policy, cfg.seed, cfg.static_fraction_clients, cfg.random_p,
        cfg.rs_base_fraction, cfg.rs_prioritize_topk, cfg.rs_scarcity_beta, cfg.rs_fairness_gamma
    )
    atk = Attacker(AttackConfig(cfg.attack_enabled, cfg.adv_frac, cfg.attack_strength, cfg.seed + 1337))

    # Global model
    global_model = model_fn().to(device)
    global_params = get_params(global_model)
    payload_bytes = model_num_bytes(global_model)
    payload_bits = float(payload_bytes * 8)

    # State for RightSized
    impact_score = {cid: float(data_sizes[cid]) for cid in range(cfg.n_clients)}  # cold start proxy
    used_bits_so_far = {cid: 0.0 for cid in range(cfg.n_clients)}

    # Logging
    rows = []
    total_time_s = 0.0
    total_bytes = 0
    total_qkd_bits = 0.0

    acc_hist = []
    skipped_empty_clients_total = 0

    for r in range(1, cfg.rounds + 1):
        # QKD generation between rounds
        net.tick_qkd(cfg.round_gap_s)

        m = max(1, int(math.ceil(cfg.clients_per_round * cfg.n_clients)))
        selected = random.sample(range(cfg.n_clients), m)
        adversaries = atk.pick_adversaries(selected)

        ctx = {
            "pools": net.pools,
            "impact_score": impact_score,
            "used_bits_so_far": used_bits_so_far,
        }
        qfrac = policy.allocate(r, selected, payload_bytes, ctx)

        local_updates = []
        local_upd_norms = []
        starvation_round = 0
        qkd_clients = 0
        qkd_partial_clients = 0
        skipped_empty_clients_round = 0

        for cid in selected:
            # comm time depends on topology
            total_time_s += net.comm_time_s(cid, payload_bytes)
            total_bytes += payload_bytes

            frac = float(max(0.0, min(1.0, qfrac.get(cid, 0.0))))
            bits_needed = frac * payload_bits
            if bits_needed > 1e-9:
                ok = net.pools[cid].try_consume(bits_needed)
                if not ok:
                    starvation_round += 1
                    frac = 0.0
                    bits_needed = 0.0
                else:
                    total_qkd_bits += bits_needed
                    used_bits_so_far[cid] += bits_needed
                    qkd_clients += 1
                    if frac < 0.999:
                        qkd_partial_clients += 1

            # local data
            idxs = parts[cid]
            if len(idxs) == 0:
                # should not happen due to fixed partition, but keep failsafe
                skipped_empty_clients_round += 1
                continue

            subset = torch.utils.data.Subset(train_ds, idxs)
            if len(subset) == 0:
                skipped_empty_clients_round += 1
                continue

            loader = torch.utils.data.DataLoader(subset, batch_size=cfg.batch_size, shuffle=True)

            # local train
            local_model = model_fn().to(device)
            set_params(local_model, global_params)
            before = get_params(local_model)

            if cfg.fl_algo == "fedavg":
                train_local_fedavg(local_model, loader, cfg.local_epochs, cfg.lr, device)
            elif cfg.fl_algo == "fedprox":
                train_local_fedprox(local_model, global_params, loader, cfg.local_epochs, cfg.lr, cfg.fedprox_mu, device)
            else:
                raise ValueError("fl_algo must be fedavg|fedprox")

            after = get_params(local_model)

            # poisoning (if adversary)
            if cid in adversaries:
                after = atk.poison_update(after, global_params)

            # impact proxy
            upd_norm = delta_norm(after, before)
            local_upd_norms.append((cid, upd_norm))

            local_updates.append((after, len(idxs)))

        skipped_empty_clients_total += skipped_empty_clients_round

        # aggregate (guard: if everything got skipped, keep global as-is)
        if len(local_updates) > 0:
            new_global = avg_params(local_updates)
            global_params = new_global
            set_params(global_model, global_params)

        # update impact score (EMA)
        for cid, upd_norm in local_upd_norms:
            impact_score[cid] = 0.7 * impact_score.get(cid, 0.0) + 0.3 * float(upd_norm)

        # eval
        test_loss, test_acc = eval_model(global_model, test_loader, device)
        acc_hist.append(test_acc)

        total_starv = sum(net.pools[c].starvation_events for c in range(cfg.n_clients))
        fairness = jain_fairness([used_bits_so_far[c] for c in range(cfg.n_clients)])
        avg_pool_fill_sel = float(np.mean([net.pools[c].pool_bits / net.pools[c].cfg.pool_capacity_bits for c in selected]))

        rows.append({
            "round": r,
            "test_acc": test_acc,
            "test_loss": test_loss,
            "total_time_s": total_time_s,
            "total_bytes": total_bytes,
            "total_qkd_bits": total_qkd_bits,
            "starvation_round": starvation_round,
            "starvation_total": total_starv,
            "qkd_clients": qkd_clients,
            "qkd_partial_clients": qkd_partial_clients,
            "fairness_jain": fairness,
            "avg_pool_fill_selected": avg_pool_fill_sel,
            "adv_frac_selected": (len(adversaries) / max(1, len(selected))),
            "skipped_empty_clients_round": skipped_empty_clients_round,
            "skipped_empty_clients_total": skipped_empty_clients_total,
        })

        if r == 1 or r % cfg.log_every == 0 or r == cfg.rounds:
            print(f"[{run_id}] r={r:04d} acc={test_acc:.4f} "
                  f"qkd={total_qkd_bits/1e6:.1f}Mbit starv={total_starv} fair={fairness:.3f} "
                  f"t={total_time_s:.1f}s adv={len(adversaries)} skipEmpty={skipped_empty_clients_total}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(run_dir, "rounds.csv"), index=False)

    # final metrics + robust metrics
    rounds_arr = df["round"].to_numpy()
    acc_arr = df["test_acc"].to_numpy()

    auc_acc = auc_trapz(acc_arr, rounds_arr) / float(cfg.rounds)  # normalized AUC
    t75 = time_to_target(acc_arr, rounds_arr, 0.75)
    t80 = time_to_target(acc_arr, rounds_arr, 0.80)

    final = df.iloc[-1].to_dict()
    final.update({
        "run_id": run_id,
        "dataset": cfg.dataset,
        "topology": cfg.topology,
        "fl_algo": cfg.fl_algo,
        "policy": policy.name,
        "seed": cfg.seed,
        "n_clients": cfg.n_clients,
        "clients_per_round": cfg.clients_per_round,
        "rounds": cfg.rounds,
        "dirichlet_alpha": cfg.dirichlet_alpha,
        "min_samples_per_client": cfg.min_samples_per_client,
        "bandwidth_mbps": cfg.bandwidth_mbps,
        "rtt_ms": cfg.rtt_ms,
        "gw_bandwidth_mbps": cfg.gw_bandwidth_mbps,
        "gw_rtt_ms": cfg.gw_rtt_ms,
        "n_gateways": cfg.n_gateways,
        "key_rate_bps_mean": cfg.key_rate_bps_mean,
        "pool_capacity_bits": cfg.pool_capacity_bits,
        "attack_enabled": cfg.attack_enabled,
        "adv_frac": cfg.adv_frac,
        "attack_strength": cfg.attack_strength,
        "auc_acc_norm": auc_acc,
        "t_acc_0p75": t75,
        "t_acc_0p80": t80,
        # include knobs for traceability
        "static_fraction_clients": cfg.static_fraction_clients,
        "random_p": cfg.random_p,
        "rs_base_fraction": cfg.rs_base_fraction,
        "rs_prioritize_topk": cfg.rs_prioritize_topk,
        "rs_scarcity_beta": cfg.rs_scarcity_beta,
        "rs_fairness_gamma": cfg.rs_fairness_gamma,
    })
    pd.DataFrame([final]).to_csv(os.path.join(run_dir, "final.csv"), index=False)

    return run_dir


# -----------------------------
# GRID: parameters
# -----------------------------
def build_full_grid(dataset: str, seeds: List[int]) -> List[ExpConfig]:
    policies = ["classical", "all_qkd", "static_split", "random", "right_sized"]
    fl_algos = ["fedavg", "fedprox"]
    topologies = ["STAR", "RING", "TWO_HOP"]

    n_clients_list = [50, 200]
    cpr_list = [0.1, 0.2]
    alphas = [0.1, 0.3, 1.0]

    # network scenarios: (bandwidth, rtt)
    net_scen = [
        (5.0,  50.0),
        (10.0, 20.0),
        (50.0, 5.0),
    ]

    # qkd scarcity: (key_rate, pool)
    qkd_scen = [
        (5e4,  5e6),   # scarce
        (2e5,  2e7),   # medium
        (8e5,  8e7),   # abundant
    ]

    attacks = [
        (False, 0.0, 1.0),
        (True,  0.2, 1.0),
    ]

    cfgs = []
    for seed in seeds:
        for pol in policies:
            for fl in fl_algos:
                for topo in topologies:
                    for n_clients in n_clients_list:
                        for cpr in cpr_list:
                            for alpha in alphas:
                                for (bw, rtt) in net_scen:
                                    for (kr, pool) in qkd_scen:
                                        for (atk_on, adv_frac, strength) in attacks:
                                            rounds = 100 if dataset == "mnist" else 300
                                            local_epochs = 1
                                            bs = 32
                                            lr = 0.01
                                            cfg = ExpConfig(
                                                dataset=dataset,
                                                fl_algo=fl,
                                                policy=pol,
                                                topology=topo,
                                                n_clients=n_clients,
                                                clients_per_round=cpr,
                                                rounds=rounds,
                                                local_epochs=local_epochs,
                                                batch_size=bs,
                                                lr=lr,
                                                fedprox_mu=0.01,
                                                dirichlet_alpha=alpha,
                                                min_samples_per_client=1,  # you can raise to 2/5 if you want
                                                bandwidth_mbps=bw,
                                                rtt_ms=rtt,
                                                gw_bandwidth_mbps=max(20.0, bw * 2.0),
                                                gw_rtt_ms=max(2.0, rtt / 2.0),
                                                n_gateways=max(5, int(n_clients / 10)),
                                                key_rate_bps_mean=kr,
                                                key_rate_hetero=0.5,
                                                pool_capacity_bits=pool,
                                                round_gap_s=1.0,
                                                static_fraction_clients=0.3,
                                                random_p=0.3,
                                                rs_base_fraction=0.9,
                                                rs_prioritize_topk=0.6,
                                                rs_scarcity_beta=0.7,
                                                rs_fairness_gamma=0.15,
                                                attack_enabled=atk_on,
                                                adv_frac=adv_frac,
                                                attack_strength=strength,
                                                seed=seed,
                                                device="auto",
                                                log_every=50 if dataset == "cifar10" else 25,
                                            )
                                            cfgs.append(cfg)
    return cfgs


# -----------------------------
# Summaries / plots / tables / ranking
# -----------------------------
def find_runs(root: str) -> List[str]:
    out = []
    for r, _d, files in os.walk(root):
        if "config.json" in files and "rounds.csv" in files and "final.csv" in files:
            out.append(r)
    return out

def load_all(root: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    run_dirs = find_runs(root)
    if not run_dirs:
        raise FileNotFoundError(f"No runs found under: {root}")

    rounds_all = []
    finals_all = []
    skipped = []

    for rd in run_dirs:
        cfg_path = os.path.join(rd, "config.json")
        rounds_path = os.path.join(rd, "rounds.csv")
        final_path = os.path.join(rd, "final.csv")

        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception as e:
            skipped.append((rd, f"config.json error: {e}"))
            continue

        try:
            rounds = pd.read_csv(rounds_path)
            final = pd.read_csv(final_path)
        except Exception as e:
            skipped.append((rd, f"csv error: {e}"))
            continue

        rounds["run_dir"] = rd
        for k in [
            "dataset","topology","fl_algo","policy","seed","n_clients","clients_per_round",
            "rounds","dirichlet_alpha","min_samples_per_client",
            "bandwidth_mbps","rtt_ms","key_rate_bps_mean",
            "pool_capacity_bits","attack_enabled","adv_frac","attack_strength"
        ]:
            rounds[k] = cfg.get(k)

        final["run_dir"] = rd

        rounds_all.append(rounds)
        finals_all.append(final)

    if skipped:
        print("\n[WARN] Skipped corrupted/incomplete runs:")
        for rd, msg in skipped:
            print(" -", rd, "=>", msg)
        print(f"[WARN] Total skipped: {len(skipped)}\n")

    if not rounds_all or not finals_all:
        raise RuntimeError("All runs were skipped — no valid data to summarize.")

    return (
        pd.concat(rounds_all, ignore_index=True),
        pd.concat(finals_all, ignore_index=True),
    )

def export_summary_tables(finals: pd.DataFrame, out_dir: str):
    ensure_dir(out_dir)
    group_cols = ["dataset","topology","fl_algo","policy","n_clients","clients_per_round",
                  "dirichlet_alpha","bandwidth_mbps","rtt_ms","key_rate_bps_mean","pool_capacity_bits",
                  "attack_enabled","adv_frac","attack_strength"]
    metrics = ["test_acc","auc_acc_norm","t_acc_0p75","t_acc_0p80",
               "total_time_s","total_qkd_bits","starvation_total","fairness_jain"]

    rows = []
    for keys, g in finals.groupby(group_cols):
        row = dict(zip(group_cols, keys))
        row["n_runs"] = len(g)
        for m in metrics:
            vals = g[m].to_numpy(dtype=np.float64)
            mu, sd, ci = mean_ci95(vals)
            row[f"{m}_mean"] = mu
            row[f"{m}_std"] = sd
            row[f"{m}_ci95"] = ci
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "summary_final.csv"), index=False)

    view_cols = group_cols + ["n_runs",
                              "test_acc_mean","test_acc_ci95",
                              "auc_acc_norm_mean","auc_acc_norm_ci95",
                              "total_qkd_bits_mean","starvation_total_mean",
                              "fairness_jain_mean","total_time_s_mean"]
    view = df[view_cols].copy()
    view["acc"] = view.apply(lambda r: f"{r['test_acc_mean']:.3f}±{r['test_acc_ci95']:.3f}", axis=1)
    view["auc"] = view.apply(lambda r: f"{r['auc_acc_norm_mean']:.3f}±{r['auc_acc_norm_ci95']:.3f}", axis=1)
    view["qkd(Mbit)"] = view["total_qkd_bits_mean"].map(lambda x: f"{x/1e6:.1f}")
    view["starv"] = view["starvation_total_mean"].map(lambda x: f"{x:.1f}")
    view["fair"] = view["fairness_jain_mean"].map(lambda x: f"{x:.3f}")
    view["time(s)"] = view["total_time_s_mean"].map(lambda x: f"{x:.1f}")

    tex_cols = group_cols + ["n_runs","acc","auc","qkd(Mbit)","starv","fair","time(s)"]
    with open(os.path.join(out_dir, "summary_final.tex"), "w", encoding="utf-8") as f:
        f.write(view[tex_cols].to_latex(index=False, escape=True))

def plot_curves_with_ci(rounds_df: pd.DataFrame, out_dir: str):
    ensure_dir(out_dir)
    scenario_cols = ["dataset","topology","fl_algo","n_clients","clients_per_round",
                     "dirichlet_alpha","bandwidth_mbps","rtt_ms","key_rate_bps_mean","pool_capacity_bits",
                     "attack_enabled","adv_frac","attack_strength"]

    for keys, g in rounds_df.groupby(scenario_cols):
        (dataset, topo, fl, n_clients, cpr, alpha, bw, rtt, kr, pool, atk, advf, strength) = keys

        tag = f"{dataset}_{topo}_{fl}_N{n_clients}_cpr{cpr}_a{alpha}_bw{bw}_rtt{rtt}_kr{int(kr)}_pool{int(pool)}_atk{int(atk)}"
        title = f"{dataset}|{topo}|{fl}|N={n_clients}|cpr={cpr}|a={alpha}|bw={bw}|rtt={rtt}|kr={kr:.0f}|pool={pool:.0f}|atk={atk}"

        # Accuracy curve
        plt.figure()
        for pol, gp in g.groupby("policy"):
            s = gp.groupby("round")["test_acc"].agg(["mean","std","count"]).reset_index()
            ci = 1.96 * (s["std"] / np.sqrt(s["count"].clip(lower=1)))
            plt.plot(s["round"], s["mean"], label=pol)
            plt.fill_between(s["round"], s["mean"]-ci, s["mean"]+ci, alpha=0.2)
        plt.xlabel("Round"); plt.ylabel("Test Accuracy"); plt.title(title); plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"acc_{tag}.png"), dpi=220)
        plt.savefig(os.path.join(out_dir, f"acc_{tag}.pdf"))
        plt.close()

        # QKD bits curve (Mbit)
        plt.figure()
        for pol, gp in g.groupby("policy"):
            s = gp.groupby("round")["total_qkd_bits"].agg(["mean","std","count"]).reset_index()
            ci = 1.96 * (s["std"] / np.sqrt(s["count"].clip(lower=1)))
            plt.plot(s["round"], s["mean"]/1e6, label=pol)
            plt.fill_between(s["round"], (s["mean"]-ci)/1e6, (s["mean"]+ci)/1e6, alpha=0.2)
        plt.xlabel("Round"); plt.ylabel("Total QKD Bits (Mbit)"); plt.title(title); plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"qkd_{tag}.png"), dpi=220)
        plt.savefig(os.path.join(out_dir, f"qkd_{tag}.pdf"))
        plt.close()

        # Starvation curve
        plt.figure()
        for pol, gp in g.groupby("policy"):
            s = gp.groupby("round")["starvation_total"].agg(["mean","std","count"]).reset_index()
            ci = 1.96 * (s["std"] / np.sqrt(s["count"].clip(lower=1)))
            plt.plot(s["round"], s["mean"], label=pol)
            plt.fill_between(s["round"], s["mean"]-ci, s["mean"]+ci, alpha=0.2)
        plt.xlabel("Round"); plt.ylabel("Key Starvation (cumulative)"); plt.title(title); plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"starv_{tag}.png"), dpi=220)
        plt.savefig(os.path.join(out_dir, f"starv_{tag}.pdf"))
        plt.close()

def plot_cdf_and_boxplots(finals: pd.DataFrame, out_dir: str):
    ensure_dir(out_dir)

    group_cols = ["dataset","topology","fl_algo","attack_enabled","adv_frac","attack_strength"]
    metrics = {
        "test_acc": ("Final Accuracy", False),
        "auc_acc_norm": ("AUC Accuracy (norm)", False),
        "total_qkd_bits": ("Total QKD Bits", True),
        "starvation_total": ("Starvation Total", True),
        "total_time_s": ("Total Time (s)", True),
    }

    for keys, g in finals.groupby(group_cols):
        dataset, topo, fl, atk, advf, strength = keys
        base_tag = f"{dataset}_{topo}_{fl}_atk{int(atk)}_adv{advf}_s{strength}"

        for m, (label, logy) in metrics.items():
            # CDF
            plt.figure()
            for pol, gp in g.groupby("policy"):
                vals = np.sort(gp[m].to_numpy(dtype=np.float64))
                y = np.arange(1, len(vals)+1) / max(len(vals), 1)
                plt.plot(vals, y, label=pol)
            plt.xlabel(label); plt.ylabel("CDF")
            plt.title(f"CDF - {label} | {dataset}|{topo}|{fl}|atk={atk}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"cdf_{m}_{base_tag}.png"), dpi=220)
            plt.savefig(os.path.join(out_dir, f"cdf_{m}_{base_tag}.pdf"))
            plt.close()

            # Boxplot
            plt.figure()
            policies = sorted(g["policy"].unique().tolist())
            data = [g[g["policy"] == p][m].to_numpy(dtype=np.float64) for p in policies]
            plt.boxplot(data, tick_labels=policies, showfliers=False)
            plt.ylabel(label)
            plt.title(f"Boxplot - {label} | {dataset}|{topo}|{fl}|atk={atk}")
            if logy:
                plt.yscale("log")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"box_{m}_{base_tag}.png"), dpi=220)
            plt.savefig(os.path.join(out_dir, f"box_{m}_{base_tag}.pdf"))
            plt.close()

def rank_policies(summary_final: pd.DataFrame, out_dir: str,
                  w_acc: float = 0.55, w_qkd: float = 0.20, w_starv: float = 0.15, w_time: float = 0.10):
    ensure_dir(out_dir)
    df = summary_final.copy()

    scenario_cols = ["dataset","topology","fl_algo","n_clients","clients_per_round",
                     "dirichlet_alpha","bandwidth_mbps","rtt_ms","key_rate_bps_mean","pool_capacity_bits",
                     "attack_enabled","adv_frac","attack_strength"]
    acc = "test_acc_mean"
    qkd = "total_qkd_bits_mean"
    starv = "starvation_total_mean"
    t = "total_time_s_mean"

    rows = []
    for keys, g in df.groupby(scenario_cols):
        g = g.copy()

        def minmax(x):
            mn, mx = float(np.min(x)), float(np.max(x))
            if abs(mx - mn) < 1e-12:
                return np.zeros_like(x, dtype=np.float64)
            return (x - mn) / (mx - mn)

        acc_n = minmax(g[acc].to_numpy(dtype=np.float64))
        qkd_n = minmax(g[qkd].to_numpy(dtype=np.float64))
        starv_n = minmax(g[starv].to_numpy(dtype=np.float64))
        t_n = minmax(g[t].to_numpy(dtype=np.float64))

        score = (w_acc * acc_n) - (w_qkd * qkd_n) - (w_starv * starv_n) - (w_time * t_n)
        g["score"] = score
        g = g.sort_values("score", ascending=False)

        out = g[scenario_cols + ["policy", acc, qkd, starv, t, "score", "n_runs"]]
        rows.append(out)

        tag = "_".join([str(k).replace(".","p") for k in keys])
        out.to_csv(os.path.join(out_dir, f"ranking_{tag}.csv"), index=False)
        with open(os.path.join(out_dir, f"ranking_{tag}.tex"), "w", encoding="utf-8") as f:
            f.write(out.to_latex(index=False, float_format="%.4f"))

    all_rank = pd.concat(rows, ignore_index=True)
    all_rank.to_csv(os.path.join(out_dir, "ranking_all.csv"), index=False)

def summarize_all(in_dir: str, out_dir: str):
    ensure_dir(out_dir)
    rounds_df, finals_df = load_all(in_dir)

    rounds_df.to_csv(os.path.join(out_dir, "all_rounds.csv"), index=False)
    finals_df.to_csv(os.path.join(out_dir, "all_final.csv"), index=False)

    export_summary_tables(finals_df, out_dir)
    summary_final = pd.read_csv(os.path.join(out_dir, "summary_final.csv"))

    figs_dir = os.path.join(out_dir, "figures_curves")
    plot_curves_with_ci(rounds_df, figs_dir)

    figs2_dir = os.path.join(out_dir, "figures_cdf_box")
    plot_cdf_and_boxplots(finals_df, figs2_dir)

    rank_dir = os.path.join(out_dir, "rankings")
    rank_policies(summary_final, rank_dir)

    print("Summary complete:", out_dir)


# -----------------------------
# Parallel grid execution
# -----------------------------
def _run_one_wrapper(args):
    cfg_dict, out_root = args
    cfg = ExpConfig(**cfg_dict)
    try:
        return run_one(cfg, out_root)
    except Exception as e:
        # Better debug info (especially under multiprocessing)
        print("\n[ERROR] Run failed with cfg:")
        print(cfg_dict)
        print("[ERROR] Exception:", repr(e), "\n")
        raise

def run_grid(dataset: str, out_root: str, n_seeds: int, seed0: int, jobs: int):
    ensure_dir(out_root)
    seeds = list(range(seed0, seed0 + n_seeds))
    cfgs = build_full_grid(dataset, seeds)

    meta = {
        "dataset": dataset,
        "seed0": seed0,
        "n_seeds": n_seeds,
        "seeds": seeds,
        "n_runs": len(cfgs),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    safe_json_dump(meta, os.path.join(out_root, "grid_meta.json"))

    print(f"GRID: {len(cfgs)} runs | dataset={dataset} | out={out_root} | jobs={jobs}")

    payload = [ (asdict(cfg), out_root) for cfg in cfgs ]

    if jobs <= 1:
        for i, item in enumerate(payload, 1):
            cfgd, _ = item
            print(f"[{i}/{len(payload)}] {cfgd['policy']} {cfgd['fl_algo']} {cfgd['topology']} "
                  f"N={cfgd['n_clients']} a={cfgd['dirichlet_alpha']} bw={cfgd['bandwidth_mbps']} rtt={cfgd['rtt_ms']} "
                  f"kr={cfgd['key_rate_bps_mean']} pool={cfgd['pool_capacity_bits']} atk={cfgd['attack_enabled']} seed={cfgd['seed']}")
            _run_one_wrapper(item)
    else:
        jobs = min(jobs, max(1, cpu_count() - 1))
        with Pool(processes=jobs) as pool:
            for i, _ in enumerate(pool.imap_unordered(_run_one_wrapper, payload), 1):
                if i % max(1, int(len(payload) / 50)) == 0 or i == len(payload):
                    print(f"Progress: {i}/{len(payload)} runs done.")

    print("GRID done. Now run summarize to generate figures/tables:")
    print(f"  python bazooka_qkd_fl_full.py summarize --in {out_root} --out {out_root}_summary")


# -----------------------------
# CLI
# -----------------------------
def build_parser():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("run-grid", help="Run the full grid (baselines + right-sized) with seeds")
    p1.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist")
    p1.add_argument("--out", required=True)
    p1.add_argument("--n-seeds", type=int, default=5)
    p1.add_argument("--seed0", type=int, default=1)
    p1.add_argument("--jobs", type=int, default=1)
    p1.set_defaults(func=lambda a: run_grid(a.dataset, a.out, a.n_seeds, a.seed0, a.jobs))

    p2 = sub.add_parser("summarize", help="Summarize existing results (plots + tables + rankings)")
    p2.add_argument("--in", dest="inp", required=True)
    p2.add_argument("--out", required=True)
    p2.set_defaults(func=lambda a: summarize_all(a.inp, a.out))

    return ap

def main():
    ap = build_parser()
    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()

