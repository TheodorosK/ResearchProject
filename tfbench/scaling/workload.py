"""Workload definitions for the scaling benchmark.  TensorFlow-free (numpy + genetics only).

A *Point* is a workload (tier x population x training steps x population mode); a *Strategy* is how it is executed
(device x mode x workers).  Paired CPU/GPU comparisons use the SAME Point, so genomes, data, seeds and budgets are
identical by construction (``workload_signature`` makes that checkable).
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .. import genetics as G
from ..evolution import make_dataset


@dataclass(frozen=True)
class TierSpec:
    name: str
    min_layers: int
    max_layers: int
    units: Tuple[int, ...]
    canonical: Tuple[int, ...]            # fixed architecture used for homogeneous (batch-friendly) populations


TIERS: Dict[str, TierSpec] = {
    "small": TierSpec("small", 1, 2, (32, 48, 64), (64, 64)),
    "medium": TierSpec("medium", 2, 4, (128, 192, 256), (256, 256, 256)),
    "large": TierSpec("large", 4, 6, (512, 768, 1024), (1024, 1024, 1024, 1024, 1024)),
}
TIER_ORDER = ("small", "medium", "large")


@dataclass(frozen=True)
class Settings:
    """Everything that defines the workload and is shared by all strategies."""
    seed: int = 1234
    batch: int = 128
    in_dim: int = 32
    train_n: int = 4096
    val_n: int = 1024
    generations: int = 3
    activations: Tuple[str, ...] = ("relu", "tanh")
    tiers: Tuple[str, ...] = TIER_ORDER
    pops: Tuple[int, ...] = (4, 8, 16, 32)
    steps: Tuple[int, ...] = (1, 5, 20, 50)
    pilot_pop: int = 8
    pilot_steps: Tuple[int, ...] = (1, 5, 20, 50)
    confirm_reps: int = 3
    confirm_points: int = 3
    max_cell_s: float = 300.0
    max_total_s: float = 1800.0
    workers: Tuple[int, ...] = (2, 4)
    max_workers: int = 4
    extend_steps: Tuple[int, ...] = ()
    skip_batched: bool = False
    skip_mp: bool = False
    grappler: str = "none"
    quick: bool = False
    design: str = "adaptive"                    # adaptive (crossover search) | capacity (memory / concurrency grid + resident probe)
    ladder: Tuple[int, ...] = (1, 2, 4, 8, 16, 32)   # capacity design: numbers of simultaneously resident models probed
    resident_tiers: Tuple[str, ...] = ("large", "medium")

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Settings":
        d = dict(d)
        for k in ("activations", "tiers", "pops", "steps", "pilot_steps", "workers", "extend_steps", "ladder", "resident_tiers"):
            if k in d:
                d[k] = tuple(d[k])
        return Settings(**d)


@dataclass(frozen=True)
class Point:
    tier: str
    population: int
    steps: int
    mode: str = "hetero"                  # hetero: random architectures within the tier | homo: one shared architecture
    generations: int = 3

    @property
    def key(self) -> str:
        return f"{self.tier}/P{self.population}/S{self.steps}/{self.mode}"

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Strategy:
    device: str                           # cpu | gpu
    mode: str = "seq"                     # seq | batched | mp
    workers: int = 1

    @property
    def name(self) -> str:
        return f"{self.device}-mp{self.workers}" if self.mode == "mp" else f"{self.device}-{self.mode}"

    @property
    def variant(self) -> str:             # worker job variant
        return f"mp{self.workers}" if self.mode == "mp" else self.mode


@dataclass(frozen=True)
class Cell:
    point: Point
    strategy: Strategy
    phase: str = "pilot"                  # pilot | expansion | confirmation
    rep: int = 0

    @property
    def cell_id(self) -> str:
        return f"{self.phase}:{self.point.key}:{self.strategy.name}:r{self.rep}"

    def as_dict(self) -> Dict[str, Any]:
        return {"point": self.point.as_dict(), "strategy": asdict(self.strategy), "phase": self.phase, "rep": self.rep}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Cell":
        return Cell(Point(**d["point"]), Strategy(**d["strategy"]), d.get("phase", "pilot"), d.get("rep", 0))


# ----------------------------------------------------------------------------- populations
def make_space(point: Point, st: Settings) -> G.SearchSpace:
    t = TIERS[point.tier]
    if point.mode == "homo":              # offspring must stay batch-compatible: one architecture, one activation
        return G.SearchSpace(min_layers=len(t.canonical), max_layers=len(t.canonical), unit_choices=(t.canonical[0],),
                             activations=(st.activations[0],))
    return G.SearchSpace(min_layers=t.min_layers, max_layers=t.max_layers, unit_choices=t.units,
                         activations=tuple(st.activations))


def initial_population(point: Point, st: Settings) -> List[G.Genome]:
    """Deterministic: identical for every strategy/device given the same Point and Settings."""
    space = make_space(point, st)
    rng = random.Random(st.seed * 7919 + point.population * 104729 + (1 if point.mode == "homo" else 0))
    pop = []
    for uid in range(point.population):
        g = G.random_genome(rng, space, uid=uid)
        if point.mode == "homo":
            t = TIERS[point.tier]
            g.layers = [G.LayerGene(u, st.activations[0]) for u in t.canonical]
            G.repair(g, space)
        pop.append(g)
    return pop


def elite_count(population: int) -> int:
    return max(1, population // 4)


def n_evaluations(population: int, generations: int) -> int:
    return population + (generations - 1) * (population - elite_count(population))


def individual_seed(base_seed: int, uid: int) -> int:
    return (base_seed * 1_000_003 + uid) % (2 ** 31 - 1)


def batch_indices(seed: int, n: int, batch: int, steps: int) -> np.ndarray:
    """(steps, batch) training-sample indices; a fresh permutation per pass over the data."""
    rng = np.random.default_rng(seed)
    nb = max(1, n // batch)
    parts, have = [], 0
    while have < steps:
        parts.append(rng.permutation(n)[: nb * batch].reshape(nb, batch))
        have += nb
    return np.concatenate(parts)[:steps]


def genome_to_dict(g: G.Genome) -> Dict[str, Any]:
    return asdict(g)


def genome_from_dict(d: Dict[str, Any]) -> G.Genome:
    d = dict(d)
    d["layers"] = [G.LayerGene(**l) for l in d["layers"]]
    d["parents"] = tuple(d.get("parents", ()))
    return G.Genome(**d)


def workload_signature(point: Point, st: Settings) -> str:
    """Hash of everything that defines the work (genomes, data spec, seed, budget) and NOTHING about the strategy."""
    pop = [(tuple((l.units, l.activation) for l in g.layers), round(g.learning_rate, 8)) for g in initial_population(point, st)]
    payload = {"point": point.as_dict(), "pop": pop, "seed": st.seed, "batch": st.batch, "in_dim": st.in_dim,
               "train_n": st.train_n, "val_n": st.val_n, "acts": list(st.activations), "opt": "simple-adam"}
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def dataset_for(st: Settings) -> Dict[str, np.ndarray]:
    return make_dataset(st.train_n, st.val_n, 64, st.in_dim, st.seed)


def params_of(g: G.Genome, st: Settings) -> int:
    return G.count_params(g, st.in_dim, 1)


def point_param_stats(point: Point, st: Settings) -> Dict[str, float]:
    ps = [params_of(g, st) for g in initial_population(point, st)]
    return {"mean": float(np.mean(ps)), "min": float(min(ps)), "max": float(max(ps)),
            "layers_mean": float(np.mean([len(g.layers) for g in initial_population(point, st)]))}


def bucket_key(g: G.Genome) -> Tuple:
    """Genomes that can be trained as ONE batched network after zero-padding widths: same depth, same activations."""
    return tuple(l.activation for l in g.layers)


def bucket_genomes(genomes: List[G.Genome]) -> List[List[int]]:
    """Indices grouped by bucket_key (order of first appearance)."""
    out: Dict[Tuple, List[int]] = {}
    for i, g in enumerate(genomes):
        out.setdefault(bucket_key(g), []).append(i)
    return list(out.values())


def padding_overhead(genomes: List[G.Genome], in_dim: int) -> float:
    """padded parameters / real parameters when the genomes are zero-padded to the widest member (>= 1)."""
    depth = len(genomes[0].layers)
    widths = [max(g.layers[j].units for g in genomes) for j in range(depth)]
    padded = sum(a * b + b for a, b in zip([in_dim] + widths, widths + [1]))
    real = sum(G.count_params(g, in_dim, 1) for g in genomes)
    return (padded * len(genomes)) / real if real else 1.0
