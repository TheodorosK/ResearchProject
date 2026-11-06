"""Genetic-algorithm core for neuroevolution.  Pure Python + numpy (no TensorFlow).

Representation
--------------
A ``Genome`` is a variable-depth MLP description plus numeric hyper-parameters:
  layers        1..max_layers hidden Dense layers (units, activation)
  learning_rate Adam learning rate (log-scale search)
  weight_sigma / weight_rate   optional weight-mutation parameters
  arch_mut_rate probability that a mutation changes the architecture

Validity
--------
``validate_genome`` defines validity; ``repair`` forces any genome into the
search space.  Every operator (random, mutate, crossover) ends with ``repair`` so
generated architectures are ALWAYS valid (this is property-tested).

Crossover of incompatible architectures
---------------------------------------
Two genomes are *compatible* when they have the same depth.
* compatible   -> layer-wise uniform crossover (each layer gene comes from either parent).
* incompatible -> "depth-aligned" crossover: the child keeps the depth of a
  randomly chosen *primary* parent; each layer position takes the primary's gene,
  or (p=0.5) the secondary's gene at the proportionally aligned position
  (index scaled by the depth ratio).  The child therefore always has a legal depth.
Numeric genes are blended (geometric for log-scale genes, arithmetic otherwise).

Weight inheritance (optional)
-----------------------------
``inherit_weights`` copies the overlapping block of each parent Dense tensor into
the child's fresh initialisation (hidden layers aligned by index; the output
layer only when depth is equal, otherwise its input semantics differ).
Shapes that do not exist in the parent keep the child's own initialisation.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ACTIVATIONS = ("relu", "tanh", "elu", "swish")


@dataclass(frozen=True)
class SearchSpace:
    min_layers: int = 1
    max_layers: int = 4
    unit_choices: Tuple[int, ...] = (8, 16, 32, 64, 128, 256)
    activations: Tuple[str, ...] = ACTIVATIONS
    lr_range: Tuple[float, float] = (1e-4, 3e-2)
    weight_sigma_range: Tuple[float, float] = (1e-3, 0.1)
    weight_rate_range: Tuple[float, float] = (0.01, 0.3)
    arch_rate_range: Tuple[float, float] = (0.05, 0.9)

    def problems(self) -> List[str]:
        out = []
        if not 1 <= self.min_layers <= self.max_layers:
            out.append("require 1 <= min_layers <= max_layers")
        if not self.unit_choices or any(u < 1 for u in self.unit_choices):
            out.append("unit_choices must be a non-empty set of positive ints")
        if not self.activations:
            out.append("activations must be non-empty")
        for name in ("lr_range", "weight_sigma_range", "weight_rate_range", "arch_rate_range"):
            lo, hi = getattr(self, name)
            if not (0 < lo <= hi):
                out.append(f"{name} must satisfy 0 < lo <= hi")
        return out


@dataclass(frozen=True)
class LayerGene:
    units: int
    activation: str


@dataclass
class Genome:
    layers: List[LayerGene]
    learning_rate: float = 1e-3
    weight_sigma: float = 0.02
    weight_rate: float = 0.1
    arch_mut_rate: float = 0.3
    uid: int = 0
    parents: Tuple[int, ...] = ()
    generation: int = 0

    def copy(self) -> "Genome":
        return replace(self, layers=list(self.layers))

    def signature(self) -> Tuple:
        """Identity for de-duplication (hyper-parameters rounded to 2 significant digits)."""
        return (tuple((l.units, l.activation) for l in self.layers),
                float(f"{self.learning_rate:.2g}"))

    def describe(self) -> str:
        arch = "-".join(f"{l.units}{l.activation[0]}" for l in self.layers)
        return f"[{arch}] lr={self.learning_rate:.2e}"


# ----------------------------------------------------------------- validity
def _clamp(x: float, lo: float, hi: float, default: float) -> float:
    if not isinstance(x, (int, float)) or not math.isfinite(x):
        return default
    return min(max(float(x), lo), hi)


def validate_genome(g: Genome, space: SearchSpace) -> List[str]:
    errs = []
    if not space.min_layers <= len(g.layers) <= space.max_layers:
        errs.append(f"depth {len(g.layers)} outside [{space.min_layers}, {space.max_layers}]")
    for i, l in enumerate(g.layers):
        if l.units not in space.unit_choices:
            errs.append(f"layer {i}: units {l.units} not allowed")
        if l.activation not in space.activations:
            errs.append(f"layer {i}: activation {l.activation!r} not allowed")
    for name, rng_ in (("learning_rate", space.lr_range), ("weight_sigma", space.weight_sigma_range),
                       ("weight_rate", space.weight_rate_range), ("arch_mut_rate", space.arch_rate_range)):
        v = getattr(g, name)
        if not (isinstance(v, (int, float)) and math.isfinite(v) and rng_[0] <= v <= rng_[1]):
            errs.append(f"{name}={v!r} outside {rng_}")
    return errs


def _nearest(units: int, choices: Sequence[int]) -> int:
    return min(choices, key=lambda c: (abs(c - units), c))


def repair(g: Genome, space: SearchSpace) -> Genome:
    """Force ``g`` into the search space (in place) and return it."""
    layers = list(g.layers)[: space.max_layers]
    while len(layers) < space.min_layers:
        layers.append(layers[-1] if layers else LayerGene(space.unit_choices[0], space.activations[0]))
    g.layers = [LayerGene(_nearest(int(l.units), space.unit_choices) if isinstance(l.units, (int, float))
                          and math.isfinite(l.units) else space.unit_choices[0],
                          l.activation if l.activation in space.activations else space.activations[0])
                for l in layers]
    lo, hi = space.lr_range
    g.learning_rate = _clamp(g.learning_rate, lo, hi, math.sqrt(lo * hi))
    lo, hi = space.weight_sigma_range
    g.weight_sigma = _clamp(g.weight_sigma, lo, hi, math.sqrt(lo * hi))
    lo, hi = space.weight_rate_range
    g.weight_rate = _clamp(g.weight_rate, lo, hi, (lo + hi) / 2)
    lo, hi = space.arch_rate_range
    g.arch_mut_rate = _clamp(g.arch_mut_rate, lo, hi, (lo + hi) / 2)
    return g


def _log_uniform(rng: random.Random, lo: float, hi: float) -> float:
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def random_genome(rng: random.Random, space: SearchSpace, uid: int = 0, generation: int = 0) -> Genome:
    depth = rng.randint(space.min_layers, space.max_layers)
    g = Genome(
        layers=[LayerGene(rng.choice(space.unit_choices), rng.choice(space.activations))
                for _ in range(depth)],
        learning_rate=_log_uniform(rng, *space.lr_range),
        weight_sigma=_log_uniform(rng, *space.weight_sigma_range),
        weight_rate=rng.uniform(*space.weight_rate_range),
        arch_mut_rate=rng.uniform(*space.arch_rate_range),
        uid=uid, generation=generation)
    return repair(g, space)


# ---------------------------------------------------------------- mutation
def mutate_numeric(g: Genome, rng: random.Random, space: SearchSpace, tau: float = 0.25) -> Genome:
    """Log-normal perturbation of all numeric genes (self-adaptive rates included)."""
    g.learning_rate *= math.exp(rng.gauss(0, tau))
    g.weight_sigma *= math.exp(rng.gauss(0, tau))
    g.weight_rate *= math.exp(rng.gauss(0, tau))
    g.arch_mut_rate *= math.exp(rng.gauss(0, tau))
    return repair(g, space)


ARCH_OPS = ("add_layer", "remove_layer", "resize_layer", "change_activation")


def mutate_architecture(g: Genome, rng: random.Random, space: SearchSpace) -> Tuple[Genome, Optional[str]]:
    """With probability ``g.arch_mut_rate`` apply ONE valid structural operator."""
    if rng.random() >= g.arch_mut_rate:
        return g, None
    ops = ["resize_layer", "change_activation"]
    if len(g.layers) < space.max_layers:
        ops.append("add_layer")
    if len(g.layers) > space.min_layers:
        ops.append("remove_layer")
    op = rng.choice(ops)
    layers = list(g.layers)
    if op == "add_layer":
        layers.insert(rng.randint(0, len(layers)),
                      LayerGene(rng.choice(space.unit_choices), rng.choice(space.activations)))
    elif op == "remove_layer":
        layers.pop(rng.randrange(len(layers)))
    elif op == "resize_layer":
        i = rng.randrange(len(layers))
        layers[i] = LayerGene(rng.choice(space.unit_choices), layers[i].activation)
    else:
        i = rng.randrange(len(layers))
        layers[i] = LayerGene(layers[i].units, rng.choice(space.activations))
    g.layers = layers
    return repair(g, space), op


def mutate(g: Genome, rng: random.Random, space: SearchSpace, uid: int, generation: int) -> Tuple[Genome, Optional[str]]:
    child = g.copy()
    child.uid, child.parents, child.generation = uid, (g.uid,), generation
    child, op = mutate_architecture(child, rng, space)
    child = mutate_numeric(child, rng, space)
    return child, op


# --------------------------------------------------------------- crossover
def compatible(a: Genome, b: Genome) -> bool:
    return len(a.layers) == len(b.layers)


def _aligned_index(i: int, n_from: int, n_to: int) -> int:
    if n_from <= 1:
        return 0
    return min(n_to - 1, round(i * (n_to - 1) / (n_from - 1)))


def crossover(a: Genome, b: Genome, rng: random.Random, space: SearchSpace,
              uid: int, generation: int) -> Tuple[Genome, Dict[str, object]]:
    """Returns (child, info).  ``info['primary']`` is the parent that supplies weights."""
    if compatible(a, b):
        layers = [rng.choice((la, lb)) for la, lb in zip(a.layers, b.layers)]
        primary, mode = (a if rng.random() < 0.5 else b), "same_depth_uniform"
    else:
        primary, secondary = (a, b) if rng.random() < 0.5 else (b, a)
        layers = []
        for i, gene in enumerate(primary.layers):
            if rng.random() < 0.5:
                layers.append(secondary.layers[_aligned_index(i, len(primary.layers), len(secondary.layers))])
            else:
                layers.append(gene)
        mode = "depth_aligned"

    def blend_log(x: float, y: float) -> float:
        w = rng.random()
        return math.exp(w * math.log(x) + (1 - w) * math.log(y))

    def blend_lin(x: float, y: float) -> float:
        w = rng.random()
        return w * x + (1 - w) * y

    child = Genome(
        layers=layers,
        learning_rate=blend_log(a.learning_rate, b.learning_rate),
        weight_sigma=blend_log(a.weight_sigma, b.weight_sigma),
        weight_rate=blend_lin(a.weight_rate, b.weight_rate),
        arch_mut_rate=blend_lin(a.arch_mut_rate, b.arch_mut_rate),
        uid=uid, parents=(a.uid, b.uid), generation=generation)
    return repair(child, space), {"mode": mode, "primary": primary.uid}


# --------------------------------------------------------------- selection
def _score(s: Optional[float]) -> float:
    return s if (s is not None and math.isfinite(s)) else float("-inf")


def tournament_select(rng: random.Random, scores: Sequence[Optional[float]], k: int) -> int:
    """Index of the best of ``k`` random contestants (higher score = fitter; failures never win
    unless every contestant failed)."""
    k = max(1, min(k, len(scores)))
    contestants = rng.sample(range(len(scores)), k)
    return max(contestants, key=lambda i: (_score(scores[i]), -i))


def elite_indices(scores: Sequence[Optional[float]], k: int) -> List[int]:
    ranked = sorted((i for i, s in enumerate(scores) if math.isfinite(_score(s))),
                    key=lambda i: (-_score(scores[i]), i))
    return ranked[: max(0, k)]


@dataclass
class Offspring:
    genome: Genome
    is_elite: bool = False
    primary_parent: Optional[int] = None     # index into the previous population (weight donor)
    operator: str = "random"


def next_generation(population: Sequence[Genome], scores: Sequence[Optional[float]],
                    rng: random.Random, space: SearchSpace, *, elite_k: int, tournament_k: int,
                    crossover_rate: float, next_uid: int, generation: int,
                    max_dedupe_tries: int = 5) -> Tuple[List[Offspring], int]:
    """Build the next population (same size).  Returns (offspring, next free uid).

    Elites are carried over UNCHANGED (same uid, never re-evaluated).  Remaining slots are
    filled by tournament selection + crossover (with prob ``crossover_rate``) + mutation;
    children identical (by signature) to an existing individual are re-mutated a few times.
    """
    n = len(population)
    elite_k = max(0, min(elite_k, n - 1)) if n > 1 else 0
    out: List[Offspring] = [Offspring(population[i], True, i, "elite") for i in elite_indices(scores, elite_k)]
    seen = {o.genome.signature() for o in out}
    uid = next_uid
    while len(out) < n:
        i = tournament_select(rng, scores, tournament_k)
        parent_idx, op = i, "mutation"
        if rng.random() < crossover_rate and n > 1:
            j = tournament_select(rng, scores, tournament_k)
            child, info = crossover(population[i], population[j], rng, space, uid, generation)
            parent_idx = i if info["primary"] == population[i].uid else j
            op = f"crossover:{info['mode']}"
            child, _ = mutate_architecture(child, rng, space)
            child = mutate_numeric(child, rng, space)
        else:
            child, _ = mutate(population[i], rng, space, uid, generation)
        tries = 0
        while child.signature() in seen and tries < max_dedupe_tries:
            child, _ = mutate(child, rng, space, uid, generation)
            child.parents = population[i].uid, population[parent_idx].uid
            tries += 1
        seen.add(child.signature())
        out.append(Offspring(child, False, parent_idx, op))
        uid += 1
    return out, uid


# ---------------------------------------------------------- weights (numpy)
def layer_dims(g: Genome, in_dim: int, out_dim: int) -> List[Tuple[int, int]]:
    dims, prev = [], in_dim
    for l in g.layers:
        dims.append((prev, l.units))
        prev = l.units
    dims.append((prev, out_dim))
    return dims


def count_params(g: Genome, in_dim: int, out_dim: int) -> int:
    return sum(i * o + o for i, o in layer_dims(g, in_dim, out_dim))


def inherit_weights(parent: Sequence[np.ndarray], parent_depth: int,
                    child_init: Sequence[np.ndarray], child_depth: int
                    ) -> Tuple[List[np.ndarray], float]:
    """Overlay overlapping blocks of parent Dense weights onto the child's initial weights.

    ``parent`` / ``child_init`` are flat lists [k0, b0, k1, b1, ..., k_out, b_out].
    Returns (new child weights, fraction of child parameters copied from the parent).
    """
    assert len(parent) == 2 * (parent_depth + 1) and len(child_init) == 2 * (child_depth + 1)
    out = [w.copy() for w in child_init]
    copied = 0
    total = sum(w.size for w in out)
    n_hidden = min(parent_depth, child_depth)
    pairs = [(i, i) for i in range(n_hidden)]
    if parent_depth == child_depth:
        pairs.append((parent_depth, child_depth))      # output layer, same semantics
    for pi, ci in pairs:
        pk, pb = parent[2 * pi], parent[2 * pi + 1]
        ck, cb = out[2 * ci], out[2 * ci + 1]
        r, c = min(pk.shape[0], ck.shape[0]), min(pk.shape[1], ck.shape[1])
        ck[:r, :c] = pk[:r, :c]
        cb[:c] = pb[:c]
        copied += r * c + c
    return out, copied / total if total else 0.0


def perturb_weights(weights: Sequence[np.ndarray], rng: np.random.Generator,
                    sigma: float, rate: float) -> List[np.ndarray]:
    """Each element is perturbed with probability ``rate`` by N(0, (sigma*std(tensor))^2)."""
    out = []
    for w in weights:
        scale = float(np.std(w)) or 1.0
        mask = rng.random(w.shape) < rate
        noise = rng.normal(0.0, sigma * scale, w.shape).astype(w.dtype)
        out.append(w + mask * noise)
    return out
