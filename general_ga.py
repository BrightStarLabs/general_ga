"""
general_ga.py

Generic, simple genetic algorithm engine for fixed-shape numpy genomes.

- First axis of every variable is population axis (A).
- Per-variable:
    * dtype: float32 | int32 | int8
    * shape: (A, ...)
    * init_mode:
        - float32: "gaussian" | "uniform" | "constant"
        - int32/int8: "int_uniform" | "constant"
    * mut_type: "additive" | "resample" | "none"
        - int32/int8: ONLY "resample" or "none" (NO additive)
    * mut_prob, sigma_base, crossover_prob: per-variable
    * trainable: if False -> no crossover, no mutation

- Global:
    * eps_adapt: epsilon for adaptive mutation scale (float additive only)
    * t_df: degrees of freedom for student-t mutation noise
    * noise_dist: "student_t" | "gaussian" | "uniform"

Bounds semantics (kept uniform/same as before):
- int32/int8 ranges use [min_val, max_val) (max_val is exclusive)
- float32 init uniform uses [min_val, max_val); clipping uses np.clip (inclusive)

Optional resample interval:
- resample_min/resample_max can narrow the range used by *resample mutation* only.
- Must satisfy: min_val <= resample_min < resample_max <= max_val
- For int32/int8, resample_min/resample_max must be integer-valued.

IMPORTANT POLICY:
- No silent fallbacks for invalid dtype/operation combos.
- If an invalid configuration slips through, we raise loudly (TypeError/ValueError).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np


# ----------------------------------------------------------------------
# Config dataclasses
# ----------------------------------------------------------------------


@dataclass
class VarSpec:
    """
    Specification for a single parameter group in the genome.

    name:          Unique variable name (used as key in genome dict).
    dtype:         "float32" | "int32" | "int8".
    shape:         Full shape including population axis as first dim (A, ...).
    trainable:     If False, GA will never crossover or mutate this variable.

    init_mode:
        - float32: "gaussian" | "uniform" | "constant"
        - int32/int8: "int_uniform" | "constant"

    min_val:       Lower bound (init, clamp, int range).
    max_val:       Upper bound (init, clamp, int range). For ints: exclusive.
    const_val:     Value for "constant" init_mode.

    resample_min/resample_max:
        Optional bounds used ONLY by mut_type="resample".

    mut_type:      "additive" | "resample" | "none".
                   - floats can use additive/resample/none.
                   - int32/int8 can use resample/none only.

    mut_prob:      Per-element mutation probability.
    sigma_base:    Base scale for adaptive mutation (float additive only).
    crossover_prob:Per-element crossover probability; True → copy from parent A.
    """
    name: str
    dtype: str
    shape: Tuple[int, ...]

    trainable: bool = True

    init_mode: str = "gaussian"
    min_val: float = -1.0
    max_val: float = 1.0
    const_val: float = 0.0

    resample_min: Optional[float] = None
    resample_max: Optional[float] = None

    mut_type: str = "additive"  # "additive" | "resample" | "none"
    mut_prob: float = 0.0
    sigma_base: float = 0.0
    crossover_prob: float = 0.0


@dataclass
class GAConfig:
    """
    Global GA configuration (all non-per-variable knobs live here).
    """
    population_size: int
    elitism: int
    bank_max: int = 0
    bank_add_every: int = 1
    immigrants_per_gen: int = 0
    tournament_k: int = 2

    eps_adapt: float = 1e-3
    t_df: float = 2.0
    noise_dist: str = "student_t"


# ----------------------------------------------------------------------
# GA engine
# ----------------------------------------------------------------------


class GA:
    """
    Generic GA engine.
    """

    def __init__(self, config: GAConfig, var_specs: List[VarSpec]):
        self.config = config
        self.var_specs: List[VarSpec] = [self._normalize_spec(v) for v in var_specs]
        self._validate_specs()

        # Population buffers: dict[name] -> np.ndarray with shape spec.shape
        self.pop_cur: Dict[str, np.ndarray] | None = None
        self.pop_next: Dict[str, np.ndarray] | None = None

        # Bank of good individuals: list of dict[name] -> np.ndarray (per-individual)
        self.bank: List[Dict[str, np.ndarray]] = []

        self.generation: int = 0
        self.rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask(self, init: bool = False) -> Dict[str, np.ndarray]:
        if self.pop_cur is None or init:
            self._init_population()
            self.generation = 0
            self.bank.clear()
        return self.pop_cur

    def tell(self, fitness: np.ndarray) -> None:
        if self.pop_cur is None:
            raise RuntimeError("tell() called before ask()/initialization.")

        fitness = np.asarray(fitness, dtype=np.float32)
        A = self.config.population_size
        if fitness.shape != (A,):
            raise ValueError(f"fitness must have shape ({A},), got {fitness.shape}")

        order = np.argsort(-fitness)
        self._update_bank(fitness, order)
        self._build_next_population(fitness, order)

        self.pop_cur, self.pop_next = self.pop_next, self.pop_cur
        self.generation += 1

    # ------------------------------------------------------------------
    # Internal: spec handling / validation
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_spec(spec: VarSpec) -> VarSpec:
        dt = str(spec.dtype).lower().strip()
        if dt in ("float32", "float"):
            norm_dtype = "float32"
        elif dt in ("int32", "int"):
            norm_dtype = "int32"
        elif dt in ("int8", "i8"):
            norm_dtype = "int8"
        else:
            raise ValueError(f"Unsupported dtype in VarSpec {spec.name}: {spec.dtype}")

        return VarSpec(
            name=spec.name,
            dtype=norm_dtype,
            shape=tuple(spec.shape),

            trainable=spec.trainable,

            init_mode=spec.init_mode,
            min_val=spec.min_val,
            max_val=spec.max_val,
            const_val=spec.const_val,

            resample_min=spec.resample_min,
            resample_max=spec.resample_max,

            mut_type=spec.mut_type,
            mut_prob=spec.mut_prob,
            sigma_base=spec.sigma_base,
            crossover_prob=spec.crossover_prob,
        )

    @staticmethod
    def _is_int_valued(x: float) -> bool:
        return float(x).is_integer()

    def _validate_specs(self) -> None:
        cfg = self.config
        A = cfg.population_size

        if A <= 0:
            raise ValueError("population_size must be > 0")
        if cfg.elitism < 0 or cfg.elitism >= A:
            raise ValueError("elitism must satisfy 0 <= elitism < population_size")
        if cfg.tournament_k < 1:
            raise ValueError("tournament_k must be >= 1")
        if cfg.noise_dist not in ("student_t", "gaussian", "uniform"):
            raise ValueError(f"Unsupported noise_dist: {cfg.noise_dist}")

        names = [v.name for v in self.var_specs]
        if len(set(names)) != len(names):
            raise ValueError("VarSpec names must be unique")

        for v in self.var_specs:
            if len(v.shape) == 0:
                raise ValueError(f"VarSpec {v.name}: shape must include population axis")
            if v.shape[0] != A:
                raise ValueError(
                    f"VarSpec {v.name}: first dim (pop axis) must be {A}, got {v.shape[0]}"
                )

            # dtype/init_mode compatibility
            if v.dtype == "float32":
                if v.init_mode not in ("gaussian", "uniform", "constant"):
                    raise ValueError(
                        f"VarSpec {v.name}: float32 requires init_mode in "
                        f"('gaussian','uniform','constant')"
                    )
            elif v.dtype in ("int32", "int8"):
                if v.init_mode not in ("int_uniform", "constant"):
                    raise ValueError(
                        f"VarSpec {v.name}: {v.dtype} requires init_mode in "
                        f"('int_uniform','constant')"
                    )
            else:
                raise ValueError(f"VarSpec {v.name}: unsupported dtype {v.dtype}")

            # Bounds sanity
            if v.min_val >= v.max_val:
                raise ValueError(
                    f"VarSpec {v.name}: min_val must be < max_val "
                    f"(got {v.min_val} >= {v.max_val})"
                )

            # Integer-valued bounds (loud fail)
            if v.dtype in ("int32", "int8"):
                if not (self._is_int_valued(v.min_val) and self._is_int_valued(v.max_val)):
                    raise ValueError(
                        f"VarSpec {v.name}: integer dtypes require integer-valued min_val/max_val; "
                        f"got {v.min_val}, {v.max_val}"
                    )
                if v.init_mode == "constant" and not self._is_int_valued(v.const_val):
                    raise ValueError(
                        f"VarSpec {v.name}: integer dtypes require integer-valued const_val; got {v.const_val}"
                    )

            # int8 representable range checks; max_val is exclusive, allow 128
            if v.dtype == "int8":
                if v.min_val < -128 or v.min_val > 127:
                    raise ValueError(
                        f"VarSpec {v.name}: int8 min_val must be in [-128, 127], got {v.min_val}"
                    )
                if v.max_val < -127 or v.max_val > 128:
                    raise ValueError(
                        f"VarSpec {v.name}: int8 max_val must be in [-127, 128] (exclusive high), got {v.max_val}"
                    )

            # Mutation type sanity
            if v.mut_type not in ("additive", "resample", "none"):
                raise ValueError(
                    f"VarSpec {v.name}: mut_type must be 'additive', 'resample', or 'none'"
                )

            # ints: only resample/none (explicit, loud fail)
            if v.dtype in ("int32", "int8") and v.mut_type == "additive":
                raise ValueError(
                    f"VarSpec {v.name}: {v.dtype} cannot use mut_type='additive'; use 'resample' or 'none'."
                )

            # resample bounds validation
            if (v.resample_min is None) ^ (v.resample_max is None):
                raise ValueError(
                    f"VarSpec {v.name}: resample_min and resample_max must be both set or both None."
                )
            if v.resample_min is not None:
                rmin = float(v.resample_min)
                rmax = float(v.resample_max)
                if not (v.min_val <= rmin < rmax <= v.max_val):
                    raise ValueError(
                        f"VarSpec {v.name}: resample bounds must satisfy "
                        f"min_val <= resample_min < resample_max <= max_val; "
                        f"got min_val={v.min_val}, resample=[{rmin}, {rmax}), max_val={v.max_val}"
                    )
                if v.dtype in ("int32", "int8"):
                    if not (self._is_int_valued(rmin) and self._is_int_valued(rmax)):
                        raise ValueError(
                            f"VarSpec {v.name}: integer dtypes require integer-valued resample_min/resample_max; "
                            f"got {rmin}, {rmax}"
                        )

            # Non-trainable vars must not have mut_prob or crossover_prob
            if not v.trainable:
                if v.mut_prob != 0.0 or v.crossover_prob != 0.0:
                    v.mut_prob = 0.0
                    v.crossover_prob = 0.0

    # ------------------------------------------------------------------
    # Internal: initialization
    # ------------------------------------------------------------------

    def _init_population(self) -> None:
        self.pop_cur = {}
        for spec in self.var_specs:
            shape = spec.shape
            if spec.dtype == "float32":
                arr = self._init_float_array(spec, shape)
            elif spec.dtype in ("int32", "int8"):
                arr = self._init_int_array(spec, shape)
            else:
                raise TypeError(f"Unsupported dtype for init: {spec.dtype} (VarSpec {spec.name})")
            self.pop_cur[spec.name] = arr

        self.pop_next = {name: np.empty_like(arr) for name, arr in self.pop_cur.items()}

    def _init_float_array(self, spec: VarSpec, shape: Tuple[int, ...]) -> np.ndarray:
        if spec.dtype != "float32":
            raise TypeError(f"_init_float_array called for non-float32 VarSpec {spec.name} ({spec.dtype})")

        if spec.init_mode == "gaussian":
            std = (spec.max_val - spec.min_val) / 4.0
            if std <= 0.0:
                std = 1.0
            arr = self.rng.normal(loc=0.0, scale=std, size=shape).astype(np.float32)
            return np.clip(arr, spec.min_val, spec.max_val)

        if spec.init_mode == "uniform":
            return self.rng.uniform(low=spec.min_val, high=spec.max_val, size=shape).astype(np.float32)

        if spec.init_mode == "constant":
            arr = np.full(shape, spec.const_val, dtype=np.float32)
            return np.clip(arr, spec.min_val, spec.max_val)

        raise ValueError(f"Float VarSpec {spec.name}: unsupported init_mode {spec.init_mode}")

    def _init_int_array(self, spec: VarSpec, shape: Tuple[int, ...]) -> np.ndarray:
        if spec.dtype not in ("int32", "int8"):
            raise TypeError(f"_init_int_array called for non-int VarSpec {spec.name} ({spec.dtype})")

        low = int(spec.min_val)
        high = int(spec.max_val)  # exclusive

        if spec.init_mode == "int_uniform":
            if high <= low:
                raise ValueError(f"Int VarSpec {spec.name}: int_uniform requires max_val > min_val")

            if spec.dtype == "int8":
                # generate in wider dtype then cast down
                return self.rng.integers(low=low, high=high, size=shape, dtype=np.int16).astype(np.int8)

            return self.rng.integers(low=low, high=high, size=shape, dtype=np.int32)

        if spec.init_mode == "constant":
            val = int(spec.const_val)

            if spec.dtype == "int8":
                tmp = np.full(shape, val, dtype=np.int16)
                tmp = np.maximum(tmp, low)
                tmp = np.minimum(tmp, high - 1)
                return tmp.astype(np.int8)

            tmp = np.full(shape, val, dtype=np.int32)
            tmp = np.maximum(tmp, low)
            tmp = np.minimum(tmp, high - 1)
            return tmp

        raise ValueError(f"Int VarSpec {spec.name}: unsupported init_mode {spec.init_mode}")

    # ------------------------------------------------------------------
    # Internal: bank handling
    # ------------------------------------------------------------------

    def _update_bank(self, fitness: np.ndarray, order: np.ndarray) -> None:
        cfg = self.config
        if cfg.bank_max <= 0:
            return
        if cfg.bank_add_every <= 0:
            return
        if (self.generation % cfg.bank_add_every) != 0:
            return

        best_idx = int(order[0])
        indiv = {spec.name: self.pop_cur[spec.name][best_idx].copy() for spec in self.var_specs}
        indiv["fitness"] = float(fitness[best_idx])
        self.bank.append(indiv)
        if len(self.bank) > cfg.bank_max:
            self.bank = self.bank[-cfg.bank_max :]

    def _sample_immigrants(self, n: int) -> List[Dict[str, np.ndarray]]:
        n = min(n, len(self.bank))
        if n <= 0:
            return []
        idxs = self.rng.choice(len(self.bank), size=n, replace=False)
        return [self.bank[i] for i in idxs]

    # ------------------------------------------------------------------
    # Internal: build next population
    # ------------------------------------------------------------------

    def _build_next_population(self, fitness: np.ndarray, order: np.ndarray) -> None:
        cfg = self.config
        A = cfg.population_size

        if self.pop_next is None or self.pop_cur is None:
            raise RuntimeError("Population buffers not initialized.")

        elite_n = cfg.elitism
        max_imm_allowed = max(0, A - elite_n)
        imm_n = min(cfg.immigrants_per_gen, max_imm_allowed, len(self.bank))
        child_n = A - elite_n - imm_n
        assert child_n >= 0

        imm_start, imm_end = 0, imm_n
        elite_start, elite_end = imm_end, imm_end + elite_n
        child_start, child_end = elite_end, A
        assert child_end == A

        for spec in self.var_specs:
            arr = self.pop_next[spec.name]
            if arr.shape != spec.shape:
                raise RuntimeError(
                    f"pop_next[{spec.name}] has wrong shape {arr.shape}, expected {spec.shape}"
                )

        immigrants = self._sample_immigrants(imm_n)
        if imm_n > 0:
            for spec in self.var_specs:
                name = spec.name
                stacked = np.stack([indiv[name] for indiv in immigrants], axis=0)
                self.pop_next[name][imm_start:imm_end] = stacked

        elite_indices = order[:elite_n]
        if elite_n > 0:
            for spec in self.var_specs:
                name = spec.name
                self.pop_next[name][elite_start:elite_end] = self.pop_cur[name][elite_indices]

        child_idx = 0
        while child_idx < child_n:
            pa = self._tournament_select(fitness)
            pb = self._tournament_select(fitness)
            dest = child_start + child_idx

            for spec in self.var_specs:
                name = spec.name
                parent_a = self.pop_cur[name][pa]
                parent_b = self.pop_cur[name][pb]

                if not spec.trainable:
                    self.pop_next[name][dest] = parent_a
                    continue

                child = self._crossover(parent_a, parent_b, spec)
                child = self._mutate(child, spec)
                child = self._clamp(child, spec)

                self.pop_next[name][dest] = child

            child_idx += 1

    # ------------------------------------------------------------------
    # Internal: selection, crossover, mutation
    # ------------------------------------------------------------------

    def _tournament_select(self, fitness: np.ndarray) -> int:
        A = fitness.shape[0]
        k = self.config.tournament_k
        idxs = self.rng.integers(0, A, size=k)
        best = idxs[np.argmax(fitness[idxs])]
        return int(best)

    def _crossover(self, parent_a: np.ndarray, parent_b: np.ndarray, spec: VarSpec) -> np.ndarray:
        if spec.crossover_prob <= 0.0:
            return parent_a.copy()

        mask = self.rng.random(parent_a.shape) < spec.crossover_prob
        child = np.where(mask, parent_a, parent_b)
        return child.astype(parent_a.dtype, copy=False)

    def _get_resample_bounds(self, spec: VarSpec) -> Tuple[float, float]:
        if spec.resample_min is None:
            return float(spec.min_val), float(spec.max_val)
        return float(spec.resample_min), float(spec.resample_max)

    def _mutate(self, child: np.ndarray, spec: VarSpec) -> np.ndarray:
        if spec.mut_type == "none" or spec.mut_prob <= 0.0 or not spec.trainable:
            return child

        mut_mask = self.rng.random(child.shape) < spec.mut_prob
        if not mut_mask.any():
            return child

        if spec.mut_type == "additive":
            # Loud fail if misconfigured (no silent fallback)
            if spec.dtype != "float32":
                raise TypeError(
                    f"VarSpec {spec.name}: mut_type='additive' requires float32, got {spec.dtype}"
                )
            if spec.sigma_base < 0.0:
                raise ValueError(f"VarSpec {spec.name}: sigma_base must be >= 0, got {spec.sigma_base}")

            if spec.sigma_base == 0.0:
                return child  # valid: mutation becomes no-op

            noise = self._draw_noise(int(mut_mask.sum()))
            scale = spec.sigma_base * (np.abs(child[mut_mask]) + self.config.eps_adapt)
            child[mut_mask] = child[mut_mask] + scale * noise
            return child

        if spec.mut_type == "resample":
            lo, hi = self._get_resample_bounds(spec)
            idx_count = int(mut_mask.sum())
            if idx_count == 0:
                return child

            if spec.dtype == "float32":
                new_vals = self._sample_float_values(spec, idx_count, lo, hi)
            elif spec.dtype in ("int32", "int8"):
                new_vals = self._sample_int_values(spec, idx_count, lo, hi)
            else:
                raise TypeError(f"VarSpec {spec.name}: resample for unsupported dtype {spec.dtype}")

            child[mut_mask] = new_vals
            return child

        raise ValueError(f"VarSpec {spec.name}: unsupported mut_type {spec.mut_type}")

    def _draw_noise(self, size: int) -> np.ndarray:
        if size < 0:
            raise ValueError(f"noise size must be >= 0, got {size}")

        dist = self.config.noise_dist
        if dist == "student_t":
            x = self.rng.standard_t(df=self.config.t_df, size=size)
        elif dist == "gaussian":
            x = self.rng.normal(loc=0.0, scale=1.0, size=size)
        elif dist == "uniform":
            x = self.rng.uniform(low=-1.0, high=1.0, size=size)
        else:
            raise ValueError(f"Unsupported noise_dist: {dist}")
        return x.astype(np.float32)

    # ------------------------------------------------------------------
    # Internal: resample helpers
    # ------------------------------------------------------------------

    def _sample_float_values(self, spec: VarSpec, size: int, lo: float, hi: float) -> np.ndarray:
        if spec.dtype != "float32":
            raise TypeError(f"_sample_float_values called for non-float32 VarSpec {spec.name} ({spec.dtype})")

        if spec.init_mode == "gaussian":
            std = (hi - lo) / 4.0
            if std <= 0.0:
                std = 1.0
            vals = self.rng.normal(loc=0.0, scale=std, size=size).astype(np.float32)
            return np.clip(vals, lo, hi)

        if spec.init_mode == "uniform":
            return self.rng.uniform(low=lo, high=hi, size=size).astype(np.float32)

        if spec.init_mode == "constant":
            vals = np.full(size, spec.const_val, dtype=np.float32)
            return np.clip(vals, lo, hi)

        raise ValueError(f"Float VarSpec {spec.name}: unsupported init_mode {spec.init_mode}")

    def _sample_int_values(self, spec: VarSpec, size: int, lo: float, hi: float) -> np.ndarray:
        if spec.dtype not in ("int32", "int8"):
            raise TypeError(f"_sample_int_values called for non-int VarSpec {spec.name} ({spec.dtype})")

        if spec.init_mode not in ("int_uniform", "constant"):
            raise ValueError(f"Int VarSpec {spec.name}: unsupported init_mode {spec.init_mode}")

        low = int(lo)
        high = int(hi)  # exclusive
        if high <= low:
            raise ValueError(f"Int VarSpec {spec.name}: requires resample_max > resample_min")

        if spec.init_mode == "int_uniform":
            if spec.dtype == "int8":
                return self.rng.integers(low=low, high=high, size=size, dtype=np.int16).astype(np.int8)
            return self.rng.integers(low=low, high=high, size=size, dtype=np.int32)

        # constant
        val = int(spec.const_val)
        if spec.dtype == "int8":
            tmp = np.full(size, val, dtype=np.int16)
            tmp = np.maximum(tmp, low)
            tmp = np.minimum(tmp, high - 1)
            return tmp.astype(np.int8)

        tmp = np.full(size, val, dtype=np.int32)
        tmp = np.maximum(tmp, low)
        tmp = np.minimum(tmp, high - 1)
        return tmp.astype(np.int32, copy=False)

    # ------------------------------------------------------------------
    # Internal: clamping
    # ------------------------------------------------------------------

    @staticmethod
    def _clamp(child: np.ndarray, spec: VarSpec) -> np.ndarray:
        if spec.dtype == "float32":
            return np.clip(child, spec.min_val, spec.max_val)

        if spec.dtype not in ("int32", "int8"):
            raise TypeError(f"_clamp called for unsupported dtype {spec.dtype} (VarSpec {spec.name})")

        low = int(spec.min_val)
        high = int(spec.max_val)  # exclusive

        if spec.dtype == "int8":
            tmp = child.astype(np.int16, copy=False)
            tmp = np.maximum(tmp, low)
            tmp = np.minimum(tmp, high - 1)
            return tmp.astype(np.int8, copy=False)

        tmp = child.astype(np.int32, copy=False)
        tmp = np.maximum(tmp, low)
        tmp = np.minimum(tmp, high - 1)
        return tmp.astype(np.int32, copy=False)


# Optional lowercase alias if you prefer `ga(...)` over `GA(...)`
ga = GA
