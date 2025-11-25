# general_ga
General Genetic Algorithm for Emergent Models

````md
# `general_ga.py` — Generic Genetic Algorithm for Fixed-Shape NumPy Genomes

A small, strict genetic algorithm (GA) engine that evolves **fixed-shape NumPy arrays** (“genomes”).  
It **does not** run any environment/simulation — it only manages populations and evolves them from fitness.

## Key rules

- **Population axis is always axis 0** for every variable: shape `(A, ...)`.
- Per-variable `dtype`: `float32`, `int32`, or `int8`.
- Per-variable init / crossover / mutation / clamp.
- **Loud failures** for invalid dtype/operation combinations (no silent fallbacks).
- Optional “bank” of best individuals + immigrants.

## Requirements

- Python 3.10+
- `numpy`

## Quick start

```python
from general_ga import GA, GAConfig, VarSpec

ga = GA(config, var_specs)

pop = ga.ask(init=True)     # generation 0 (random init)
# ... evaluate pop -> fitness shape (A,)
ga.tell(fitness)

pop = ga.ask()              # next generation
# ... evaluate ...
ga.tell(fitness)
````

## Genome format

`ask()` returns:

```python
pop: dict[str, np.ndarray]
```

Each array is shaped `(A, ...)`, where `A = GAConfig.population_size`.

## Configuration

### `GAConfig` (global)

* `population_size`: population size `A`
* `elitism`: number of top individuals copied unchanged each generation
* `tournament_k`: tournament size for parent selection
* `bank_max`: max bank size (0 disables the bank)
* `bank_add_every`: add best-of-generation to bank every N generations
* `immigrants_per_gen`: number of bank individuals inserted per generation

Noise settings (used only for **float32 additive mutation**):

* `noise_dist`: `"student_t"` | `"gaussian"` | `"uniform"`
* `t_df`: Student-t degrees of freedom
* `eps_adapt`: adaptive scaling uses `|x| + eps_adapt`

### `VarSpec` (per-variable)

Core:

* `name`: unique variable key
* `dtype`: `"float32" | "int32" | "int8"`
* `shape`: must start with population axis `(A, ...)`
* `trainable`: if `False` → no crossover/mutation; copied from parent A

Initialization:

* `init_mode`:

  * `float32`: `"gaussian"` | `"uniform"` | `"constant"`
  * `int32/int8`: `"int_uniform"` | `"constant"`
* `min_val`, `max_val`:

  * `float32`: init + clamp bounds
  * `int32/int8`: **exclusive high bound**; value range is `[min_val, max_val)`
* `const_val`: used when `init_mode="constant"`

Mutation:

* `mut_type`:

  * `float32`: `"additive"` | `"resample"` | `"none"`
  * `int32/int8`: `"resample"` | `"none"` (**additive is invalid**)
* `mut_prob`: per-element mutation probability
* `sigma_base`: float additive scale (allowed to be `0.0`)
* `crossover_prob`: per-element crossover probability (if `0.0`, child copies parent A)

Resample-only bounds (optional):

* `resample_min`, `resample_max`: used only when `mut_type="resample"`.

  * ints: new values sampled from `[resample_min, resample_max)`
  * floats: new values sampled/clipped within `[resample_min, resample_max]` (implementation-defined)
  * must satisfy `min_val <= resample_min < resample_max <= max_val`

## Example

Float weights + int8 discrete rules:

```python
A = 256

var_specs = [
    VarSpec(
        name="W",
        dtype="float32",
        shape=(A, 32, 32),
        init_mode="uniform",
        min_val=-1.0, max_val=1.0,
        mut_type="additive",
        mut_prob=0.2,
        sigma_base=0.01,
        crossover_prob=0.02,
    ),
    VarSpec(
        name="RULES",
        dtype="int8",
        shape=(A, 64),
        init_mode="int_uniform",
        min_val=-128, max_val=128,          # exclusive max => values in [-128..127]
        mut_type="resample",
        mut_prob=0.05,
        crossover_prob=0.02,
        resample_min=-10, resample_max=11,  # values in [-10..10]
    ),
]

config = GAConfig(
    population_size=A,
    elitism=16,
    tournament_k=3,
    bank_max=128,
    immigrants_per_gen=2,
    noise_dist="student_t",
    t_df=2.0,
)

ga = GA(config, var_specs)
```

## How each generation is built

The next population is filled in this order:

1. **Immigrants** (from bank, if enabled)
2. **Elites** (best `elitism` copied unchanged)
3. **Children** (tournament select parents → crossover → mutation → clamp)

## Strict error behavior

This GA intentionally **fails loudly** on misconfiguration, e.g.:

* integer dtype with `mut_type="additive"`
* invalid init modes for a dtype
* invalid bounds (`min_val >= max_val`, bad resample interval, non-integer bounds for integer dtypes, etc.)
* unsupported `noise_dist`

## FAQ (short)

**Why use `int(mut_mask.sum())`?**
It counts how many elements are selected for mutation, so resampling generates exactly that many new values.

**Does boolean masking scramble indices?**
No. `child[mask] = values` writes values back in the same deterministic order that NumPy uses to read `child[mask]`.

**What’s the integer range rule?**
For ints, `max_val` is **exclusive**: values are always in `[min_val, max_val)`.

```
```
