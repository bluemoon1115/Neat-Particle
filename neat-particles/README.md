## NEAT Particles (interactive particle-system evolution)

This folder contains a lightweight, **interactive evolutionary computation (IEC)** framework inspired by the paper in this directory: `neat-particle paper.pdf`.

It is designed for learning: you can browse 9 evolving particle “species” (genomes) at a time, manually select which ones breed the next generation, then switch into sequential plane search (SPS) to fine-tune how inputs are fed into a chosen particle ANN.

### What you can do

- Display **nine candidates** at once (3×3 grid).
- Display the **genome/ANN diagram** next to each candidate.
- Click (or press `1..9`) to **select genomes** to preserve.
- Press `N` to generate a new batch of 9 offspring from the selected parents.
- Press `R` to reset to a fresh random batch; press `W` to randomize weights of the current batch.
- Press `Space` to pause/unpause time (“freeze”).
- Press `Tab` or `B` to enter SPS mode and tune input transforms for the selected genome.

### Particle system implemented

Each grid cell is one NEAT genome controlling one particle-system instance.
All particles within a system share the same genome, but the ANN is activated **once per particle per frame**.

- **Generic** (`generic`): explosion/fire/smoke-like point particles.
  - Inputs: `Px, Py, Pz, Dc, Bias`
  - Outputs: `Vx, Vy, Vz, R, G, B`

Motion uses the linear model you specified:
`P_t = P_{t-1} + S * V * T`

### Requirements

This example uses **pygame** for visualization (optional dependency, only for examples).

From the repository root:
```bash
pip install -e .
pip install pygame
```

### Run

```bash
cd neat-particles
python interactive_neat_particles.py
```

### Controls

#### IEC mode

- Mouse click or keys `1..9`: toggle selection for a candidate.
- `N`: new generation (offspring from selected parents; if none selected, does nothing).
- `B`: bind the selected candidate to SPS mode; if none is selected, candidate 1 is used.
- `Tab`: switch to SPS mode (binds candidate 1 if no genome has been bound yet).
- `R`: reset (new random batch of 9).
- `W`: randomize weights (keeps topology).
- `Space`: pause/unpause (freeze time).
- `Esc`: quit.

#### SPS mode

- Mouse click or keys `1..9`: choose the preferred SPS sample and advance to the next search plane.
- `Tab`: return to IEC mode.
- `B`: rebind SPS to the currently selected IEC genome; if none is selected, candidate 1 is used.
- `R`: reset the SPS search around the bound genome.
- `Space`: pause/unpause (freeze time).
- `Esc`: quit.

### Sequential plane search mode

Sequential plane search is a second interactive optimizer layered on top of the genome gallery. IEC changes the genome by breeding and mutation; SPS keeps one chosen genome fixed and searches around an input-transform vector with 9 visible samples on each plane.

SPS tunes 9 normalized input-transform parameters:

1. Scale for `Px`, `Py`, `Pz`, and `Dc`.
2. Offset for `Px`, `Py`, `Pz`, and `Dc`.
3. Replacement value for the final bias input.

At runtime, the raw Generic ANN input:

```text
(Px, Py, Pz, Dc, 1.0)
```

is transformed into:

```text
(Px * sx + ox,
 Py * sy + oy,
 Pz * sz + oz,
 Dc * sd + od,
 bias_value)
```

The decoded ranges are:

- `scale in [0.1, 3.0]`
- `offset in [-1.0, 1.0]`
- `bias_value in [-1.0, 1.0]`

The genome key, topology, weights, and node biases do not change during SPS. The visual differences come from preprocessing the ANN inputs differently before `net.activate(...)`.

The intended workflow is:

1. Use IEC mode to evolve or find an interesting particle genome.
2. Select that candidate and press `B` to bind it into SPS.
3. In SPS mode, the 3x3 gallery shows the same bound genome key with 9 input-transform samples from the current search plane.
4. Click the preferred cell; the system records that choice as better than representative points on the plane and generates the next plane around the chosen sample.

This implementation follows the sequential-plane idea without adding Bayesian optimization dependencies. It uses a lightweight preference-guided acquisition approximation over the recorded choices and rejected samples.

### Notes / next steps

This is intentionally a **framework**:
- Rendering is a 2D preview (not a full 3D billboarded renderer).
- The network drawing is a compact genome visualization (inputs at top, outputs at bottom).

---

## Learning notes: NEAT in `neat-python` (and how this example uses it)

This is written as a learning note about the NEAT algorithm as implemented in this repository.


### 1) What NEAT is doing (conceptually)

NEAT evolves neural networks while simultaneously:
1. **Optimizing weights** (tuning parameters),
2. **Complexifying topology** over time (adding nodes/connections).

Two mechanisms make topology evolution workable:
- **Speciation**: new structural innovations are protected by grouping similar genomes so they don’t compete directly against very different, mature topologies.
- **Historical markings (innovation numbers)**: every structural mutation is tagged, allowing meaningful crossover alignment between different topologies.

In this repo a NEAT “network” is represented as:
- A **genome** (genotype): node genes + connection genes (`neat\genome.py`, `neat\genes.py`)
- A **phenotype network** built from the genome: e.g. `FeedForwardNetwork` (`neat\nn\feed_forward.py`)

### 2) The standard NEAT engine flow (`neat.Population`)

If you use the core engine (fitness-driven, non-interactive), the high-level loop is implemented in:
- `C:\Users\jonat\neat-python\neat\population.py` → `Population.run(...)`

The loop is (directly from the code):
1. Evaluate fitness of all genomes (your fitness function assigns `genome.fitness`).
2. Check termination criterion (fitness threshold / generation limit).
3. Generate next generation (reproduction).
4. Partition the new generation into species (speciation).
5. Repeat.

Core objects involved:
- Config parsing: `C:\Users\jonat\neat-python\neat\config.py` → `Config(...)`
- Reproduction: `C:\Users\jonat\neat-python\neat\reproduction.py` → `DefaultReproduction.reproduce(...)`
- Speciation: `C:\Users\jonat\neat-python\neat\species.py` → `DefaultSpeciesSet.speciate(...)`
- Stagnation: `C:\Users\jonat\neat-python\neat\stagnation.py` → `DefaultStagnation.update(...)`
- Genome logic: `C:\Users\jonat\neat-python\neat\genome.py` → `DefaultGenome` methods (mutation/distance/crossover)

### 3) What this example uses from the NEAT core engine

This example is **interactive evolutionary computation (IEC)**: you are the “fitness function” by selecting parents visually.

So it intentionally does **not** call:
- `Population.run(...)`
- `DefaultReproduction.reproduce(...)`
- `DefaultSpeciesSet.speciate(...)`
- `DefaultStagnation.update(...)`

Instead it calls these core pieces directly:

**Config (to define genome shape + mutation rules)**
- `neat.Config(...)` in `C:\Users\jonat\neat-python\neat-particles\interactive_neat_particles.py`

**Genome creation and mutation**
- `DefaultGenome.configure_new(...)` (random initialization)
- `DefaultGenome.mutate(...)` (structural + parameter mutation)

**Phenotype network build**
- `neat.nn.FeedForwardNetwork.create(genome, config)`

**Innovation bookkeeping**
- `neat.innovation.InnovationTracker` (the example uses one tracker across the session and calls `reset_generation()` per new generation)

So you still get NEAT-style “complexify over time” mutations, but **speciation/fitness sharing are not running** in the IEC path unless you refactor to use `Population`.

#### Switching this project to the standard engine (automatic fitness)

If you want “full NEAT” (speciation + crossover + fitness sharing), the intended pattern is:

```python
import neat

config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    "path/to/config.ini",
)

pop = neat.Population(config)

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        # run particle simulation and assign a scalar fitness:
        genome.fitness = 0.0

winner = pop.run(eval_genomes, n=50)
```

In an IEC setting, you can still use `Population` by defining “fitness” as a score derived from your selections (e.g., selected candidates get higher fitness, or you let the user rank 1–9 and map rank → fitness).

### 4) How the neural network shape is defined (inputs/outputs)

In this repo, input/output counts are set in the config file:
- `[DefaultGenome] num_inputs`
- `[DefaultGenome] num_outputs`

`DefaultGenomeConfig` then defines fixed node keys:
- Inputs are negative keys: `[-1, -2, ..., -num_inputs]`
- Outputs are non-negative keys: `[0, 1, ..., num_outputs-1]`

See:
- `C:\Users\jonat\neat-python\neat\genome.py` → `DefaultGenomeConfig.__init__` (`input_keys`, `output_keys`)

When you build a phenotype network with:
- `C:\Users\jonat\neat-python\neat\nn\feed_forward.py` → `FeedForwardNetwork.create(...)`

then at runtime:
- `FeedForwardNetwork.activate(inputs)` expects `len(inputs) == num_inputs`
- it assigns those values to input keys using `zip(input_keys, inputs)`
- it returns outputs in the order of `output_keys`.

That means your **semantic mapping** (what each input/output “means”) is entirely up to your application code.

#### How the feed-forward phenotype is constructed (what “settled” means here)

When you call `FeedForwardNetwork.create(genome, config)`:
1. It gathers all **enabled** connections from `genome.connections`.
2. It computes a feed-forward evaluation order with:
   - `C:\Users\jonat\neat-python\neat\graphs.py` → `feed_forward_layers(...)`
3. For each node in each layer, it builds a “node evaluation instruction” tuple:
   - `(node_id, activation_fn, aggregation_fn, bias, response, incoming_links)`
4. `activate(...)` executes those instructions in order:
   - multiply incoming node values by weights
   - aggregate them (sum in our configs)
   - compute `activation(bias + response * aggregated_sum)` (sigmoid in our configs)

All of that logic is visible in:
- `C:\Users\jonat\neat-python\neat\nn\feed_forward.py`

### 5) How this example maps particle systems to ANN inputs/outputs

Mappings are implemented in:
- `C:\Users\jonat\neat-python\neat-particles\particle_systems.py`

The Generic system uses sigmoid activation (locked in the config), so ANN outputs are typically in `[0, 1]`.

This example interprets outputs as:
- velocities: remap `[0, 1] → [-1, 1]`, then scale by a speed constant
- colors: clamp to `[0, 1]` and convert to 0–255

#### Generic

Config:
- `C:\Users\jonat\neat-python\neat-particles\config-generic.ini`
  - `num_inputs = 5`
  - `num_outputs = 6`

Input vector order:
1. `x`
2. `y`
3. `z`
4. distance from center (`Dc`)
5. bias (= `1.0`)

Output vector order:
1. `Vx`
2. `Vy`
3. `Vz`
4. `R`
5. `G`
6. `B`

### 6) Meaning of the configuration options (what the knobs do)

These example configs follow the same conventions as the other examples in this repo.

#### `[NEAT]`

- `fitness_criterion`: how “best fitness” is defined in the standard engine (`max`/`min`/`mean`).
- `fitness_threshold`: termination threshold in the standard engine (ignored when `no_fitness_termination = True`).
- `pop_size`: population size used by `Population`; this IEC demo always displays 9.
- `reset_on_extinction`: whether to restart if all species die out in standard engine.
- `no_fitness_termination`: if `True`, don’t early-stop on `fitness_threshold`.
- `seed`: optional reproducibility seed supported by `Population`.

#### `[DefaultGenome]` (the most important section)

Activation/aggregation:
- `activation_default`, `activation_options`, `activation_mutate_rate`
- `aggregation_default`, `aggregation_options`, `aggregation_mutate_rate`

In this example both are locked to simplify learning:
- activation = sigmoid only
- aggregation = sum only

Topology mutation rates:
- `conn_add_prob`, `conn_delete_prob`
- `node_add_prob`, `node_delete_prob`

Weight/bias mutation rules (handled by `neat\attributes.py`):
- `weight_*` parameters control how weights initialize and mutate
- `bias_*` parameters control how biases initialize and mutate
- `enabled_*` parameters control how connections toggle on/off

Shape:
- `num_inputs`, `num_outputs` define I/O size
- `num_hidden` sets the initial hidden node count (classic NEAT often starts at 0)

Connectivity / cycles:
- `feed_forward = True` prevents adding connections that would create cycles.
- `initial_connection = full_direct` starts with all inputs connected to all outputs.

Mutation scheduling:
- `single_structural_mutation`:
  - `false`: add-node/add-conn/delete-node/delete-conn are attempted independently (each by its own probability)
  - `true`: at most one structural mutation happens per `mutate()` call

#### `[DefaultSpeciesSet]` (speciation)

- `compatibility_threshold` controls how strict species matching is (lower = more species).

#### `[DefaultStagnation]`

- `species_fitness_func`: how species fitness is summarized (`max`/`mean`/etc).
- `max_stagnation`: generations without improvement before species is marked stagnant.
- `species_elitism`: protects top species from removal.

#### `[DefaultReproduction]`

- `elitism`: best genomes copied unchanged into next generation (per species).
- `survival_threshold`: fraction of each species eligible to be parents.
- `min_species_size`: species get at least this many offspring.
- `interspecies_crossover_prob`: chance parent2 comes from a different species.

### 7) Mutation and crossover (how genomes change)

#### Parameter mutation (weights/biases/etc.)

Numeric mutation is handled by `FloatAttribute`:
- with probability `*_mutate_rate`: add Gaussian noise with std `*_mutate_power`
- else with probability `*_replace_rate`: replace with a freshly initialized value
- clamp to `[*_min_value, *_max_value]`

See:
- `C:\Users\jonat\neat-python\neat\attributes.py` → `FloatAttribute.mutate_value(...)`
- `C:\Users\jonat\neat-python\neat\genes.py` → `BaseGene.mutate(...)`

#### Structural mutation (complexification)

In `C:\Users\jonat\neat-python\neat\genome.py` → `DefaultGenome.mutate(...)`, the structural operations are:
- `mutate_add_node(...)`: split one existing connection and insert a new node
- `mutate_add_connection(...)`: add a new connection between two nodes
- `mutate_delete_node(...)`: delete a random hidden node (never deletes outputs)
- `mutate_delete_connection(...)`: delete a random connection

Innovation tracking matters here:
- Connection genes carry an `innovation` integer (`DefaultConnectionGene(..., innovation=...)`).
- When multiple genomes make the “same” structural change in one generation, they should share innovation numbers to keep crossover alignment meaningful.

#### Parent pairing (is it random?)

Standard engine:
- In `DefaultReproduction.reproduce(...)`, parents are chosen randomly from the top `survival_threshold` fraction within each species.

This IEC example:
- You choose parents by selecting candidates.
- Offspring are currently produced by cloning a selected parent and calling `mutate(...)` (no crossover yet).

If you want crossover in IEC, the next step is to use:
- `DefaultGenome.configure_crossover(parent1, parent2, ...)` then `mutate(...)`

### 8) How particles are simulated and rendered in this project

Simulation/rendering code:
- `C:\Users\jonat\neat-python\neat-particles\particle_systems.py`

The common loop is:
1. For each particle, build an input vector.
2. Call `net.activate(inputs)` to get outputs.
3. Interpret outputs as velocity and color.
4. Update position using: `P_t = P_{t-1} + S * V * T`.
5. Draw a 2D preview using pygame primitives.

System behavior summary:
- `GenericSystem`: independent particles, TTL respawn, and bounds-based respawn.

### 9) What the ANN diagram is showing

Genome drawing code:
- `C:\Users\jonat\neat-python\neat-particles\draw_genome.py`

The labels next to circles are **node keys**:
- inputs: negative IDs
- outputs: `0..num_outputs-1`
- hidden nodes: positive IDs assigned as nodes are added

This is different from:
- the **genome key** (unique ID for the whole genome; shown as `key=...` in the UI)
- the **connection innovation number** (`cg.innovation`), which is not currently displayed in the UI.
