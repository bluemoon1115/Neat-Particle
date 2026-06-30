## NEAT Particles (interactive particle-system evolution)

This folder contains a lightweight, **interactive evolutionary computation (IEC)** framework inspired by the paper: `neat-particle`.

It is designed for learning: you can browse 9 evolving particle “species” (genomes) at a time, manually select which ones breed the next generation, then use Sequential Plane Search (SPS) to fine-tune selected Generic genome connection weights.

### What you can do

- Display **nine candidates** at once (3×3 grid).
- Display the **genome/ANN diagram** next to each candidate.
- Click (or press `1..9`) to **select genomes** to preserve.
- Press `N` to generate a new batch of 9 offspring from the selected parents.
- Press `R` to reset to a fresh random batch; press `W` to randomize weights of the current batch.
- Press `Space` to pause/unpause time (“freeze”).
- Press `Tab` or `B` to enter **SPS weight tuning** for the selected genome.
- Press `E` to export a genome target file for automated SPS selection.

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

To show genetic distance from an exported target genome on every grid candidate:

```bash
python interactive_neat_particles.py --target targets/[target_file]
```

### Controls

- Mouse click or keys `1..9`: IEC toggles selection; SPS chooses the preferred sample.
- `Tab`: switch between IEC and SPS. From IEC, this binds SPS to the selected genome.
- `B`: bind SPS to the selected IEC genome, or candidate 1 if none is selected.
- `N`: new IEC generation (offspring from selected parents; if none selected, does nothing).
- `R`: reset the active mode.
- `W`: randomize IEC weights (keeps topology).
- `E`: export a manual auto-selection target genome.
- `Space`: pause/unpause (freeze time).
- `Esc`: quit.

### Sequential Plane Search weight tuning

SPS is a second interaction mode layered on top of the IEC gallery. IEC changes genomes through NEAT mutation. SPS keeps the selected Generic genome topology fixed and generates nine variants by changing selected connection weights.

Current SPS target:
- Connection weights whose source input is one of the Generic `Px`, `Py`, or `Pz` inputs.
- If those links are missing or disabled, SPS falls back to enabled connection weights so the mode still has something to tune.

The SPS design vector is the selected weights normalized into `[0, 1]`. The UI decodes each coordinate back into the configured NEAT weight range from `config-generic.ini` (`weight_min_value` to `weight_max_value`) before building each genome variant.

Hybrid workflow:
1. Use IEC to find an interesting particle genome.
2. Select it and press `Tab` or `B`.
3. In SPS, click the best-looking of the nine weight variants.
4. SPS records that preference and creates the next 3×3 plane around the chosen weights.

### Manual target export and automated SPS selection

The interactive UI can export a genome as a target for automated SPS experiments:

- Press `E` in IEC mode to export the selected candidate. If nothing is selected, candidate 1 is exported.
- Press `E` in SPS mode to export the center SPS candidate. The center candidate is index `4` in the 3×3 SPS plane.
- Exported files are written to `targets/` with timestamped names such as `target_20260627_120000_key_4.json`.
- There is no `latest_target.json` shortcut. Pass the exact exported file path to the auto-selection script.

The exported target file stores the original NEAT genome data, not only the runnable network. This includes node genes, connection genes, innovation numbers, weights, enabled flags, fitness, config shape, generation/mode metadata, and the candidate key. The auto-selection script needs this genome data so it can use `DefaultGenome.distance(...)` for similarity.

Run automated SPS selection from the repository root:

```bash
python neat-particles/auto_sps_select.py --target neat-particles/targets/target_20260627_120000_key_4.json --config neat-particles/config-generic.ini --seed 1 --threshold 0.05 --max-steps 100
```

To inspect one exported target genome visually:

```bash
python neat-particles/view_genome_target.py neat-particles/targets/target_20260627_120000_key_4.json
```

The viewer opens a pygame window with the particle animation, the ANN/genome graph, input/output key labels, export metadata, config shape, and connection summary.

Useful options:

- `--target <path>`: required target genome JSON exported from the UI.
- `--config <path>`: NEAT config file; defaults to `config-generic.ini`.
- `--seed <int>`: repeatable random initial SPS center.
- `--threshold <float>`: stop when genome distance is at or below this value; default is `0.05`.
- `--max-steps <int>`: stop after this many SPS choices if the threshold is not reached; default is `100`.
- `--output-dir <path>`: directory for the final selected genome and run report; defaults to `auto-runs/`.
- `--no-view`: skip the final pygame comparison window.

The automated script does not render during search. It creates a random initial genome, generates the same nine-candidate SPS plane layout as the interactive SPS mode, chooses the candidate with the smallest genome distance to the exported target, and repeats until `--threshold` or `--max-steps` stops the run. It records elapsed search time as `elapsed_seconds` in the console output and run report; this measures the headless search only, not the final pygame viewing time. The report history is compacted to every five steps, plus the final termination step. After completion, it saves a run report and final genome export, then opens a pygame comparison window unless `--no-view` is used. The comparison view shows the target particle/genome and the final selected particle/genome side by side.

### Notes / next steps

This is intentionally a **framework**:
- Rendering is a 2D preview (not a full 3D billboarded renderer).
- The network drawing is a compact genome visualization (inputs at top, outputs at bottom).
- Non-Generic systems are kept out of the active UI path so SPS is easier to reason about while testing.

---

## Learning notes: NEAT in `neat-python` (and how this example uses it)

This is written as a learning note about the NEAT algorithm as implemented in this repository.

Important honesty note: I did not extract text from `neat-particle paper.pdf` (there’s no PDF text-extraction tool available in this environment). The notes below are based on the actual `neat-python` core implementation in `C:\Users\jonat\neat-python\neat\` and the particle example code in `C:\Users\jonat\neat-python\neat-particles\`.

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

All systems use sigmoid activation (locked in the configs), so ANN outputs are typically in `[0, 1]`.

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

The current interactive UI intentionally assumes Generic only. Older Trail/Beam/Plane scaffold code is not part of the active SPS path.

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
1. For each particle (or control point / quad corner), build an input vector.
2. Call `net.activate(inputs)` to get outputs.
3. Interpret outputs as velocity and color.
4. Update position using: `P_t = P_{t-1} + S * V * T`.
5. Draw a 2D preview using pygame primitives.

Active system behavior summary:
- `GenericSystem`: independent particles, TTL respawn, position bounds, and raw inputs `(Px, Py, Pz, Dc, 1.0)`.

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
