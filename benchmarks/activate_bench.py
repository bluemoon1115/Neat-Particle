"""Wall-time benchmark for FeedForwardNetwork.activate().

Builds a fixed genome via deterministic seed, evaluates `N_CALLS` activations
over a fixed input set, prints median-of-REPEATS wall-clock time in seconds
as the last non-empty line of stdout.

Used as the autoloop metric for the `neat/nn/feed_forward.py` optimization
target. The harness file itself must not be edited by experiments — only
`neat/nn/feed_forward.py` is in the editable surface.
"""
import os
import random
import statistics
import sys
import time

import neat
from neat.innovation import InnovationTracker

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(REPO_ROOT, "examples", "xor", "config-feedforward")

N_CALLS = 20000
REPEATS = 7
SEED = 20260418


def build_fixed_network():
    random.seed(SEED)
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        CONFIG_PATH,
    )
    config.genome_config.innovation_tracker = InnovationTracker()
    genome = config.genome_type(0)
    genome.configure_new(config.genome_config)
    # Grow the network so activate() does meaningful work: several mutate_add_node
    # calls against a full-init genome produce a multi-layer topology.
    for _ in range(6):
        genome.mutate_add_node(config.genome_config)
    for _ in range(10):
        genome.mutate_add_connection(config.genome_config)
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    return net


def run_once(net, inputs_list):
    t0 = time.perf_counter()
    for xi in inputs_list:
        net.activate(xi)
    return time.perf_counter() - t0


def main():
    net = build_fixed_network()
    random.seed(SEED + 1)
    inputs_list = [
        tuple(random.uniform(-1.0, 1.0) for _ in range(2)) for _ in range(N_CALLS)
    ]

    # Warm-up call to trigger any first-call dict materialization.
    net.activate(inputs_list[0])

    times = [run_once(net, inputs_list) for _ in range(REPEATS)]
    median = statistics.median(times)

    print(f"# repeats={REPEATS} n_calls={N_CALLS} times={[round(t, 4) for t in times]}", flush=True)
    print(f"{median:.6f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
