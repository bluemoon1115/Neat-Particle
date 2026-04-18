"""Holdout correctness check for FeedForwardNetwork.activate() edits.

Builds the same fixed network as activate_bench.py, runs a grid of inputs,
compares outputs against a reference computed with an independent
straightforward evaluator, and prints the maximum absolute deviation as
the last non-empty line of stdout.

The autoloop treats this as a "minimize" metric — regressions (larger
deviation) block winner promotion. A clean edit should keep deviation at
machine-epsilon level (< 1e-12).
"""
import os
import random
import sys

import neat
from neat.innovation import InnovationTracker

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(REPO_ROOT, "examples", "xor", "config-feedforward")

SEED = 20260418
N_GRID = 64


def build_fixed_network_and_evals():
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
    for _ in range(6):
        genome.mutate_add_node(config.genome_config)
    for _ in range(10):
        genome.mutate_add_connection(config.genome_config)
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    return net, config


def reference_activate(net, inputs):
    """Minimal reference evaluator — mirrors the v1.x loop verbatim.

    Used as the ground truth the edited activate() must match.
    """
    values = {k: 0.0 for k in net.input_nodes + net.output_nodes}
    for k, v in zip(net.input_nodes, inputs):
        values[k] = v
    for node, act_func, agg_func, bias, response, links in net.node_evals:
        node_inputs = [values[i] * w for i, w in links]
        s = agg_func(node_inputs)
        values[node] = act_func(bias + response * s)
    return [values[i] for i in net.output_nodes]


def main():
    net, _ = build_fixed_network_and_evals()
    random.seed(SEED + 2)
    inputs_grid = [
        tuple(random.uniform(-1.0, 1.0) for _ in range(2)) for _ in range(N_GRID)
    ]

    max_dev = 0.0
    for xi in inputs_grid:
        ref = reference_activate(net, xi)
        # Re-build the network to reset per-call state, then activate.
        net2, _ = build_fixed_network_and_evals()
        got = net2.activate(xi)
        if len(got) != len(ref):
            print(f"LENGTH MISMATCH: got={len(got)} ref={len(ref)}", flush=True)
            print("inf", flush=True)
            return 1
        for a, b in zip(got, ref):
            dev = abs(a - b)
            if dev > max_dev:
                max_dev = dev

    print(f"# grid={N_GRID}", flush=True)
    print(f"{max_dev:.3e}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
