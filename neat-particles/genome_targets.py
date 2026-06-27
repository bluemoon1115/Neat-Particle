from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import neat


TARGET_FORMAT_VERSION = 1


def _gene_attributes(gene) -> Dict[str, Any]:
    return {attribute.name: getattr(gene, attribute.name) for attribute in gene._gene_attributes}


def genome_to_target_data(
    genome: neat.DefaultGenome,
    config: neat.Config,
    *,
    candidate_key: Optional[int] = None,
    mode: Optional[str] = None,
    generation: Optional[int] = None,
    label: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert a DefaultGenome into JSON-safe target data."""
    genome_config = config.genome_config
    nodes = []
    for key, node in sorted(genome.nodes.items()):
        node_data = {"key": key}
        node_data.update(_gene_attributes(node))
        nodes.append(node_data)

    connections = []
    for key, connection in sorted(genome.connections.items()):
        input_key, output_key = key
        connection_data = {
            "key": [input_key, output_key],
            "innovation": connection.innovation,
        }
        connection_data.update(_gene_attributes(connection))
        connections.append(connection_data)

    return {
        "format": "neat-particles-genome-target",
        "format_version": TARGET_FORMAT_VERSION,
        "created_timestamp": datetime.now().isoformat(timespec="seconds"),
        "metadata": {
            "genome_key": genome.key,
            "candidate_key": candidate_key,
            "mode": mode,
            "generation": generation,
            "label": label,
            "fitness": genome.fitness,
        },
        "config_shape": {
            "num_inputs": genome_config.num_inputs,
            "num_outputs": genome_config.num_outputs,
            "input_keys": list(genome_config.input_keys),
            "output_keys": list(genome_config.output_keys),
            "feed_forward": genome_config.feed_forward,
        },
        "nodes": nodes,
        "connections": connections,
    }


def target_data_to_genome(data: Dict[str, Any], config: neat.Config) -> neat.DefaultGenome:
    """Rebuild a DefaultGenome from target data and validate its config shape."""
    validate_target_data(data, config)
    metadata = data.get("metadata", {})
    genome = neat.DefaultGenome(metadata.get("genome_key"))
    genome.fitness = metadata.get("fitness")

    node_type = config.genome_config.node_gene_type
    connection_type = config.genome_config.connection_gene_type

    for node_data in data["nodes"]:
        node = node_type(int(node_data["key"]))
        for attribute in node._gene_attributes:
            setattr(node, attribute.name, node_data[attribute.name])
        genome.nodes[node.key] = node

    for connection_data in data["connections"]:
        input_key, output_key = connection_data["key"]
        connection = connection_type(
            (int(input_key), int(output_key)),
            innovation=int(connection_data["innovation"]),
        )
        for attribute in connection._gene_attributes:
            setattr(connection, attribute.name, connection_data[attribute.name])
        genome.connections[connection.key] = connection

    return genome


def validate_target_data(data: Dict[str, Any], config: neat.Config) -> None:
    """Validate target JSON before using it in automated selection."""
    if data.get("format") != "neat-particles-genome-target":
        raise ValueError("Target file is not a neat-particles genome target")
    if data.get("format_version") != TARGET_FORMAT_VERSION:
        raise ValueError(f"Unsupported target format version: {data.get('format_version')!r}")

    shape = data.get("config_shape", {})
    genome_config = config.genome_config
    expected = {
        "num_inputs": genome_config.num_inputs,
        "num_outputs": genome_config.num_outputs,
        "input_keys": list(genome_config.input_keys),
        "output_keys": list(genome_config.output_keys),
        "feed_forward": genome_config.feed_forward,
    }
    for key, value in expected.items():
        if shape.get(key) != value:
            raise ValueError(
                f"Target config mismatch for {key}: expected {value!r}, got {shape.get(key)!r}"
            )


def save_genome_target(
    filepath: str,
    genome: neat.DefaultGenome,
    config: neat.Config,
    *,
    candidate_key: Optional[int] = None,
    mode: Optional[str] = None,
    generation: Optional[int] = None,
    label: Optional[str] = None,
) -> str:
    """Save a genome target JSON file and return the absolute path."""
    data = genome_to_target_data(
        genome,
        config,
        candidate_key=candidate_key,
        mode=mode,
        generation=generation,
        label=label,
    )
    abs_path = os.path.abspath(filepath)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    with open(abs_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
        file.write("\n")
    return abs_path


def load_genome_target(filepath: str, config: neat.Config) -> Tuple[neat.DefaultGenome, Dict[str, Any]]:
    """Load a target genome JSON file."""
    with open(filepath, "r", encoding="utf-8") as file:
        data = json.load(file)
    genome = target_data_to_genome(data, config)
    return genome, data


def timestamped_target_path(target_dir: str, candidate_key: Optional[int]) -> str:
    """Build the standard timestamped target export path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    key = "unknown" if candidate_key is None else str(candidate_key)
    return os.path.join(target_dir, f"target_{timestamp}_key_{key}.json")
