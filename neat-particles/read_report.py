import argparse
import json
import os
import matplotlib.pyplot as plt

def parse_arguments():
    parser = argparse.ArgumentParser(description="Read JSON file and extract specific information")
    parser.add_argument("--target", type=str, required=True, help="Path to the json file")
    return parser.parse_args()

def load_path(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        print("Path read")
        return json.load(f)
    
def extract_metrics(data):
    steps = [entry["step"] for entry in data["history"]]
    distances = [entry["best_distance"] for entry in data["history"]]
    print("data extracted")
    return steps, distances

def plot_graph(step, distance):
    plt.figure(figsize=(10, 5))
    plt.plot(step, distance, marker='o', linestyle='-', color='b', label='score')
    plt.xlabel('step')
    plt.ylabel('distance')
    plt.title('best distance over iter')
    plt.grid(True)
    plt.legend()
    plt.show()

def main():
    args = parse_arguments()
    target_path = os.path.abspath(args.target)
    data = load_path(target_path)
    steps, distances = extract_metrics(data)
    plot_graph(steps, distances)
    
if __name__ == "__main__":
    raise SystemExit(main())

