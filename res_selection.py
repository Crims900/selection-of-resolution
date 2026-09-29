import math
import os
import re
import sys
from collections import Counter
from typing import List, Tuple

import numpy as np
import pandas as pd


def extract_resolution(filename: str) -> float:
    """Extract a numeric resolution from names such as res0.5, res_0.5, or res-0.5.csv."""
    match = re.search(r"res[-_]?(\d+(?:\.\d+)?)", filename, re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not extract a resolution from filename: {filename}")
    return float(match.group(1))


def read_cluster_labels(filepath: str) -> List[int]:
    """Read cluster labels from a CSV file produced by the Louvain scripts."""
    df = pd.read_csv(filepath)
    if "cluster" in df.columns:
        labels = df["cluster"].values
    elif df.shape[1] >= 2:
        labels = df.iloc[:, 1].values
    else:
        raise ValueError(f"CSV file must contain at least two columns: {filepath}")

    try:
        labels = labels.astype(int)
    except (TypeError, ValueError):
        # Keep string labels if they cannot be converted to integers.
        pass
    return list(labels)


def compute_cluster_sizes(labels: List) -> Tuple[List[int], int]:
    """Return cluster sizes in descending order and the total number of cells."""
    counter = Counter(labels)
    sizes = sorted(counter.values(), reverse=True)
    total = sum(sizes)
    return sizes, total


def compute_entropy_contributions(sizes: List[int], total: int) -> Tuple[List[float], float]:
    """Calculate E_l = -p * log(p) for each cluster and the total entropy."""
    entropies = []
    for size in sizes:
        p = size / total
        entropies.append(0.0 if p == 0 else -p * math.log(p))
    return entropies, sum(entropies)


def compute_effective_clusters(sizes: List[int], total: int) -> int:
    """Determine the effective number of clusters using the 5 percent entropy rule.

    Each cluster's entropy contribution is normalized by the total entropy so
    that the contributions sum to 1. Consequently, the 0.05 thresholds are
    interpreted as 5 percent of the total cluster entropy, and the remaining
    entropy is simply 1 - cumulative.
    """
    entropies, total_entropy = compute_entropy_contributions(sizes, total)
    if total_entropy <= 0:
        # A single-cluster (zero-entropy) result has no effective substructure.
        return 0
    entropies = [e / total_entropy for e in entropies]
    n_clusters = len(sizes)
    cumulative = 0.0
    effective = 0

    for index, entropy_value in enumerate(entropies, start=1):
        if entropy_value < 0.05:
            remaining_entropy = 1.0 - cumulative
            effective = index - 1 if remaining_entropy < 0.05 else 0
            break
        if index == n_clusters:
            effective = n_clusters
        cumulative += entropy_value

    return effective


def select_resolution(directory: str) -> Tuple[float, int]:
    """Select the best resolution and its effective cluster count from CSV files."""
    all_files = [name for name in os.listdir(directory) if name.lower().endswith(".csv")]
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in directory: {directory}")

    file_info = []
    for filename in all_files:
        try:
            tau = extract_resolution(filename)
            file_info.append((tau, os.path.join(directory, filename)))
        except ValueError as exc:
            print(f"Warning: skipping {filename}: {exc}")

    if not file_info:
        raise ValueError("No CSV files with a resolution in the filename were found")

    file_info.sort(key=lambda item: item[0])
    resolutions = [item[0] for item in file_info]
    file_paths = [item[1] for item in file_info]

    total_entropies = []
    cluster_sizes_list = []
    total_cells_list = []

    for tau, filepath in zip(resolutions, file_paths):
        labels = read_cluster_labels(filepath)
        sizes, total_cells = compute_cluster_sizes(labels)
        _, total_entropy = compute_entropy_contributions(sizes, total_cells)
        total_entropies.append(total_entropy)
        cluster_sizes_list.append(sizes)
        total_cells_list.append(total_cells)
        print(f"Resolution {tau}: total entropy = {total_entropy:.4f}, clusters = {len(sizes)}")

    k_values = []
    for i in range(1, len(resolutions)):
        entropy_delta = total_entropies[i] - total_entropies[i - 1]
        resolution_delta = resolutions[i] - resolutions[i - 1]
        k = float("inf") if resolution_delta == 0 else entropy_delta / resolution_delta
        k_values.append((k, resolutions[i]))

    if not k_values:
        raise RuntimeError("At least two distinct resolutions are required")

    k_values.sort(key=lambda item: item[0], reverse=True)
    top_k_count = max(1, int(np.ceil(len(k_values) * 0.05)))
    candidate_taus = sorted({tau for _, tau in k_values[:top_k_count]})
    print(f"Candidate resolutions (top {top_k_count} of {len(k_values)} k values): {candidate_taus}")

    tau_to_sizes = dict(zip(resolutions, cluster_sizes_list))
    tau_to_total_cells = dict(zip(resolutions, total_cells_list))
    final_candidates = []

    for tau in candidate_taus:
        n_effective = compute_effective_clusters(tau_to_sizes[tau], tau_to_total_cells[tau])
        if n_effective > 0:
            final_candidates.append((tau, n_effective))
            print(f"Resolution {tau}: effective clusters = {n_effective}")
        else:
            print(f"Resolution {tau} was excluded because its effective cluster count is 0")

    if not final_candidates:
        print("Warning: all candidates were invalid; using the largest resolution")
        tau_star = max(resolutions)
        n_star = compute_effective_clusters(tau_to_sizes[tau_star], tau_to_total_cells[tau_star])
    else:
        tau_star, n_star = max(final_candidates, key=lambda item: item[0])

    print(f"Final selection: tau* = {tau_star}, N(tau*) = {n_star}")
    return tau_star, n_star


if __name__ == "__main__":
    # Example: python res_selection.py /Data/Programs/SpaGCN_stabilization/stabilization_part/151507/first100/louvain_results/
    if len(sys.argv) != 2:
        print("Usage: python res_selection.py <csv_directory>")
        sys.exit(1)

    selected_tau, selected_clusters = select_resolution(sys.argv[1])
    print(f"Selected resolution: {selected_tau}")
    print(f"Effective cluster count: {selected_clusters}")
