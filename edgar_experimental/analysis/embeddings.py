"""Tools for generating embeddings and analyzing candidate program diversity.

This module provides functions to extract docstrings from model code, obtain
embeddings using Google GenAI, and compute geometric diversity metrics (e.g., Average
Pairwise Distance, Centroid Dispersion, Participation Ratio, HDBSCAN Cluster Entropy)
to analyze similarity and paradigms among evolved programs.

Example:
    >>> from edgar-experimental.analysis.embeddings import embed_programs, analyze_model_diversity
    >>> embeddings_dict = embed_programs(population)
    >>> metrics = analyze_model_diversity(embeddings_dict, program_names)
"""

from __future__ import annotations

import ast
import os
from typing import Sequence
import numpy as np
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.decomposition import PCA
from sklearn.cluster import HDBSCAN
import umap
import matplotlib.pyplot as plt
from google import genai
from google.genai import types
import json

from edgar.evolution.population import Population


def extract_model_docstring(code_str: str | None) -> str:
    """Extracts the docstring from the 'model' function in the code string.

    Locates the `model` function inside the source code using the AST and returns
    its docstring. Falls back to the module-level docstring if `model` is not found
    or has no docstring.

    Args:
        code_str: The Python source code containing the model function.

    Returns:
        The extracted docstring, or an empty string if none is found or on parsing error.
    """
    if not code_str:
        return ""
    try:
        tree = ast.parse(code_str)
    except SyntaxError:
        return ""

    # Look for 'model' function definition
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "model":
            doc = ast.get_docstring(node)
            if doc:
                return doc.strip()

    return ""


def embed(
    strings: Sequence[str],
    model_name: str = "gemini-embedding-001",
    api_key: str | None = None,
) -> np.ndarray:
    """Generates embeddings for all strings passed in.

    Args:
        strings: Sequence of strings to embed.
        model_name: The embedding model to use. Defaults to "gemini-embedding-001".
        api_key: Optional Google API key. Defaults to GOOGLE_API_KEY env variable.

    Returns:
        A numpy array of shape (N, D) containing the embeddings.
    """
    # Initialize Google GenAI client
    key = api_key or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise ValueError("GOOGLE_API_KEY is not set. Export it in your environment.")

    client = genai.Client(api_key=key)
    # Batch the strings (max request size is 100)
    batch_size = 100
    responses = []
    for i in range(0, len(strings), batch_size):
        batch = strings[i : i + batch_size]
        # Call Gemini batch embedding
        response = client.models.embed_content(
            model=model_name,
            contents=batch,
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
        )
        responses.append(response)

    values = [emb.values for response in responses for emb in response.embeddings]
    return np.stack(values)


def analyze_model_diversity(
    embeddings: np.ndarray,
    distance_metric: str = "cosine",
    plot: bool = False,
    save_path: str | None = None,
    min_cluster_size: int = 5,
    min_samples: int = 1,
) -> dict:
    """Analyzes high-dimensional model embeddings, computing diversity metrics and plots.

    Args:
        embeddings: NumPy array of shape (N, D) containing the embeddings.
        distance_metric: The metric to use for computing distances between embeddings (cosine or euclidean). Defaults to "cosine".
        plot: Whether to display interactive UMAP and Dendrogram plots.
        save_path: Optional file path to save the generated figure.
        min_cluster_size: Minimum cluster size for HDBSCAN clustering. Defaults to 5. Smaller values allow for more clusters, larger values yield fewer clusters.
        min_samples: Minimum samples for HDBSCAN clustering. Defaults to 1. Lower encourages merging of clusters

    Returns:
        A dictionary of computed diversity metrics and cluster assignments.
    """
    N, _ = embeddings.shape
    if N < 2:
        return {
            "error": "At least 2 embedded programs are required for diversity analysis."
        }

    if distance_metric == "cosine":
        dist_func = cosine_distances
    elif distance_metric == "euclidean":
        dist_func = euclidean_distances
    else:
        raise ValueError(
            "Unsupported distance metric. Please use 'cosine' or 'euclidean'."
        )

    # 1. Calculate Distance Matrix
    dist_matrix = dist_func(embeddings)

    # Metric A: Average Pairwise Distance
    triu_indices = np.triu_indices(N, k=1)
    avg_pairwise_dist = float(np.mean(dist_matrix[triu_indices]))

    # Metric B: Mean Centroid Dispersion
    centroid = np.mean(embeddings, axis=0, keepdims=True)
    centroid_dist = dist_func(embeddings, centroid)
    mean_dispersion = float(np.mean(centroid_dist))

    # Metric C: Participation Ratio (Effective Dimensionality)
    pca = PCA().fit(embeddings)
    eigenvalues = pca.explained_variance_
    sum_eigen = np.sum(eigenvalues)
    effective_dim = (
        float((sum_eigen**2) / np.sum(eigenvalues**2)) if sum_eigen > 0 else 1.0
    )

    # Metric D: HDBSCAN Clustering & Entropy
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=min_samples, metric="precomputed"
    )
    cluster_indices = clusterer.fit_predict(dist_matrix.astype(np.float64))

    unique_labels, counts = np.unique(
        cluster_indices[cluster_indices >= 0], return_counts=True
    )
    probs = counts / np.sum(counts) if np.sum(counts) > 0 else np.array([])
    cluster_entropy = float(-np.sum(probs * np.log2(probs))) if len(probs) > 1 else 0.0

    metrics = {
        "num_models_analyzed": N,
        "avg_pairwise_dist": avg_pairwise_dist,
        "mean_dispersion": mean_dispersion,
        "effective_dim": effective_dim,
        "num_clusters": len(unique_labels),
        "cluster_entropy": cluster_entropy,
        "cluster_indices": cluster_indices,
        "unclustered_percent": int(np.sum(cluster_indices == -1)) / N * 100,
    }

    if plot or save_path:
        # Avoid crashing headless environments
        if not plot:
            plt.switch_backend("Agg")

        fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))

        # 2D UMAP Projection
        reducer = umap.UMAP(
            n_neighbors=min(15, N - 1), metric=distance_metric, random_state=42
        )
        embedding_2d = reducer.fit_transform(embeddings)

        ax1.scatter(
            embedding_2d[:, 0],
            embedding_2d[:, 1],
            c=cluster_indices,
            cmap="tab10",
            s=70,
            alpha=0.8,
        )
        # for i, program_idx in enumerate(program_indices):
        #     ax1.annotate(program_idx, (embedding_2d[i, 0], embedding_2d[i, 1]), fontsize=8, alpha=0.7)
        ax1.set_title(f"UMAP Projection of Model Embeddings ({distance_metric})")
        ax1.set_xlabel("UMAP Dimension 1")
        ax1.set_ylabel("UMAP Dimension 2")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        if plot:
            plt.show()
        else:
            plt.close()

    return metrics


def label_clusters(cluster_indices: np.ndarray, docstrings: Sequence[str]) -> list[str]:
    """Assigns natural language labels to clusters based on LLM analysis of all docstrings

    Args:
        cluster_indices: Array of cluster indices for each program.
        docstrings: List of docstrings corresponding to each program.

    Returns:
        A list of natural language labels where the index corresponds to the cluster ID.
    """
    import json

    # Group docstrings by cluster ID
    cluster_to_docstrings = {}
    for idx, cluster_id in enumerate(cluster_indices):
        if cluster_id >= 0:
            cluster_to_docstrings.setdefault(int(cluster_id), []).append(
                docstrings[idx]
            )

    sorted_cluster_ids = sorted(cluster_to_docstrings.keys())

    # Initialize Google GenAI client
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise ValueError("GOOGLE_API_KEY is not set. Export it in your environment.")

    client = genai.Client(api_key=key)

    # Build the prompt matching the required structure
    prompt_task = (
        "Identify the common mathematical or scientific theme for each of the following clusters of mathematical models. "
        "For each cluster, analyze the model descriptions which belong to that cluster and assign a concise, descriptive natural language "
        "label (2 to 5 words) that summarizes the scientific paradigm or mathematical form shared by the models in that cluster and make"
        "them distinct from other clusters."
    )
    prompt_parts = [f"{prompt_task}:"]
    for cluster_id in sorted_cluster_ids:
        docs = cluster_to_docstrings[cluster_id]
        n_models = len(docs)
        docs_joined = "\n".join(f"- {d}" for d in docs)
        prompt_parts.append(f"Cluster {cluster_id} ({n_models} models):\n{docs_joined}")

    prompt = "\n\n".join(prompt_parts)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=list[str],
        ),
    )

    parsed_labels = json.loads(response.text)

    # Map the generated labels back to the list by cluster ID
    max_cluster_id = max(sorted_cluster_ids)
    cluster_labels = [""] * (max_cluster_id + 1)
    for i, cluster_id in enumerate(sorted_cluster_ids):
        if i < len(parsed_labels):
            cluster_labels[cluster_id] = parsed_labels[i]
        else:
            cluster_labels[cluster_id] = "Unknown Cluster (LLM token limit exceeded)"

    return cluster_labels


def run_semantic_analysis(
    run_path: str,
    distance_metric: str = "cosine",
    plot: bool = False,
    min_cluster_size: int = 5,
    min_samples: int = 1,
) -> dict:
    pop = Population.load(f"{run_path}/population.jsonl")
    print(
        f"Loaded population with {len(pop)} programs from {run_path}/population.jsonl"
    )
    docstrings = [extract_model_docstring(p.model_code) for p in pop]
    valid_indices = [i for i, d in enumerate(docstrings) if d != ""]
    docstrings = [docstrings[i] for i in valid_indices]
    print(
        f"Running semantic analysis on {len(docstrings)} programs with valid docstrings..."
    )
    program_indices = [pop[i].idx for i in valid_indices]
    if not os.path.exists(f"{run_path}/embeddings.npy"):
        print(f"Computing embeddings for {len(docstrings)} programs...")
        embeddings = embed(docstrings)
        np.save(f"{run_path}/embeddings.npy", embeddings)
        print(f"Embeddings saved to {run_path}/embeddings.npy")
    else:
        print(f"Loading precomputed embeddings from {run_path}/embeddings.npy...")
        embeddings = np.load(f"{run_path}/embeddings.npy")
    distance_metric = "cosine"
    metrics = analyze_model_diversity(
        embeddings,
        distance_metric=distance_metric,
        plot=plot,
        save_path=f"{run_path}/diversity_analysis_{distance_metric}.png",
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )
    print(f"Diversity analysis completed. Metrics: {metrics}")
    if plot:
        print(f"UMAP plot saved to {run_path}/diversity_analysis_{distance_metric}.png")
    if not os.path.exists(f"{run_path}/cluster_labels.json"):
        print("Generating cluster labels using LLM...")
        cluster_labels = label_clusters(metrics["cluster_indices"], docstrings)
        with open(f"{run_path}/cluster_labels.json", "w") as f:
            json.dump(cluster_labels, f, indent=2)
        print(f"Cluster labels saved to {run_path}/cluster_labels.json")
    else:
        with open(f"{run_path}/cluster_labels.json", "r") as f:
            cluster_labels = json.load(f)

    # Print a summary table of the clusters
    print("\n" + "=" * 80)
    print("CLUSTER SUMMARY TABLE")
    print("=" * 80)

    cluster_to_programs = {}
    cluster_to_docs = {}
    for j, c_id in enumerate(metrics["cluster_indices"]):
        if c_id >= 0:
            cluster_to_programs.setdefault(int(c_id), []).append(program_indices[j])
            cluster_to_docs.setdefault(int(c_id), []).append(docstrings[j])

    for c_id in sorted(cluster_to_programs.keys()):
        cluster_name = cluster_labels[c_id] if c_id < len(cluster_labels) else "Unknown"
        prog_ids = cluster_to_programs[c_id]
        example_doc = (
            cluster_to_docs[c_id][0]
            if cluster_to_docs[c_id]
            else "No docstring available"
        )

        prog_ids_str = ", ".join(map(str, prog_ids))
        indented_doc = "\n".join(
            "    " + line for line in example_doc.strip().splitlines()
        )

        print(f"Cluster {c_id}: {cluster_name}")
        print(f"  Program indices: {prog_ids_str}")
        print(f"  Example docstring:\n{indented_doc}")
        print("-" * 80)

    return metrics
