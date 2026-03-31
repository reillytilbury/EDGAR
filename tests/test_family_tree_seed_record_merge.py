import json

from src.monitoring.family_tree import create_family_tree


def test_family_tree_merges_duplicate_seed_records_without_dropping_metrics(tmp_path):
    log_path = tmp_path / "program_generation_log.jsonl"
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    seed_base = {
        "iteration_number": -1,
        "birth_island": -1,
        "batch_index": 0,
        "train_loss": 0.12,
        "test_loss": 0.34,
        "initial_loss": 0.56,
        "n_params": 7,
        "complexity_penalty": 0.07,
        "mode": "seed",
    }
    seed_update = {
        "iteration_number": -1,
        "birth_island": -1,
        "batch_index": 0,
        "train_fit_loss": 0.11,
        "test_fit_loss": 0.33,
        "is_seed": True,
    }
    candidate = {
        "iteration_number": 0,
        "birth_island": 0,
        "batch_index": 0,
        "parent1_id": [-1, -1, 0],
        "parent2_id": None,
        "train_loss": 0.22,
        "test_loss": 0.44,
        "initial_loss": 0.66,
        "n_params": 9,
        "complexity_penalty": 0.09,
        "mode": "explore",
    }

    with log_path.open("w") as f:
        for record in (seed_base, seed_update, candidate):
            f.write(json.dumps(record) + "\n")

    create_family_tree(str(log_path), str(output_dir), n_islands=1)

    html = (output_dir / "family_tree.html").read_text()
    assert '"train_loss": 0.12' in html
    assert '"test_loss": 0.34' in html
    assert '"initial_loss": 0.56' in html
    assert '"n_params": 7' in html
    assert '"complexity_penalty": 0.07' in html
    assert '"train_fit_loss": 0.11' in html
    assert '"test_fit_loss": 0.33' in html
