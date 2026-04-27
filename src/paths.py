import logging
import os
from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class RunPaths:
    base_dir: str
    date_stamp: str
    time_stamp: str
    full_dir: str
    image_prompts_dir: str
    image_param_est_vs_gd_dir: str
    image_param_est_refine_dir: str
    image_family_tree_fits_dir: str
    generation_log_path: str
    best_loss_path: str


def create_run_paths(save_path: str) -> RunPaths:
    base_dir = os.path.abspath(save_path)
    print("Base directory:", base_dir)
    os.makedirs(base_dir, exist_ok=True)

    date_stamp = pd.Timestamp.now().strftime("%m-%d")
    time_stamp = pd.Timestamp.now().strftime("%H-%M-%S")
    full_dir = os.path.join(base_dir, date_stamp, time_stamp)
    os.makedirs(full_dir, exist_ok=True)
    print("Created folder:", full_dir)

    image_feedback_dir = os.path.join(full_dir, "image_feedback")
    image_prompts_dir = os.path.join(image_feedback_dir, "prompts")
    image_param_est_vs_gd_dir = os.path.join(image_feedback_dir, "param_est_vs_gd")
    image_param_est_refine_dir = os.path.join(image_feedback_dir, "param_est_refinement")
    image_family_tree_fits_dir = os.path.join(image_feedback_dir, "family_tree_fits")
    for path in (
        image_feedback_dir,
        image_prompts_dir,
        image_param_est_vs_gd_dir,
        image_param_est_refine_dir,
        image_family_tree_fits_dir,
    ):
        os.makedirs(path, exist_ok=True)

    print("Created image feedback folder:", image_feedback_dir)
    print("Created image prompts folder:", image_prompts_dir)
    print("Created param-est vs gd folder:", image_param_est_vs_gd_dir)
    print("Created param-est refinement folder:", image_param_est_refine_dir)
    print("Created family tree fits folder:", image_family_tree_fits_dir)

    return RunPaths(
        base_dir=base_dir,
        date_stamp=date_stamp,
        time_stamp=time_stamp,
        full_dir=full_dir,
        image_prompts_dir=image_prompts_dir,
        image_param_est_vs_gd_dir=image_param_est_vs_gd_dir,
        image_param_est_refine_dir=image_param_est_refine_dir,
        image_family_tree_fits_dir=image_family_tree_fits_dir,
        generation_log_path=os.path.join(full_dir, "program_generation_log.jsonl"),
        best_loss_path=os.path.join(full_dir, "best_loss_log.csv"),
    )


def configure_file_logging(full_dir: str, verbose: bool = False) -> None:
    log_file = os.path.join(full_dir, "hypothesis_engine.log")
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(filename=log_file, level=level, format="%(message)s")
