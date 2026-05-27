"""Monitoring: family tree + loss/gd-effect HTML reports built from a Population.

    write_family_tree(population, census, out_dir, task_name="...")
    write_progress(population, census, out_dir, task_name="...")
"""
from .family_tree import write_family_tree
from .progress import write_progress

__all__ = ["write_family_tree", "write_progress"]
