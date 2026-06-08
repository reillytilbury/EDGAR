"""
The `edgar` package serves as the root for the EDGAR (Equation Discovery with Generative AI and Reinforcement learning) system.

It organizes all core components, including the evolutionary algorithm, LLM interactions,
I/O operations, and the dashboard, into submodules for a structured and scalable architecture.

Usage Example:
    # While this __init__.py file is largely for package structure,
    # the EDGAR system is typically invoked via its command-line interface (CLI).
    #
    # To initialize a new EDGAR project:
    # edgar init-project my_scientific_task
    #
    # To run an EDGAR experiment:
    # edgar run projects/my_scientific_task/config.yaml
    #
    # To launch the dashboard for a running or completed experiment:
    # edgar dashboard experiments/my_scientific_task/MM-DD/HH-MM-SS
    #
    # Individual modules can be imported for direct use in advanced scenarios, e.g.:
    # from edgar.evolution.program import Program
    # from edgar.scoring.scoring import score
"""
