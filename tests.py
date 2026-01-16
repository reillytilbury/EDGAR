import pandas as pd
import numpy as np
import jax.numpy as jnp
import pytest
from src.genetic_helpers_v2 import (
    compare_programs, compute_intersection, perform_island_deduplication,
    perform_population_pruning, perform_probabilistic_migration, remove_duplicates,
    Program, Island, Archipelago, MigrationTopology, EvolutionConfig, ProgramRegistry
)


# =============================================================================
# TEST DATA SETUP
# Programs are defined by their BEHAVIOR (evaluation matrix), not their code.
# Two programs with identical outputs are duplicates, regardless of code.
# =============================================================================

n_cells = 10
n_trials = 8

# Parameters with 2 params per cell
params_2 = jnp.arange(20, dtype=jnp.float32).reshape((n_cells, 2)) + 1

# Parameters with 3 params per cell (structurally different)
params_3 = jnp.arange(30, dtype=jnp.float32).reshape((n_cells, 3)) + 1

# Create evaluation matrices that represent different behaviors
# Each matrix is (n_trials, n_cells) - output for each cell across trials

# Behavior A: linear response pattern (varies across trials AND cells)
eval_matrix_A = jnp.array([[float(t * c + 1) for c in range(n_cells)] for t in range(n_trials)], dtype=jnp.float32)

# Behavior B: negative linear response (clearly different from A)
eval_matrix_B = jnp.array([[float(-t * c - 1) for c in range(n_cells)] for t in range(n_trials)], dtype=jnp.float32)

# Behavior A': nearly identical to A (should be detected as duplicate)
eval_matrix_A_prime = eval_matrix_A * 1.001  # 0.1% difference

# Behavior C: quadratic response pattern
eval_matrix_C = jnp.array([[float(t * c**2 + 1) for c in range(n_cells)] for t in range(n_trials)], dtype=jnp.float32)


def make_program(code: str, eval_matrix, params, train_loss: float = 0.1,
                 birth_island: int = 0, iteration: int = 0, batch: int = 0) -> pd.Series:
    """Helper to create test program Series."""
    return pd.Series({
        'program_code_string': code,
        'params': params,
        'train_loss': train_loss,
        'evaluation_matrix': eval_matrix,
        'birth_island': birth_island,
        'iteration_number': iteration,
        'batch_index': batch,
        'program': lambda x: x,  # Placeholder
    })


# Define test programs with clear behavioral relationships
program_A1 = make_program("def model_a(x, a, b): return a * x + b", eval_matrix_A, params_2, 
                          train_loss=0.05, birth_island=0)
program_A2 = make_program("def model_a_v2(x, a, b): return a * x + b", eval_matrix_A, params_2,
                          train_loss=0.05, birth_island=1)  # Same behavior, different code
program_A3 = make_program("def model_a(x, a, b): return a * x + b", eval_matrix_A_prime, params_2,
                          train_loss=0.051, birth_island=2)  # Nearly identical behavior

program_B1 = make_program("def model_b(x, a, b): return -a * x + b", eval_matrix_B, params_2,
                          train_loss=0.05, birth_island=0)  # Different behavior

program_C1 = make_program("def model_c(x, a, b, c): return a * x**2", eval_matrix_C, params_3,
                          train_loss=0.05, birth_island=0)  # Different structure (3 params)


# =============================================================================
# PRINCIPLED BEHAVIORAL TESTS
# =============================================================================

class TestBehavioralEquivalence:
    """
    Tests for the core concept: programs are duplicates if they produce the same outputs.
    This is the principled approach - we care about WHAT programs do, not HOW they're written.
    """
    
    def test_identical_code_is_duplicate(self):
        """Exact same code string → always duplicate (fast path)."""
        p1 = make_program("def f(x): return x", eval_matrix_A, params_2)
        p2 = make_program("def f(x): return x", eval_matrix_A, params_2)
        
        assert compare_programs(p1, p2, mode='complicated') == True
        assert compare_programs(p1, p2, mode='simple') == True
    
    def test_same_behavior_different_code_is_duplicate(self):
        """
        Different code but identical outputs → IS a duplicate.
        This is the key insight: behavioral equivalence matters, not syntax.
        """
        # program_A1 and program_A2 have same eval_matrix but different code
        assert compare_programs(program_A1, program_A2, mode='complicated') == True
    
    def test_different_behavior_is_not_duplicate(self):
        """Different outputs → NOT a duplicate, even if loss is similar."""
        # program_A1 (positive slope) vs program_B1 (negative slope) 
        assert compare_programs(program_A1, program_B1, mode='complicated') == False
    
    def test_nearly_identical_behavior_is_duplicate(self):
        """Outputs within cosine tolerance → duplicate."""
        # program_A1 and program_A3 have nearly identical eval matrices
        assert compare_programs(program_A1, program_A3, mode='complicated') == True
    
    def test_different_param_count_is_not_duplicate(self):
        """Structurally different programs (different # params) → never duplicate."""
        # program_A1 has 2 params/cell, program_C1 has 3 params/cell
        assert compare_programs(program_A1, program_C1, mode='complicated') == False
        assert compare_programs(program_A1, program_C1, mode='simple') == False
    
    def test_simple_mode_only_compares_code(self):
        """
        Simple mode: only code string comparison (weak, but fast).
        Same behavior but different code → NOT duplicate in simple mode.
        """
        # Different code strings
        assert compare_programs(program_A1, program_A2, mode='simple') == False
        # Same code strings
        assert compare_programs(program_A1, program_A1, mode='simple') == True


class TestComputeIntersection:
    """Tests for finding duplicate programs across islands."""
    
    def test_finds_behavioral_duplicates(self):
        """Should find programs with same behavior across islands."""
        island_1 = pd.Series([program_A1, program_B1])
        island_2 = pd.Series([program_A2, program_C1])  # A2 is behavioral duplicate of A1
        
        intersection = compute_intersection(island_1, island_2, mode='complicated')
        assert intersection == [0], f"Expected [0], got {intersection}"
    
    def test_no_intersection_with_different_behaviors(self):
        """Islands with no behavioral overlap should have empty intersection."""
        island_1 = pd.Series([program_A1])
        island_2 = pd.Series([program_B1, program_C1])
        
        intersection = compute_intersection(island_1, island_2, mode='complicated')
        assert intersection == [], f"Expected [], got {intersection}"
    
    def test_all_duplicates(self):
        """All programs in island_2 are duplicates of island_1."""
        island_1 = pd.Series([program_A1, program_A2, program_A3])
        island_2 = pd.Series([program_A1, program_A2])  # All are duplicates
        
        intersection = compute_intersection(island_1, island_2, mode='complicated')
        assert set(intersection) == {0, 1}


class TestIslandDeduplication:
    """Tests for removing duplicate programs within and across islands."""
    
    def test_removes_within_island_duplicates(self):
        """Should remove behavioral duplicates within same island."""
        island = pd.Series([program_A1, program_A2, program_B1])
        
        deduped = remove_duplicates(island, mode='complicated')
        
        # A1 and A2 are duplicates, should keep only one + B1
        assert len(deduped) == 2
    
    def test_keeps_best_loss_when_deduplicating(self):
        """When removing duplicates, keep the one with better loss."""
        # Losses must be within loss_tol (1% relative) to trigger behavioral comparison
        p_good = make_program("def f(x): return x", eval_matrix_A, params_2, train_loss=0.100)
        p_bad = make_program("def g(x): return x", eval_matrix_A, params_2, train_loss=0.101)
        island = pd.Series([p_bad, p_good])
        
        deduped = remove_duplicates(island, mode='complicated')
        
        assert len(deduped) == 1
        assert deduped.iloc[0]['train_loss'] == 0.100
    
    def test_cross_island_deduplication(self):
        """Should remove duplicates across islands with overlap threshold."""
        island_1 = pd.Series([program_A1, program_B1])
        island_2 = pd.Series([program_A2, program_C1])  # A2 duplicates A1
        
        deduped = perform_island_deduplication([island_1, island_2], 
                                                overlap_threshold=1, mode='complicated')
        
        assert len(deduped) == 2
        assert len(deduped[0]) == 2  # Island 1 unchanged
        # Island 2 should have A2 removed (it's a duplicate of A1 in island 1)
        assert len(deduped[1]) == 1


# =============================================================================
# Keep original test functions for backward compatibility
# (These use the old test data format)
# =============================================================================

# Legacy test data
params_true = jnp.arange(20, dtype=jnp.float32).reshape((n_cells, 2)) + 1
ref_program = pd.Series({
    'birth_island': 0,
    'iteration_number': 0,
    'batch_index': 0,
    'program_code_string': 'def neuron_model(x, a, b): return a * x + b',
    'params': params_true,
    'cost': 0.1,
    'program': lambda x, a, b: a * x + b,
    'train_loss': 0.05,
    'evaluation_matrix': jnp.array([[i * a + b for a, b in params_true] for i in range(10)], dtype=jnp.float32)
})

program_1 = ref_program.copy()
program_1['birth_island'] = 1

program_2 = program_1.copy()
program_2['cost'] = 0.11

program_3 = program_2.copy()
program_3['program_code_string'] = 'def neuron_model(x, a, b): return a * x + b + 0.0'

program_4 = program_3.copy()
program_4['params'] = program_3['params'] + 1e-4

program_5 = program_4.copy()
program_5['birth_island'] = 2
program_5['program_code_string'] = 'def neuron_model(x, a, b): return -a * x + b'
program_5['program'] = lambda x, a, b: -a * x + b
program_5['evaluation_matrix'] = jnp.array([[-(i * a) + b for a, b in program_5['params']] for i in range(10)], dtype=jnp.float32)

program_6 = program_4.copy()
program_6['birth_island'] = 3
program_6['program_code_string'] = 'def neuron_model(x, b, a): return b * x + a'
program_6['params'] = jnp.arange(30, dtype=jnp.float32).reshape((n_cells, 3)) + 1
program_6['evaluation_matrix'] = jnp.array([[i * b + a + 0.0 for b, a, c in program_6['params']] for i in range(10)], dtype=jnp.float32)


def test_compare_programs():
    """Legacy tests updated for principled behavioral comparison."""
    # Same code = duplicate
    assert compare_programs(ref_program, program_1, mode='complicated') == True
    assert compare_programs(ref_program, program_2, mode='complicated') == True
    
    # Same behavior (same eval matrix) = duplicate in complicated mode
    assert compare_programs(ref_program, program_4, mode='complicated') == True
    
    # Different behavior (different eval matrix) = NOT duplicate
    assert compare_programs(ref_program, program_5, mode='complicated') == False
    
    # Different param count = NOT duplicate
    assert compare_programs(ref_program, program_6, mode='complicated') == False
    
    # Simple mode only compares code strings (not behavior)
    assert compare_programs(ref_program, program_1, mode='simple') == True  # Same code
    assert compare_programs(program_1, program_2, mode='simple') == True  # Same code
    # program_3 has different code (+ 0.0), so simple mode says NOT duplicate
    assert compare_programs(program_2, program_3, mode='simple') == False  # Different code strings
    assert compare_programs(program_4, program_5, mode='simple') == False  # Different code


def test_compute_intersection():
    """Test intersection detection."""
    island_1 = pd.Series([program_1, program_5])
    island_2 = pd.Series([program_2, program_6])
    
    # program_2 is behaviorally same as program_1
    intersection = compute_intersection(island_1, island_2, mode='complicated')
    assert intersection == [0]
    
    # program_5 and program_6 have different behaviors from program_1-4
    island_1 = pd.Series([program_1, program_2, program_3, program_4])
    island_2 = pd.Series([program_5, program_6])
    intersection = compute_intersection(island_1, island_2, mode='complicated')
    assert intersection == []


def test_perform_island_deduplication():
    """Test island deduplication."""
    island_1 = pd.Series([program_1, program_2, program_5])
    island_2 = pd.Series([program_3, program_4, program_6])
    islands = [island_1, island_2]
    
    deduplicated_islands = perform_island_deduplication(islands, overlap_threshold=4, mode='complicated')
    
    # Within-island dedup: programs 1,2 are duplicates, 3,4 are duplicates
    assert len(deduplicated_islands[0]) == 2  # Keep one of (1,2) + program_5
    assert len(deduplicated_islands[1]) == 2  # Keep one of (3,4) + program_6


if __name__ == "__main__":
    test_compare_programs()
    test_compute_intersection()
    test_perform_island_deduplication()
    print("All tests passed successfully.")


# =============================================================================
# COMPATIBILITY WRAPPER TESTS
# Tests for the DataFrame-based API used by hypothesis_engine.py
# =============================================================================

class TestCompatibilityWrappers:
    """Tests for v1 API compatibility wrapper functions."""
    
    def test_remove_duplicates_wrapper(self):
        """Test remove_duplicates with behavioral comparison."""
        # Create programs with same behavior (same eval matrix)
        eval_same = jnp.ones((5, 10))
        eval_diff = jnp.zeros((5, 10))
        
        # Losses must be within loss_tol (1% relative) for behavioral comparison
        df = pd.DataFrame({
            'program_code_string': ['def f(X): return 1', 'def g(X): return 1', 'def h(X): return 2'],
            'train_loss': [0.100, 0.101, 0.300],  # First two close, third different
            'params': [params_2, params_2, params_2],
            'evaluation_matrix': [eval_same, eval_same, eval_diff],
            'birth_island': [0, 0, 0]
        })
        
        result = remove_duplicates(df, mode='complicated')
        
        # First two are behavioral duplicates, should keep one + the third
        assert len(result) == 2
    
    def test_perform_population_pruning_wrapper(self):
        """Test perform_population_pruning wrapper function."""
        df = pd.DataFrame({
            'program_code_string': [f'def f{i}(X): return {i}' for i in range(10)],
            'train_loss': [float(i) for i in range(10)],
            'birth_island': [0] * 10
        })
        
        result = perform_population_pruning([df], max_population=3)
        
        assert len(result[0]) == 3
        # Should keep the 3 best (lowest loss)
        assert list(result[0]['train_loss']) == [0.0, 1.0, 2.0]
    
    def test_perform_probabilistic_migration_wrapper(self):
        """Test perform_probabilistic_migration wrapper function."""
        df1 = pd.DataFrame({
            'program_code_string': [f'def f{i}(X): return {i}' for i in range(5)],
            'train_loss': [float(i) for i in range(5)],
            'birth_island': [0] * 5
        })
        df2 = pd.DataFrame({
            'program_code_string': [f'def g{i}(X): return {i * 2}' for i in range(3)],
            'train_loss': [float(i) for i in range(3)],
            'birth_island': [1] * 3
        })
        
        # Migrate using ring topology (0 -> 1, 1 -> 0)
        result = perform_probabilistic_migration([df1, df2], n_migrants=2, destination_islands=[1, 0])
        
        assert len(result) == 2
        # Total programs should increase (migrants are copied)
        total_before = len(df1) + len(df2)
        total_after = len(result[0]) + len(result[1])
        assert total_after >= total_before


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_empty_dataframe_remove_duplicates(self):
        """Test remove_duplicates with empty input."""
        df = pd.DataFrame({
            'program_code_string': [],
            'train_loss': [],
            'evaluation_matrix': []
        })
        
        result = remove_duplicates(df, mode='complicated')
        assert len(result) == 0
    
    def test_single_program_remove_duplicates(self):
        """Test remove_duplicates with single program."""
        df = pd.DataFrame({
            'program_code_string': ['def f(x): return x'],
            'train_loss': [0.1],
            'params': [params_2],
            'evaluation_matrix': [eval_matrix_A]
        })
        
        result = remove_duplicates(df, mode='complicated')
        assert len(result) == 1
    
    def test_migration_preserves_data(self):
        """Test that migration doesn't corrupt program data."""
        df1 = pd.DataFrame({
            'program_code_string': ['def f(x): return x'],
            'train_loss': [0.1],
            'birth_island': [0]
        })
        df2 = pd.DataFrame({
            'program_code_string': ['def g(x): return x + 1'],
            'train_loss': [0.2],
            'birth_island': [1]
        })
        
        # Migrate with ring topology (0 -> 1, 1 -> 0)
        result = perform_probabilistic_migration([df1, df2], n_migrants=1, destination_islands=[1, 0])
        
        # Migration should add programs to destination islands
        # Island 0 should receive from island 1, island 1 should receive from island 0
        assert len(result) == 2
        # Programs should be preserved, and migrants added
        assert len(result[0]) >= 1  # At least original
        assert len(result[1]) >= 1  # At least original


