import pandas as pd
import jax
import jax.numpy as jnp
from genetic_helpers import compare_programs, compute_intersection, perform_island_deduplication


# Define a set of test programs to compare
n_cells = 10
params_true = jnp.arange(20, dtype=jnp.float32).reshape((n_cells, 2)) + 1 # 10 cells, 2 parameters each
program_true = lambda x, a, b: a * x + b

def generate_evaluation_matrix_and_loss(program, params):
    key = jax.random.PRNGKey(seed=42)
    eval_matrix_x = jax.random.uniform(key, shape=(n_cells, 100), minval=0.0, maxval=1.0).astype(jnp.float32)
    eval_matrix = program(eval_matrix_x, params[:, 0:1], params[:, 1:2])
    train_loss = jnp.mean((eval_matrix - eval_matrix_x) ** 2)
    return eval_matrix, train_loss

eval_matrix_true, train_loss_true = generate_evaluation_matrix_and_loss(program_true, params_true)

ref_program = pd.Series({
    'birth_island': 0,
    'iteration_number': 0,
    'batch_index': 0,
    'program_code_string': 'def neuron_model(x, a, b): return a * x + b',
    'params': params_true,
    'program': program_true, 
    'evaluation_matrix': eval_matrix_true,
    'train_loss': train_loss_true
})
# program 1 is just ref program with a different birth_island
program_1 = ref_program.copy()
program_1['birth_island'] = 1

# program 2 is just program_1 with a different train_loss
program_2 = program_1.copy()
program_2['train_loss'] = train_loss_true + 0.005  # slightly different loss, but within tolerance

# program_3 is just program_2 with a different (but equivalent) code string
program_3 = program_2.copy()
program_3['program_code_string'] = 'def neuron_model(x, a, b): return a * x + b + 0.0'  # equivalent code

# program_4 is just program_3 with slightly different parameters
program_4 = program_3.copy()
program_4['params'] = program_3['params'] - 1e-5

# program_5 is program_4 with a different code string that is not equivalent, and different program reflecting this
program_5 = program_4.copy()
program_5['birth_island'] = 2  # different birth island
program_5['program_code_string'] = 'def neuron_model(x, a, b): return -a * x + b'  # not equivalent code
program_5['program'] = lambda x, a, b: -a * x + b  # not equivalent program
program_5['evaluation_matrix'], program_5['train_loss'] = generate_evaluation_matrix_and_loss(program_5['program'], program_5['params'])

# program_6 has a different number of params and should fail early on
program_6 = program_4.copy()
program_6['birth_island'] = 3  # different birth island
program_6['program_code_string'] = 'def neuron_model(x, b, a): return b * x + a'  # different (but equivalent) code string
program_6['params'] = jnp.arange(30, dtype=jnp.float32).reshape((n_cells, 3)) + 1  # 10 cells, 3 parameters each
program_6['evaluation_matrix'], program_6['train_loss'] = generate_evaluation_matrix_and_loss(program_6['program'], program_6['params'])

def test_compare_programs():
    assert compare_programs(ref_program, program_1, mode='complex') == True, "Test failed: ref_program and program 1 should be equivalent modulo birth_island."
    assert compare_programs(ref_program, program_2, mode='complex') == True, f"Test failed: ref_program and program 2 should be equivalent modulo train_loss. diff ={abs(program_1['train_loss'] - program_2['train_loss'])}"
    assert compare_programs(ref_program, program_3, mode='complex') == True, "Test failed: ref_program and program 3 should be equivalent modulo code string."
    assert compare_programs(ref_program, program_4, mode='complex') == True, "Test failed: ref_program and program 4 should be equivalent modulo parameters."
    assert compare_programs(ref_program, program_5, mode='complex') == False, "Test failed: ref_program and program 5 should not be equivalent due to different code and program."
    assert compare_programs(ref_program, program_6, mode='complex') == False, "Test failed: ref_program and program 6 should not be equivalent due to different number of parameters."
    assert compare_programs(ref_program, program_1, mode='simple') == True, "Test failed: simple mode should consider ref_program and program_1 equivalent as they have the same code but diff unique identifiers."
    assert compare_programs(program_1, program_2, mode='simple') == True, "Test failed: simple mode should consider program_1 and program_2 equivalent as they have the same code and unique identifiers."
    assert compare_programs(program_2, program_3, mode='simple') == True, "Test failed: simple mode should consider program_2 and program_3 equivalent as they have the same unique identifiers."
    assert compare_programs(program_4, program_5, mode='simple') == False, "Test failed: simple mode should not consider program_4 and program_5 equivalent due to different code and unique identifiers."

def test_compute_intersection():
    # compute intersection
    island_1 = pd.Series([program_1, program_5])
    island_2 = pd.Series([program_2, program_6])

    # intersection returns indices in island_2 that are contained in island_1. 
    # in this case, since program_2 ~ program_1, we expect the intersection to contain program_2's index in island_2, so [0]
    intersection = compute_intersection(island_1, island_2)
    assert intersection == [0], f"Test failed: Expected intersection to be [0], got {intersection}."

    island_1 = pd.Series([program_1, program_2, program_3, program_4])
    island_2 = pd.Series([program_5, program_6])
    intersection = compute_intersection(island_1, island_2)
    assert intersection == [], f"Test failed: Expected intersection to be [], got {intersection}."

    island_1 = pd.Series([program_1, program_2, program_3])
    island_2 = pd.Series([program_4, program_5, program_6])
    intersection = compute_intersection(island_1, island_2)
    assert intersection == [0], f"Test failed: Expected intersection to be [0], got {intersection}."

    island_2 = pd.Series([program_5, program_4, program_6])
    intersection = compute_intersection(island_1, island_2)
    assert intersection == [1], f"Test failed: Expected intersection to be [1], got {intersection}."

    island_2 = pd.Series([program_5, program_6, program_4])
    intersection = compute_intersection(island_1, island_2)
    assert intersection == [2], f"Test failed: Expected intersection to be [2], got {intersection}."

    island_2 = island_1.copy()
    intersection = compute_intersection(island_1, island_2)
    assert intersection == [0, 1, 2], f"Test failed: Expected intersection to be [0, 1, 2], got {intersection}."

def test_perform_island_deduplication():
    island_1 = pd.Series([program_1, program_2, program_5])
    island_2 = pd.Series([program_3, program_4, program_6])
    # Perform deduplication
    # this will first remove duplicates within each island, leaving island_1 = [program_1, program_5] and island_2 = [program_3, program_6]
    # then if overlap_threshold = 4, will not do any cross-island deduplication since there are not enough programs in each island to hit overlap of 4
    islands = [island_1, island_2]
    deduplicated_islands = perform_island_deduplication(islands, overlap_threshold=4)
    assert len(deduplicated_islands) == 2, "Test failed: Expected 2 islands after deduplication."
    assert deduplicated_islands[0].equals(pd.Series([program_1, program_5])), "Test failed: Island 1 did not deduplicate correctly."
    assert deduplicated_islands[1].equals(pd.Series([program_3, program_6])), "Test failed: Island 2 did not deduplicate correctly."

    # now test with overlap_threshold = 1
    # this removes duplicates from the island with the higher index, so island_1 will stay the same, and island_2 will be reduced to [program_6]
    deduplicated_islands = perform_island_deduplication(islands, overlap_threshold=1, min_island_size=1)
    assert len(deduplicated_islands) == 2, "Test failed: Expected 2 islands after deduplication with overlap_threshold=1."
    assert deduplicated_islands[0].equals(pd.Series([program_1, program_5])), "Test failed: Island 1 did not deduplicate correctly with overlap_threshold=1."
    assert deduplicated_islands[1].equals(pd.Series([program_6])), "Test failed: Island 2 did not deduplicate correctly with overlap_threshold=1."

if __name__ == "__main__":
    test_compare_programs()
    test_compute_intersection()
    test_perform_island_deduplication()
    print("All tests passed successfully.")