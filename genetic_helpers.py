import jax.numpy as jnp
import numpy as np
import logging
import pandas as pd
import utils
import itertools

def compare_programs(program_a, program_b, loss_tol=0.02, cosine_tol=0.95, mode='complicated'):
    """
    Compare two programs similarity based on their unique identifiers, code strings, losses, and predictions.

    If using simple mode, only checks if they have the same unique identifiers or code strings, or different param shapes.

    if using complex mode, if it is inconclusive, it checks if the losses are within tolerance, and if the predictions are similar across cells.

    Args:
        program_a (pd.Series): First program to compare.
        program_b (pd.Series): Second program to compare.
        loss_tol (float): Tolerance for loss comparison.
        corr_tol (float): Tolerance for correlation comparison.
        mode (str): Mode of comparison, can be 'simple' or 'complicated'.
    Returns:
        bool: True if the programs are equivalent.
    """
    assert mode in ['simple', 'complicated'], "Mode must be either 'simple' or 'complicated'."

    # return false immediately if the programs have different number of free parameters, regardless of mode
    params_a = program_a['params']
    params_b = program_b['params']
    parameter_shape_mismatch = params_a.shape != params_b.shape
    if parameter_shape_mismatch:
        return False

    id_match = (program_a['birth_island'] == program_b['birth_island'] and
                program_a['iteration_number'] == program_b['iteration_number'] and
                program_a['batch_index'] == program_b['batch_index'])
    code_string_match = program_a['program_code_string'] == program_b['program_code_string']
    
    if mode == 'simple':
        return id_match or code_string_match
    
    # if we reach here, we are in complex mode
    # 1. check if losses are very different, if so, return False
    if abs(program_a['train_loss'] - program_b['train_loss']) >= loss_tol:
        return False
    
    # 2. check if predictions are very different, if so, return False
    y_eval_a = program_a['evaluation_matrix']
    y_eval_b = program_b['evaluation_matrix']
    y_a_normed = y_eval_a / jnp.linalg.norm(y_eval_a, axis=1, keepdims=True)
    y_b_normed = y_eval_b / jnp.linalg.norm(y_eval_b, axis=1, keepdims=True)
    cosine_similarity = jnp.sum(y_a_normed * y_b_normed, axis=1)
    if jnp.mean(cosine_similarity) < cosine_tol:
        return False
    
    # if we reach here, the programs are equivalent
    return True

def remove_duplicates(island, mode='complicated', loss_tol=0.01, cosine_tol=0.99, loss_type='train_loss'):
    """ Remove duplicate programs within island
    Args:
        island (pd.DataFrame): DataFrame containing programs in the island.
        mode (str): Mode of comparison, can be 'simple' or 'complicated'.
        loss_tol (float): Tolerance for loss comparison.
        cosine_tol (float): Tolerance for cosine similarity comparison.
        loss_type (str): Type of loss to use for comparison, e.g., 'train_loss' or 'test_loss'.
    Returns:
        island (pd.DataFrame): DataFrame with duplicates removed.
    """
    n_programs = len(island)
    indices_to_remove = set()
    for i in range(n_programs):
        p_i = island.iloc[i]
        for j in range(i + 1, n_programs):
            p_j = island.iloc[j]
            if not compare_programs(p_i, p_j, mode=mode, loss_tol=loss_tol, cosine_tol=cosine_tol):
                continue
            # if programs are equivalent, mark the one with higher loss for removal
            if p_i[loss_type] < p_j[loss_type]:
                indices_to_remove.add(j)
            elif p_i[loss_type] > p_j[loss_type]:
                indices_to_remove.add(i)
            # if losses are equal, remove the one with higher index
            else:
                indices_to_remove.add(j)
    
    # Drop all marked indices at once
    island = island.drop(index=list(indices_to_remove)).reset_index(drop=True)
    return island

def compute_intersection(island_a, island_b, mode='complicated'):
    # this is symmetric if and only if island_a and island_b do not harbour any duplicates
    duplicate_indices_in_b = []
    n_programs_a, n_programs_b = len(island_a), len(island_b)
    for i in range(n_programs_b):
        ref_program = island_b.iloc[i]
        for j in range(n_programs_a):
            candidate_match = island_a.iloc[j]
            if compare_programs(ref_program, candidate_match, mode=mode):
                duplicate_indices_in_b.append(i)
                break
    return duplicate_indices_in_b

def perform_island_deduplication(islands, overlap_threshold=6, mode='complicated'):
    """
    Perform deduplication of programs 1. within each island and 2. between islands
    """

    # 1. within island deduplication
    n_islands = len(islands)
    for i in range(n_islands):
        islands[i] = remove_duplicates(islands[i])

    # 2. between islands deduplication
    for i, j in itertools.product(range(n_islands), range(n_islands)):
        if j <= i:
            continue
        duplicate_indices_in_j = compute_intersection(islands[i], islands[j], mode)
        programs_in_j = len(islands[j])
        if len(duplicate_indices_in_j) < overlap_threshold:
            continue
        # ensure >= 2 programs left after deduplication
        duplicate_indices_to_drop = [idx for k, idx in enumerate(duplicate_indices_in_j) if k < programs_in_j - 2]
        islands[j] = islands[j].drop(index=duplicate_indices_to_drop).reset_index(drop=True)
        logging.info(f"Removed indices {duplicate_indices_to_drop} from island {j} due to overlap with island {i}.")
        print(f"Removed indices {duplicate_indices_to_drop} from island {j} due to overlap with island {i}. \nRemaining programs in island {j}: {len(islands[j])}")
                    
    return islands

def perform_population_pruning(islands: list[pd.DataFrame], critical_population_size=12, 
                               large_lm_name="", min_wise_population_size=0):
    """
    Prune the population of each island to ensure that it does not exceed the critical population size.
    Ensure that each island keeps a reserve of at least `min_wise_population_size` programs that are wise (i.e., trained with a large model).
    """
    assert min_wise_population_size <= critical_population_size, \
        f"min_wise_population_size ({min_wise_population_size}) must be less than or equal to critical_population_size ({critical_population_size})."
    
    for j, current_island in enumerate(islands):
        if len(current_island) <= critical_population_size:
            logging.info(f"Island {j} has fewer programs than the critical population size, skipping pruning.")
            continue
        wise_programs = current_island[current_island['llm_name'] == large_lm_name]
        top_wise_programs = wise_programs.nsmallest(min_wise_population_size, 'train_loss').reset_index(drop=True)
        # remove top_wise from current_island
        current_island = current_island[~current_island.index.isin(top_wise_programs.index)].reset_index(drop=True)
        n_vacancies = critical_population_size - len(top_wise_programs)
        top_overflow_programs = current_island.nsmallest(n_vacancies, 'train_loss').reset_index(drop=True)
        # concatenate top_wise and top_overflow_programs and reset index
        islands[j] = pd.concat([top_wise_programs, top_overflow_programs], ignore_index=True).reset_index(drop=True)
    return islands

def perform_population_pruning_old(islands: list[pd.DataFrame], critical_population_size=12):
    for j, current_island in enumerate(islands):
        if len(current_island) <= critical_population_size:
            logging.info(f"Island {j} has fewer programs than the critical population size, skipping pruning.")
            continue
        current_island = current_island.nsmallest(critical_population_size, 'train_loss').reset_index(drop=True)
        islands[j] = current_island
    return islands

def perform_probabilistic_migration(islands, n_migrants, destination_islands:list[int], temperature=1.0):
    n_islands = len(islands)
    if destination_islands is None:
        logging.info("No destination islands provided, using default migration strategy.")
        destination_islands = [(i + 1) % n_islands for i in range(n_islands)]

    # calculate migration probabilities based on relative losses
    temp = max(temperature, 1e-3)
    relative_losses = [np.array(island['train_loss'] - island['train_loss'].min()) for island in islands]
    rel_loss_std = [np.std(losses) for losses in relative_losses]
    losses = [relative_losses[i] / (rel_loss_std[i] + 1e-6) for i in range(n_islands)]
    migration_prob = [np.exp(-(losses[i] / temp)) for i in range(n_islands)]
    migration_prob = [prob / np.sum(prob) for prob in migration_prob]

    # create a list of migrants for each island
    migrants_list = []
    for island_id in range(n_islands):
        n_programs = len(islands[island_id])
        n_nonzero_probs = np.sum(migration_prob[island_id] > 0)
        n_migrants_i = min(n_migrants, n_nonzero_probs)
        sampled_indices = np.random.choice(np.arange(n_programs), size=n_migrants_i, replace=False, p=migration_prob[island_id])
        migrants = islands[island_id].iloc[sampled_indices].reset_index(drop=True) 
        migrants_list.append(migrants)
    # now we have a list of migrants for each island, we can migrate them to their destination islands
    for island_id in range(n_islands):
        dest_id = destination_islands[island_id]
        islands[dest_id] = pd.concat([islands[dest_id], migrants_list[island_id]], ignore_index=True)
    return islands