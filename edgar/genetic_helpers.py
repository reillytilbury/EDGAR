import itertools
import logging
import numpy as np
import jax.numpy as jnp
from .entities import Island, Program


def _get_attr(program, attr):
    if isinstance(program, Program):
        return getattr(program, attr)
    return program[attr]


def _ensure_params(program):
    params = _get_attr(program, "params")
    if params is None:
        return None
    return params

def compare_programs(program_a, program_b, loss_tol=0.02, cosine_tol=0.95, mode='complicated'):
    """
    Compare two programs similarity based on their unique identifiers, code strings, losses, and predictions.

    If using simple mode, only checks if they have the same unique identifiers or code strings, or different param shapes.

    if using complex mode, if it is inconclusive, it checks if the losses are within tolerance, and if the predictions are similar across cells.

    Args:
        program_a (Program): First program to compare.
        program_b (Program): Second program to compare.
        loss_tol (float): Tolerance for loss comparison.
        corr_tol (float): Tolerance for correlation comparison.
        mode (str): Mode of comparison, can be 'simple' or 'complicated'.
    Returns:
        bool: True if the programs are equivalent.
    """
    assert mode in ['simple', 'complicated'], "Mode must be either 'simple' or 'complicated'."

    # return false immediately if the programs have different number of free parameters, regardless of mode
    params_a = _ensure_params(program_a)
    params_b = _ensure_params(program_b)
    if params_a is None or params_b is None:
        return False
    if params_a.shape != params_b.shape:
        return False

    id_match = (_get_attr(program_a, 'birth_island') == _get_attr(program_b, 'birth_island') and
                _get_attr(program_a, 'generation') == _get_attr(program_b, 'generation') and
                _get_attr(program_a, 'batch_index') == _get_attr(program_b, 'batch_index'))
    code_string_match = _get_attr(program_a, 'function_code_string') == _get_attr(program_b, 'function_code_string')
    
    if mode == 'simple':
        return id_match or code_string_match
    
    # if we reach here, we are in complex mode
    # 1. check if losses are very different, if so, return False
    if abs(_get_attr(program_a, 'train_loss') - _get_attr(program_b, 'train_loss')) >= loss_tol:
        return False
    
    # 2. check if predictions are very different, if so, return False
    y_eval_a = _get_attr(program_a, 'evaluation_matrix')
    y_eval_b = _get_attr(program_b, 'evaluation_matrix')
    if y_eval_a is None or y_eval_b is None:
        return False
    y_a_normed = y_eval_a / jnp.linalg.norm(y_eval_a, axis=1, keepdims=True)
    y_b_normed = y_eval_b / jnp.linalg.norm(y_eval_b, axis=1, keepdims=True)
    cosine_similarity = jnp.sum(y_a_normed * y_b_normed, axis=1)
    if jnp.mean(cosine_similarity) < cosine_tol:
        return False
    
    # if we reach here, the programs are equivalent
    return True

def remove_duplicates(island: Island, mode='complicated', loss_tol=0.01, cosine_tol=0.99, loss_type='train_loss'):
    """Remove duplicate programs within an island."""
    programs = island.programs
    n_programs = len(programs)
    indices_to_remove = set()
    for i in range(n_programs):
        p_i = programs[i]
        for j in range(i + 1, n_programs):
            p_j = programs[j]
            if not compare_programs(p_i, p_j, mode=mode, loss_tol=loss_tol, cosine_tol=cosine_tol):
                continue
            loss_i = _get_attr(p_i, loss_type)
            loss_j = _get_attr(p_j, loss_type)
            if loss_i < loss_j:
                indices_to_remove.add(j)
            elif loss_i > loss_j:
                indices_to_remove.add(i)
            else:
                indices_to_remove.add(j)
    island.drop_indices(list(indices_to_remove))
    return island

def compute_intersection(island_a: Island, island_b: Island, mode='complicated'):
    # this is symmetric if and only if island_a and island_b do not harbour any duplicates
    duplicate_indices_in_b = []
    programs_a = island_a.programs
    programs_b = island_b.programs
    n_programs_a, n_programs_b = len(programs_a), len(programs_b)
    for i in range(n_programs_b):
        ref_program = programs_b[i]
        for j in range(n_programs_a):
            candidate_match = programs_a[j]
            if compare_programs(ref_program, candidate_match, mode=mode):
                duplicate_indices_in_b.append(i)
                break
    return duplicate_indices_in_b

def perform_island_deduplication(islands: list[Island], overlap_threshold=6, mode='complicated'):
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
        islands[j].drop_indices(duplicate_indices_to_drop)
        logging.info(f"Removed indices {duplicate_indices_to_drop} from island {j} due to overlap with island {i}.")
        print(f"Removed indices {duplicate_indices_to_drop} from island {j} due to overlap with island {i}. \nRemaining programs in island {j}: {len(islands[j])}")
                    
    return islands

def perform_population_pruning(islands: list[Island], critical_population_size=12, 
                               large_lm_name="", min_wise_population_size=0):
    """
    Prune the population of each island to ensure that it does not exceed the critical population size.
    Ensure that each island keeps a reserve of at least `min_wise_population_size` programs that are wise (i.e., trained with a large model).
    """
    assert min_wise_population_size <= critical_population_size, \
        f"min_wise_population_size ({min_wise_population_size}) must be less than or equal to critical_population_size ({critical_population_size})."
    
    for j, current_island in enumerate(islands):
        population = current_island.programs
        if len(population) <= critical_population_size:
            logging.info(f"Island {j} has fewer programs than the critical population size, skipping pruning.")
            continue
        wise_programs = [p for p in population if p.llm_name == large_lm_name]
        wise_programs.sort(key=lambda p: p.train_loss)
        reserved = wise_programs[:min_wise_population_size]
        reserved_set = set(reserved)
        remaining = [p for p in population if p not in reserved_set]
        remaining.sort(key=lambda p: p.train_loss)
        keep = reserved + remaining[:critical_population_size - len(reserved)]
        current_island._programs = keep
    return islands

def perform_probabilistic_migration(islands: list[Island], n_migrants, destination_islands:list[int], temperature=1.0):
    n_islands = len(islands)
    if destination_islands is None:
        logging.info("No destination islands provided, using default migration strategy.")
        destination_islands = [(i + 1) % n_islands for i in range(n_islands)]

    # calculate migration probabilities based on relative losses
    temp = max(temperature, 1e-3)
    relative_losses = []
    for island in islands:
        losses = np.array([p.train_loss for p in island.programs])
        if len(losses) == 0:
            relative_losses.append(np.array([]))
            continue
        relative_losses.append(losses - losses.min())
    rel_loss_std = [np.std(losses) if len(losses) else 0.0 for losses in relative_losses]
    losses_normed = [relative_losses[i] / (rel_loss_std[i] + 1e-6) if len(relative_losses[i]) else np.array([]) for i in range(n_islands)]
    migration_prob = [np.exp(-(losses_normed[i] / temp)) if len(losses_normed[i]) else np.array([]) for i in range(n_islands)]
    migration_prob = [prob / np.sum(prob) if prob.size and np.sum(prob) > 0 else prob for prob in migration_prob]

    # create a list of migrants for each island
    migrants_list = []
    for island_id in range(n_islands):
        n_programs = len(islands[island_id])
        probs = migration_prob[island_id]
        if n_programs == 0 or probs.size == 0:
            migrants_list.append([])
            continue
        n_nonzero_probs = np.sum(probs > 0)
        n_migrants_i = min(n_migrants, n_nonzero_probs)
        sampled_indices = np.random.choice(np.arange(n_programs), size=n_migrants_i, replace=False, p=probs)
        migrants = [islands[island_id].programs[idx].clone() for idx in sampled_indices]
        migrants_list.append(migrants)
    # now we have a list of migrants for each island, we can migrate them to their destination islands
    for island_id in range(n_islands):
        dest_id = destination_islands[island_id]
        migrants = migrants_list[island_id]
        islands[dest_id].extend(migrants)
    return islands
