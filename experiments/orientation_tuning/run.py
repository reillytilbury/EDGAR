import inspect
import os
import logging
import asyncio
import numpy as np
import jax, jax.numpy as jnp
import timeout_decorator
import jaxopt, optax
import pandas as pd
from pathlib import Path
import utils, diagnostic, seed_programs, genetic_helpers, loss_functions
from tqdm import tqdm
from google import genai
from dotenv import load_dotenv
import warnings
import time
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.*"
)
print(jax.default_backend())    # should print "gpu"
print(jax.devices())

async def main(n_iterations=9, time_limit=60, k_max=2, n_islands=8, batch_size=6, 
                critical_population_size=12, min_wise_population_size=0, 
                n_migrants=2, fit_params=True, tol=1e-6, exploit_point=0.5,
                param_penalty_weight=0.01, FAILED_PROGRAM_COST=np.inf,
                use_image_feedback=True, use_param_estimator=True,
                exploration_topology = [1, 2, 3, 4, 5, 6, 7, 0],
                exploitation_topology = [1, 2, 3, 4, 5, 6, 7, 0],
                tiny_lm_name = 'gemini-2.0-flash-lite',
                little_lm_name = 'gemini-2.0-flash',
                large_lm_name = 'gemini-2.5-flash',
                use_large_every = 3,
                conc_thresh = 0.55, activity_thresh = 0.4,
                data_path = '/home/reilly/Downloads/8279387/gratings_drifting_GT1_2019_04_12_1.npy'):
    """ 
    Main function to run the hypothesis engine.
    """
    # load api keys
    load_dotenv()
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    # load and preprocess data
    neural_data = np.load(data_path, allow_pickle=True)
    neural_data = neural_data.item()
    response = utils.extract_stimulus_related_response(neural_data, n_pcs=0)
    angles = neural_data['istim']
    n_trials = response.shape[1]
    n_trials_small = int(n_trials * activity_thresh)

    # filter 
    active = (response > 0).astype(np.float32)
    firing_probs = np.mean(active, axis=1)
    conc = np.abs(np.sum(np.exp(2j * angles)[np.newaxis, :] * response, axis=1) / np.sum(response, axis=1))
    good_cells = np.where((firing_probs > activity_thresh) & (conc > conc_thresh))[0]
    n_good_cells = len(good_cells)

    # update angles and response to be (n_cells_small, n_trials_small) and (n_cells_small, n_trials_small)
    response_cropped, angles_cropped = np.zeros((len(good_cells), n_trials_small)), np.zeros((len(good_cells), n_trials_small))
    for i, cell in enumerate(good_cells):
        active_trials = response[cell] > 0
        active_trials_idx = np.where(active_trials)[0][:n_trials_small]
        response_cropped[i] = response[cell, active_trials_idx]
        angles_cropped[i] = angles[active_trials_idx]
        
    # update response and angles to be the cropped versions and convert to JAX arrays, normalize and split into train/test
    response, angles = jnp.asarray(response_cropped), jnp.asarray(angles_cropped)
    response = 100 * response / jnp.linalg.norm(response, axis=1, keepdims=True)  # normalize response
    key = jax.random.PRNGKey(42)
    training_size = n_good_cells // 2
    shuffled_indices = jax.random.permutation(key, jnp.arange(n_good_cells))
    training_cells, test_cells = shuffled_indices[:training_size], shuffled_indices[training_size:]
    response_train, response_test = response[training_cells, :], response[test_cells, :]
    angles_train, angles_test = angles[training_cells, :], angles[test_cells, :]
    print(f"Selected {len(good_cells)} cells with activity > {activity_thresh} and concentration > {conc_thresh}.")
    print(f"Using {len(training_cells)} cells for training and {len(test_cells)} cells for testing.")

    # create a dataframe to store the programs in each island
    islands = []
    for _ in range(n_islands):
        islands.append(pd.DataFrame(columns=['program_code_string', 'program', 'parameter_estimator_code_string', 'parameter_estimator',
                                             'iteration_number', 'birth_island', 'batch_index', 'train_loss', 'test_loss', 'params',
                                             'initial_loss', 'initial_params', 'llm_name', 'parent1_id', 'parent2_id', 'evaluation_matrix']))
    initial_programs = pd.DataFrame([])

    # wherever you run “python script.py” from…
    base_dir = os.path.join(os.getcwd(), 'program_databases')
    print("Base directory:", base_dir)
    os.makedirs(base_dir, exist_ok=True)
    date_stamp = pd.Timestamp.now().strftime("%m-%d")
    time_stamp = pd.Timestamp.now().strftime("%H-%M-%S")
    full_dir = os.path.join(base_dir, date_stamp, time_stamp)
    os.makedirs(full_dir, exist_ok=True)
    print("Created folder:", full_dir)
    # create a directory for image diagnostics
    image_feedback_dir = os.path.join(full_dir, 'image_feedback')
    os.makedirs(image_feedback_dir, exist_ok=True)
    print("Created image feedback folder:", image_feedback_dir)

    # census[i] = [generation, island, batch_index, llm_name, loss, time, parent1_id, parent2_id, evaluation_matrix, n_free_params]
    census = []
    
    # store and compute loss of 2 initial programs
    t_start = time.time()
    numpy_programs = [seed_programs.neuron_model_1, seed_programs.neuron_model_2]
    jax_programs = [seed_programs.neuron_model_1_jax, seed_programs.neuron_model_2_jax]
    param_estimators = [seed_programs.parameter_estimator_1, seed_programs.parameter_estimator_2]
    seed_losses = np.zeros(2)
    for i in range(2):
        # get the program, parameter estimator, and jax program
        program_num = numpy_programs[i]
        param_est = param_estimators[i]
        program_jax = jax_programs[i]
        # score the initial program
        loss_init, params_init, loss, params = objective(program_jax, param_est, 
                                        loss_func=loss_functions.quadratic_loss, 
                                        x=angles_train, y=response_train, 
                                        fit_params=fit_params, param_penalty_weight=param_penalty_weight, tol=tol,
                                        use_param_estimator=use_param_estimator)
        seed_losses[i] = loss
        # format strings
        import_string = "import numpy as np \n"
        import_string_jax = "import jax.numpy as jnp \n"
        program_name = program_num.__name__
        param_est_name = param_est.__name__
        program_jax_name = program_jax.__name__
        program_code_string = inspect.getsource(program_num).replace(f'def {program_name}(', f'def neuron_model_v{i+1}(')
        program_code_string = import_string + program_code_string
        parameter_estimator_code_string = inspect.getsource(param_est).replace(f'def {param_est_name}(', f'def parameter_estimator_v{i+1}(')
        parameter_estimator_code_string = import_string + parameter_estimator_code_string
        program_jax_code_string = inspect.getsource(program_jax).replace(f'def {program_jax_name}(', f'def neuron_model_v{i+1}(')
        program_jax_code_string = import_string_jax + program_jax_code_string
        y_eval = compute_evaluation_matrix(program_jax, params, n_evaluation_points=100)

        new_program_df = pd.DataFrame({'program_code_string': program_code_string,
                                    'program': program_jax,
                                    'parameter_estimator_code_string': parameter_estimator_code_string,
                                    'parameter_estimator': param_est,
                                    'iteration_number': -1,
                                    'birth_island': -1,  # Birth island is set to a special value for initial programs
                                    'batch_index': i,
                                    'train_loss': loss, 
                                    'test_loss': None,  # all test losses will be computed at the end
                                    'llm_name': None,
                                    'params': [params],
                                    'initial_loss': loss_init,
                                    'initial_params': [params_init],
                                    'parent1_id': None,
                                    'parent2_id': None,
                                    'evaluation_matrix': [y_eval]})
        initial_programs = pd.concat([initial_programs, new_program_df], ignore_index=True)
        print(f"Initial program {i + 1} loss: {loss:.2f}")
        census.append([-1, -1, i, None, loss, time.time() - t_start, None, None, y_eval, params.shape[1]])

    # seed each island with the initial programs
    for i in range(n_islands):
        islands[i] = pd.concat([islands[i], initial_programs], ignore_index=True)

    # Reset logging configuration
    log_file = os.path.join(full_dir, 'hypothesis_engine.log')
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(message)s')
    diagnostic.plot_model_fits(programs_df=initial_programs,
                               loss_function=loss_functions.quadratic_loss,
                               x=angles_train, y=response_train,
                               cell_selection=np.random.choice(len(angles_train), size=9, replace=False),
                               save_path=os.path.join(image_feedback_dir, 'initial_programs.png'),
                               labels=['seed_1', 'seed_2'],
                               colours=['tab:green', 'tab:red'],
                               dpi=100.0,
                               title="Seed Programs",
                               legend_fontsize=20,
                               line_alpha=0.9,
                               line_width=4,)

    # -----------------------------
    # HYPOTHESIS ENGINE
    # -----------------------------
    for i in tqdm(range(n_iterations), desc="Hypothesis Engine Iterations"):
        # check if time limit is reached
        if time.time() - t_start > time_limit * 60:
            logging.info(f"Time limit of {time_limit} minutes reached. Stopping iterations.")
            break
        logging.info(f"Iteration {i}")
        if use_large_every > 0 and i % use_large_every == 0:
            llm_name = large_lm_name
            logging.info(f"Using large LLM: {llm_name}")
        else:
            llm_name = little_lm_name
            logging.info(f"Using little LLM: {llm_name}")
        mode = 'explore' if i < n_iterations * exploit_point else 'exploit'
        temperature = 1 + np.exp(-i / n_iterations)
        model_image_dirs = np.empty((n_islands, batch_size), dtype=object)
        # param_est_image_dirs = np.empty((n_islands, batch_size), dtype=object)
        for island_idx in range(n_islands):
            for j in range(batch_size):
                if use_image_feedback:
                    model_image_dirs[island_idx, j] = os.path.join(image_feedback_dir, f'iter_{i}_island_{island_idx}_batch_{j}.png')
                    # param_est_image_dirs[island_idx, j] = os.path.join(image_feedback_dir, f'iter_{i}_island_{island_idx}_batch_{j}_param_est.png')
                else:
                    model_image_dirs[island_idx, j] = None
                    # param_est_image_dirs[island_idx, j] = None
        # generate new programs
        neuron_model_generation_tasks = [generate_new_neuron_model(islands[island_idx], 
                                                                   llm_name=llm_name, 
                                                                   client=client, 
                                                                   mode=mode, 
                                                                   k_max=k_max, 
                                                                   temp=temperature,
                                                                   spike_matrix=response_train, 
                                                                   stimuli=angles_train,
                                                                   img_dir=model_image_dirs[island_idx, j]) 
                                         for island_idx in range(n_islands) for j in range(batch_size)]
        logging.info(f"Generating {n_islands * batch_size} new programs... Model: {llm_name}, mode: {mode}, temperature: {temperature:.2f}")
        print(f"Generating {n_islands * batch_size} new programs... Model: {llm_name}, mode: {mode}, temperature: {temperature:.2f}")
        neuron_model_results = await asyncio.gather(*neuron_model_generation_tasks)
        neuron_model_code_strings = [result[0] for result in neuron_model_results]
        neuron_model_prompts = [result[1] for result in neuron_model_results]
        parent_ids = [result[2] for result in neuron_model_results]
        
        # convert to jax
        neuron_model_function_translation_tasks = [translate_to_jax(code_string, client, tiny_lm_name) for code_string in neuron_model_code_strings]
        jax_results = await asyncio.gather(*neuron_model_function_translation_tasks)
        neuron_model_results = [(neuron_model_code_strings[j], neuron_model_prompts[j], jax_results[j][0], jax_results[j][1]) for j in range(n_islands * batch_size)]
        
        # build parameter‑estimator tasks
        param_estimation_tasks = [
            generate_new_parameter_estimator(
                current_island=islands[island_idx],
                neuron_model_code_string=neuron_model_code_strings[island_idx * batch_size + j],
                llm_name=little_lm_name,  # same model used for programs
                client=client,
                spike_matrix=response_train, # training data
                stimuli=angles_train,
                k_max=2,
                temp=temperature,
                param_estimator_max_lines=100,
                img_dir=None # no image feedback for parameter estimator generation
            )
            for island_idx in range(n_islands)
            for j in range(batch_size)
        ]

        logging.info(
            f"Generating {n_islands * batch_size} parameter estimators "
            f"(LLM={llm_name}, mode={mode}, T={temperature:.2f})"
        )
        logging.info(f"Generating {n_islands * batch_size} new parameter estimators... Model: {little_lm_name}, mode: {mode}, temperature: {temperature:.2f}")
        param_est_results = await asyncio.gather(*param_estimation_tasks)
        # combine results
        island_results = [[neuron_model_results[island_idx * batch_size + j] + param_est_results[island_idx * batch_size + j] for j in range(batch_size)] for island_idx in range(n_islands)]

        # now loop through the results and compute losses
        success_rate = 0.0
        for island_idx, j in np.ndindex(n_islands, batch_size):
            logging.info(f"id={i},{island_idx},{j}")
            neuron_model_code_string, prompt, neuron_model_code_string_jax, neuron_model_new, param_est_code_string, param_est_new = island_results[island_idx][j]
            parent1_id, parent2_id = parent_ids[island_idx * batch_size + j]
            if neuron_model_new is None or param_est_new is None:
                logging.info(f"Skipping island {island_idx}, batch {j} due to LLM generation failure.")
                logging.info('-' * 50)
                continue
            
            initial_loss, initial_params, loss, optimized_params = objective(neuron_model_new, param_est_new, 
                                                                                loss_func=loss_functions.quadratic_loss,
                                                                                x=angles_train, y=response_train,
                                                                                param_penalty_weight=param_penalty_weight,
                                                                                fit_params=fit_params, tol=tol, 
                                                                                use_param_estimator=use_param_estimator)
            if loss == FAILED_PROGRAM_COST:
                logging.info('-' * 50)
                continue

            y_eval = compute_evaluation_matrix(neuron_model_new, optimized_params, n_evaluation_points=100)
            logging.info(f"Prompt: \n{prompt}\n")
            logging.info(f"Loss: {loss:.2f}\n")
            logging.info(f"Neuron Model: \n{neuron_model_code_string}\n")
            logging.info(f"Neuron Model (JAX): \n{neuron_model_code_string_jax}\n")
            logging.info(f"Parameter Estimator: \n{param_est_code_string}\n")


            # plot the fits of the neuron model and parameter estimator if using image feedback
            if use_image_feedback:
                diagnostic.plot_model_fits(
                    programs_df=pd.DataFrame({'program': [neuron_model_new, neuron_model_new], 'params': [initial_params, optimized_params]}),
                    loss_function=loss_functions.quadratic_loss,
                    x=angles_train,
                    y=response_train,
                    cell_selection=np.random.choice(len(angles_train), size=4, replace=False),
                    colours=['tab:green', 'tab:red'],
                    labels=['Param Estimator', 'Gradient Descent'],
                    line_alpha=1.0,
                    line_width=5.0,
                    point_alpha=0.2,
                    point_size=120,
                    legend_fontsize=20,
                    title=f"Updated Parameter Estimator and Gradient Descent Fit \n"
                        f"Initial Loss: {initial_loss:.2f}, Final Loss: {loss:.2f}",
                    save_path=os.path.join(image_feedback_dir, f'iter_{i}_island_{island_idx}_batch_{j}_updated_param_est.png')
                )
            
            param_names = [n for n in inspect.signature(neuron_model_new).parameters if n != "theta"]
            if optimized_params.shape[1] == len(param_names):
                df = pd.DataFrame(np.array(optimized_params)[:10], columns=param_names)
                logging.info(f"Optimized Parameters for 10 cells:\n{df}\n")
            t_added = time.time() - t_start
            new_program_df = pd.DataFrame({'program_code_string': neuron_model_code_string,
                                        'program': neuron_model_new,
                                        'parameter_estimator_code_string': param_est_code_string,
                                        'parameter_estimator': param_est_new,
                                        'iteration_number': i,
                                        'birth_island': island_idx,
                                        'batch_index': j,
                                        'train_loss': loss,
                                        'test_loss': None,  # will be filled later
                                        'llm_name': llm_name,
                                        'params': [optimized_params],
                                        'initial_loss': initial_loss,
                                        'initial_params': [initial_params],
                                        'parent1_id': [parent1_id],
                                        'parent2_id': [parent2_id],
                                        'evaluation_matrix': [y_eval]
                                        })
            
            islands[island_idx] = pd.concat([islands[island_idx], new_program_df], ignore_index=True)
            census.append([i, island_idx, j, llm_name, loss, t_added, parent1_id, parent2_id, y_eval, optimized_params.shape[1]])
            success_rate += 1 / (n_islands * batch_size)
            print(f"iteration {i}, island {island_idx}, batch {j}, loss: {loss:.2f}")
            print('-' * 50)
            logging.info("-" * 50)
        print("Success rate:", success_rate)

        # sort each island by loss
        for island_idx in range(n_islands):
            islands[island_idx] = islands[island_idx].sort_values(by='train_loss').reset_index(drop=True)
        logging.info(f"Iteration {i} complete. The proportion of programs that successfully ran and received a loss is {success_rate:.2f}.")
        logging.info('-' * 50)
        # migrate and prune programs (better here for temperature to be in [0, 1] range)
        islands = genetic_helpers.perform_island_deduplication(islands, overlap_threshold=int(0.75 * critical_population_size))
        islands = genetic_helpers.perform_population_pruning(islands, critical_population_size=critical_population_size - n_migrants,
                                                min_wise_population_size=min_wise_population_size,)
        islands = genetic_helpers.perform_probabilistic_migration(islands, 
                                                                  n_migrants=n_migrants,
                                                                  destination_islands=exploration_topology if mode == 'explore' else exploitation_topology, 
                                                                  temperature=(temperature - 1.0)**4)

                                                             
        # save diagnostics
        iteration_dir = os.path.join(full_dir, 'iteration_updates', f'iteration_{i}')
        os.makedirs(iteration_dir, exist_ok=True)
        for island_idx in range(n_islands):
            pg_info = islands[island_idx][['iteration_number', 'birth_island', 'batch_index', 'train_loss']].to_string(index=False, header=False)
            print(f"Iter {i}, Island {island_idx} programs:\n{pg_info}\n")
            logging.info(f"Iter {i}, Island {island_idx} programs:\n{pg_info}\n")
        
            # Save plots of top programs
            top_df = islands[island_idx].sort_values(by='train_loss').head(3).reset_index(drop=True)
            top_df = top_df.sort_values(by='train_loss', ascending=False).reset_index(drop=True)
            sup_title = f"Iteration {i}, Island {island_idx}, Top {len(top_df)} Programs\n"
            sup_title += "\n".join([f"model {j+1}: iter {top_df['iteration_number'][j]}, birth island {top_df['birth_island'][j]}, batch {top_df['batch_index'][j]}, loss: {top_df['train_loss'][j]:.2f}" for j in range(len(top_df))])
            diagnostic.plot_model_fits(
                programs_df=top_df,
                loss_function=loss_functions.quadratic_loss,
                x=angles_train,
                y=response_train,
                cell_selection=np.random.choice(response_train.shape[0], size=9, replace=False),
                title=sup_title,
                save_path=os.path.join(iteration_dir, f'island_{island_idx}_top_programs.png'),
                dpi=300.0)
        
        all_programs = pd.concat([islands[idx] for idx in range(n_islands)], ignore_index=True)
        top_programs = all_programs.sort_values(by='train_loss').head(3).reset_index(drop=True)
        top_programs = top_programs.sort_values(by='train_loss', ascending=False).reset_index(drop=True)
        sup_title = f"Iteration {i}, Top 3 Programs Overall\n"
        sup_title += "\n".join([f"model {j+1}: iter {top_programs['iteration_number'][j]}, birth island {top_programs['birth_island'][j]}, batch {top_programs['batch_index'][j]}, loss: {top_programs['train_loss'][j]:.2f}" for j in range(len(top_programs))])
        diagnostic.plot_model_fits(
            programs_df=top_programs,
            loss_function=loss_functions.quadratic_loss,
            x=angles_train,
            y=response_train,
            cell_selection=np.random.choice(response_train.shape[0], size=9, replace=False),
            title=sup_title,
            save_path=os.path.join(iteration_dir, 'top_programs_overall.png'),
            dpi=300.0)
        
        # save census
        census_path = os.path.join(iteration_dir, 'census.npy')
        census_np = np.array(census, dtype=object)
        np.save(census_path, census_np)

    # -----------------------------
    # now carry out the loss calculation on the test cells
    logging.info("Calculating loss on test set...")
    for island_idx in range(n_islands):
        logging.info(f"Island {island_idx} programs:")
        for j in range(len(islands[island_idx])):
            program = islands[island_idx].iloc[j]
            neuron_model = program['program']
            param_estimator = program['parameter_estimator']
            # compute the test loss
            _, _, test_loss, optimized_params = objective(neuron_model, param_estimator,
                                                          loss_func=loss_functions.quadratic_loss,
                                                          x=angles_test, y=response_test, fit_params=fit_params,
                                                          max_iter=2_000, 
                                                          param_penalty_weight=param_penalty_weight, tol=tol,
                                                          use_param_estimator=use_param_estimator)
            islands[island_idx].at[j, 'test_loss'] = test_loss
            islands[island_idx].at[j, 'params'] = optimized_params
            islands[island_idx].at[j, 'mean_loss'] = np.mean(test_loss)
            print(f"Test loss: {test_loss:.2f}")

    # group all islands together and save
    combined_dir = os.path.join(base_dir, date_stamp, time_stamp, 'combined')
    os.makedirs(combined_dir, exist_ok=True)
    combined_programs_dataframe = pd.concat(islands, ignore_index=True)
    combined_programs_dataframe = genetic_helpers.remove_duplicates(combined_programs_dataframe, mode='complicated', loss_tol=0.025, cosine_tol=0.99, loss_type='test_loss')
    # combined_programs_dataframe = combined_programs_dataframe.sort_values(by='test_loss').reset_index(drop=True)
    # sort by mean loss
    combined_programs_dataframe = combined_programs_dataframe.sort_values(by='mean_loss').reset_index(drop=True)
    # save the combined programs dataframe, reordering columns to have order:
    # iteration_number, birth_island, batch_index, train_loss, test_loss, program_code_string, parameter_estimator_code_string, program, parameter_estimator, params, parent1_id, parent2_id
    combined_programs_dataframe = combined_programs_dataframe[['iteration_number', 'birth_island', 'batch_index',
                                                                'train_loss', 'test_loss',
                                                                'program_code_string', 'parameter_estimator_code_string',
                                                                'program', 'parameter_estimator', 'params',
                                                                'parent1_id', 'parent2_id', 'llm_name']]
    combined_programs_dataframe.to_csv(os.path.join(combined_dir, 'programs_db.csv'), index=False)

    # save census npy array
    census_path = os.path.join(combined_dir, 'census.npy')
    census_np = np.array(census, dtype=object)
    np.save(census_path, census_np)

    # save island-specific results
    for island_id, island_df in enumerate(islands):
        island_dir = os.path.join(base_dir, date_stamp, time_stamp, f'island_{island_id}' if island_id < n_islands else 'meta_island')
        os.makedirs(island_dir, exist_ok=True)
        island_df.to_csv(os.path.join(island_dir, 'programs_db.csv'), index=False)

    # ---------------------------
    # save losses plot    
    diagnostic.plot_train_vs_test_loss(programs_df=combined_programs_dataframe,
                                       island_labels=[f'Island {i}' for i in range(n_islands)] + ['garden_of_eden'],
                                       save_path=os.path.join(combined_dir, 'train_vs_test_loss.png'))
    
    # ---------------------------
    df_list = [combined_programs_dataframe] + islands
    combined_dir = [os.path.join(base_dir, date_stamp, time_stamp, "combined")] 
    island_dirs = [os.path.join(base_dir, date_stamp, time_stamp, f'island_{i}') for i in range(n_islands)]
    df_dirs = combined_dir + island_dirs
    config_str = f"n_islands={n_islands}, batch_size={batch_size}, n_iterations={n_iterations},\n"
    config_str += f"llm_names={little_lm_name, large_lm_name}, fit_params={fit_params}, \n"
    config_str += f"critical_population_size={critical_population_size}.\n"

    for i, df in enumerate(df_list):
        df_sup = config_str
        df = df.head(3)
        df = df.sort_values(by='test_loss', ascending=False).reset_index(drop=True)
        df_sup += "".join([f"model {len(df) - i}: iter {df['iteration_number'][i]}, birth_island {df['birth_island'][i]}, batch {df['batch_index'][i]}, total loss {0.5 * (df['test_loss'][i] + df['train_loss'][i]):.2f}\n" for i in range(min(3, len(df)))])
        diagnostic.plot_model_fits(
            programs_df=df,
            loss_function=loss_functions.quadratic_loss,
            x=angles_test,
            y=response_test,
            cell_selection=np.random.choice(response_test.shape[0], size=9, replace=False),
            title=df_sup,
            save_path=os.path.join(df_dirs[i], 'top_model_fits.png')
        )
        # plot top 3 models separately
        for j in range(min(3, len(df))):
            birth_island = df['birth_island'][j]
            iteration_number = df['iteration_number'][j]
            batch_index = df['batch_index'][j]
            cell_selection = np.random.choice(response_test.shape[0], size=9, replace=False)
            diagnostic.plot_single_model_fit(
                model=df['program'][j],
                loss_function=loss_functions.quadratic_loss,
                x=angles_test[cell_selection],
                y=response_test[cell_selection],
                params=df['params'][j][cell_selection],
                title=f"Island {birth_island}, Iteration {iteration_number}, Batch {batch_index}, loss: {df['test_loss'][j]:.2f}",
                save_path=os.path.join(df_dirs[i], f'top_model_fit_{min(3, len(df)) - j}.png')
            )

if __name__ == "__main__":
    for i in range(4):
        print("running with standard params")
        asyncio.run(main(n_iterations=9, time_limit=60, use_image_feedback=True, use_large_every=0,
                         param_penalty_weight=0.01,
                         exploration_topology=[1, 2, 3, 4, 5, 6, 7, 0], exploit_point=0.7))