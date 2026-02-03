"""
Test the objective function to ensure parameter estimation is working correctly.

This test suite checks:
1. Whether compute_initial_params produces reasonable outputs
2. Whether the objective function handles data correctly
3. Whether there are any data transformation issues
4. Whether parameter estimators produce values in the expected range
5. Whether LLM-generated parameter estimators (chat vs legacy) produce reasonable code
"""

import numpy as np
import jax.numpy as jnp
import sys
import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hypothesis_engine import compute_initial_params, objective, compute_default_params
from src import loss_functions, llm_helper, utils
from src.prompt_manager import PromptManager
from experiments.orientation_tuning import seed_programs
from google import genai


def test_parameter_estimator_basic():
    """Test that seed parameter estimators produce reasonable outputs."""
    print("\n" + "="*80)
    print("TEST 1: Basic Parameter Estimator Output")
    print("="*80)
    
    # Create simple synthetic data
    n_samples = 10
    n_trials = 100
    
    # Generate angles uniformly
    theta = np.random.uniform(0, 2*np.pi, n_trials)
    
    # Generate synthetic responses using known parameters
    true_theta_pref = np.pi
    true_baseline = 1.0
    true_amplitude = 5.0
    true_tuning_width = 0.8
    
    # Use the model to generate clean data
    X = theta.reshape(1, -1)  # (1, n_trials)
    response = seed_programs.neuron_model_1(X, true_theta_pref, true_baseline, 
                                           true_amplitude, true_tuning_width)
    
    # Add some noise
    response = response + np.random.normal(0, 0.5, n_trials)
    response = np.clip(response, 0, None)  # Keep non-negative
    
    print(f"\nTrue parameters:")
    print(f"  theta_pref: {true_theta_pref:.3f}")
    print(f"  baseline: {true_baseline:.3f}")
    print(f"  amplitude: {true_amplitude:.3f}")
    print(f"  tuning_width: {true_tuning_width:.3f}")
    print(f"\nResponse stats:")
    print(f"  min: {response.min():.3f}, max: {response.max():.3f}, mean: {response.mean():.3f}")
    
    # Test parameter estimator
    estimated_params = seed_programs.parameter_estimator_1(X, response)
    
    print(f"\nEstimated parameters:")
    print(f"  theta_pref: {estimated_params[0]:.3f}")
    print(f"  baseline: {estimated_params[1]:.3f}")
    print(f"  amplitude: {estimated_params[2]:.3f}")
    print(f"  tuning_width: {estimated_params[3]:.3f}")
    
    # Check that estimates are in reasonable range
    assert 0 <= estimated_params[0] <= 2*np.pi, f"theta_pref out of range: {estimated_params[0]}"
    assert 0 <= estimated_params[1] <= 100, f"baseline out of range: {estimated_params[1]}"
    assert 0 <= estimated_params[2] <= 100, f"amplitude out of range: {estimated_params[2]}"
    assert 0.01 <= estimated_params[3] <= 10, f"tuning_width out of range: {estimated_params[3]}"
    
    print("\n✓ Parameter estimator produces values in reasonable range")


def test_compute_initial_params_shape():
    """Test that compute_initial_params produces correct shapes."""
    print("\n" + "="*80)
    print("TEST 2: compute_initial_params Shape and Values")
    print("="*80)
    
    n_samples = 5
    n_trials = 80
    
    # Create data with shape (n_samples, n_features=1, n_trials)
    X_data = np.random.uniform(0, 2*np.pi, (n_samples, 1, n_trials))
    
    # Create synthetic responses
    y_data = np.zeros((n_samples, n_trials))
    for i in range(n_samples):
        # Each sample has slightly different parameters
        theta_pref = np.random.uniform(0, 2*np.pi)
        baseline = np.random.uniform(0.5, 2.0)
        amplitude = np.random.uniform(3.0, 8.0)
        tuning_width = np.random.uniform(0.5, 1.5)
        
        y_data[i] = seed_programs.neuron_model_1(
            X_data[i], theta_pref, baseline, amplitude, tuning_width
        ) + np.random.normal(0, 0.3, n_trials)
        y_data[i] = np.clip(y_data[i], 0, None)
    
    print(f"\nInput data shapes:")
    print(f"  X: {X_data.shape} (n_samples, n_features, n_trials)")
    print(f"  y: {y_data.shape} (n_samples, n_trials)")
    print(f"\nResponse stats per sample:")
    for i in range(n_samples):
        print(f"  Sample {i}: min={y_data[i].min():.2f}, max={y_data[i].max():.2f}, mean={y_data[i].mean():.2f}")
    
    # Call compute_initial_params
    initial_params = compute_initial_params(
        seed_programs.parameter_estimator_1,
        seed_programs.neuron_model_1_jax,
        X_data,
        y_data
    )
    
    print(f"\nInitial params shape: {initial_params.shape}")
    print(f"Expected shape: ({n_samples}, 4)")
    
    assert initial_params.shape == (n_samples, 4), \
        f"Wrong shape: {initial_params.shape} vs ({n_samples}, 4)"
    
    # Check parameter values
    print(f"\nEstimated parameters for each sample:")
    for i in range(n_samples):
        params = initial_params[i]
        print(f"  Sample {i}: theta_pref={params[0]:.3f}, baseline={params[1]:.3f}, "
              f"amplitude={params[2]:.3f}, tuning_width={params[3]:.3f}")
        
        # Sanity checks
        assert 0 <= params[0] <= 2*np.pi, f"Sample {i}: theta_pref out of range"
        assert 0 <= params[1] <= 100, f"Sample {i}: baseline out of range: {params[1]}"
        assert 0 <= params[2] <= 100, f"Sample {i}: amplitude out of range: {params[2]}"
        assert 0.01 <= params[3] <= 10, f"Sample {i}: tuning_width out of range: {params[3]}"
    
    print("\n✓ All parameters are in reasonable ranges")


def test_objective_with_seed_programs():
    """Test the objective function end-to-end with seed programs."""
    print("\n" + "="*80)
    print("TEST 3: Objective Function with Seed Programs")
    print("="*80)
    
    n_samples = 8
    n_trials = 100
    
    # Create data
    X_data = np.random.uniform(0, 2*np.pi, (n_samples, 1, n_trials))
    y_data = np.zeros((n_samples, n_trials))
    
    print(f"\nGenerating synthetic data for {n_samples} samples...")
    for i in range(n_samples):
        theta_pref = np.random.uniform(0, 2*np.pi)
        baseline = np.random.uniform(0.5, 2.0)
        amplitude = np.random.uniform(3.0, 8.0)
        tuning_width = np.random.uniform(0.5, 1.5)
        
        y_data[i] = seed_programs.neuron_model_1_jax(
            X_data[i], theta_pref, baseline, amplitude, tuning_width
        ) + np.random.normal(0, 0.3, n_trials)
        y_data[i] = np.clip(y_data[i], 0, None)
    
    print(f"Data generated. Response range: [{y_data.min():.2f}, {y_data.max():.2f}]")
    
    # Run objective function
    print(f"\nRunning objective function (fit_params=False to test param estimator only)...")
    initial_loss, initial_params, final_loss, final_params = objective(
        model=seed_programs.neuron_model_1_jax,
        param_estimator=seed_programs.parameter_estimator_1,
        loss_func=loss_functions.quadratic_loss,
        x=X_data,
        y=y_data,
        fit_params=False,  # Don't optimize, just test param estimator
        use_param_estimator=True,
        max_iter=100
    )
    
    print(f"\nResults:")
    print(f"  Initial loss: {initial_loss:.4f}")
    print(f"  Final loss: {final_loss:.4f}")
    print(f"  Initial params shape: {initial_params.shape}")
    
    # Check that loss is reasonable (not catastrophically high)
    print(f"\nChecking loss values...")
    assert initial_loss < 1000, f"Initial loss too high: {initial_loss} (expected < 1000)"
    assert final_loss < 1000, f"Final loss too high: {final_loss} (expected < 1000)"
    
    # Check parameter values
    print(f"\nParameter statistics:")
    param_names = ['theta_pref', 'baseline', 'amplitude', 'tuning_width']
    for j, name in enumerate(param_names):
        values = initial_params[:, j]
        print(f"  {name}: min={values.min():.3f}, max={values.max():.3f}, mean={values.mean():.3f}")
    
    # Check amplitudes specifically (this is where the problem appears in the images)
    amplitudes = initial_params[:, 2]
    print(f"\nAmplitude check (this is where the problem appears in the images):")
    print(f"  Data max firing rate: {y_data.max():.2f}")
    print(f"  Estimated amplitude range: [{amplitudes.min():.2f}, {amplitudes.max():.2f}]")
    print(f"  Estimated amplitude mean: {amplitudes.mean():.2f}")
    
    # The amplitude should be roughly in the same ballpark as the data max
    # If amplitudes are ~500 when data max is ~10, we have a problem!
    assert amplitudes.max() < y_data.max() * 50, \
        f"Amplitudes way too high! Max amplitude: {amplitudes.max():.2f}, Data max: {y_data.max():.2f}"
    
    print("\n✓ Objective function produces reasonable results with seed programs")


def test_with_normalized_data():
    """Test with normalized data similar to what's used in practice."""
    print("\n" + "="*80)
    print("TEST 4: With Normalized Data (matching actual pipeline)")
    print("="*80)
    
    n_samples = 10
    n_trials = 100
    
    # Create data
    X_data = np.random.uniform(0, 2*np.pi, (n_samples, 1, n_trials))
    y_data = np.zeros((n_samples, n_trials))
    
    for i in range(n_samples):
        theta_pref = np.random.uniform(0, 2*np.pi)
        baseline = np.random.uniform(0.5, 2.0)
        amplitude = np.random.uniform(3.0, 8.0)
        tuning_width = np.random.uniform(0.5, 1.5)
        
        y_data[i] = seed_programs.neuron_model_1_jax(
            X_data[i], theta_pref, baseline, amplitude, tuning_width
        ) + np.random.normal(0, 0.3, n_trials)
        y_data[i] = np.clip(y_data[i], 0, None)
    
    # Apply the same normalization as in data_parser.py
    print(f"\nBefore normalization:")
    print(f"  Response range: [{y_data.min():.2f}, {y_data.max():.2f}]")
    print(f"  Response mean: {y_data.mean():.2f}")
    
    # Normalize: multiply by 100 and divide by L2 norm
    y_normalized = 100 * y_data / np.linalg.norm(y_data, axis=1, keepdims=True)
    
    print(f"\nAfter normalization:")
    print(f"  Response range: [{y_normalized.min():.2f}, {y_normalized.max():.2f}]")
    print(f"  Response mean: {y_normalized.mean():.2f}")
    
    # Run objective with normalized data
    print(f"\nRunning objective function with normalized data...")
    initial_loss, initial_params, final_loss, final_params = objective(
        model=seed_programs.neuron_model_1_jax,
        param_estimator=seed_programs.parameter_estimator_1,
        loss_func=loss_functions.quadratic_loss,
        x=X_data,
        y=y_normalized,
        fit_params=False,
        use_param_estimator=True,
        max_iter=100
    )
    
    print(f"\nResults with normalized data:")
    print(f"  Initial loss: {initial_loss:.4f}")
    print(f"  Final loss: {final_loss:.4f}")
    
    # Check amplitudes
    amplitudes = initial_params[:, 2]
    baselines = initial_params[:, 1]
    
    print(f"\nEstimated parameters on normalized data:")
    print(f"  Baseline range: [{baselines.min():.2f}, {baselines.max():.2f}]")
    print(f"  Amplitude range: [{amplitudes.min():.2f}, {amplitudes.max():.2f}]")
    print(f"  Normalized data range: [{y_normalized.min():.2f}, {y_normalized.max():.2f}]")
    
    # The amplitudes should still be reasonable relative to normalized data
    assert amplitudes.max() < y_normalized.max() * 10, \
        f"Amplitudes too high even after normalization! Amplitude max: {amplitudes.max():.2f}"
    
    assert initial_loss < 1000, f"Loss too high with normalized data: {initial_loss}"
    
    print("\n✓ Objective function handles normalized data correctly")


def test_parameter_scale_mismatch():
    """Test if there's a scaling issue between what param_estimator returns and what model expects."""
    print("\n" + "="*80)
    print("TEST 5: Parameter Scale Mismatch Detection")
    print("="*80)
    
    # Create a single sample with known properties
    n_trials = 200
    X = np.random.uniform(0, 2*np.pi, (1, n_trials))
    
    # Generate data with known params
    true_params = {
        'theta_pref': 1.5,
        'baseline': 1.0,
        'amplitude': 5.0,
        'tuning_width': 0.8
    }
    
    y = seed_programs.neuron_model_1(X, **true_params)
    y = y + np.random.normal(0, 0.2, n_trials)
    y = np.clip(y, 0, None)
    
    print(f"\nTrue parameters: {true_params}")
    print(f"Data range: [{y.min():.2f}, {y.max():.2f}], mean: {y.mean():.2f}")
    
    # Get estimated parameters
    est_params = seed_programs.parameter_estimator_1(X, y)
    
    print(f"\nEstimated parameters: theta_pref={est_params[0]:.3f}, baseline={est_params[1]:.3f}, "
          f"amplitude={est_params[2]:.3f}, tuning_width={est_params[3]:.3f}")
    
    # Evaluate model with estimated parameters
    y_pred = seed_programs.neuron_model_1(X, *est_params)
    
    print(f"\nModel prediction with estimated params:")
    print(f"  Predicted range: [{y_pred.min():.2f}, {y_pred.max():.2f}], mean: {y_pred.mean():.2f}")
    print(f"  Actual range: [{y.min():.2f}, {y.max():.2f}], mean: {y.mean():.2f}")
    print(f"  Ratio of max values: {y_pred.max() / y.max():.2f}")
    
    # If there's a huge mismatch, we have a problem
    ratio = y_pred.max() / y.max()
    assert 0.1 < ratio < 10.0, \
        f"Huge scale mismatch! Predicted max / actual max = {ratio:.2f}"
    
    # Calculate loss
    mse = np.mean((y_pred - y)**2)
    print(f"  MSE: {mse:.4f}")
    
    print("\n✓ No obvious parameter scale mismatch")


async def test_llm_generated_parameter_estimator():
    """Test that LLM-generated parameter estimators produce reasonable code and values.
    
    This test simulates multiple iterations with chat mode to see if quality degrades
    over time as chat history accumulates.
    """
    print("\n" + "="*80)
    print("TEST 6: LLM-Generated Parameter Estimator (Chat Mode Over Multiple Iterations)")
    print("="*80)
    
    # Load API key
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("⚠️  SKIPPING: No GOOGLE_API_KEY found in environment")
        return
    
    client = genai.Client(api_key=api_key)
    
    # Load prompt manager
    prompts_config_path = Path(__file__).parent.parent / "experiments" / "orientation_tuning" / "configs" / "prompts.yaml"
    prompt_manager = PromptManager(config_path=prompts_config_path)
    
    # Create synthetic data for testing
    n_samples = 5
    n_trials = 100
    X_data = np.random.uniform(0, 2*np.pi, (n_samples, 1, n_trials))
    y_data = np.zeros((n_samples, n_trials))
    
    for i in range(n_samples):
        theta_pref = np.random.uniform(0, 2*np.pi)
        baseline = np.random.uniform(0.5, 2.0)
        amplitude = np.random.uniform(3.0, 8.0)
        tuning_width = np.random.uniform(0.5, 1.5)
        
        y_data[i] = seed_programs.neuron_model_1_jax(
            X_data[i], theta_pref, baseline, amplitude, tuning_width
        ) + np.random.normal(0, 0.3, n_trials)
        y_data[i] = np.clip(y_data[i], 0, None)
    
    # Normalize data as in the actual pipeline
    y_normalized = 100 * y_data / np.linalg.norm(y_data, axis=1, keepdims=True)
    
    print(f"\nNormalized data range: [{y_normalized.min():.2f}, {y_normalized.max():.2f}]")
    print(f"Normalized data mean: {y_normalized.mean():.2f}")
    
    # Create a mock island with seed programs
    initial_params_1 = compute_initial_params(
        seed_programs.parameter_estimator_1,
        seed_programs.neuron_model_1_jax,
        X_data, y_data
    )
    initial_params_2 = compute_initial_params(
        seed_programs.parameter_estimator_2,
        seed_programs.neuron_model_2_jax,
        X_data, y_data
    )
    
    # Score the seed programs
    _, _, loss_1, _ = objective(
        seed_programs.neuron_model_1_jax, seed_programs.parameter_estimator_1,
        loss_functions.quadratic_loss, X_data, y_normalized,
        fit_params=False, use_param_estimator=True
    )
    _, _, loss_2, _ = objective(
        seed_programs.neuron_model_2_jax, seed_programs.parameter_estimator_2,
        loss_functions.quadratic_loss, X_data, y_normalized,
        fit_params=False, use_param_estimator=True
    )
    
    # Format seed program code
    model_name = prompt_manager.get_model_name()
    program_1_code = utils.format_function_source(
        seed_programs.neuron_model_1, f'{model_name}_v1', 'import numpy as np'
    )
    program_2_code = utils.format_function_source(
        seed_programs.neuron_model_2, f'{model_name}_v2', 'import numpy as np'
    )
    param_est_1_code = utils.format_function_source(
        seed_programs.parameter_estimator_1, 'parameter_estimator_v1', 'import numpy as np'
    )
    param_est_2_code = utils.format_function_source(
        seed_programs.parameter_estimator_2, 'parameter_estimator_v2', 'import numpy as np'
    )
    
    mock_island = pd.DataFrame({
        'program_code_string': [program_1_code, program_2_code],
        'parameter_estimator_code_string': [param_est_1_code, param_est_2_code],
        'train_loss': [loss_1, loss_2],
        'initial_params': [initial_params_1, initial_params_2]
    })
    
    # Sample programs (sorted worst to best)
    random_programs = mock_island.sort_values(by='train_loss', ascending=False).reset_index(drop=True)
    
    # Model code for the new parameter estimator (just use neuron_model_1 as the "new" model)
    new_model_code = program_1_code
    
    
    # Create IslandChatManager (this is what's used in practice)
    print(f"\nCreating IslandChatManager with explore mode system instruction...")
    chat_manager = llm_helper.IslandChatManager(
        client=client,
        get_system_instruction=prompt_manager.get_system_instruction,
        small_model_name='gemini-2.0-flash',
        large_model_name='gemini-2.5-flash',
        explore_temperature=1.5,
        exploit_temperature=0.7,
        thinking_budget_fraction=1.0,
        chat_token_limit=50000,
        batch_size=2
    )
    
    print(f"Testing chat mode with MULTIPLE iterations to simulate real usage...")
    print(f"(In practice, the problem may occur after chat history accumulates)\n")
    
    # Simulate 3 iterations
    n_iterations = 3
    island_id = 0
    batch_id = 0
    
    all_results = []
    
    for iteration in range(n_iterations):
        print(f"\n{'='*70}")
        print(f"ITERATION {iteration}: Simulating parameter estimator generation")
        print(f"{'='*70}")
        
        # Create fresh synthetic data for this iteration
        n_samples = 5
        n_trials = 100
        X_data = np.random.uniform(0, 2*np.pi, (n_samples, 1, n_trials))
        y_data = np.zeros((n_samples, n_trials))
        
        for i in range(n_samples):
            theta_pref = np.random.uniform(0, 2*np.pi)
            baseline = np.random.uniform(0.5, 2.0)
            amplitude = np.random.uniform(3.0, 8.0)
            tuning_width = np.random.uniform(0.5, 1.5)
            
            y_data[i] = seed_programs.neuron_model_1_jax(
                X_data[i], theta_pref, baseline, amplitude, tuning_width
            ) + np.random.normal(0, 0.3, n_trials)
            y_data[i] = np.clip(y_data[i], 0, None)
        
        # Normalize data as in the actual pipeline
        y_normalized = 100 * y_data / np.linalg.norm(y_data, axis=1, keepdims=True)
        
        print(f"Normalized data range: [{y_normalized.min():.2f}, {y_normalized.max():.2f}]")
        
        # Create mock island (would have accumulated programs from previous iterations)
        initial_params_1 = compute_initial_params(
            seed_programs.parameter_estimator_1,
            seed_programs.neuron_model_1_jax,
            X_data, y_data
        )
        initial_params_2 = compute_initial_params(
            seed_programs.parameter_estimator_2,
            seed_programs.neuron_model_2_jax,
            X_data, y_data
        )
        
        _, _, loss_1, _ = objective(
            seed_programs.neuron_model_1_jax, seed_programs.parameter_estimator_1,
            loss_functions.quadratic_loss, X_data, y_normalized,
            fit_params=False, use_param_estimator=True
        )
        _, _, loss_2, _ = objective(
            seed_programs.neuron_model_2_jax, seed_programs.parameter_estimator_2,
            loss_functions.quadratic_loss, X_data, y_normalized,
            fit_params=False, use_param_estimator=True
        )
        
        model_name = prompt_manager.get_model_name()
        program_1_code = utils.format_function_source(
            seed_programs.neuron_model_1, f'{model_name}_v1', 'import numpy as np'
        )
        program_2_code = utils.format_function_source(
            seed_programs.neuron_model_2, f'{model_name}_v2', 'import numpy as np'
        )
        param_est_1_code = utils.format_function_source(
            seed_programs.parameter_estimator_1, 'parameter_estimator_v1', 'import numpy as np'
        )
        param_est_2_code = utils.format_function_source(
            seed_programs.parameter_estimator_2, 'parameter_estimator_v2', 'import numpy as np'
        )
        
        mock_island = pd.DataFrame({
            'program_code_string': [program_1_code, program_2_code],
            'parameter_estimator_code_string': [param_est_1_code, param_est_2_code],
            'train_loss': [loss_1, loss_2],
            'initial_params': [initial_params_1, initial_params_2]
        })
        
        random_programs = mock_island.sort_values(by='train_loss', ascending=False).reset_index(drop=True)
        new_model_code = program_1_code  # The "new" model to create param estimator for
        
        print(f"Mock island losses: {loss_1:.2f}, {loss_2:.2f}")
        
        # Generate parameter estimator using CHAT MODE (get_parameter_estimator_prompt)
        chat_prompt = prompt_manager.get_parameter_estimator_prompt(
            random_programs, model_code_string=new_model_code,
            max_lines=100, use_image=False
        )
        
        print(f"Chat prompt length: {len(chat_prompt)} chars (relies on system instruction)")
        print(f"Calling LLM via chat manager (iteration {iteration})...")
        
        # Use the SAME chat session across iterations (simulating real usage)
        chat_output = await chat_manager.ask_island(
            island_id=island_id, prompt=chat_prompt, batch_id=batch_id,
            mode='explore', use_large_model=False
        )
        
        chat_code = utils.extract_code_block(chat_output)
        if not chat_code:
            print(f"✗ Failed to extract code from LLM output")
            continue
            
        chat_code = chat_code.replace('def parameter_estimator_v3(', 'def parameter_estimator(')
        print(f"✓ Generated parameter estimator ({len(chat_code)} chars)")
        
        # Test the generated parameter estimator
        try:
            param_est_func = utils.str_to_func(chat_code, 'parameter_estimator')
            
            # Test on all samples
            test_X = X_data[0]
            test_y = y_normalized[0]
            
            estimated_params = param_est_func(test_X, test_y)
            
            print(f"\nEstimated params: {estimated_params}")
            print(f"Data range: [{test_y.min():.2f}, {test_y.max():.2f}]")
            
            if len(estimated_params) >= 3:
                baseline = estimated_params[1] if len(estimated_params) > 1 else 0
                amplitude = estimated_params[2] if len(estimated_params) > 2 else 0
                
                print(f"Baseline: {baseline:.2f}, Amplitude: {amplitude:.2f}")
                
                # Check for the scaling issue
                amplitude_ratio = amplitude / test_y.max()
                
                if amplitude > test_y.max() * 50:
                    print(f"❌ CRITICAL: Amplitude ({amplitude:.2f}) is WAY too high!")
                    print(f"   Data max: {test_y.max():.2f}, Ratio: {amplitude_ratio:.1f}x")
                    status = "CRITICAL_ERROR"
                elif amplitude > test_y.max() * 10:
                    print(f"⚠️  WARNING: Amplitude ({amplitude:.2f}) is too high")
                    print(f"   Data max: {test_y.max():.2f}, Ratio: {amplitude_ratio:.1f}x")
                    status = "WARNING"
                else:
                    print(f"✓ Amplitude is reasonable (ratio: {amplitude_ratio:.2f}x)")
                    status = "OK"
                
                # Test in objective function
                try:
                    initial_loss, initial_params_all, _, _ = objective(
                        seed_programs.neuron_model_1_jax, param_est_func,
                        loss_functions.quadratic_loss, X_data[:3], y_normalized[:3],
                        fit_params=False, use_param_estimator=True
                    )
                    
                    print(f"Initial loss in objective: {initial_loss:.4f}")
                    
                    if initial_loss > 10000:
                        print(f"❌ CRITICAL: Loss is catastrophically high!")
                        status = "CRITICAL_ERROR"
                    elif initial_loss > 1000:
                        print(f"⚠️  WARNING: Loss is very high")
                        status = "WARNING"
                    else:
                        print(f"✓ Loss is reasonable")
                    
                    all_results.append({
                        'iteration': iteration,
                        'amplitude': amplitude,
                        'amplitude_ratio': amplitude_ratio,
                        'data_max': test_y.max(),
                        'initial_loss': initial_loss,
                        'status': status
                    })
                    
                except Exception as e:
                    print(f"✗ Failed in objective function: {e}")
                    all_results.append({
                        'iteration': iteration,
                        'amplitude': amplitude,
                        'amplitude_ratio': amplitude_ratio,
                        'data_max': test_y.max(),
                        'initial_loss': float('inf'),
                        'status': "OBJECTIVE_ERROR"
                    })
                    
            else:
                print(f"✗ Unexpected number of parameters: {len(estimated_params)}")
                all_results.append({
                    'iteration': iteration,
                    'status': "PARAM_COUNT_ERROR"
                })
                
        except Exception as e:
            print(f"✗ Error creating/testing parameter estimator: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                'iteration': iteration,
                'status': "CREATION_ERROR"
            })
    
    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY: Chat Mode Parameter Estimator Quality Across Iterations")
    print(f"{'='*70}")
    
    for result in all_results:
        if 'amplitude_ratio' in result:
            print(f"Iteration {result['iteration']}: "
                  f"Amplitude ratio: {result['amplitude_ratio']:.2f}x, "
                  f"Loss: {result['initial_loss']:.2f}, "
                  f"Status: {result['status']}")
        else:
            print(f"Iteration {result['iteration']}: Status: {result['status']}")
    
    # Check if quality degraded
    critical_errors = sum(1 for r in all_results if r.get('status') == 'CRITICAL_ERROR')
    warnings = sum(1 for r in all_results if r.get('status') == 'WARNING')
    
    if critical_errors > 0:
        print(f"\n❌ {critical_errors} CRITICAL ERROR(S) detected - parameter estimation failed catastrophically!")
    elif warnings > 0:
        print(f"\n⚠️  {warnings} warning(s) detected - some parameter estimates were suboptimal")
    else:
        print(f"\n✓ All iterations produced reasonable parameter estimators")
    
    print("\n✓ Chat mode parameter estimator test complete")


if __name__ == "__main__":
    import asyncio
    
    print("\n" + "="*80)
    print("TESTING OBJECTIVE FUNCTION AND PARAMETER ESTIMATION")
    print("="*80)
    
    try:
        test_parameter_estimator_basic()
        test_compute_initial_params_shape()
        test_objective_with_seed_programs()
        test_with_normalized_data()
        test_parameter_scale_mismatch()
        
        # Run async test
        asyncio.run(test_llm_generated_parameter_estimator())
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED ✓")
        print("="*80)
        print("\nThe objective function appears to be working correctly.")
        print("Check the LLM-generated parameter estimator results above to see")
        print("if there are differences between chat mode and legacy mode.")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        print("\nThis test failure indicates where the problem is!")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise
