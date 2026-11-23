# Selected program from the smoke run (generation 0, island 2, batch 4), NumPy source as proposed.
import numpy as np

def neuron_model(theta, theta_pref=0.0, baseline=0.0, amplitude_1=1.0,
                    width_1=0.6, sharpness_1=2.0, skew_1=0.0,
                    amplitude_2=0.0, width_2=0.6, sharpness_2=2.0,
                    skew_2=0.0, second_offset=0.0,
                    broad_amplitude=0.0, broad_phase=0.0):
    """
    parent_model_1: circular_gaussian_single_lobe_model
    parent_model_2: antipodal_equal_width_two_gaussian_model

    Image analysis:
    In Cell 754, Cell 490, and Cell 88, the blue binned mean has one dominant
    lobe, and both parents already follow it well, so this model must retain a
    clean single-lobe limit. In Cell 456, Cell 523, Cell 436, and Cell 119, the
    green single-lobe parent fails by inflating baseline/width to bridge two
    separated lobes; the red two-lobe parent is much better because it keeps the
    troughs low while representing opposing peaks. However, the red parent still
    assumes both peaks have the same width and are exactly antipodal. Cell 456
    shows a tall sharp first peak and a weaker, broader second peak; Cell 881 and
    Cell 59 show weak broad shoulders/bumps that are not perfectly antipodal;
    Cell 119 and Cell 436 show shallow nonzero valleys and broad flank structure
    that the red curve slightly underestimates.

    This model improves on the parents by using two circular generalized
    Gaussian lobes with independent amplitudes, widths, tail/peak sharpness, and
    left/right skew. A small broad sinusoidal background captures slow baseline
    modulation and shallow valleys without forcing the main peaks to widen. The
    second lobe is near, but not rigidly fixed to, the antipode. For single-peak
    cells, amplitude_2=broad_amplitude=0, skew=0, and sharpness=2 recover a
    single circular Gaussian response.

    Equation:
      Let wrap(x) = atan2(sin(x), cos(x)), mu1 = theta_pref,
      mu2 = wrap(theta_pref + pi + second_offset),
      d_j = wrap(theta - mu_j),
      sigma_j(d) = clip(width_j * exp(skew_j * d / pi), 0.03, pi),
      lobe_j(d) = amplitude_j * exp(-0.5 * (abs(d) / sigma_j(d))^sharpness_j),
      background(theta) = broad_amplitude * (1 + cos(theta - broad_phase)) / 2.

      rate(theta) = baseline + background(theta) + lobe_1(d1) + lobe_2(d2)

    Free parameters:
      theta_pref, baseline, amplitude_1, width_1, sharpness_1, skew_1,
      amplitude_2, width_2, sharpness_2, skew_2, second_offset,
      broad_amplitude, broad_phase.
    """
    theta = np.asarray(theta, dtype=float)

    theta_pref = np.clip(theta_pref, 0.0, 2.0 * np.pi)
    baseline = np.clip(baseline, 0.0, 50.0)

    amplitude_1 = np.clip(amplitude_1, 0.0, 100.0)
    width_1 = np.clip(width_1, 0.03, np.pi)
    sharpness_1 = np.clip(sharpness_1, 0.7, 4.0)
    skew_1 = np.clip(skew_1, -1.0, 1.0)

    amplitude_2 = np.clip(amplitude_2, 0.0, 100.0)
    width_2 = np.clip(width_2, 0.03, np.pi)
    sharpness_2 = np.clip(sharpness_2, 0.7, 4.0)
    skew_2 = np.clip(skew_2, -1.0, 1.0)

    second_offset = np.clip(second_offset, -0.5 * np.pi, 0.5 * np.pi)

    broad_amplitude = np.clip(broad_amplitude, 0.0, 50.0)
    broad_phase = np.clip(broad_phase, 0.0, 2.0 * np.pi)

    def wrap(x):
        return np.arctan2(np.sin(x), np.cos(x))

    mu_1 = theta_pref
    mu_2 = wrap(theta_pref + np.pi + second_offset)

    d_1 = wrap(theta - mu_1)
    d_2 = wrap(theta - mu_2)

    sigma_1 = np.clip(width_1 * np.exp(skew_1 * d_1 / np.pi), 0.03, np.pi)
    sigma_2 = np.clip(width_2 * np.exp(skew_2 * d_2 / np.pi), 0.03, np.pi)

    lobe_1 = amplitude_1 * np.exp(
        -0.5 * np.power(np.abs(d_1) / sigma_1, sharpness_1)
    )
    lobe_2 = amplitude_2 * np.exp(
        -0.5 * np.power(np.abs(d_2) / sigma_2, sharpness_2)
    )

    broad_background = broad_amplitude * (
        0.5 + 0.5 * np.cos(theta - broad_phase)
    )

    return baseline + broad_background + lobe_1 + lobe_2

# JAX translation used for fitting
# import jax
# import jax.numpy as jnp
# 
# def neuron_model(theta, theta_pref=0.0, baseline=0.0, amplitude_1=1.0,
#                     width_1=0.6, sharpness_1=2.0, skew_1=0.0,
#                     amplitude_2=0.0, width_2=0.6, sharpness_2=2.0,
#                     skew_2=0.0, second_offset=0.0,
#                     broad_amplitude=0.0, broad_phase=0.0):
#     theta = jnp.asarray(theta, dtype=jnp.float32)
# 
#     theta_pref = jnp.clip(theta_pref, 0.0, 2.0 * jnp.pi)
#     baseline = jnp.clip(baseline, 0.0, 50.0)
# 
#     amplitude_1 = jnp.clip(amplitude_1, 0.0, 100.0)
#     width_1 = jnp.clip(width_1, 0.03, jnp.pi)
#     sharpness_1 = jnp.clip(sharpness_1, 0.7, 4.0)
#     skew_1 = jnp.clip(skew_1, -1.0, 1.0)
# 
#     amplitude_2 = jnp.clip(amplitude_2, 0.0, 100.0)
#     width_2 = jnp.clip(width_2, 0.03, jnp.pi)
#     sharpness_2 = jnp.clip(sharpness_2, 0.7, 4.0)
#     skew_2 = jnp.clip(skew_2, -1.0, 1.0)
# 
#     second_offset = jnp.clip(second_offset, -0.5 * jnp.pi, 0.5 * jnp.pi)
# 
#     broad_amplitude = jnp.clip(broad_amplitude, 0.0, 50.0)
#     broad_phase = jnp.clip(broad_phase, 0.0, 2.0 * jnp.pi)
# 
#     def wrap(x):
#         return jnp.arctan2(jnp.sin(x), jnp.cos(x))
# 
#     mu_1 = theta_pref
#     mu_2 = wrap(theta_pref + jnp.pi + second_offset)
# 
#     d_1 = wrap(theta - mu_1)
#     d_2 = wrap(theta - mu_2)
# 
#     sigma_1 = jnp.clip(width_1 * jnp.exp(skew_1 * d_1 / jnp.pi), 0.03, jnp.pi)
#     sigma_2 = jnp.clip(width_2 * jnp.exp(skew_2 * d_2 / jnp.pi), 0.03, jnp.pi)
# 
#     lobe_1 = amplitude_1 * jnp.exp(
#         -0.5 * jnp.power(jnp.abs(d_1) / sigma_1, sharpness_1)
#     )
#     lobe_2 = amplitude_2 * jnp.exp(
#         -0.5 * jnp.power(jnp.abs(d_2) / sigma_2, sharpness_2)
#     )
# 
#     broad_background = broad_amplitude * (
#         0.5 + 0.5 * jnp.cos(theta - broad_phase)
#     )
# 
#     return baseline + broad_background + lobe_1 + lobe_2
