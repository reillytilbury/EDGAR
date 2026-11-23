# Selected program from the fast run (1_2_5), NumPy source as proposed.
import numpy as np

def neuron_model(theta, theta_pref=0.0, baseline=0.0, amplitude_1=1.0, amplitude_2=0.0,
                    tuning_width=1.0, width_2_ratio=1.0, sharpness=2.0, second_offset=0.0,
                    skew=0.0):
    """
    parent_model_1: flexible_asymmetric_double_lobe_model
    parent_model_2: symmetric_double_gaussian_model

    Image analysis:
    Across the subplots, the symmetric double Gaussian parent is usually the
    stronger baseline: it tracks the sky-blue binned mean better than the more
    complex flexible parent in Cell 90, Cell 23, Cell 128, Cell 17, and Cell 13.
    Its advantage is especially clear where the response is a narrow, nearly
    symmetric peak: Cell 17 and Cell 13 show the red curve matching the peak
    height and width well, while the green flexible parent is too broad or has
    an inappropriate plateau. Cell 128 also shows that the red model captures
    the wrap-around peak near theta=0/2pi better than the green model.

    However, the symmetric parent has systematic weaknesses. In Cell 44, the
    red second lobe is too large near 5.4 rad while the blue mean is smaller,
    and the green model has an implausible elevated baseline between peaks. In
    Cell 37 and Cell 23, the two lobes have different amplitudes and widths,
    but the red parent is forced to share one width. In Cell 118, the blue mean
    has three sharp boundary/central lobes; both parents miss parts of this,
    but the red parent is much less distorted away from the peaks. In Cell 103
    the blue mean is noisy and asymmetric, and both models are too smooth, but
    the red curve follows the dominant two-lobe structure better. Cell 17 and
    Cell 13 also show that the green model's extra skew/exponent flexibility can
    produce flat-topped or shifted peaks that do not match the smooth blue mean.

    Therefore this model keeps the simple, robust two-Gaussian circular form of
    the better parent, but adds only the low-risk flexibility supported by the
    image: independent second-lobe width, a mild shared peak sharpness, a small
    second-peak offset, and a mild shared skew. It avoids the over-flexible
    per-lobe exponent/skew parameterization of the flexible parent, which caused
    unstable shapes and excessive baseline/shoulder errors.

    Equation:
        d1 = wrap(theta - theta_pref)
        d2 = wrap(theta - theta_pref - pi - second_offset)

        w1 = tuning_width
        w2 = tuning_width * width_2_ratio
        wj_eff = wj * (1 + skew * tanh(dj / 0.35))

        R(theta) = baseline
                 + amplitude_1 * exp(-0.5 * abs(d1 / w1_eff) ** sharpness)
                 + amplitude_2 * exp(-0.5 * abs(d2 / w2_eff) ** sharpness)

    Free parameters: theta_pref, baseline, amplitude_1, amplitude_2,
    tuning_width, width_2_ratio, sharpness, second_offset, skew.

    Improvement:
    This model preserves the strong symmetric Gaussian fit while correcting its
    main failure modes: unequal lobe amplitudes are already allowed by the
    better parent, and this model additionally allows unequal lobe widths,
    modest non-Gaussian sharpness, slight non-antipodal spacing, and mild
    asymmetry. The parameter ranges are intentionally conservative to avoid the
    green parent's failure in Cell 44, Cell 17, and Cell 13, where excessive
    flexibility produced broad shoulders, plateaus, and poor baselines.
    """
    theta = np.asarray(theta, dtype=float)

    theta_pref = np.mod(np.clip(float(theta_pref), 0.0, 2.0 * np.pi), 2.0 * np.pi)
    baseline = np.clip(float(baseline), 0.0, 20.0)
    amplitude_1 = np.clip(float(amplitude_1), 0.0, 30.0)
    amplitude_2 = np.clip(float(amplitude_2), 0.0, 30.0)
    tuning_width = np.clip(float(tuning_width), 0.08, 2.5)
    width_2_ratio = np.clip(float(width_2_ratio), 0.35, 2.5)
    sharpness = np.clip(float(sharpness), 1.4, 3.2)
    second_offset = np.clip(float(second_offset), -0.35, 0.35)
    skew = np.clip(float(skew), -0.35, 0.35)

    def circular_difference(x):
        return np.arctan2(np.sin(x), np.cos(x))

    def lobe(d, amp, width):
        effective_width = width * (1.0 + skew * np.tanh(d / 0.35))
        effective_width = np.maximum(effective_width, 1e-3)
        return amp * np.exp(-0.5 * np.abs(d / effective_width) ** sharpness)

    second_center = theta_pref + np.pi + second_offset
    dist_1 = circular_difference(theta - theta_pref)
    dist_2 = circular_difference(theta - second_center)

    return (
        baseline
        + lobe(dist_1, amplitude_1, tuning_width)
        + lobe(dist_2, amplitude_2, tuning_width * width_2_ratio)
    )

# JAX translation used for fitting
# import jax
# import jax.numpy as jnp
# 
# 
# def neuron_model(theta, theta_pref=0.0, baseline=0.0, amplitude_1=1.0, amplitude_2=0.0,
#                  tuning_width=1.0, width_2_ratio=1.0, sharpness=2.0, second_offset=0.0,
#                  skew=0.0):
#     theta = jnp.asarray(theta, dtype=jnp.float32)
# 
#     theta_pref = jnp.mod(jnp.clip(theta_pref, 0.0, 2.0 * jnp.pi), 2.0 * jnp.pi)
#     baseline = jnp.clip(baseline, 0.0, 20.0)
#     amplitude_1 = jnp.clip(amplitude_1, 0.0, 30.0)
#     amplitude_2 = jnp.clip(amplitude_2, 0.0, 30.0)
#     tuning_width = jnp.clip(tuning_width, 0.08, 2.5)
#     width_2_ratio = jnp.clip(width_2_ratio, 0.35, 2.5)
#     sharpness = jnp.clip(sharpness, 1.4, 3.2)
#     second_offset = jnp.clip(second_offset, -0.35, 0.35)
#     skew = jnp.clip(skew, -0.35, 0.35)
# 
#     def circular_difference(x):
#         return jnp.arctan2(jnp.sin(x), jnp.cos(x))
# 
#     def lobe(d, amp, width):
#         effective_width = width * (1.0 + skew * jnp.tanh(d / 0.35))
#         effective_width = jnp.maximum(effective_width, 1e-3)
#         return amp * jnp.exp(-0.5 * jnp.abs(d / effective_width) ** sharpness)
# 
#     second_center = theta_pref + jnp.pi + second_offset
#     dist_1 = circular_difference(theta - theta_pref)
#     dist_2 = circular_difference(theta - second_center)
# 
#     return (
#         baseline
#         + lobe(dist_1, amplitude_1, tuning_width)
#         + lobe(dist_2, amplitude_2, tuning_width * width_2_ratio)
#     )
