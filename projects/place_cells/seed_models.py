import numpy as np


def neuron_model_simple_place(
    stimuli,
    center_x=0.0,
    center_y=0.0,
    amplitude=1.0,
    sigma=0.3,
    baseline=0.0,
    hd_pref=0.0,
    hd_gain=0.0,
    speed_gain=0.0,
):
    """
    Single place field with simple head-direction and speed modulation.

    rate = baseline + place * (1 + hd_gain * cos(hd - hd_pref)) * (1 + speed_gain * speed)
    """
    baseline = np.clip(baseline, 0.0, None)
    amplitude = np.clip(amplitude, 0.0, None)
    sigma = np.clip(sigma, 1e-6, None)
    hd_gain = np.clip(hd_gain, -1.0, 1.0)
    speed_gain = np.clip(speed_gain, 0.0, 5.0)

    stim = np.asarray(stimuli)
    x = stim[:, 0]
    y = stim[:, 1]
    hd = stim[:, 2] if stim.shape[1] > 2 else 0.0
    speed = stim[:, 3] if stim.shape[1] > 3 else 0.0

    r2 = (x - center_x) ** 2 + (y - center_y) ** 2
    place = amplitude * np.exp(-0.5 * r2 / (sigma ** 2))
    hd_mod = 1.0 + hd_gain * np.cos(hd - hd_pref)
    speed_mod = 1.0 + speed_gain * speed
    return baseline + place * hd_mod * speed_mod


def parameter_estimator_simple_place(stimuli, spike_counts):
    """Estimate [center_x, center_y, amplitude, sigma, baseline, hd_pref, hd_gain, speed_gain]."""
    def _rate_map(pos_xy, spikes, bins=40, min_occ=3, smoothing_sigma=1.2, eps=1e-6):
        def _gaussian_kernel(sigma):
            if sigma <= 0:
                return None
            size = int(max(3, np.ceil(6 * sigma)))
            if size % 2 == 0:
                size += 1
            ax = np.arange(size) - size // 2
            xx, yy = np.meshgrid(ax, ax)
            kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
            kernel /= np.sum(kernel)
            return kernel

        def _fft_convolve2d(a, b):
            out_shape = (a.shape[0] + b.shape[0] - 1, a.shape[1] + b.shape[1] - 1)
            fa = np.fft.rfftn(a, s=out_shape)
            fb = np.fft.rfftn(b, s=out_shape)
            out = np.fft.irfftn(fa * fb, s=out_shape)
            start0 = (b.shape[0] - 1) // 2
            start1 = (b.shape[1] - 1) // 2
            return out[start0:start0 + a.shape[0], start1:start1 + a.shape[1]]

        occ, xedges, yedges = np.histogram2d(pos_xy[:, 0], pos_xy[:, 1], bins=bins)
        spk, _, _ = np.histogram2d(pos_xy[:, 0], pos_xy[:, 1], bins=[xedges, yedges], weights=spikes)
        rate = spk / (occ + eps)
        rate = np.where(occ >= min_occ, rate, np.nan)
        kernel = _gaussian_kernel(smoothing_sigma)
        if kernel is not None:
            rate_filled = np.nan_to_num(rate, nan=0.0)
            occ_mask = np.isfinite(rate).astype(float)
            rate_s = _fft_convolve2d(rate_filled, kernel)
            norm = _fft_convolve2d(occ_mask, kernel)
            rate = rate_s / (norm + eps)
            rate = np.where(norm > 0, rate, np.nan)
        return rate.T, xedges, yedges

    stim = np.asarray(stimuli)
    pos = stim[:, :2]
    hd = stim[:, 2] if stim.shape[1] > 2 else np.zeros(len(stim))
    speed = stim[:, 3] if stim.shape[1] > 3 else np.zeros(len(stim))
    spikes = np.asarray(spike_counts)

    rate, xedges, yedges = _rate_map(pos, spikes)
    finite = np.isfinite(rate)
    if not np.any(finite):
        return np.zeros(8)

    rate_f = rate[finite]
    baseline = float(np.percentile(rate_f, 20))
    peak = float(np.nanmax(rate))
    amplitude = max(peak - baseline, 1e-6)

    peak_idx = np.nanargmax(rate)
    iy, ix = np.unravel_index(peak_idx, rate.shape)
    x_centers = 0.5 * (xedges[:-1] + xedges[1:])
    y_centers = 0.5 * (yedges[:-1] + yedges[1:])
    center_x = float(x_centers[ix])
    center_y = float(y_centers[iy])

    yy, xx = np.meshgrid(y_centers, x_centers, indexing="ij")
    weights = np.clip(rate - baseline, 0.0, None)
    weights = np.nan_to_num(weights, nan=0.0)
    wsum = np.sum(weights) + 1e-8
    var_x = np.sum(((xx - center_x) ** 2) * weights) / wsum
    var_y = np.sum(((yy - center_y) ** 2) * weights) / wsum
    sigma = float(np.sqrt(0.5 * (var_x + var_y) + 1e-12))

    spike_w = np.clip(spikes - np.min(spikes), 0.0, None)
    ssum = np.sum(spike_w)
    if ssum > 0:
        hd_pref = float(np.arctan2(np.sum(spike_w * np.sin(hd)), np.sum(spike_w * np.cos(hd))))
    else:
        hd_pref = 0.0

    n_bins = 18
    hd_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    hd_idx = np.digitize(hd, hd_bins) - 1
    hd_idx = np.clip(hd_idx, 0, n_bins - 1)
    hd_sum = np.bincount(hd_idx, weights=spikes, minlength=n_bins)
    hd_count = np.bincount(hd_idx, minlength=n_bins)
    hd_rate = np.zeros(n_bins, dtype=float)
    valid = hd_count > 0
    hd_rate[valid] = hd_sum[valid] / hd_count[valid]
    if np.max(hd_rate) > 0:
        hd_gain = float((np.max(hd_rate) - np.min(hd_rate)) / (np.max(hd_rate) + 1e-8))
    else:
        hd_gain = 0.0

    if np.std(speed) > 1e-6:
        slope = np.cov(speed, spikes)[0, 1] / (np.var(speed) + 1e-8)
        speed_gain = float(max(0.0, slope / (np.mean(spikes) + 1e-8)))
    else:
        speed_gain = 0.0

    return np.array([center_x, center_y, amplitude, sigma, baseline, hd_pref, hd_gain, speed_gain])


def neuron_model_speed_exp(stimuli,
                    center_x1=0.25,
                    center_y1=0.25,
                    amp1=1.0,
                    sigma1=1.0,
                    center_x2=-0.25,
                    center_y2=-0.25,
                    amp2=1.0,
                    sigma2=1.0,
                    baseline=0.0,
                    hd_pref=0.0,
                    hd_kappa=0.5,
                    speed_exp=1.0
                    ):
    """
    Equation:
    place1 = amp1 * exp(-0.5 * ((x - center_x1)^2 + (y - center_y1)^2) / sigma1^2)
    place2 = amp2 * exp(-0.5 * ((x - center_x2)^2 + (y - center_y2)^2) / sigma2^2)
    hd_mod = exp(hd_kappa * cos(hd - hd_pref))
    speed_gain = np.clip(speed_gain, 0.0, None)
    speed_mod = speed ** speed_exp
    rate = baseline + (place1 + place2) * hd_mod * speed_mod
    """
    baseline = np.clip(baseline, 0.0, None)
    amp1 = np.clip(amp1, 0.0, None)
    sigma1 = np.clip(sigma1, 1e-6, None)
    amp2 = np.clip(amp2, 0.0, None)
    sigma2 = np.clip(sigma2, 1e-6, None)
    speed_exp = np.clip(speed_exp, 0.0, 5.0) # biologically plausible
    speed_gain = np.clip(speed_gain, 0.0, None)

    stim = np.asarray(stimuli)
    x = stim[:, 0]
    y = stim[:, 1]
    hd = stim[:, 2] if stim.shape[1] > 2 else 0.0
    speed = stim[:, 3] if stim.shape[1] > 3 else 0.0

    # First Gaussian place field
    dx1 = x - center_x1
    dy1 = y - center_y1
    r2_1 = dx1 ** 2 + dy1 ** 2
    place1 = amp1 * np.exp(-0.5 * r2_1 / (sigma1 ** 2))

    # Second Gaussian place field
    dx2 = x - center_x2
    dy2 = y - center_y2
    r2_2 = dx2 ** 2 + dy2 ** 2
    place2 = amp2 * np.exp(-0.5 * r2_2 / (sigma2 ** 2))
    
    place_sum = place1 + place2

    # Multiplicative modulation terms
    hd_mod = np.exp(hd_kappa * np.cos(hd - hd_pref))
    speed_mod = speed ** speed_exp
    

    # Combine all components
    rate = baseline + place_sum * hd_mod * speed_mod

    return rate


def parameter_estimator_speed_exp(stimuli, spike_counts):
    """Estimates parameters for neuron_model based on rate map analysis.

    Returns: [center_x1, center_y1, amp1, sigma1, center_x2, center_y2, amp2, sigma2, baseline, hd_pref, hd_kappa, speed_exp]
    """
    from scipy.ndimage import gaussian_filter, maximum_filter
    def _rate_map(pos_xy, spikes, bins=40, min_occ=3, smoothing_sigma=1.2):
        H, xedges, yedges = np.histogram2d(pos_xy[:, 0], pos_xy[:, 1], bins=bins)
        H_s, _, _ = np.histogram2d(pos_xy[:, 0], pos_xy[:, 1], bins=[xedges, yedges], weights=spikes)
        
        # Dwell Time & Min Occupation Mask
        H_occupancy = H.copy()
        occupancy_filter = H > min_occ
        
        # Replace values with their counts, with safety value for inf/NaN division
        with np.errstate(divide='ignore', invalid='ignore'):  # Handle inf/NaN cases
            H = np.where(H_occupancy>min_occ, H_s / (H+ 1e-12), np.zeros_like(H))
        # Apply Smoothing/blurr (helps make Gauss fit, also removes jitter and over counts for low rates in sparse data sets)
        H = gaussian_filter(H, sigma=smoothing_sigma)
        
        H_occupancy = np.where(occupancy_filter==True,H_occupancy,np.zeros_like(H))
        return H.T, xedges, yedges
        
    stim = np.asarray(stimuli)
    pos = stim[:, :2]
    hd = stim[:, 2] if stim.shape[1] > 2 else np.zeros(len(stim))
    speed = stim[:, 3] if stim.shape[1] > 3 else np.zeros(len(stim))
    spikes = np.asarray(spike_counts)

    rate, xedges, yedges = _rate_map(pos, spikes, bins=40, smoothing_sigma=1.2)
    finite = np.isfinite(rate)
    if not np.any(finite): return np.zeros(13)

    baseline = float(np.percentile(rate, 10))
    rate = np.nan_to_num(rate, nan=baseline)
    
    neighborhood_size = 3
    threshold = baseline
    peak_local_max = lambda x: x > threshold

    
    maxima = maximum_filter(rate, footprint=np.ones((neighborhood_size, neighborhood_size)), mode='constant', cval=0.0)
    peaks = (maxima == rate) & peak_local_max(maxima) #

    peaks_indices = np.argwhere(peaks)

    if peaks_indices.shape[0] >= 2:

        # Take only top 2 rate mapped indices to use for our guassian locations, take absolute
        peaks_abs = np.array([abs(rate[x[0],x[1]]) for x in peaks_indices], dtype=float)
        peaks_argsorted = np.argsort(peaks_abs)[::-1][:2] 
    
    
        ix1 = int(peaks_indices[peaks_argsorted[0]][1])
        iy1 = int(peaks_indices[peaks_argsorted[0]][0])

        ix2 = int(peaks_indices[peaks_argsorted[1]][1])
        iy2 = int(peaks_indices[peaks_argsorted[1]][0])
    
    
    else: 
        # In the case there is one/zero location of rate coding activity make defaults
        max_indices = np.unravel_index(np.argmax(rate), rate.shape)
        ix1 = int(max_indices[1])
        iy1 = int(max_indices[0])
        
        ix2 = int(ix1) #Set place 1 location to position 2
        iy2 = int(iy1)


    # Bin Conversion  xedges[0,1]  # x/yedges[:-1] + xedges[1:]) / 2  Convert
    x_centers = 0.5 * (xedges[:-1] + xedges[1:])
    y_centers = 0.5 * (yedges[:-1] + yedges[1:])

    # Position (x,y) Locations to build gaussians for fitting: xy indexing used
    center1 = np.array([x_centers[ix1], y_centers[iy1]]) #Position/ location peak firing rate is.
    center2 = np.array([x_centers[ix2], y_centers[iy2]])

    amp1 = float(rate[iy1, ix1] - baseline) 
    amp2 = float(rate[iy2, ix2] - baseline)
    sigma = 0.075 * max(np.ptp(pos[:, 0]), np.ptp(pos[:, 1]))

    w = np.clip(spikes - baseline, 0.0, None)
    wsum_s = np.sum(w) + 1e-8 #add l1 bias


    # -- Estimate HD preference --
    c = float(np.sum(w * np.cos(hd)) / wsum_s) #calculate angular moments 
    s = float(np.sum(w * np.sin(hd)) / wsum_s) 
    hd_pref = float(np.arctan2(s, c))
    hd_kappa = 0.5 # head_direction modulation
    
    # -speed mod-, (L1 BIAS added), #Linear least squares, L1 to stabilize & allow
    
    # Estimating speed_exp
    speed_exp = 1.0
    if np.any(speed > 0) and np.any(spikes > 0):
      # Only consider times when the cell fired and the animal moved.
      active_times = (spikes > baseline) & (speed > 0.1)
      if np.sum(active_times) > 10:  # Need some active data
          log_speed = np.log(speed[active_times])
          log_spikes = np.log(spikes[active_times] - baseline)
          #Simple speed exponent regression with clip on max spike value as that indicates rate is not position sensitive
          speed_exp = float(np.cov(log_speed, log_spikes, bias=True)[0, 1] / (np.var(log_speed) + 1e-8))
          speed_exp = np.clip(speed_exp, 0.1, 2.0)
    
    return np.array([center1[0], center1[1], amp1, sigma, center2[0], center2[1], amp2, sigma, baseline, hd_pref, hd_kappa, speed_exp])

def neuron_model_speed_gain(stimuli,
                    # Field 1 parameters
                    center_x_1=0.25,
                    center_y_1=0.25,
                    place_amp_1=1.0,
                    sigma_x_1=1.0,
                    sigma_y_1=1.0,
                    # Field 2 parameters
                    center_x_2=0.75,
                    center_y_2=0.75,
                    place_amp_2=1.0,
                    sigma_x_2=1.0,
                    sigma_y_2=1.0,
                    # Modulation parameters
                    baseline=0.0,
                    hd_pref=0.0,
                    hd_amp=0.5,
                    speed_gain=0.0,
                    speed_exp = 1.0):
    """
    This model represents the firing rate as a sum of two anisotropic Gaussian place
    fields, with multiplicative head-direction (HD) gain modulation and additive speed
    modulation.

    rate = baseline + (G_1 + G_2) * (1 + hd_amp * cos(hd - hd_pref)) + speed_gain * speed^speed_exp
    where G_i is an anisotropic 2D Gaussian for field i.

    """
    # Clip parameters to plausible ranges
    place_amp_1 = np.clip(place_amp_1, 0.0, None)
    sigma_x_1 = np.clip(sigma_x_1, 1e-6, None)
    sigma_y_1 = np.clip(sigma_y_1, 1e-6, None)
    place_amp_2 = np.clip(place_amp_2, 0.0, None)
    sigma_x_2 = np.clip(sigma_x_2, 1e-6, None)
    sigma_y_2 = np.clip(sigma_y_2, 1e-6, None)
    baseline = np.clip(baseline, 0.0, None)
    hd_amp = np.clip(hd_amp, 0.0, 1.0)
    speed_gain = np.clip(speed_gain, 0.0, None)
    speed_exp = np.clip(speed_exp, 0.0, 5.0)
    
    # Unpack stimuli
    stim = np.asarray(stimuli)
    x = stim[:, 0]
    y = stim[:, 1]
    hd = stim[:, 2] if stim.shape[1] > 2 else 0.0
    speed = stim[:, 3] if stim.shape[1] > 3 else 0.0

    # Calculate place field 1
    dx1 = x - center_x_1
    dy1 = y - center_y_1
    r2_1 = (dx1 ** 2) / (sigma_x_1 ** 2) + (dy1 ** 2) / (sigma_y_1 ** 2)
    place_1 = place_amp_1 * np.exp(-0.5 * r2_1)

    # Calculate place field 2
    dx2 = x - center_x_2
    dy2 = y - center_y_2
    r2_2 = (dx2 ** 2) / (sigma_x_2 ** 2) + (dy2 ** 2) / (sigma_y_2 ** 2)
    place_2 = place_amp_2 * np.exp(-0.5 * r2_2)

    total_place = place_1 + place_2

    # Calculate multiplicative HD modulation
    hd_modulation = 1.0 + hd_amp * np.cos(hd - hd_pref)
    
    # Calculate speed modulation
    speed_modulation = speed_gain * speed**speed_exp
    
    # Combine terms
    rate = baseline + total_place * hd_modulation + speed_modulation

    return rate


def parameter_estimator_speed_gain(stimuli, spike_counts):
    """Estimates parameters for neuron_model based on rate map analysis and vector statistics.

    Returns: [center_x_1, center_y_1, place_amp_1, sigma_x_1, sigma_y_1, center_x_2, center_y_2, place_amp_2, sigma_x_2, sigma_y_2, baseline, hd_pref, hd_amp, speed_gain, speed_exp]
    """
    from scipy.ndimage import gaussian_filter, maximum_filter

    def _rate_map(pos_xy, spikes, bins=40, min_occ=3, smoothing_sigma=1.2):
        H, xedges, yedges = np.histogram2d(pos_xy[:, 0], pos_xy[:, 1], bins=bins)
        H_s, _, _ = np.histogram2d(pos_xy[:, 0], pos_xy[:, 1], bins=[xedges, yedges], weights=spikes)

        H_occupancy = H.copy()
        occupancy_filter = H > min_occ

        with np.errstate(divide='ignore', invalid='ignore'):
            H = np.where(H_occupancy > min_occ, H_s / (H + 1e-12), np.zeros_like(H))
        H = gaussian_filter(H, sigma=smoothing_sigma)
        H_occupancy = np.where(occupancy_filter, H_occupancy, np.zeros_like(H))
        return H.T, xedges, yedges

    stim = np.asarray(stimuli)
    pos = stim[:, :2]
    hd = stim[:, 2] if stim.shape[1] > 2 else np.zeros(len(stim))
    speed = stim[:, 3] if stim.shape[1] > 3 else np.zeros(len(stim))
    spikes = np.asarray(spike_counts)

    rate, xedges, yedges = _rate_map(pos, spikes, bins=40, smoothing_sigma=1.2)
    finite = np.isfinite(rate)
    if not np.any(finite):
      return np.zeros(15)

    baseline = float(np.percentile(rate, 10))
    rate = np.nan_to_num(rate, nan=baseline)

    maxima = maximum_filter(rate, footprint=np.ones((3, 3)), mode='constant', cval=0.0)
    peaks = (maxima == rate) & (rate > baseline)

    peak_indices = np.argwhere(peaks)
    if peak_indices.shape[0] >= 2:
      peak_vals = np.array([rate[i, j] for i, j in peak_indices])
      top_peaks = peak_indices[np.argsort(peak_vals)[::-1][:2]]
      iy1, ix1 = top_peaks[0]
      iy2, ix2 = top_peaks[1]
    else:
      max_idx = np.unravel_index(np.argmax(rate), rate.shape)
      iy1, ix1 = max_idx
      ix2, iy2 = ix1, iy1
    x_centers = 0.5 * (xedges[:-1] + xedges[1:])
    y_centers = 0.5 * (yedges[:-1] + yedges[1:])
    center1 = np.array([x_centers[ix1], y_centers[iy1]])
    center2 = np.array([x_centers[ix2], y_centers[iy2]])

    amp1 = float(rate[iy1, ix1] - baseline)
    amp2 = float(rate[iy2, ix2] - baseline)
    sigma_x = sigma_y = 0.075 * max(np.ptp(pos[:, 0]), np.ptp(pos[:, 1]))
    
    # Improved circular heading (preferred dir) via vector means
    active_hd = hd[spikes > baseline]
    if len(active_hd) > 0:
        hd_pref = float(np.arctan2(np.mean(np.sin(active_hd)), np.mean(np.cos(active_hd))))
    else:
        hd_pref = 0.0
    hd_amp = 0.5

    # Refined speed modulation estimation using quantiles and a correction factor
    if np.sum(spikes > baseline) > 10:
        speed_threshold = np.quantile(speed[spikes > baseline], 0.25)
        active = (spikes > baseline) & (speed > speed_threshold)

        if np.sum(active) > 5:
            mean_speed = np.mean(speed[active])
            mean_spikes = np.mean(spikes[active]) - baseline

            if mean_speed > 0:
              # Correction factor based on dynamic range of speed and spike rates
              correction_factor = np.clip(np.ptp(spikes) / np.ptp(speed), 0.1, 5) 
              speed_gain = mean_spikes / (mean_speed + 1e-6) * correction_factor
              speed_exp = 1.0 
            else:
                speed_gain = 0.0
                speed_exp = 1.0
        else:
            speed_gain = 0.0
            speed_exp = 1.0
    else:
        speed_gain = 0.0
        speed_exp = 1.0
    
    return np.array([center1[0], center1[1], amp1, sigma_x, sigma_y, center2[0], center2[1], amp2, sigma_x, sigma_y, baseline, hd_pref, hd_amp, speed_gain, speed_exp])
