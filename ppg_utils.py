import numpy as np
import scipy.signal as sp
import torch
import wfdb
import os
from matplotlib import pyplot as plt
from pathlib import Path
from sklearn.preprocessing import minmax_scale
from vital_sqi.sqi.standard_sqi import (
    skewness_sqi,
    zero_crossings_rate_sqi,
    perfusion_sqi,
)


def find_peaks_ppg_li(ppg, fs):
    """
    Find systolic peaks in a PPG signal using Li's derivative-based detection algorithm.

    This is a Python implementation of the systolic peak detection portion of the
    automatic delineator algorithm for PPG signals, originally described by Li et al. (2010).
    The algorithm uses derivative analysis and adaptive thresholding to identify systolic peaks in PPG waveforms.

    Primary reference:
        Li, B. N., Dong, M. C., & Vai, M. I. (2010).
        On an automatic delineator for arterial blood pressure waveforms.
        Biomedical Signal Processing and Control, 5(1), 76–81.
        doi:10.1016/j.bspc.2009.06.002

    Args:
        ppg: PPG signal
        fs: Sampling frequency of the signal

    Returns:
        numpy.ndarray: Array of indices locating the systolic peaks in the signal.
            Returns empty array if no peaks are found.
    """

    # Calculate window sizes of different milliseconds in samples
    ms_200 = int(0.2 * fs)
    ms_50 = int(0.05 * fs)
    ms_300 = int(0.3 * fs)

    # Calculate derivatives using one-order amplitude differences as specified in paper
    derivative = np.diff(ppg, prepend=ppg[0])
    second_derivative = np.diff(derivative, prepend=derivative[0])

    # Find maximal inflections (zero-crossings of second derivative with negative slope)
    pos_to_neg = (second_derivative[:-1] > 0) & (second_derivative[1:] < 0)
    maximal_inflections = np.nonzero(pos_to_neg)[0]

    signal_length = len(ppg)

    # Adaptive thresholding with 2-second divisions (as specified in paper)
    division_length = 2 * fs
    n_divisions = max(1, signal_length // division_length)
    ppg_divisions = np.array_split(ppg, n_divisions)
    derivative_divisions = np.array_split(derivative, n_divisions)

    # Amplitude threshold calculation
    amplitude_thresholds = np.array([np.ptp(division) for division in ppg_divisions])
    amplitude_threshold = np.mean(amplitude_thresholds)

    # Interval threshold calculation
    interval_thresholds = []
    for der_division in derivative_divisions:
        zero_crossings = np.nonzero(np.diff(np.signbit(der_division)))[0]
        if len(zero_crossings) >= 2:
            intervals = np.diff(zero_crossings)
            interval_thresholds.append(np.mean(intervals))

    interval_threshold = (
        np.mean(interval_thresholds) if interval_thresholds else 0.5 * fs
    )

    # Peak detection using zero-crossings after maximal inflections
    peaks = []
    last_peak = -interval_threshold  # Initialize with negative interval

    # Iterate through maximal inflections
    for inflection_idx in maximal_inflections:
        # Look for zero crossing in derivative within 200ms window after maximal inflection
        search_end = min(signal_length - 1, inflection_idx + ms_200)
        zero_crossings = np.nonzero(
            (derivative[inflection_idx : search_end - 1] > 0)
            & (derivative[inflection_idx + 1 : search_end] <= 0)
        )[0]

        # Skip if no zero-crossings found
        if len(zero_crossings) == 0:
            continue

        # Take the first zero-crossing as peak candidate
        peak_candidate = inflection_idx + zero_crossings[0]

        # Skip if too close to last detected peak
        if peak_candidate - last_peak < interval_threshold:
            continue

        # Find local maximum (systolic peak) in vicinity of zero-crossing (50ms window)
        start_search = max(0, peak_candidate - ms_50)
        end_search = min(signal_length, peak_candidate + ms_50)
        local_max_idx = start_search + np.argmax(ppg[start_search:end_search])

        # Calculate lower bound for searching minimum that precedes the potential peak (300ms window)
        search_min = max(0, local_max_idx - ms_300)

        # Skip if there is no minimum before the peak
        if local_max_idx <= search_min:
            continue

        # Calculate peak amplitude relative to preceding minimum
        peak_amplitude = ppg[local_max_idx] - np.min(ppg[search_min:local_max_idx])

        # Check if peak amplitude exceeds threshold
        if peak_amplitude >= amplitude_threshold:
            peaks.append(local_max_idx)
            last_peak = local_max_idx

            # Update amplitude threshold adaptively
            amplitude_threshold = 0.8 * amplitude_threshold + 0.2 * peak_amplitude

    return np.array(peaks)


def find_peaks_ppg_billauer(ppg, delta=0.5):
    """
    Find peaks and valleys in a PPG signal using Billauer's peak detection algorithm.

    This is a Python implementation of the peak detection algorithm originally written
    in MATLAB by Eli Billauer. The algorithm detects local maxima (peaks) and minima
    (valleys) in a signal by identifying points that are followed by a decrease/increase
    of at least 'delta' in amplitude.

    Original source:
    Billauer, E. (2012). peakdet: Peak detection using MATLAB.
    http://billauer.co.il/peakdet.html (accessed November 29th, 2024)

    Args:
        ppg: PPG signal
        delta (float, optional): Minimum difference in amplitude required to detect
            a peak or valley. Higher values are more restrictive and detect fewer
            peaks. Defaults to 0.5.

    Returns:
        tuple: Two numpy arrays containing:
            - peaks: Array of [position, value] pairs for each detected peak
            - valleys: Array of [position, value] pairs for each detected valley
    """

    peaks = []
    valleys = []

    # Initialize variables to store min and max values
    min_value = float("inf")
    max_value = float("-inf")
    max_value_position = 0
    min_value_position = 0

    # Initialize flag to indicate whether we are currently searching for a peak
    searching_for_peak = True

    # Iterate over the signal to find peaks and valleys
    for i in range(len(ppg)):
        current_value = ppg[i]

        # Update min and max values
        if current_value > max_value:
            max_value = current_value
            max_value_position = i
        if current_value < min_value:
            min_value = current_value
            min_value_position = i

        if searching_for_peak:
            if current_value < max_value - delta:
                # If we're searching for a maximum and find a point that's significantly
                # lower than our max (difference > delta), we've found a peak
                peaks.append([max_value_position, max_value])

                # Reset minimum value tracking and switch to searching for a valley
                min_value = current_value
                min_value_position = i
                searching_for_peak = False
        else:
            if current_value > min_value + delta:
                # If we're searching for a minimum and find a point that's significantly
                # higher than our min (difference > delta), we've found a valley
                valleys.append([min_value_position, min_value])

                # Reset maximum value tracking and switch to searching for a peak
                max_value = current_value
                max_value_position = i
                searching_for_peak = True

    # Return peaks and valleys as numpy arrays
    return np.array(peaks), np.array(valleys)


def matched_peak_detection_sqi(ppg, fs, tolerance_samples=2):
    """
    Calculate M_SQI (Matching of multiple systolic wave detection algorithms)
    for PPG signal as described in Elgendi M. (2016). (see assess_ppg_quality)

    M_SQI = (S_Li ∩ S_Billauer) / S_Li


    where S_Li denotes the set of peaks detected by Li's algorithm and
    S_Billauer denotes the set of peaks detected by Billauer's algorithm.

    Args:
        ppg: PPG signal
        fs: Sampling frequency of the signal
        tolerance_samples: Number of samples tolerance for peak matching (default 2).
        This tolerance is not specified in the original paper
        The paper doesn't specify a tolerance window, but 2 samples was chosen because:
        - At our sampling rate of 50 Hz (after resampling), 2 samples = 40ms
        - This provides enough tolerance to account for slight differences in peak
          detection between algorithms while being narrow enough to avoid
          matching incorrect peaks
        - Too small a tolerance might miss genuine matches due to minor timing differences
          between algorithms

    Returns:
        m_sqi: M_SQI score
    """

    # Get peaks from both algorithms
    peaks_billauer, _ = find_peaks_ppg_billauer(ppg)
    peaks_li = find_peaks_ppg_li(ppg, fs)

    # Return 0 if no peaks detected by either algorithm
    if len(peaks_billauer) == 0 or len(peaks_li) == 0:
        return 0

    # Extract just the peak indices
    peak_indices_billauer = peaks_billauer[:, 0].astype(int)
    peak_indices_li = peaks_li.astype(int)

    # Find matching peaks
    matched_peaks = []

    for li_peak in peak_indices_li:
        # Check if there's a Billauer peak within tolerance
        matches = np.where(
            np.abs(peak_indices_billauer - li_peak) <= tolerance_samples
        )[0]
        if len(matches) > 0:
            # If multiple peaks within tolerance, take the closest one
            closest_match = peak_indices_billauer[
                matches[np.argmin(np.abs(peak_indices_billauer[matches] - li_peak))]
            ]
            matched_peaks.append(closest_match)

    matched_peaks = np.array(matched_peaks)

    # Calculate M_SQI
    if len(peak_indices_li) == 0:
        m_sqi = 0
    else:
        m_sqi = len(matched_peaks) / len(peak_indices_li)

    return m_sqi


def assess_ppg_quality(raw_ppg, filtered_ppg, fs):
    """
    Assess the quality of PPG signals using three key quality indices from:
    Elgendi M. (2016).
    Optimal Signal Quality Index for Photoplethysmogram Signals.
    Bioengineering (Basel, Switzerland), 3(4), 21.
    https://doi.org/10.3390/bioengineering3040021

    The paper finds that the following three metrics are best at distinguishing between higher and lower quality PPG signals:
    1. Skewness (S_SQI)
    2. Zero crossing rate (Z_SQI)
    3. Matching of multiple systolic wave detection algorithms (M_SQI)

    Additionally, I am also using the gold standard for assessing the quality of PPG signals: the Perfusion Index

    Args:
        raw_ppg: Raw PPG signal
        filtered_ppg: Bandpass filtered PPG signal
        fs: Sampling frequency of the signal

    Returns:
        tuple: (is_high_quality, metrics_dict)
            - is_high_quality: Boolean indicating if signal meets quality thresholds
            - metrics_dict: Dictionary containing calculated quality metrics
    """

    skewness = skewness_sqi(filtered_ppg)
    zero_crossing_rate = zero_crossings_rate_sqi(np.array(filtered_ppg))
    matched_peak_detection = matched_peak_detection_sqi(filtered_ppg, fs)
    perfusion_index = perfusion_sqi(raw_ppg, filtered_ppg)

    # Quality thresholds based on empirical analysis (see quality metrics distribution plot in plots directory)
    is_high_quality = (
        0.3 < skewness < 0.7
        and zero_crossing_rate < 0.025
        and matched_peak_detection > 0.7
        and 230 < perfusion_index < 310
    )

    metrics = {
        "skewness": skewness,
        "zero_crossing_rate": zero_crossing_rate,
        "matched_peak_detection": matched_peak_detection,
        "perfusion_index": perfusion_index,
    }

    return is_high_quality, metrics


def filter_ppg(ppg, fs):
    """
    Applies bandpass filtering to the PPG signal to remove noise.

    Args:
        ppg: Raw PPG signal
        fs: Sampling frequency of the signal

    Returns:
        Filtered PPG signal
    """

    lpf_cutoff = 1
    hpf_cutoff = 15

    sos_ppg = sp.butter(
        4, [lpf_cutoff, hpf_cutoff], btype="bandpass", output="sos", fs=fs
    )

    # Apply zero-phase filtering
    ppg_filtered = sp.sosfiltfilt(sos_ppg, ppg)

    return ppg_filtered


def load_ppg(
    metadata,
    record_name,
    start_seconds,
    end_seconds,
    window_size,
    target_fs,
):
    """
    Loads a no_sec_to_load second segment of PPG signal from a WFDB record. # TODO
    The signal is bandpass filtered and resampled to target_fs.

    Args:
        metadata: WFDB record header containing signal metadata
        record_name: Name/path of the record to load
        start_seconds: Starting point in seconds from where to load the signal
        end_seconds: Ending point in seconds until the signal is to be loaded
        window_size: Size of individual PPG window in seconds
        target_fs: Target sampling frequency to resample the signal to

    Returns:
        Resampled PPG signal if quality is high and it doesn't contain NaN values, otherwise None
        Metrics dictionary containing the calculated metrics for tracking signal quality
    """

    # Original sampling frequency of the signal
    original_fs = round(metadata.fs)

    # Multiply by sampling frequency to get the sample start and end points
    sampfrom = start_seconds * original_fs
    sampto = end_seconds * original_fs

    raw_data = wfdb.rdrecord(record_name=record_name, sampfrom=sampfrom, sampto=sampto)

    # Find the index of the PPG signal
    for sig_no in range(len(raw_data.sig_name)):
        if "PLETH" in raw_data.sig_name[sig_no]:
            break

    raw_ppg = raw_data.p_signal[:, sig_no]

    # Filter out signals with NaN values
    if np.isnan(raw_ppg).any():
        return None, None

    filtered_ppg = filter_ppg(raw_ppg, original_fs)

    # Resample the signal to target_fs
    num_samples = int(len(filtered_ppg) * target_fs / original_fs)
    resampled_ppg = sp.resample(filtered_ppg, num_samples)

    is_high_quality, metrics = assess_ppg_quality(resampled_ppg, target_fs)

    # Return None if signal quality is low to exclude signal, but still return metrics for tracking
    if not is_high_quality:
        return None, metrics

    # normalize the signal
    resampled_ppg = (resampled_ppg - np.mean(resampled_ppg)) / (
        np.std(resampled_ppg) + 1e-8
    )

    # Rescaling the signal to [0, 1] is required for tokenization approach
    resampled_ppg = minmax_scale(resampled_ppg, feature_range=(0, 1))

    # Create windows of PPG signal
    window_samples = int(window_size * target_fs)  # Number of samples in each window
    stride = int(1 * window_samples)  # Stride in samples (60% overlap)

    # Calculate number of windows
    n_windows = ((len(resampled_ppg) - window_samples) // stride) + 1

    # Create sliding windows
    windows = np.lib.stride_tricks.sliding_window_view(
        resampled_ppg[: n_windows * stride + window_samples], window_samples
    )[::stride]

    # Convert to list of torch tensors
    windowed_signals = [torch.from_numpy(window.copy()) for window in windows]

    return windowed_signals, metrics


def plot_ppg(ppg, fs, filename, metrics):
    """
    Visualize PPG signal with time on x-axis and amplitude on y-axis.
    Also displays calculated quality metrics and sampling frequency on the plot.
    Implementation based on the WFDB tutorial by Peter H Carlton (2022)
    https://wfdb.io/mimic_wfdb_tutorials/tutorial/notebooks/differentiation.html
    Args:
        ppg: PPG signal
        fs: Sampling frequency of the signal
        filename: Name of the file to save the plot
        metrics: Dictionary containing calculated quality metrics
    Returns:
        None: Displays matplotlib plot
    """

    # Create directory to save plots if it doesn't exist
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, f"{filename}.png")

    time = np.arange(len(ppg)) / fs * 1000  # Time in milliseconds

    plt.figure(figsize=(10, 5))
    plt.plot(time, ppg)
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")
    plt.title("PPG Signal")

    # Display quality metrics and sampling frequency on the plot
    plt.text(
        1.02,
        0.95,
        (
            f"Sampling Frequency: {fs} Hz\n"
            f"Skewness SQI: {metrics['skewness']:.2f}\n"
            f"Zero Crossing Rate SQI: {metrics['zero_crossing_rate']:.2f}\n"
            f"Matched Peak Detection SQI: {metrics['matched_peak_detection']:.2f}\n"
            f"Perfusion Index SQI: {metrics['perfusion_index']:.2f}"
        ),
        ha="left",
        va="top",
        transform=plt.gca().transAxes,
        bbox=dict(facecolor="white", alpha=0.8),
    )

    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
