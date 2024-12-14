import torch
from torch.utils.data import Dataset, DataLoader
import wfdb  # For reading PhysioNet data
import os
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
from ppg_utils import load_ppg
import traceback
import matplotlib.pyplot as plt
import seaborn as sns


# Parts of this code were inspired by this tutorial: https://wfdb.io/mimic_wfdb_tutorials/tutorials.html by Peter H Carlton © Copyright 2022.
# The corresponding repository: https://github.com/wfdb/mimic_wfdb_tutorials


class PPGDataset(Dataset):
    """
    Custom PyTorch Dataset class for PPG (Photoplethysmography) signals.
    """

    def __init__(self, signals, labels):
        self.signals = signals
        self.labels = labels.to(torch.float32)  # Convert to float for BCE loss

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]


def analyze_and_visualize_metrics(quality_metrics):
    """
    Analyze and visualize the distribution of PPG signal quality metrics using histograms with overlaid density plots.

    This function creates a figure with three subplots, one for each quality metric. Each subplot contains:
    - A histogram showing the distribution of the metric values
    - A kernel density estimation (KDE) curve overlaid on the histogram
    - Key statistical measures (mean, standard deviation, quartiles) in a text box

    The visualization excludes extreme outliers (beyond 3 * IQR from quartiles) to prevent crashes
    from plotting too many points, while maintaining all data points in the statistical calculations

    Args:
        quality_metrics: Dictionary containing quality metrics (skewness, zero_crossing_rate, matched_peak_detection)
    """

    print("Analyzing signal quality metrics...")

    # Create subplot for each metric
    fig, axes = plt.subplots(1, 4, figsize=(15, 10))
    axes = axes.ravel()

    for i, (key, values) in enumerate(quality_metrics.items()):
        values = np.array(values)

        # Calculate IQR and bounds for outlier removal
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        bounds = (q1 - 3 * iqr, q3 + 3 * iqr)

        # Remove outliers only for visualization
        filtered_values = values[(values >= bounds[0]) & (values <= bounds[1])]

        # Plot distribution
        sns.histplot(filtered_values, kde=True, ax=axes[i])
        axes[i].set_title(f"{key} Distribution")

        # Calculate statistics using ALL values
        stats_text = (
            f"Mean: {np.mean(values):.3f}\n"
            f"Std: {np.std(values):.3f}\n"
            f"25%: {np.percentile(values, 25):.3f}\n"
            f"50%: {np.percentile(values, 50):.3f}\n"
            f"75%: {np.percentile(values, 75):.3f}"
        )

        # Position text box with statistics in upper right corner
        axes[i].text(
            0.95,
            0.95,
            stats_text,
            transform=axes[i].transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, f"quality_metrics_distribution.png")

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


def load_data(
    path_to_data,
    label,
    distance_from_start_and_end=100,
    window_size=10,
    target_fs=50,
):
    """
    Loads and processes PPG signals from a directory of WFDB records.

    Args:
        path_to_data: Directory containing the WFDB records
        label: Label to assign to all signals from this directory (1 for PD, 0 for non-PD)
        distance_from_start_and_end: Number of seconds to skip from the start and end of each segment
        window_size: Size of each individual PPG window in seconds
        target_fs: Target sampling frequency to resample the signal to
    Returns:
        Tuple of (signals, labels, patient_ids, quality_metrics)
    """

    print("Loading data from " + path_to_data + "\n")
    all_signals = []
    all_labels = []
    all_patient_ids = []  # Track patient IDs for patient-level splits
    quality_metrics = {
        "skewness": [],
        "zero_crossing_rate": [],
        "matched_peak_detection": [],
        "perfusion_index": [],
    }  # Track quality metrics for signal analysis

    # Find all master header files
    files_to_process = []
    for dirpath, _, filenames in os.walk(path_to_data):
        for file in filenames:
            if "-" in file and not file.endswith("n.hea"):
                files_to_process.append((dirpath, file))

    # Process each record by iterating through segment list in master header file
    for dirpath, file in tqdm(
        files_to_process, desc="Processing files", position=0, leave=True
    ):
        if "-" in file and not file.endswith("n.hea"):
            try:
                patient_id = dirpath.split("/")[-1]
                record_data = wfdb.rdheader(
                    record_name=os.path.join(dirpath, file[:-4]), rd_segments=True
                )

                # Skip if none of the segments master header links to contain PPG data
                if not "PLETH" in record_data.sig_name:
                    continue

                segments = record_data.seg_name
                # Skip empty segments denoted by "~"
                non_empty_segments = [segment for segment in segments if segment != "~"]

                for segment in tqdm(non_empty_segments, leave=False):
                    record_name = os.path.join(dirpath, segment)
                    segment_metadata = wfdb.rdheader(record_name=record_name)
                    segment_length = segment_metadata.sig_len / segment_metadata.fs

                    minimum_segment_length = (
                        2 * distance_from_start_and_end + window_size
                    )  # distance_from_start_and_end is multiplied by 2 because it is skipped from both start and end

                    # Skip if segment is shorter than required length
                    if segment_length < minimum_segment_length:
                        continue

                    signals_present = segment_metadata.sig_name

                    # Check again if PPG data is present in current segment because master header only indicates presence in at least one segment
                    if "PLETH" in signals_present:
                        # Start and end points for loading the segment
                        start_seconds = distance_from_start_and_end
                        end_seconds = int(segment_length - distance_from_start_and_end)

                        windowed_ppg, metrics = load_ppg(
                            segment_metadata,
                            record_name,
                            label,
                            start_seconds,
                            end_seconds,
                            window_size,
                            target_fs,
                        )

                        # Append quality metrics for this signal if available for quality metric tracking
                        if metrics is not None:
                            quality_metrics["skewness"].append(metrics["skewness"])
                            quality_metrics["zero_crossing_rate"].append(
                                metrics["zero_crossing_rate"]
                            )
                            quality_metrics["matched_peak_detection"].append(
                                metrics["matched_peak_detection"]
                            )
                            quality_metrics["perfusion_index"].append(
                                metrics["perfusion_index"]
                            )

                        # Skip if signal is None (due to NaN values or low quality)
                        if windowed_ppg is not None:
                            all_signals.extend(windowed_ppg)
                            all_labels.extend([label] * len(windowed_ppg))
                            all_patient_ids.extend([patient_id] * len(windowed_ppg))

            except Exception as e:
                print(f"Error loading {segment} from {file[:-4]}: {e}")
                traceback.print_exc()
                continue

    return (
        torch.stack(all_signals),
        torch.tensor(all_labels),
        all_patient_ids,
        quality_metrics,
    )


def get_dataloaders(cfg):
    """
    Creates PyTorch DataLoaders for train, validation, and test sets.

    Args:
        cfg (Config): Configuration object containing model settings and data paths
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """

    train_dataset = torch.load(os.path.join(cfg.data_path, "train_dataset.pt"))
    val_dataset = torch.load(os.path.join(cfg.data_path, "val_dataset.pt"))
    test_dataset = torch.load(os.path.join(cfg.data_path, "test_dataset.pt"))

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers,
    )

    return train_loader, val_loader, test_loader


def tokenize_signals(signals):
    # Tokenize signals by rounding to nearest integer (Tokens: 0-100)
    scale = 100 * signals
    return torch.round(scale).to(torch.int32)


def find_optimal_patient_subset(patient_IDs, target_number_of_segments, patient_info):
    """
    Find an optimal subset of patients whose total segment count is closest to the target number.
    Iteratively tries different starting points to find the best consecutive sequence of patients
    that minimizes the difference between their total segment count and the target number.

    Args:
        patient_IDs (list): List of patient identifiers
        target_number_of_segments (int): Desired total number of segments to achieve
        patient_info: Dictionary mapping patient IDs to their information
                    Format: patient_id: {
                            "label": int, 0 or 1 for non-PD/PD
                            "segment_count": int, number of segments for this patient
                            "indices": List[int] indices of segments belonging to this patient
                        }

    Returns:
        list: Selected patient IDs whose combined segment count best matches the target
    """

    # Sort patient IDs by segment count
    patient_IDs = sorted(patient_IDs, key=lambda x: patient_info[x]["segment_count"])

    # Initialize variables to track best selection and difference
    best_difference = float("inf")
    best_selection = []

    # Try each possible starting point to find the optimal subset
    for start_idx in range(len(patient_IDs)):
        current_selection = []
        current_sum = 0

        # Iterate through patients starting from current start_idx
        for current_patient_id in patient_IDs[start_idx:]:
            # Add patients while staying under target
            if (
                current_sum + patient_info[current_patient_id]["segment_count"]
                <= target_number_of_segments
            ):
                current_selection.append(current_patient_id)
                current_sum += patient_info[current_patient_id]["segment_count"]

        # Update best solution if current selection is closer to target
        current_difference = abs(current_sum - target_number_of_segments)
        if current_difference < best_difference:
            best_difference = current_difference
            best_selection = current_selection

    return best_selection


def split_by_patient(
    signals,
    labels,
    patient_ids,
    test_size=0.2,
    val_size=0.1,
):
    """
    Split data ensuring all segments from the same patient stay together while maintaining class balance
    and target split sizes.

    Args:
        signals: Tensor of signal data
        labels: Tensor of labels
        patient_ids: List of patient IDs corresponding to each signal
        test_size: Proportion of data to use for testing
        val_size: Proportion of training data to use for validation

    Returns:
        Tuple of (train_signals, train_labels, val_signals, val_labels, test_signals, test_labels)
    """

    # Get unique patient IDs and their segment counts
    unique_patient_ids = list(set(patient_ids))
    patient_info = {}
    for patient_id in unique_patient_ids:
        indices = [
            i for i, pid in enumerate(patient_ids) if pid == patient_id
        ]  # Get indices of this patient's segments
        label = labels[indices[0]].item()
        segment_count = len(indices)
        patient_info[patient_id] = {
            "label": label,
            "segment_count": segment_count,
            "indices": indices,
        }

    # Separate patients by class
    pd_patient_ids = [pid for pid, info in patient_info.items() if info["label"] == 1]
    non_pd_patient_ids = [
        pid for pid, info in patient_info.items() if info["label"] == 0
    ]

    # Calculate total number of segments for each class
    total_pd_segments = sum(
        patient_info[pid]["segment_count"] for pid in pd_patient_ids
    )
    total_non_pd_segments = sum(
        patient_info[pid]["segment_count"] for pid in non_pd_patient_ids
    )

    # Calculate target class-distribution ratio per split (use overall dataset ratio as target)
    target_pd_ratio = total_pd_segments / (total_pd_segments + total_non_pd_segments)
    target_non_pd_ratio = 1 - target_pd_ratio

    # Calculate target number of segments for each split
    total_segments = total_pd_segments + total_non_pd_segments
    target_test_segments = total_segments * test_size
    remaining_segments = total_segments - target_test_segments
    val_size = val_size / (
        1 - test_size
    )  # Adjust validation size based on size used for test split
    target_val_segments = remaining_segments * val_size

    # Create test split
    # Calculate target number of segments for test split per class
    target_test_pd_segments = int(target_test_segments * target_pd_ratio)
    target_test_non_pd_segments = int(target_test_segments * target_non_pd_ratio)

    # Get patient IDs used for test split for each class
    test_pd_ids = find_optimal_patient_subset(
        pd_patient_ids, target_test_pd_segments, patient_info
    )
    test_non_pd_ids = find_optimal_patient_subset(
        non_pd_patient_ids, target_test_non_pd_segments, patient_info
    )

    # Remove test patients from available patient IDs
    remaining_pd_ids = [p for p in pd_patient_ids if p not in test_pd_ids]
    remaining_non_pd_ids = [p for p in non_pd_patient_ids if p not in test_non_pd_ids]

    # Create validation split
    # Calculate target number of segments for validation split per class
    target_val_pd_segments = int(target_val_segments * target_pd_ratio)
    target_val_non_pd_segments = int(target_val_segments * target_non_pd_ratio)

    # Get patient IDs used for validation split for each class
    val_pd_ids = find_optimal_patient_subset(
        remaining_pd_ids, target_val_pd_segments, patient_info
    )
    val_non_pd_ids = find_optimal_patient_subset(
        remaining_non_pd_ids, target_val_non_pd_segments, patient_info
    )

    # Use remaining patients for training split
    train_pd_ids = [pid for pid in remaining_pd_ids if pid not in val_pd_ids]
    train_non_pd_ids = [
        pid for pid in remaining_non_pd_ids if pid not in val_non_pd_ids
    ]

    def get_indices_for_split(patient_ids):
        indices = []
        for patient_id in patient_ids:
            indices.extend(patient_info[patient_id]["indices"])
        return indices

    # Get indices for each split based on patient IDs assigned to each class per split
    train_indices = get_indices_for_split(train_pd_ids + train_non_pd_ids)
    val_indices = get_indices_for_split(val_pd_ids + val_non_pd_ids)
    test_indices = get_indices_for_split(test_pd_ids + test_non_pd_ids)

    # Split signals and labels based on indices
    train_signals, train_labels = signals[train_indices], labels[train_indices]
    val_signals, val_labels = signals[val_indices], labels[val_indices]
    test_signals, test_labels = signals[test_indices], labels[test_indices]

    # Print distribution statistics
    for split_name, split_labels in [
        ("Train", train_labels),
        ("Validation", val_labels),
        ("Test", test_labels),
    ]:
        total = len(split_labels)
        pd_count = split_labels.sum().item()
        non_pd_count = total - pd_count
        print(f"\n{split_name} set:")
        print(f"Total: {total}")
        print(f"PD: {pd_count} ({pd_count/total*100:.1f}%)")
        print(f"Non-PD: {non_pd_count} ({non_pd_count/total*100:.1f}%)")

    return (
        train_signals,
        train_labels,
        val_signals,
        val_labels,
        test_signals,
        test_labels,
    )


def main():
    os.makedirs("data", exist_ok=True)

    print("Processing PD data...")
    pd_signals, pd_labels, pd_patient_ids, pd_quality_metrics = load_data(
        "data/waveform_data/PD/", label=1
    )

    torch.save((pd_signals, pd_labels, pd_patient_ids), "data/pd_data.pt")
    print(
        f"Saved {len(pd_signals)} PD segments, from {len(set(pd_patient_ids))} patients, each of length 10 seconds"
    )

    print("Processing non-PD data...")
    non_pd_signals, non_pd_labels, non_pd_patient_ids, non_pd_quality_metrics = (
        load_data("data/waveform_data/non_PD/", label=0)
    )
    torch.save(
        (non_pd_signals, non_pd_labels, non_pd_patient_ids),
        "data/non_pd_data.pt",
    )
    print(
        f"Saved {len(non_pd_signals)} non-PD segments, from {len(set(non_pd_patient_ids))} patients, each of length 10 seconds"
    )

    quality_metrics = {
        "skewness": pd_quality_metrics["skewness"] + non_pd_quality_metrics["skewness"],
        "zero_crossing_rate": pd_quality_metrics["zero_crossing_rate"]
        + non_pd_quality_metrics["zero_crossing_rate"],
        "matched_peak_detection": pd_quality_metrics["matched_peak_detection"]
        + non_pd_quality_metrics["matched_peak_detection"],
        "perfusion_index": pd_quality_metrics["perfusion_index"]
        + non_pd_quality_metrics["perfusion_index"],
    }
    torch.save(quality_metrics, "data/quality_metrics.pt")
    analyze_and_visualize_metrics(quality_metrics)

    # Concatenate PD and non-PD data
    all_signals = torch.cat([pd_signals, non_pd_signals])
    all_labels = torch.cat([pd_labels, non_pd_labels])
    all_patient_ids = pd_patient_ids + non_pd_patient_ids

    # Clear memory
    del (
        pd_signals,
        pd_labels,
        pd_patient_ids,
        pd_quality_metrics,
        non_pd_signals,
        non_pd_labels,
        non_pd_patient_ids,
        non_pd_quality_metrics,
        quality_metrics,
    )
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    tokenized_signals = tokenize_signals(all_signals)

    # Clear memory
    del all_signals
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Split data by patient
    train_signals, train_labels, val_signals, val_labels, test_signals, test_labels = (
        split_by_patient(
            tokenized_signals, all_labels, all_patient_ids, test_size=0.2, val_size=0.1
        )
    )

    # Clear memory
    del tokenized_signals, all_labels, all_patient_ids
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    train_dataset = PPGDataset(train_signals, train_labels)
    val_dataset = PPGDataset(val_signals, val_labels)
    test_dataset = PPGDataset(test_signals, test_labels)

    print("Saving data...")
    torch.save(train_dataset, "data/train_dataset.pt", pickle_protocol=4)
    torch.save(val_dataset, "data/val_dataset.pt", pickle_protocol=4)
    torch.save(test_dataset, "data/test_dataset.pt", pickle_protocol=4)

    print("Data saved successfully")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    print("Data preprocessing complete")


if __name__ == "__main__":
    main()
