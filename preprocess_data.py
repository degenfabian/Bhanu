import torch
from torch.utils.data import Dataset, DataLoader
import wfdb  # For reading PhysioNet data
import os
import scipy.signal as sp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
import numpy as np
from tqdm import tqdm

# Parts of this code were heavily inspired by this tutorial: https://wfdb.io/mimic_wfdb_tutorials/tutorials.html by Peter H Carlton © Copyright 2022.
# The corresponding repository: https://github.com/wfdb/mimic_wfdb_tutorials


class PPGDataset(Dataset):
    """
    Custom PyTorch Dataset class for PPG (Photoplethysmography) signals.
    Implements required methods for PyTorch DataLoader compatibility.
    """

    def __init__(self, signals, labels):
        self.signals = signals
        self.labels = labels.to(torch.float32)  # Convert to float for BCE loss

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]


def load_ppg(metadata, record_name, start_seconds=100, no_sec_to_load=10, target_fs=50):
    """
    Loads a no_sec_to_load second segment of PPG signal from a WFDB record.

    Args:
        metadata: WFDB record header containing signal metadata
        record_name: Name/path of the record to load
        start_seconds: Starting point in seconds from where to load the signal
        no_sec_to_load: Number of seconds of signal to load
        target_fs: Target sampling frequency to resample the signal to

    Returns:
        numpy array containing the PPG signal segment
    """

    fs = round(metadata.fs)

    # Multiply by sampling frequency to get the sample start and end points
    sampfrom = start_seconds * fs
    sampto = (start_seconds + no_sec_to_load) * fs

    raw_data = wfdb.rdrecord(record_name=record_name, sampfrom=sampfrom, sampto=sampto)

    # Find the index of the PPG signal
    for sig_no in range(len(raw_data.sig_name)):
        if "PLETH" in raw_data.sig_name[sig_no]:
            break

    ppg = raw_data.p_signal[:, sig_no]

    # Resample the signal to target_fs
    num_samples = int(len(ppg) * target_fs / fs)
    ppg = sp.resample(ppg, num_samples)

    return ppg


def filter_ppg(ppg, metadata):
    """
    Applies bandpass filtering to the PPG signal to remove noise.

    Args:
        ppg: Raw PPG signal
        metadata: WFDB record header containing signal metadata

    Returns:
        Filtered PPG signal
    """

    lpf_cutoff = 1
    hpf_cutoff = 15

    sos_ppg = sp.butter(
        10, [lpf_cutoff, hpf_cutoff], btype="bandpass", output="sos", fs=metadata.fs
    )

    # Apply zero-phase filtering
    ppg_filtered = sp.sosfiltfilt(sos_ppg, ppg)

    return ppg_filtered


def load_data(
    path_to_data,
    label,
    required_signals=["PLETH"],
    no_sec_to_load=10,
    distance_from_start_and_end=100,
):
    """
    Loads and processes PPG signals from a directory of WFDB records.

    Args:
        path_to_data: Directory containing the WFDB records
        label: Label to assign to all signals from this directory (0 or 1)
        required_signals: List of required signal types (default: ["PLETH"])
        no_sec_to_load: Number of seconds to load for each segment
        distance_from_start_and_end: Number of seconds to skip from the start and end of each segment

    Returns:
        Tuple of (signals, labels) as PyTorch tensors
    """

    print("Loading data from " + path_to_data + "\n")
    all_signals = []
    all_labels = []
    all_patient_ids = []  # Track patient IDs for patient-level splits

    # Find all master header files
    files_to_process = []
    for dirpath, dirnames, filenames in os.walk(path_to_data):
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

                # Skip if signal doesn't contain required signals
                if not all(x in record_data.sig_name for x in required_signals):
                    continue

                segments = record_data.seg_name

                # Skip empty segments denoted by "~"
                non_empty_segments = [segment for segment in segments if segment != "~"]

                for segment in tqdm(non_empty_segments, leave=False):
                    record_name = os.path.join(dirpath, segment)
                    segment_metadata = wfdb.rdheader(record_name=record_name)
                    segment_length = segment_metadata.sig_len / segment_metadata.fs

                    # Skip if segment is shorter than required length
                    if segment_length < (2 * distance_from_start_and_end + no_sec_to_load):
                        continue

                    signals_present = segment_metadata.sig_name

                    # Check again if all required signals are present because master header doesn't indicate that for all segments it links to
                    if all(x in signals_present for x in required_signals):
                        # Load the segment in chunks of no_sec_to_load seconds
                        while (
                            segment_length >= 2 * distance_from_start_and_end + no_sec_to_load 
                        ):
                            ppg = load_ppg(
                                segment_metadata,
                                record_name,
                                start_seconds=distance_from_start_and_end,
                            )
                            ppg = filter_ppg(ppg, segment_metadata)

                            # Skip if any NaN values are present
                            if np.isnan(ppg).any():
                                start_seconds += no_sec_to_load
                                segment_length -= no_sec_to_load
                                continue

                            ppg = torch.from_numpy(ppg.copy())
                            all_signals.append(ppg)
                            all_labels.append(label)
                            all_patient_ids.append(patient_id)

                            start_seconds += no_sec_to_load
                            segment_length -= no_sec_to_load

            except Exception as e:
                print(f"Error loading {segment} from {file[:-4]}")
                continue

    return torch.stack(all_signals), torch.tensor(all_labels), all_patient_ids


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
    return np.round(scale).astype(int)


def split_by_patient(
    signals, labels, patient_ids, test_size=0.2, val_size=0.1, random_state=42
):
    """
    Split data ensuring all segments from the same patient stay together.

    Args:
        signals: Tensor of signal data
        labels: Tensor of labels
        patient_ids: List of patient IDs corresponding to each signal
        test_size: Proportion of data to use for testing
        val_size: Proportion of training data to use for validation
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_signals, train_labels, val_signals, val_labels, test_signals, test_labels)
    """

    # Get unique patient IDs and map each patient to their label
    unique_patient_ids = list(set(patient_ids))
    patient_to_label = {
        patient_id: labels[patient_ids.index(patient_id)]
        for patient_id in unique_patient_ids
    }

    # Split patients into train and test set
    train_val_patients, test_patients = train_test_split(
        unique_patient_ids,
        test_size=test_size,
        random_state=random_state,
        stratify=[patient_to_label[patient_id] for patient_id in unique_patient_ids],
    )

    # Further split train set into train and validation set
    val_size = val_size / (1 - test_size)  # Adjust val_size relative to train set
    train_patients, val_patients = train_test_split(
        train_val_patients,
        test_size=val_size,
        random_state=random_state,
        stratify=[patient_to_label[patient_id] for patient_id in train_val_patients],
    )

    # Get indices of each patient in the train, val, and test sets
    train_indices = [
        i for i, patient_id in enumerate(patient_ids) if patient_id in train_patients
    ]
    val_indices = [
        i for i, patient_id in enumerate(patient_ids) if patient_id in val_patients
    ]
    test_indices = [
        i for i, patient_id in enumerate(patient_ids) if patient_id in test_patients
    ]

    # Split signals and labels based on patient indices
    train_signals, train_labels = signals[train_indices], labels[train_indices]
    val_signals, val_labels = signals[val_indices], labels[val_indices]
    test_signals, test_labels = signals[test_indices], labels[test_indices]

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
    pd_signals, pd_labels, pd_patient_ids = load_data("data/waveform_data/PD/", label=1)
    torch.save((pd_signals, pd_labels, pd_patient_ids), "data/pd_data.pt")
    print(
        f"Saved {len(pd_signals)} PD segments, from {len(set(pd_patient_ids))} patients, each of length 10 seconds"
    )

    print("Processing non-PD data...")
    non_pd_signals, non_pd_labels, non_pd_patient_ids = load_data(
        "data/waveform_data/non_PD/", label=0
    )
    torch.save(
        (non_pd_signals, non_pd_labels, non_pd_patient_ids), "data/non_pd_data.pt"
    )
    print(
        f"Saved {len(non_pd_signals)} non-PD segments, from {len(set(non_pd_patient_ids))} patients, each of length 10 seconds"
    )

    # Concatenate PD and non-PD data
    all_signals = torch.cat([pd_signals, non_pd_signals])
    all_labels = torch.cat([pd_labels, non_pd_labels])
    all_patient_ids = pd_patient_ids + non_pd_patient_ids

    # Clear memory
    del pd_signals, pd_labels, non_pd_signals, non_pd_labels
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("Normalizing and scaling data...")
    normalized_signals = (all_signals - all_signals.mean(dim=1, keepdim=True)) / (
        all_signals.std(dim=1, keepdim=True) + 1e-8
    )
    scaled_signals = minmax_scale(normalized_signals, (0, 1), axis=1)
    tokenized_signals = tokenize_signals(scaled_signals)

    # Clear memory
    del normalized_signals, all_signals, scaled_signals
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
