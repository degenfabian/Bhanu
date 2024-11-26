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


def load_ppg(metadata, record_name, start_seconds=100, no_sec_to_load=4):
    """
    Loads a no_sec_to_load second segment of PPG signal from a WFDB record.

    Args:
        metadata: WFDB record header containing signal metadata
        record_name: Name/path of the record to load
        start_seconds: Starting point in seconds from where to load the signal
        no_sec_to_load: Number of seconds of signal to load

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

    lpf_cutoff = 0.7
    hpf_cutoff = 10

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
    no_sec_to_load=4,
    offset_from_start_to_load=100,
):
    """
    Loads and processes PPG signals from a directory of WFDB records.

    Args:
        path_to_data: Directory containing the WFDB records
        label: Label to assign to all signals from this directory (0 or 1)
        required_signals: List of required signal types (default: ["PLETH"])
        no_sec_to_load: Number of seconds to load for each segment
        offset_from_start_to_load: Number of seconds to skip from start of recording

    Returns:
        Tuple of (signals, labels) as PyTorch tensors
    """

    print("Loading data from " + path_to_data + "\n")
    all_signals = []
    all_labels = []

    # Find all master header files
    files_to_process = []
    for dirpath, dirnames, filenames in os.walk(path_to_data):
        for file in filenames:
            if "-" in file and not file.endswith("n.hea"):
                files_to_process.append((dirpath, file))

    # Process each record by iterating through segment list in master header file
    for dirpath, file in tqdm(files_to_process, desc="Processing files"):
        if "-" in file and not file.endswith("n.hea"):
            try:
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
                    if segment_length < (offset_from_start_to_load + no_sec_to_load):
                        continue

                    signals_present = segment_metadata.sig_name

                    # Check again if all required signals are present because master header doesn't indicate that for all segments it links to
                    if all(x in signals_present for x in required_signals):
                        start_seconds = offset_from_start_to_load
                        # Load the segment in chunks of no_sec_to_load seconds
                        while (
                            segment_length >= offset_from_start_to_load + no_sec_to_load
                        ):
                            ppg = load_ppg(
                                segment_metadata,
                                record_name,
                                start_seconds=start_seconds,
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

                            start_seconds += no_sec_to_load
                            segment_length -= no_sec_to_load

            except Exception as e:
                print(f"Error loading {segment} from {file[:-4]}")
                continue

    return torch.stack(all_signals), torch.tensor(all_labels)


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


def main():
    os.makedirs("data", exist_ok=True)

    print("Processing PD data...")
    pd_signals, pd_labels = load_data("waveform_data/PD/", label=1)
    torch.save((pd_signals, pd_labels), "data/pd_data.pt")
    print(f"Saved {len(pd_signals)} PD segments")

    print("Processing non-PD data...")
    non_pd_signals, non_pd_labels = load_data("waveform_data/non_PD/", label=0)
    torch.save((non_pd_signals, non_pd_labels), "data/non_pd_data.pt")
    print(f"Saved {len(non_pd_signals)} non-PD segments")

    # Concatenate PD and non-PD data
    all_signals = torch.cat([pd_signals, non_pd_signals])
    all_labels = torch.cat([pd_labels, non_pd_labels])

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

    train_prelim_signals, test_signals, train_prelim_labels, test_labels = (
        train_test_split(
            tokenized_signals,
            all_labels,
            test_size=0.2,
            random_state=42,
            stratify=all_labels,
        )
    )
    train_signals, val_signals, train_labels, val_labels = train_test_split(
        train_prelim_signals,
        train_prelim_labels,
        test_size=0.125,
        random_state=42,
        stratify=train_prelim_labels,
    )

    # Clear memory
    del train_prelim_signals, train_prelim_labels, all_labels
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
