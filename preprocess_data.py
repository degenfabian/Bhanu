import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import data.schema_ppmi_pb2
import neurokit2 as nk
import numpy as np
import logging
import multiprocessing

from sklearn.preprocessing import minmax_scale
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from e2epyppg.ppg_sqa import sqa

QUALITY_ASSESSMENT_WINDOW_SIZE = 30


class PPGDataset(Dataset):
    """
    Custom PyTorch Dataset class for PPG (Photoplethysmography) signals.
    """

    def __init__(self, signals, labels):
        self.signals = signals.to(torch.int32)
        self.labels = labels.to(
            torch.float32
        )  # Convert labels to float32 for compatibility with BCEWithLogitsLoss

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]


class PPGProcessor:
    def __init__(
        self,
        path_to_load_data,
        path_to_save_data,
        original_sampling_rate=30,
        target_sampling_rate=50,
        window_size=30,
        stride=30,
        skip_start_and_end=100,
        num_processes=None,
    ):
        self.path_to_load_data = path_to_load_data
        self.path_to_save_data = path_to_save_data
        self.original_sampling_rate = original_sampling_rate
        self.target_sampling_rate = target_sampling_rate
        self.window_size = window_size
        self.stride = stride
        self.skip_start_and_end = skip_start_and_end

        # If num_processes is None, use all available CPU cores
        self.num_processes = (
            num_processes if num_processes is not None else multiprocessing.cpu_count()
        )

        # Check if the specified directories exist
        if not os.path.exists(path_to_load_data):
            raise FileNotFoundError(
                f"Input directory {path_to_load_data} does not exist."
            )

        # Check if the specified window size is valid
        if window_size > 30:
            raise ValueError(
                "Window size must be smaller than 30 because the quality "
                "assessment function works with 30 second segments and therefore"
                " cannot assess the quality of windows larger than 30 seconds"
            )

    def process_dataset(self):
        """
        Process the entire dataset of PPG signals from all patients.

        This method:
        1. Walks through the input directory to find all patient directories
        2. Creates a multiprocessing pool to process patients in parallel
        3. Processes each patient's data using the _process_patient_wrapper method

        The processing includes: reading raw data, cleaning signals, segmenting into windows,
        quality assessment, and saving processed data to the output directory.

        Returns:
            None
        """

        logging.info("Starting dataset processing...")

        patients_information = []
        import time

        for dirpath, _, filenames in os.walk(self.path_to_load_data):
            current_patient_id = dirpath.split("/")[-1]
            current_file = os.path.join(
                self.path_to_save_data, current_patient_id, "ppg.pt"
            )
            # Skip directories that are not patient data
            if not current_patient_id.isnumeric():
                continue

            patients_information.append((dirpath, filenames, current_patient_id))

        # Process each patient directory in parallel
        with multiprocessing.Pool(processes=self.num_processes) as pool:
            list(
                tqdm(
                    pool.imap_unordered(
                        self._process_patient_wrapper, patients_information
                    ),
                    total=len(patients_information),
                    desc="Processing patients",
                    position=0,
                    leave=True,
                )
            )
        logging.info("Dataset processing completed.")

    def _get_patient_logger(self, current_patient_id):
        """
        Create a patient-specific logger for the current processing thread.

        This method configures a logger that writes to a patient-specific log file,
        which helps organize logging information when processing multiple patients
        in parallel.

        Args:
            current_patient_id (str): ID of the patient to create a logger for

        Returns:
            logging.Logger: Configured logger for the current patient
        """

        patient_logger = multiprocessing.get_logger()

        # If the logger already has handlers, clear them to avoid duplicate logs
        if patient_logger.handlers:
            patient_logger.handlers.clear()

        patient_logger.setLevel(logging.INFO)
        handler = logging.FileHandler(os.path.join("logs", f"{current_patient_id}.log"))
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        patient_logger.addHandler(handler)

        return patient_logger

    def _process_patient_wrapper(self, current_patient_information):
        """
        Process a single patient's data in a parallel processing environment.

        This wrapper function is designed to be called by a multiprocessing pool.
        It handles the entire workflow for processing a single patient:
        1. Sets up a patient-specific logger
        2. Creates an output directory for the patient
        3. Calls process_patient to extract and process the patient's PPG data
        4. Saves the processed data to disk

        The function includes error handling to log any issues that occur during processing.

        Args:
            current_patient_information (tuple): A tuple containing:
                - dirpath (str): Path to the patient's directory
                - filenames (list): List of filenames in the patient's directory
                - current_patient_id (str): ID of the current patient

        Returns:
            None: Results are saved to disk rather than returned
        """

        dirpath, filenames, current_patient_id = current_patient_information

        path_to_save_current_patient = os.path.join(
            self.path_to_save_data, current_patient_id
        )
        os.makedirs(path_to_save_current_patient, exist_ok=True)

        patient_logger = self._get_patient_logger(current_patient_id)
        patient_logger.info(f"Processing patient {current_patient_id}")

        if len(os.listdir(path_to_save_current_patient)) > 0:
            patient_logger.info(
                f"Skipping patient {current_patient_id} as data already exists."
            )
            return

        current_patient_ppg = self._process_patient(dirpath, filenames, patient_logger)

        if current_patient_ppg is not None:
            patient_logger.info(f"Saving PPG data for patient {current_patient_id}")

            try:
                torch.save(
                    current_patient_ppg,
                    os.path.join(path_to_save_current_patient, "ppg.pt"),
                )
                patient_logger.info(
                    f"PPG data for patient {current_patient_id} saved successfully."
                )
            except Exception as e:
                patient_logger.error(
                    f"Error saving PPG data for patient {current_patient_id}: {e}"
                )
                return

        else:
            # Delete the directory if no valid data was found
            os.rmdir(path_to_save_current_patient)
            patient_logger.warning(
                f"Deleting empty directory for patient {current_patient_id} as no valid data was found."
            )

    def _process_patient(self, dirpath, filenames, patient_logger):
        """
        Process all PPG data files for a single patient.

        This method:
        1. Iterates through all files for the given patient
        2. Processes each file to extract high-quality PPG segments
        3. Combines all high-quality segments into a single tensor

        Args:
            dirpath (str): Path to the patient's directory
            filenames (list): List of filenames in the patient's directory
            patient_logger (logging.Logger): Logger for the current patient

        Returns:
            torch.Tensor or None: Tensor containing all high-quality PPG segments for the patient,
                                or None if no valid data was found
        """

        current_patient_ppg = []
        current_patient_id = dirpath.split("/")[-1]

        for file in tqdm(
            filenames,
            desc=f"Processing files for current patient {current_patient_id}",
            position=(os.getpid() % self.num_processes) + 1,
            leave=False,
        ):
            high_quality_ppg_current_file = self._process_file(
                os.path.join(dirpath, file), patient_logger
            )

            # If high quality PPG segments were found, add them to the current patient's data
            if len(high_quality_ppg_current_file) > 0:
                current_patient_ppg.extend(high_quality_ppg_current_file)

        if len(current_patient_ppg) == 0:
            patient_logger.warning(
                f"No valid/high quality PPG data was found for current patient."
            )
            return None

        return torch.tensor(np.array(current_patient_ppg), dtype=torch.float32)

    def _process_file(self, filepath, patient_logger):
        """
        Process a single PPG data file to extract high-quality signal segments.

        This method:
        1. Reads a protocol buffer file containing PPG data
        2. Extracts PPG signals and timestamps
        3. Skips skip_start_and_end (class variable) seconds from start and end of the recording
        4. Verifies the actual sampling rate based on the timestamps matches the expected rate
        5. Resamples the signal to the target sampling rate (class variable)
        6. Cleans the PPG signal using neurokit2
        7. Creates windows of the specified size and assesses quality

        Args:
            filepath (str): Path to the PPG data file
            patient_logger (logging.Logger): Logger for the current patient

        Returns:
            list: List of high-quality PPG signal windows
        """

        # Instantiate protocol buffer object for reading out subject data
        subject_data = data.schema_ppmi_pb2.SubjectDayDataType()

        with open(filepath, "rb") as f:
            subject_data.ParseFromString(f.read())

        samples = subject_data.data[0].samples

        current_file_ppg = []
        current_file_timestamps = []

        for sample in samples:
            current_file_ppg.append(sample.ppg.green)
            current_file_timestamps.append(sample.measurement_time_millis)

        # PPG signal needs to be long enough to skip the first and last skip_start_and_end seconds
        # and still have enough data for the window size needed for quality assessment
        # (Because self.window_size <= QUALITY_ASSESSMENT_WINDOW_SIZE we have to use the latter for the calculation,
        # for more information see the docstring of the _create_windows method)
        minimum_length = (
            self.skip_start_and_end * self.original_sampling_rate * 2
            + QUALITY_ASSESSMENT_WINDOW_SIZE * self.original_sampling_rate
        )

        if len(current_file_ppg) < minimum_length:
            patient_logger.warning(
                f"Skipping file: File too short to skip {self.skip_start_and_end} seconds at both ends."
            )
            return []

        # Skip the first and last skip_start_and_end seconds of the PPG signal
        current_file_ppg = current_file_ppg[
            self.skip_start_and_end
            * self.original_sampling_rate : -self.skip_start_and_end
            * self.original_sampling_rate
        ]

        # Skip the first and last skip_start_and_end seconds of the timestamps
        # to only keep the timestamps that correspond to the PPG signal
        current_file_timestamps = current_file_timestamps[
            self.skip_start_and_end
            * self.original_sampling_rate : -self.skip_start_and_end
            * self.original_sampling_rate
        ]

        # Calculate sampling rate of current file
        current_file_sampling_rate = 1000 / (np.mean(np.diff(current_file_timestamps)))

        # Check if sampling rate of current file matches (tolerance of 10% on both ends) empirically derived global sampling rate
        # If not, skip the file
        if (
            abs(current_file_sampling_rate - self.original_sampling_rate)
            > 0.1 * self.original_sampling_rate
        ):
            patient_logger.warning(
                f"Skipping file: Expected sampling rate {self.original_sampling_rate}, but got {current_file_sampling_rate}."
            )
            return []

        current_file_ppg = np.array(current_file_ppg)

        # Resample the PPG signal to the target sampling rate (class variable) such that it matches
        # the sampling rate used for pretraining the HeartGPT model (50 Hz) for later fine-tuning
        current_file_ppg_resampled = nk.signal_resample(
            current_file_ppg,
            sampling_rate=self.original_sampling_rate,
            desired_sampling_rate=self.target_sampling_rate,
        )

        ppg_cleaned = nk.ppg_clean(
            current_file_ppg_resampled, sampling_rate=self.target_sampling_rate
        )

        return self._create_windows(ppg_cleaned)

    def _create_windows(self, ppg_cleaned):
        """
        Segment cleaned PPG signal into high-quality windows of window_size seconds (class variable).

        This method implements a two-stage windowing approach:
        1. First segments the signal into 30-second windows with no overlap for quality assessment
        2. Then further divides high-quality 30-second segments into smaller windows of user-defined
        size with user-defined stride
        3. Each window of user-defined size then get's scaled to [0, 1] for later tokenization.

        The quality assessment is performed using the signal quality assessment (sqa) function
        from the e2epyppg library. The function internally uses a 30-second window for quality assessment.
        If a signal longer than 30 seconds is put into it, the function segments it with a 28 seconds overlap.
        This significantly increases the runtime of the function and is not feasible for the amount of data
        we are working with. Hence, we first segment the signal into 30-second windows with no overlap ourselves
        before then further segmenting them into the user-defined size and stride.

        Args:
            ppg_cleaned (numpy.ndarray): Cleaned PPG signal

        Returns:
            list: List of high-quality PPG signal windows, each normalized to range [0,1]

        Note:
            As mentioned above, PPG signals larger than 30 seconds get segmented into 30 second windows
            by the quality assessment function. Therefore, we limit the size of user-defined windows to
            maximally 30 seconds, since with larger window sizes we would also have to check whether the
            clean_segments returned by the function are consecutive or not.
        """

        num_samples_per_qa_window = int(
            QUALITY_ASSESSMENT_WINDOW_SIZE * self.target_sampling_rate
        )
        num_windows_for_qa = (
            (len(ppg_cleaned) - num_samples_per_qa_window) // num_samples_per_qa_window
        ) + 1

        # Create sliding windows for quality assessment with no overlap
        windows_for_qa = np.lib.stride_tricks.sliding_window_view(
            ppg_cleaned[
                : num_windows_for_qa * num_samples_per_qa_window
                + num_samples_per_qa_window
            ],
            num_samples_per_qa_window,
        )[::num_samples_per_qa_window]

        # User defined window size and stride
        num_samples_per_user_window = int(self.window_size * self.target_sampling_rate)
        num_samples_per_stride = int(self.stride * self.target_sampling_rate)

        # Process each 30 second window
        quality_windows = []
        for qa_window in windows_for_qa:
            # No need to filter the signal, since the cleaning function already does that
            clean_indices, _ = sqa(
                qa_window, sampling_rate=self.target_sampling_rate, filter_signal=False
            )

            if (
                len(clean_indices)
                > 0  # len(clean_indices) == 1 represents that the single 30 second segment we have put in is clean
            ):
                # Further segment the current 30 second window into user-defined window_size and stride
                for i in range(
                    0,
                    len(qa_window) - num_samples_per_user_window + 1,
                    num_samples_per_stride,
                ):
                    # Scale the window to the range [0, 1] for tokenization
                    user_window = qa_window[i : i + num_samples_per_user_window]
                    user_window_scaled = minmax_scale(user_window)
                    quality_windows.append(user_window_scaled)

        return quality_windows

    def create_datasets(
        self, ppg_data_path, test_size=0.2, val_size=0.1, seed=42, save_to_disk=True
    ):
        """
        Create datasets for training, validation, and testing.

        This method loads the processed PPG data and creates train/val/test splits while ensuring:
        1. All segments from the same patient stay together (avoiding data leakage)
        2. Class balance is maintained across all splits

        Args:
            ppg_data_path (str): Path to the directory containing processed PPG data
            test_size (float): Proportion of ALL data to use for testing (0.0-1.0)
            val_size (float): Proportion of ALL data to use for validation (0.0-1.0)
            seed (int): Random seed for reproducibility
            save_to_disk (bool): If True, saves the datasets to disk as .pt files at ppg_data_path

        Returns:
            tuple: (train_dataset, val_dataset, test_dataset) PyTorch Dataset objects

        Raises:
            FileNotFoundError: If participant status file or patient data cannot be found
            ValueError: If no patients are found
        """

        logging.info("Creating datasets from processed patient data...")

        try:
            path_of_patients_status = os.path.join(
                ppg_data_path, "participant_status.csv"
            )
            patients_status = pd.read_csv(path_of_patients_status)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Participant status file not found at {path_of_patients_status}."
                "Please ensure the file exists, is named 'participant_status.csv', and that the path is correct."
            )
        except Exception as e:
            raise ValueError(f"Error reading participant status file: {e}")

        np.random.seed(seed)
        torch.manual_seed(seed)

        patient_ids = [
            patient_id
            for patient_id in os.listdir(ppg_data_path)
            if os.path.isdir(os.path.join(ppg_data_path, patient_id))
            and patient_id.isnumeric()
        ]

        if len(patient_ids) == 0:
            raise ValueError(f"No patients found in directory {ppg_data_path}")

        logging.info(f"Found {len(patient_ids)} patients in directory {ppg_data_path}")

        # Load data and create labels
        ppg_signals, labels, patient_ids_per_sample = self._load_data_and_create_labels(
            ppg_data_path, patients_status, patient_ids
        )

        logging.info(
            f"Loaded {len(ppg_signals)} {self.window_size} second segments from {len(set(patient_ids_per_sample))} patients"
        )

        (
            train_signals,
            train_labels,
            val_signals,
            val_labels,
            test_signals,
            test_labels,
        ) = self._split_by_patient(
            ppg_signals,
            labels,
            patient_ids_per_sample,
            test_size,
            val_size,
        )

        train_dataset = PPGDataset(train_signals, train_labels)
        val_dataset = PPGDataset(val_signals, val_labels)
        test_dataset = PPGDataset(test_signals, test_labels)

        logging.info("Datasets created successfully.")

        if save_to_disk:
            try:
                torch.save(
                    train_dataset, os.path.join(ppg_data_path, "train_dataset.pt")
                )
                torch.save(val_dataset, os.path.join(ppg_data_path, "val_dataset.pt"))
                torch.save(test_dataset, os.path.join(ppg_data_path, "test_dataset.pt"))

            except Exception as e:
                logging.error(f"Error saving datasets to disk: {e}")

            logging.info("Datasets saved to disk successfully.")

        return train_dataset, val_dataset, test_dataset

    def _load_data_and_create_labels(self, ppg_data_path, patients_status, patient_ids):
        """
        Load processed PPG data for each patient and create corresponding labels.

        This method:
        1. Loads processed PPG data for each patient from the saved files
        2. Determines the patient's disease status (PD or HC) from the status DataFrame
        3. Creates and assigns appropriate labels based on the cohort
        4. Returns concatenated signals, labels, and corresponding patient IDs

        Args:
            ppg_data_path (str): Path to the directory containing processed PPG data
            patients_status (pd.DataFrame): DataFrame containing patient status information
            patient_ids (list): List of patient ids to process

        Returns:
            tuple: (
                torch.Tensor: Concatenated tokenized PPG signals,
                torch.Tensor: Labels for each signal (1 for PD, 0 for HC),
                list: Patient IDs corresponding to each signal
            )
        """

        PD_LABEL = 1
        HC_LABEL = 0

        ppg_signals = []
        labels = []
        patient_ids_per_sample = []

        for patient_id in patient_ids:
            try:
                current_patient_ppg = torch.load(
                    os.path.join(ppg_data_path, patient_id, "ppg.pt")
                )
            except FileNotFoundError:
                logging.error(
                    f"PPG file for patient {patient_id} not found in {ppg_data_path}. Skipping this patient."
                )
                continue

            # Get data entry for current patient
            current_patient_status = patients_status[
                patients_status["PATNO"] == int(patient_id)
            ]

            cohort = current_patient_status["COHORT_DEFINITION"].squeeze()
            num_signals = current_patient_ppg.shape[0]

            if len(cohort) == 0:
                logging.error(
                    f"Cohort information not found for patient {patient_id}. Skipping this patient."
                )
                continue

            if cohort == "Healthy Control":
                ppg_signals.append(self._tokenize_ppg(current_patient_ppg))
                labels.extend([HC_LABEL] * num_signals)
                patient_ids_per_sample.extend([patient_id] * num_signals)

            elif cohort == "Parkinson's Disease":
                # Because there are 4 times as many segments for PD patients, we only take 1/4 of the segments
                # to balance the dataset (due to computational constraints I can't oversample the HC patients)
                ppg_signals.append(
                    self._tokenize_ppg(current_patient_ppg[: num_signals // 4, :])
                )
                labels.extend([PD_LABEL] * (num_signals // 4))
                patient_ids_per_sample.extend([patient_id] * (num_signals // 4))
            else:
                logging.warning(
                    f"Only including Parkinson's Disease and Healthy Control in this study. Skipping this patient."
                )
                continue

        return (
            torch.cat(ppg_signals),
            torch.tensor(labels, dtype=torch.float32),
            patient_ids_per_sample,
        )

    def _tokenize_ppg(self, signals):
        """
        Tokenize PPG signals by scaling and rounding to integers.

        This method converts normalized PPG signal values (in range [0,1]) to integer tokens
        in the range [0,100] for use in tokenization-based models.

        Args:
            signals (torch.Tensor): Tensor containing normalized PPG signals (values in [0,1])

        Returns:
            torch.Tensor: Tokenized signals as integer values in range [0-100]
        """
        scale = 100 * signals
        return torch.round(scale).to(torch.int32)

    def _split_by_patient(
        self,
        ppg_signals,
        labels,
        patient_ids_per_sample,
        test_size=0.2,
        val_size=0.1,
    ):
        """
        Split data ensuring all segments from the same patient stay together while maintaining class balance
        and target split sizes.

        Args:
            signals: Tensor of PPG signal data
            labels: Tensor of labels (1 for PD, 0 for HC)
            patient_ids_per_sample: List of patient ids corresponding to each sample of the signal
            test_size (float): Proportion of ALL data to use for testing (0.0-1.0)
            val_size (float): Proportion of ALL data to use for validation (0.0-1.0)

        Returns:
            Tuple of (train_signals, train_labels, val_signals, val_labels, test_signals, test_labels)
        """

        patient_info = self._build_patient_info_dict(patient_ids_per_sample, labels)

        # Calculate target sizes for splits
        (
            pd_patient_ids,
            hc_patient_ids,
            target_pd_ratio,
            target_test_segments,
            target_val_segments,
        ) = self._calculate_split_targets(patient_info, test_size, val_size)

        test_pd_ids, test_hc_ids = self._create_split(
            pd_patient_ids,
            hc_patient_ids,
            patient_info,
            target_test_segments,
            target_pd_ratio,
        )

        # Remove test patients from available patient IDs
        remaining_pd_ids = [p for p in pd_patient_ids if p not in test_pd_ids]
        remaining_hc_ids = [p for p in hc_patient_ids if p not in test_hc_ids]

        val_pd_ids, val_hc_ids = self._create_split(
            remaining_pd_ids,
            remaining_hc_ids,
            patient_info,
            target_val_segments,
            target_pd_ratio,
        )

        # Use remaining patients for training split
        train_pd_ids = [pid for pid in remaining_pd_ids if pid not in val_pd_ids]
        train_hc_ids = [pid for pid in remaining_hc_ids if pid not in val_hc_ids]

        def _get_indices_for_split(patient_ids):
            indices = []
            for patient_id in patient_ids:
                indices.extend(patient_info[patient_id]["indices"])
            return indices

        # Get indices for each split based on patient IDs assigned to each class per split
        train_indices = _get_indices_for_split(train_pd_ids + train_hc_ids)
        val_indices = _get_indices_for_split(val_pd_ids + val_hc_ids)
        test_indices = _get_indices_for_split(test_pd_ids + test_hc_ids)

        # Split signals and labels based on indices
        train_signals, train_labels = ppg_signals[train_indices], labels[train_indices]
        val_signals, val_labels = ppg_signals[val_indices], labels[val_indices]
        test_signals, test_labels = ppg_signals[test_indices], labels[test_indices]

        self._print_distribution_statistics(train_labels, val_labels, test_labels)

        return (
            train_signals,
            train_labels,
            val_signals,
            val_labels,
            test_signals,
            test_labels,
        )

    def _build_patient_info_dict(self, patient_ids_per_sample, labels):
        """
        Build a dictionary containing patient-level information, including indices and labels.

        Args:
            patient_ids_per_sample (list): List of patient IDs corresponding to each sample
            labels (torch.Tensor): Labels for each sample

        Returns:
            dict: Dictionary with patient IDs as keys and dictionaries containing:
                - label: Class label (0 for HC, 1 for PD)
                - segment_count: Number of segments for this patient
                - indices: List of indices in the original arrays for this patient
        """
        unique_patient_ids = list(set(patient_ids_per_sample))
        patient_info = {}

        for patient_id in unique_patient_ids:
            indices = [
                i for i, pid in enumerate(patient_ids_per_sample) if pid == patient_id
            ]
            label = labels[indices[0]].item()
            segment_count = len(indices)

            patient_info[patient_id] = {
                "label": label,
                "segment_count": segment_count,
                "indices": indices,
            }

        return patient_info

    def _calculate_split_targets(self, patient_info, test_size=0.2, val_size=0.1):
        """
        Calculate target sizes for train/val/test splits while preserving class ratios.

        Args:
            patient_info (dict): Dictionary containing patient-level information
            test_size (float): Proportion of ALL data to use for testing (0.0-1.0)
            val_size (float): Proportion of ALL data to use for validation (0.0-1.0)

        Returns:
            tuple: (
                pd_patient_ids (list): List of PD patient IDs,
                hc_patient_ids (list): List of HC patient IDs,
                target_pd_ratio (float): Target ratio of PD samples,
                target_test_segments (int): Target number of segments for test split,
                target_val_segments (int): Target number of segments for validation split
            )
        """
        # Separate patients by class
        pd_patient_ids = [
            pid for pid, info in patient_info.items() if info["label"] == 1
        ]
        hc_patient_ids = [
            pid for pid, info in patient_info.items() if info["label"] == 0
        ]

        # Calculate total number of segments for each class
        total_pd_segments = sum(
            patient_info[pid]["segment_count"] for pid in pd_patient_ids
        )
        total_hc_segments = sum(
            patient_info[pid]["segment_count"] for pid in hc_patient_ids
        )

        # Calculate target class-distribution ratio
        target_pd_ratio = total_pd_segments / (total_pd_segments + total_hc_segments)

        # Calculate target number of segments for each split
        total_segments = total_pd_segments + total_hc_segments
        target_test_segments = total_segments * test_size

        # Adjust validation size based on size used for test split
        adjusted_val_size = val_size / (1 - test_size)
        target_val_segments = (
            total_segments - target_test_segments
        ) * adjusted_val_size

        return (
            pd_patient_ids,
            hc_patient_ids,
            target_pd_ratio,
            target_test_segments,
            target_val_segments,
        )

    def _create_split(
        self, pd_ids, hc_ids, patient_info, num_target_segments, target_pd_ratio
    ):
        """
        Create a balanced data split by selecting appropriate subsets of patients from each class.

        This method creates a dataset split (train/val/test) by selecting patient IDs from both
        classes (PD and HC) while maintaining the desired class balance and target segment count.
        For each class, it calls _find_optimal_patient_subset to identify the best combination
        of patients that achieves the target number of segments for that class.

        Args:
            pd_ids (list): List of available patient IDs with Parkinson's Disease
            hc_ids (list): List of available patient IDs who are Healthy Controls
            patient_info (dict): Dictionary mapping patient IDs to their information including:
                - label: Class label (0 for HC, 1 for PD)
                - segment_count: Number of segments for this patient
                - indices: List of indices in the original arrays for this patient
            num_target_segments (int): Total number of segments desired for this split
            target_pd_ratio (float): Target ratio of PD segments in the split (0.0-1.0)

        Returns:
            tuple: (
                split_pd_ids (list): Selected PD patient IDs for this split
                split_hc_ids (list): Selected HC patient IDs for this split
            )

        Note:
            The actual number of segments in the resulting split may differ slightly from
            the target due to the constraint that patients cannot be split across different
            dataset partitions.
        """

        target_hc_ratio = 1 - target_pd_ratio

        # Calculate target number of segments for split per class
        target_pc_segments = int(num_target_segments * target_pd_ratio)
        target_hc_segments = int(num_target_segments * target_hc_ratio)

        # Get patient ids used for split for each class
        split_pd_ids = self._find_optimal_patient_subset(
            pd_ids, target_pc_segments, patient_info
        )
        split_hc_ids = self._find_optimal_patient_subset(
            hc_ids, target_hc_segments, patient_info
        )

        return split_pd_ids, split_hc_ids

    def _find_optimal_patient_subset(
        self, patient_ids, num_target_segments, patient_info
    ):
        """
        Find an optimal subset of patients whose total segment count is closest to the target number.
        Iteratively tries different starting points to find the best consecutive sequence of patients
        that minimizes the difference between their total segment count and the target number.

        Args:
            patient_ids (list): List of patient identifiers
            target_number_of_segments (int): Desired total number of segments to achieve
            patient_info: Dictionary mapping patient IDs to their information
                        Format: patient_id: {
                                "label": int, 0 or 1 for HC/PD
                                "segment_count": int, number of segments for this patient
                                "indices": List[int] indices of segments belonging to this patient
                            }

        Returns:
            list: Selected patient IDs whose combined segment count best matches the target
        """

        # Sort patient ids by segment count
        patient_ids = sorted(
            patient_ids, key=lambda x: patient_info[x]["segment_count"]
        )

        # Initialize variables to track best selection and difference
        best_difference = float("inf")
        best_selection = []

        # Try each possible starting point to find the optimal subset
        for start_idx in range(len(patient_ids)):
            current_selection = []
            current_sum = 0

            # Iterate through patients starting from current start_idx
            for current_patient_id in patient_ids[start_idx:]:
                # Add patients while staying under target
                if (
                    current_sum + patient_info[current_patient_id]["segment_count"]
                    <= num_target_segments
                ):
                    current_selection.append(current_patient_id)
                    current_sum += patient_info[current_patient_id]["segment_count"]

            # Update best solution if current selection is closer to target
            current_difference = abs(current_sum - num_target_segments)
            if current_difference < best_difference:
                best_difference = current_difference
                best_selection = current_selection

        return best_selection

    def _print_distribution_statistics(self, train_labels, val_labels, test_labels):
        logging.info("Distribution statistics:")
        # Print distribution statistics
        for split_name, split_labels in [
            ("Train", train_labels),
            ("Validation", val_labels),
            ("Test", test_labels),
        ]:
            total = len(split_labels)
            pd_count = split_labels.sum().item()
            non_pd_count = total - pd_count
            logging.info(f"\n{split_name} set:")
            logging.info(f"Total: {total}")
            logging.info(f"PD: {pd_count} ({pd_count / total * 100:.1f}%)")
            logging.info(f"Non-PD: {non_pd_count} ({non_pd_count / total * 100:.1f}%)")


def get_dataloaders(cfg):
    """
    Creates PyTorch DataLoaders for train, validation, and test sets.

    Args:
        cfg (Config): Configuration object containing model settings and data paths
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """

    train_dataset = torch.load(
        os.path.join(cfg.data_path, "train_dataset.pt"), weights_only=False
    )
    val_dataset = torch.load(
        os.path.join(cfg.data_path, "val_dataset.pt"), weights_only=False
    )
    test_dataset = torch.load(
        os.path.join(cfg.data_path, "test_dataset.pt"), weights_only=False
    )

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


def main():
    """
    Main entry point for the PPG signal processing pipeline.

    This function orchestrates the end-to-end process of:
    1. Setting up logging infrastructure
    2. Configuring parameters
    3. Initializing the PPG processor
    4. Processing raw PPG data for all patients
    5. Creating datasets for training, validation, and testing
    6. Saving datasets to disk

    The processing pipeline has two main stages:
    1. Signal processing: Reading, cleaning, filtering, and segmenting PPG signals
    2. Dataset creation: Splitting processed data into train/val/test sets
    """
    os.makedirs("logs", exist_ok=True)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(processName)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("logs/ppg_processing.log"),
            logging.StreamHandler(),
        ],
    )

    path_to_load_data = "/Volumes/Elements/STUDYWATCH/"
    path_to_save_data = "data/"
    original_sampling_rate = 30
    target_sampling_rate = 50
    window_size = 10
    stride = 10
    skip_start_and_end = 100
    num_processes = 12

    processor = PPGProcessor(
        path_to_load_data,
        path_to_save_data,
        original_sampling_rate=original_sampling_rate,
        target_sampling_rate=target_sampling_rate,
        window_size=window_size,
        stride=stride,
        skip_start_and_end=skip_start_and_end,
        num_processes=num_processes,
    )
    #processor.process_dataset()

    test_size = 0.2
    val_size = 0.1
    seed = 42
    processor.create_datasets(
        path_to_save_data,
        test_size=test_size,
        val_size=val_size,
        seed=seed,
        save_to_disk=True,
    )


if __name__ == "__main__":
    main()
