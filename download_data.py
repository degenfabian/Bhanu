"""
MIMIC-III Waveform Data Processor

This script processes MIMIC-III clinical and waveform data for research purposes.
To use this script, you must:
1. Have valid PhysioNet credentials
2. Have signed the MIMIC-III data use agreement
3. Have completed required CITI training
4. Have local access to the MIMIC-III clinical database

For more information about MIMIC-III access requirements:
https://physionet.org/content/mimiciii/
"""

import pandas as pd
import wfdb
import os
from tqdm import tqdm
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler


def read_data(mimic_path):
    """
    Reads MIMIC-III clinical data files and processes patient diagnoses.

    Args:
        mimic_path: Path to MIMIC-III clinical database files

    Returns:
        tuple: (patients_without_PD, patients_with_PD, pd_subject_ids)
            - patients_without_PD: DataFrame of control patients
            - patients_with_PD: DataFrame of PD patients
            - pd_subject_ids: Array of subject IDs for PD patients
    """

    diagnoses = pd.read_csv(os.path.join(mimic_path, "DIAGNOSES_ICD.csv.gz"))
    admissions = pd.read_csv(os.path.join(mimic_path, "ADMISSIONS.csv.gz"))
    patients = pd.read_csv(os.path.join(mimic_path, "PATIENTS.csv.gz"))

    # Merge patient data with admission times and remove duplicates
    patients_with_admittime = patients.merge(
        admissions, on="SUBJECT_ID"
    ).drop_duplicates(subset="SUBJECT_ID")

    # Convert admission time and date of birth to datetime objects
    patients_with_admittime["ADMITTIME"] = pd.to_datetime(
        patients_with_admittime["ADMITTIME"]
    ).dt.date

    patients_with_admittime["DOB"] = pd.to_datetime(
        patients_with_admittime["DOB"]
    ).dt.date

    # Calculate patient age at time of admission
    # Taken from: https://stackoverflow.com/questions/56720783/how-to-fix-overflowerror-overflow-in-int64-addition
    patients["AGE"] = (
        patients_with_admittime["ADMITTIME"].to_numpy()
        - patients_with_admittime["DOB"].to_numpy()
    ).astype("timedelta64[D]").astype(int) // 365

    patients_diagnoses = patients.merge(diagnoses, on="SUBJECT_ID")

    # Extract the subject IDs of patients with Parkinson's disease
    patients_with_PD = patients_diagnoses[
        patients_diagnoses["ICD9_CODE"] == "3320"
    ].drop_duplicates(subset="SUBJECT_ID")
    pd_subject_ids = patients_with_PD["SUBJECT_ID"].unique()

    # Extract the subject IDs of patients without Parkinson's disease
    patients_without_PD = patients_diagnoses[
        ~patients_diagnoses["SUBJECT_ID"].isin(pd_subject_ids)
    ].drop_duplicates(subset="SUBJECT_ID")

    return patients_without_PD, patients_with_PD, pd_subject_ids


def match_healthy_to_PD(non_pd, pd, matching_criteria=["GENER", "AGE"]):
    """
    Matches control patients to PD patients based on demographic characteristics.
    Uses gender and age for matching criteria by default.

    Args:
        non_pd: DataFrame containing control patient data
        pd: DataFrame containing PD patient data
        matching_criteria: List of columns to use for matching patients

    Returns:
        tuple: (matched_controls, control_subject_ids)
            - matched_controls: DataFrame of matched control patients
            - control_subject_ids: Array of matched control subject IDs
    """

    # Map Gender information to integers
    pd["GENDER"] = pd["GENDER"].map({"M": 1, "F": 0})
    non_pd["GENDER"] = non_pd["GENDER"].map({"M": 1, "F": 0})

    # Extract features used for matching
    pd_features = pd[matching_criteria].to_numpy()
    non_pd_features = non_pd[matching_criteria].to_numpy()

    all_features = np.vstack([pd_features, non_pd_features])

    # Standardize features to ensure equal weighting in distance calculation
    scaler = StandardScaler()
    scaler.fit(all_features)
    normalized_PD = scaler.transform(pd_features)
    normalized_non_PD = scaler.transform(non_pd_features)

    distances = cdist(normalized_PD, normalized_non_PD, "euclidean")

    # Match each PD patient with the closest unmatched control patient
    matched_indices = set()
    for i in range(len(pd)):
        indices = np.argsort(distances[i])

        # Find the closest unmatched control patient
        for j in indices:
            if j not in matched_indices:
                matched_indices.add(j)
                break

    # Select the matched control patients
    matched_controls = non_pd.iloc[list(matched_indices)]

    print("Matching results:")
    print(f"PD patients: {len(pd)}")
    print(f"Matched controls: {len(matched_controls)}")
    print("\nAge distributions:")
    print("PD:", pd["AGE"].describe())
    print("Controls:", matched_controls["AGE"].describe())
    print("\nGender distributions:")
    print("PD:", pd["GENDER"].value_counts(normalize=True))
    print("Controls:", matched_controls["GENDER"].value_counts(normalize=True))

    return matched_controls, matched_controls["SUBJECT_ID"].unique()


def download_patient_waveforms(subject_id, target_dir="/data/waveform_data/"):
    """
    Downloads waveform data for a specific patient from MIMIC-III database.

    Args:
        subject_id: Patient's subject ID
        target_dir: Base directory for saving downloaded waveform data

    Returns:
        bool: True if download successful or data exists, False if download fails
    """

    # Format subject ID and construct directory paths following MIMIC-III structure
    subject_id = subject_id.zfill(
        6
    )  # Ensure subject ID is 6 digits, pad with zeros if necessary
    parent_dir = (
        "p" + subject_id[:2] + "/"
    )  # Group by first two digits of subject ID for MIMIC-III structure
    subject_dir = parent_dir + "p" + subject_id + "/"
    full_target_dir = target_dir + subject_dir

    # Check if data already exists to avoid redundant downloads
    if os.path.exists(full_target_dir):
        # Check if directory contains any files
        if any(os.scandir(full_target_dir)):
            print(f"Skipping patient {subject_id} - data already exists")
            return True
        else:
            # Clean up directory if empty
            os.rmdir(full_target_dir)

    # Set up temporary download directory
    temp_dir = full_target_dir[:-1] + "_temp" + "/"
    try:
        os.makedirs(temp_dir, exist_ok=True)

        # Download waveform data from MIMIC-III
        wfdb.dl_database("mimic3wdb/matched/" + subject_dir, temp_dir)

        # Rename temporary directory to final target directory
        os.rename(temp_dir, full_target_dir)
        print(f"Successfully downloaded data for patient {subject_id}")
        return True

    except Exception as e:
        print(f"Error downloading data for patient {subject_id}: {str(e)}")
        # Clean up failed download
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)
        return False


def main():
    """
    Main execution function that:
    1. Loads and processes MIMIC-III clinical data
    2. Identifies PD patients and matches control patients
    3. Downloads waveform data for both groups
    """

    # Process clinical data and match patients
    non_pd, pd, pd_subject_ids = read_data()
    non_pd, non_pd_subject_ids = match_healthy_to_PD(non_pd, pd)

    total_patients = len(pd_subject_ids) + len(non_pd_subject_ids)
    successful_downloads = 0

    # Download waveforms for PD patients
    for subject_id in tqdm(pd_subject_ids):
        if download_patient_waveforms(
            str(subject_id), target_dir="/data/waveform_data/PD/"
        ):
            successful_downloads += 1

    # Download waveforms for matched control patients
    for subject_id in tqdm(non_pd_subject_ids):
        if download_patient_waveforms(
            str(subject_id), target_dir="/data/waveform_data/non_PD/"
        ):
            successful_downloads += 1

    print(
        f"Successfully downloaded data for {successful_downloads} out of {total_patients} PD patients"
    )


if __name__ == "__main__":
    main()
