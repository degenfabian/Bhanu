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

Johnson, A., Pollard, T., & Mark, R. (2016).
MIMIC-III Clinical Database (version 1.4).
PhysioNet. https://doi.org/10.13026/C2XW26.

Moody, B., Moody, G., Villarroel, M., Clifford, G. D., & Silva, I. (2020).
MIMIC-III Waveform Database (version 1.0).
PhysioNet. https://doi.org/10.13026/c2607m.

Johnson, A., Pollard, T., Shen, L. et al.
MIMIC-III, a freely accessible critical care database.
Sci Data 3, 160035 (2016).
https://doi.org/10.1038/sdata.2016.35

Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000).
PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals.
Circulation [Online]. 101 (23), pp. e215–e220.
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
    subject_id = str(subject_id).zfill(
        6
    )  # Ensure subject ID is 6 digits, pad with zeros if necessary
    parent_dir = (
        "p" + subject_id[:2]
    )  # Group by first two digits of subject ID for MIMIC-III structure
    subject_dir = parent_dir + "/p" + subject_id
    full_target_dir = os.path.join(target_dir, subject_dir)

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
    temp_dir = os.path.join(full_target_dir, "_temp")
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
            shutil.rmtree(temp_dir)
        return False


def print_dataset_statistics(
    pd_patient_data,
    non_pd_patient_data,
):
    """
    Prints statistics about the dataset including demographics and download status.

    Args:
        pd_patient_data: DataFrame containing Parkinson's disease patient data
        non_pd_patient_data: DataFrame containing control patient data

    Returns:
        None
    """

    # Map gender back to string values for better readability
    pd_patient_data["GENDER"] = pd_patient_data["GENDER"].map({1: "M", 0: "F"})
    non_pd_patient_data["GENDER"] = non_pd_patient_data["GENDER"].map(
        {1: "M", 0: "F"}
    )

    # Every patient aged over 89 is set to 300 in MIMIC-III for privacy reasons
    # I set them back to 90 in order for the statistic calculations carried out later to be more accurate
    pd_patient_data.loc[pd_patient_data["AGE"] == 300, "AGE"] = 90
    non_pd_patient_data.loc[non_pd_patient_data["AGE"] == 300, "AGE"] = 90

    print(f"\nDownload statistics:")
    print(f"PD patients: {len(pd_patient_data)}")
    print(f"Control patients: {len(non_pd_patient_data)}")
    print(f"Total: {len(pd_patient_data) + len(non_pd_patient_data)}")

    # Age statistics
    pd_age = pd_patient_data["AGE"].describe()
    control_age = non_pd_patient_data["AGE"].describe()

    print("\nAge Statistics:")
    print("              PD      Controls")
    print(f"Mean age:   {pd_age['mean']:6.1f}  {control_age['mean']:6.1f}")
    print(f"Std dev:    {pd_age['std']:6.1f}  {control_age['std']:6.1f}")
    print(f"Median age: {pd_age['50%']:6.1f}  {control_age['50%']:6.1f}")
    print(f"Min age:      {pd_age['min']:6.1f}  {control_age['min']:6.1f}")
    print(f"Max age:      {pd_age['max']:6.1f}  {control_age['max']:6.1f}")

    # Gender statistics
    pd_gender = pd_patient_data["GENDER"].value_counts(normalize=True) * 100
    control_gender = non_pd_patient_data["GENDER"].value_counts(normalize=True) * 100

    print("\nGender Distribution (%):")
    print("              PD      Controls")
    print(f"Female:      {pd_gender['F']:6.1f}  {control_gender['F']:6.1f}")
    print(f"Male:        {pd_gender['M']:6.1f}  {control_gender['M']:6.1f}")


def main():
    """
    Main execution function that:
    1. Loads and processes MIMIC-III clinical data
    2. Identifies PD patients and matches control patients
    3. Downloads waveform data for both groups
    """

    # Process clinical data and match patients
    non_pd_patient_data, pd_patient_data, pd_subject_ids = read_data(
        "/Users/degenfabian/Documents/VSCODE/Bhanu/data/mimic-iii-clinical-database-1.4"
    )  # Input the path to your MIMIC-III clinical database directory here

    # Match control patients to PD patients according to demographic criteria
    matched_non_pd_with_backup = match_healthy_to_PD(
        non_pd_patient_data,
        pd_patient_data,
        matching_criteria=["GENDER", "AGE"],
        n_fallback=12,
    )

    # Track PD patients that were successfully downloaded to download exactly one control patient per PD patient
    downloaded_pd_patient_ids = []

    # Download waveforms for PD patients
    for subject_id in tqdm(pd_subject_ids):
        if download_patient_waveforms(
            str(subject_id), target_dir="data/waveform_data/PD/"
        ):
            downloaded_pd_patient_ids.append(subject_id)

    # Track control patients that were successfully downloaded for calculating dataset statistics
    downloaded_non_pd_patient_ids = []

    # Download waveforms for matched control patients
    for pd_subject_id in tqdm(downloaded_pd_patient_ids):
        # Find the index of the PD patient in the original data to find the matched control patient (and backups) for that specific PD patient
        pd_index = np.where(pd_subject_ids == pd_subject_id)[0][0]

        # Extract the matched control patient (and backups) for this PD patient
        matched_non_pd_patients = matched_non_pd_with_backup[pd_index]

        # Iterate through the matched control patient (and backups) for this PD patient and stop after the first successful download
        for non_pd_subject_id in matched_non_pd_patients:
            if download_patient_waveforms(
                str(non_pd_subject_id), target_dir="data/waveform_data/non_PD/"
            ):
                downloaded_non_pd_patient_ids.append(non_pd_subject_id)
                break
    )


if __name__ == "__main__":
    main()
