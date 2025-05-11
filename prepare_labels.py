### Prepare labels for the DKIM dataset to predict age/sex

import os
import pandas as pd
from tqdm import tqdm

# --- Config ---
ftp_root = "/Volumes/T7/FTP"               # Path where original folders are
input_xlsx = "raw_labels_sex_age.xlsx"        # Your original metadata
output_xlsx = "labels_sex_age.xlsx"  # Where to save expanded file

# --- Load metadata ---
df = pd.read_excel(input_xlsx)
participant_ids = set(df["participant_id"].astype(str))

# --- Prepare expanded records ---
records = []

for date_folder in tqdm(os.listdir(ftp_root)):
    full_date_path = os.path.join(ftp_root, date_folder)
    if not os.path.isdir(full_date_path):
        continue

    for filename in os.listdir(full_date_path):
        if not filename.endswith(".dcm"):
            continue

        parts = filename.split("_")
        if not parts or not parts[0].isdigit():
            continue

        pid = parts[0]
        if pid not in participant_ids:
            continue

        # Lookup metadata
        metadata_row = df[df["participant_id"] == int(pid)].iloc[0]
        full_path = os.path.join(full_date_path, filename)

        records.append({
            "participant_id": pid,
            "gender": metadata_row["gender"],
            "age": metadata_row["age"],
            "filepath": full_path
        })

# --- Save as both .xlsx and .csv ---
expanded_df = pd.DataFrame(records)
expanded_df.to_excel(output_xlsx, index=False)

print(f"Labels saved to: {output_xlsx}")
