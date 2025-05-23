import os
import pydicom
from pydicom.errors import InvalidDicomError

ftp_root = "/Volumes/T7/FTP"
log_file = "corrupted_dicom_files.txt"

corrupted = []
total = 0

for root, _, files in os.walk(ftp_root):
    for fname in files:
        if not fname.lower().endswith(".dcm"):
            continue

        total += 1
        path = os.path.join(root, fname)

        try:
            ds = pydicom.dcmread(path, stop_before_pixels=False)
            _ = ds.pixel_array
        except (InvalidDicomError, AttributeError, ValueError, Exception) as e:
            print(f"Corrupted: {path} | Reason: {str(e)}")
            corrupted.append(path)

if corrupted:
    with open(log_file, "w") as f:
        for c in corrupted:
            f.write(c + "\n")
    print(f"\nCorrupted file paths saved to: {log_file}")
