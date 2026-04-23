import pandas as pd
import numpy as np
from pathlib import Path

# Paths
clinical_csv = Path("data/clinical/adni_clinical.csv")
labels_csv = Path("data/processed/networks/labels.csv")
save_dir = Path("data/clinical")

# Load data
clinical_df = pd.read_csv(clinical_csv)
labels_df = pd.read_csv(labels_csv)

# Build RID → SubjectID mapping from labels
labels_df["RID"] = labels_df["SubjectID"].apply(lambda x: int(x.split("_")[-1]))

rid_to_subject = dict(zip(labels_df["RID"], labels_df["SubjectID"]))

# Keep only rows with matching RID
clinical_df = clinical_df[clinical_df["RID"].isin(rid_to_subject.keys())]

# Encode gender
clinical_df["PTGENDER"] = clinical_df["PTGENDER"].map({"Male": 0, "Female": 1})

# Encode diagnosis
label_map = {"AD": 0, "CN": 1, "MCI": 2, "EMCI": 2, "LMCI": 2}
clinical_df["DX_BL"] = clinical_df["DX_BL"].map(label_map)

# Drop invalid rows
clinical_df = clinical_df.dropna(subset=["AGE", "PTGENDER", "DX_BL"])

# Create SubjectID properly
clinical_df["SubjectID"] = clinical_df["RID"].map(rid_to_subject)

# Extract features
features = clinical_df[["AGE", "PTGENDER"]].values
subject_ids = clinical_df["SubjectID"].values.astype(str)

# Save
np.save(save_dir / "clinical_features.npy", features)
np.save(save_dir / "clinical_subject_ids.npy", subject_ids)

print("Clinical features generated correctly!")
