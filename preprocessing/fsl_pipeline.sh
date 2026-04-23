#!/bin/bash

# FSL-based preprocessing pipeline for ADNI DTI data.
# This script performs skull stripping, eddy current correction,
# diffusion tensor fitting, and FA registration to MNI space
# for all subject directories under data/raw/DTI/.

set -euo pipefail

# Resolve project root as the parent of the preprocessing directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RAW_DTI_DIR="${PROJECT_ROOT}/data/raw/DTI"
PROCESSED_FA_DIR="${PROJECT_ROOT}/data/processed/FA"
PROCESSED_MD_DIR="${PROJECT_ROOT}/data/processed/MD"

mkdir -p "${PROCESSED_FA_DIR}" "${PROCESSED_MD_DIR}"

echo "=== Starting FSL DTI preprocessing pipeline ==="
echo "Project root: ${PROJECT_ROOT}"
echo "Raw DTI directory: ${RAW_DTI_DIR}"

if [ ! -d "${RAW_DTI_DIR}" ]; then
  echo "ERROR: Raw DTI directory '${RAW_DTI_DIR}' does not exist." >&2
  exit 1
fi

shopt -s nullglob
SUBJECT_DIRS=("${RAW_DTI_DIR}"/*)

if [ "${#SUBJECT_DIRS[@]}" -eq 0 ]; then
  echo "ERROR: No subject directories found in '${RAW_DTI_DIR}'." >&2
  exit 1
fi

for SUBJECT_PATH in "${SUBJECT_DIRS[@]}"; do
  if [ ! -d "${SUBJECT_PATH}" ]; then
    continue
  fi

  SUBJECT_ID="$(basename "${SUBJECT_PATH}")"
  echo "---- Processing subject: ${SUBJECT_ID} ----"

  # Assumed raw DTI file path:
  DTI_INPUT="${SUBJECT_PATH}/DTI.nii.gz"
  if [ ! -f "${DTI_INPUT}" ]; then
    echo "WARNING: DTI file not found for subject '${SUBJECT_ID}' at '${DTI_INPUT}', skipping." >&2
    continue
  fi

  # Output prefixes and paths per subject.
  SUBJECT_WORKDIR="${SUBJECT_PATH}/fsl_preproc"
  mkdir -p "${SUBJECT_WORKDIR}"

  BET_OUTPUT="${SUBJECT_WORKDIR}/DTI_brain"
  EDDY_OUTPUT="${SUBJECT_WORKDIR}/DTI_eddy"
  DTIFIT_PREFIX="${SUBJECT_WORKDIR}/dti"

  # You may need to adjust these to match your ADNI file names.
  BVEC_FILE="${SUBJECT_PATH}/bvecs"
  BVAL_FILE="${SUBJECT_PATH}/bvals"

  if [ ! -f "${BVEC_FILE}" ] || [ ! -f "${BVAL_FILE}" ]; then
    echo "WARNING: Missing bvecs/bvals for subject '${SUBJECT_ID}', expected at:" >&2
    echo "         ${BVEC_FILE}" >&2
    echo "         ${BVAL_FILE}" >&2
    echo "         Skipping subject." >&2
    continue
  fi

  echo "Step 1: Skull strip the DTI brain image (BET)."
  echo "Running: bet \"${DTI_INPUT}\" \"${BET_OUTPUT}\" -f 0.3"
  bet "${DTI_INPUT}" "${BET_OUTPUT}" -f 0.3

  echo "Step 2: Eddy current correction."
  echo "Running: eddy_correct \"${BET_OUTPUT}.nii.gz\" \"${EDDY_OUTPUT}.nii.gz\" 0"
  eddy_correct "${BET_OUTPUT}.nii.gz" "${EDDY_OUTPUT}.nii.gz" 0

  echo "Step 3: Fit diffusion tensor model (DTIFIT)."
  echo "Running: dtifit --data=\"${EDDY_OUTPUT}.nii.gz\" --out=\"${DTIFIT_PREFIX}\" --mask=\"${BET_OUTPUT}_mask.nii.gz\" --bvecs=\"${BVEC_FILE}\" --bvals=\"${BVAL_FILE}\""
  dtifit \
    --data="${EDDY_OUTPUT}.nii.gz" \
    --out="${DTIFIT_PREFIX}" \
    --mask="${BET_OUTPUT}_mask.nii.gz" \
    --bvecs="${BVEC_FILE}" \
    --bvals="${BVAL_FILE}"

  # This generates dti_FA.nii.gz and dti_MD.nii.gz in SUBJECT_WORKDIR.

  echo "Step 4: Register FA map to MNI standard space."
  FA_INPUT="${DTIFIT_PREFIX}_FA.nii.gz"
  FA_MNI_OUTPUT="${PROCESSED_FA_DIR}/${SUBJECT_ID}_dti_FA_MNI.nii.gz"
  AFFINE_MAT="${SUBJECT_WORKDIR}/affine.mat"

  echo "Running: flirt -in \"${FA_INPUT}\" -ref \"${FSLDIR}/data/standard/MNI152_T1_1mm_brain\" -out \"${FA_MNI_OUTPUT}\" -omat \"${AFFINE_MAT}\""
  flirt \
    -in "${FA_INPUT}" \
    -ref "${FSLDIR}/data/standard/MNI152_T1_1mm_brain" \
    -out "${FA_MNI_OUTPUT}" \
    -omat "${AFFINE_MAT}"

  # Optionally copy MD map to processed directory for later use.
  MD_INPUT="${DTIFIT_PREFIX}_MD.nii.gz"
  MD_OUTPUT="${PROCESSED_MD_DIR}/${SUBJECT_ID}_dti_MD.nii.gz"
  if [ -f "${MD_INPUT}" ]; then
    cp "${MD_INPUT}" "${MD_OUTPUT}"
  fi

  echo "---- Finished subject: ${SUBJECT_ID} ----"
done

echo "=== Done: FSL DTI preprocessing pipeline ==="

