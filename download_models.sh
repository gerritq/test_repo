#!/bin/bash

TO DO: check folder, no need for len
cd models generalise/data/detect_len
mkdir -p models
cd models

# TO DO
FILE_ID="YOUR_GOOGLE_DRIVE_FILE_ID"
ZIP_NAME="hp_len.zip"

if [ ! -d "hp_len" ]; then
  echo "Downloading model archive from Google Drive..."
  gdown --id "$FILE_ID" -O "$ZIP_NAME"
  echo "Unzipping..."
  unzip -q "$ZIP_NAME"
  rm "$ZIP_NAME"
  echo "Models downloaded and extracted to models/hp_len/"
else
  echo "Model directory already exists. Skipping download."
fi