#!/bin/bash
# Downloads file 3 of version 4 of the Zenodo record into Data_Files/ and checks it.
set -e
cd "$(dirname "$0")"
mkdir -p Data_Files
curl -L https://zenodo.org/api/records/23496955/files/3.cleaned_dataset_per_device.csv/content -o Data_Files/cleaned_dataset_per_device.csv
printf '%s  %s\n' 638d918ffc3a15bb10564c852df6e6b0 Data_Files/cleaned_dataset_per_device.csv | md5sum --check
