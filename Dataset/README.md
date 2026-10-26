# Dataset access

The cleaned dataset used by the analysis is archived on Zenodo:

- [Latest dataset release (concept DOI)](https://doi.org/10.5281/zenodo.15349730)
- [Exact release used here: version 4, published 20 October 2026](https://doi.org/10.5281/zenodo.23496955)
- [Data description in IEEE Access](https://doi.org/10.1109/ACCESS.2025.3569164)

Download file 3 of version 4, `3.cleaned_dataset_per_device.csv`, as `Data_Files/cleaned_dataset_per_device.csv` at the repository root and check it:

```bash
mkdir -p Data_Files
curl -L https://zenodo.org/api/records/23496955/files/3.cleaned_dataset_per_device.csv/content -o Data_Files/cleaned_dataset_per_device.csv
printf '%s  %s\n' 638d918ffc3a15bb10564c852df6e6b0 Data_Files/cleaned_dataset_per_device.csv | md5sum --check
```

The file contains 2,660,274 cleaned observations from 1 October 2024 through 30 September 2025, at spreading factors 7 to 12, with the transmit power of each uplink in the column `tx_power`, and is distributed under CC BY 4.0.
