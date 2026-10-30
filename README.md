# Indoor LoRaWAN path loss modeling

Fade margins for indoor LoRaWAN links as a function of what the predictor knows: propagation models, machine learning regressors and the link's own earlier packets, evaluated on a later period, with the margins calibrated and tested as quantile forecasts.

[![Dataset](https://img.shields.io/badge/dataset-Zenodo-1682D4?logo=zenodo&logoColor=white)](https://doi.org/10.5281/zenodo.15349730)
[![Data paper](https://img.shields.io/badge/data%20paper-IEEE%20Access-00629B?logo=ieee&logoColor=white)](https://doi.org/10.1109/ACCESS.2025.3569164)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)

## What is here

One year of path loss measurements from six fixed links on one office floor of the University of Siegen: 2,660,274 uplinks from 1 October 2024 to 30 September 2025, spreading factors 7 to 12, five environmental sensors at each device. The notebooks fit five propagation models and six machine learning model families, first on static inputs (distance, wall counts, channel frequency, the sensors), then on cumulative rungs of link history: the link's earlier packets, the history of the same channel, the clock, the spreading factor, the SNR of earlier packets, the sensors, and the level and geometry of the link. One chronological protocol throughout: the earliest 80 % of the packets train, five forward-chaining folds select, the latest 20 % are held out and used once. A fade margin is the empirical quantile of the out-of-fold residuals (the split-conformal shift of `margin_tools.py`), evaluated on the hold-out by coverage within operating states, pinball loss and the clustering of the misses. The same procedure runs on two public datasets, the Medellín measurements of González-Palacio et al. and the campus metadata of the University of Virginia.

## Data

Download file 3 of version 4 of the Zenodo record and check it:

```bash
mkdir -p Data_Files
curl -L https://zenodo.org/api/records/23496955/files/3.cleaned_dataset_per_device.csv/content -o Data_Files/cleaned_dataset_per_device.csv
printf '%s  %s\n' 638d918ffc3a15bb10564c852df6e6b0 Data_Files/cleaned_dataset_per_device.csv | md5sum --check
```

The file has 2,660,274 rows with the RSSI and SNR of the project gateway, pressure in hPa, the transmit power the network server set for each uplink (`tx_power`) and the path loss computed from it (`exp_pl`). The [data pipeline](https://github.com/nahshonmokua/LoRaWAN-Indoor-PathLoss-Dataset-IEEEACCESS) regenerates it from the raw export. Notebook 19 reads `LoRaWAN_PathLossMeasurements.csv` from [magonzalezudem/MDPI_LoRaWAN_Dataset_With_Environmental_Variables](https://github.com/magonzalezudem/MDPI_LoRaWAN_Dataset_With_Environmental_Variables), placed in `Data_Files/`; notebook 21 reads `lorawan_combined_dataset.parquet` and `deployment_weather.parquet` from the [Dataverse record](https://doi.org/10.18130/V3/RFTICK), placed in `Data_Files/uva_lorawan/`.

## Environments

Python 3.12. The main environment from [`requirements.txt`](requirements.txt), kernel `lorawan-pathloss`; notebooks 06 and 14 need the RAPIDS environment from [`requirements-rapids-cu12.txt`](requirements-rapids-cu12.txt), kernel `lorawan-pathloss-rapids`:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name lorawan-pathloss --display-name "Python (LoRaWAN path loss)"
```

```bash
python3.12 -m venv .venv-rapids && source .venv-rapids/bin/activate
python -m pip install -r requirements-rapids-cu12.txt
python -m ipykernel install --user --name lorawan-pathloss-rapids --display-name "Python (LoRaWAN path loss, RAPIDS)"
```

## Notebooks

Run them in order from the repository root. They write to `Data_Files/`, `Models/`, `Residuals/`, `CV_Results/`, `Shadow_Results/`, `FM_Results/`, `Figures/` and `Tables/`, which are not versioned; the executed cells show the reported runs.

| Notebook | What it does | Main outputs |
|---|---|---|
| `00_Data_Preparation` | Chronological split and the forward-chaining folds | `Data_Files/train.csv`, `test.csv`, fold indices |
| `01_Empirical_PLMs` | Free-space loss, log-distance, Motley-Keenan, COST 231 multi-wall and ITU-R P.1238, fitted and tested | `Models/plm_*.json`, `Residuals/` |
| `02_MLR` | OLS, Ridge, Lasso and Elastic Net; the Lasso selected by the folds | `Residuals/residuals_MLR_*.csv` |
| `03_RF`, `04_XGBoost`, `05_LightGBM` | Bayesian search on the folds, final fit, residuals | `CV_Results/`, `Models/`, `Residuals/` |
| `06_kNN` (RAPIDS) | Search over k, metric, weights and scaling on the GPU | `CV_Results/`, `Models/`, `Residuals/` |
| `07_ANN` | Two-stage architecture screen, multi-seed selection, 30-seed final fit | `CV_Results/`, `Models/`, `Residuals/` |
| `08_SHADOW FADING ANALYSIS` | Normality and tail statistics, mixtures with the number of components by cross-validation | `Shadow_Results/` |
| `09_FADE MARGIN ANALYSIS` | Margins as empirical quantiles with block-bootstrap intervals, mixture sensitivity, hold-out validation | `FM_Results/` |
| `10_General_Comparisons` | Search results across the families | `Figures/` |
| `11_Bootstrap_Model_Comparison` | Paired block bootstrap of the hold-out RMSE | `CV_Results/bootstrap_test_comparison.csv` |
| `12_History_Features` | Causal link-history inputs and the rungs (`history_features.py`) | `Data_Files/history_*.csv` |
| `13_History_Models` | Linear, random forest, XGBoost, LightGBM and ANN on rungs H1 to H7; a link left out of training | `Residuals/`, `CV_Results/history_benchmark.csv` |
| `14_History_kNN` (RAPIDS) | The same for k-nearest neighbours | `Residuals/`, `CV_Results/history_benchmark_KNN.csv` |
| `15_Information_by_Family` | Rung by family tables, the ladder and residual-tail figures, unseen links, quantile margins, sparser uplinks | `CV_Results/information_by_family.csv`, `Figures/` |
| `16_Margin_Calibration` | Five calibrations of the margin on rolling windows (`margin_tools.py`), coverage by state, clustering of the misses | `CV_Results/margin_*.csv`, `Figures/` |
| `17_ADR_Estimators` | The estimators of adaptive data rate schemes against LightGBM | `CV_Results/adr_estimators.csv`, `Figures/` |
| `18_Feature_Groups` | What each group of inputs adds, static subsets and history steps, with intervals | `CV_Results/static_feature_groups.csv`, `history_group_steps.csv` |
| `19_Medellin_Ladder` | The ladder on the Medellín data | `CV_Results/medellin_*.csv` |
| `20_Error_Metrics` | MAE, bias and Pearson correlation, pooled and within link | `CV_Results/error_metrics.csv` |
| `21_UVA_Ladder` | The ladder and the margins under drift on the campus data | `CV_Results/uva_*.csv`, `Figures/` |
| `22_Paper_Tables` | Dataset by device, spreading-factor schedule, selected hyperparameters and software, as CSV and LaTeX; the generalization figure | `CV_Results/`, `Tables/`, `Figures/` |
| `23_History_Retuning` | LightGBM and XGBoost tuned on the inputs of each rung H4 to H7 | `CV_Results/history_retuning.csv` |

Wall-clock on 32 cores, 60 GB of RAM and one RTX 5090: `03` about 5 h, `06` 21.5 h, `07` 3.3 h, `09` 3.5 h, `13` 6.4 h, `15` 71 min; the other notebooks take minutes.

## Results

Hold-out window 21 July to 30 September 2025, 532,055 packets; RMSE and the 99 % margin in dB.

| Model | Inputs | Held-out RMSE | 99 % margin | Source |
|---|---|---|---|---|
| Per-link mean | none | 4.94 | 18.2 | `CV_Results/static_feature_groups.csv` |
| COST 231 multi-wall, calibrated | geometry, channel | 6.37 | 18.3 | `Residuals/residuals_PLM_COST231_MWM_*.csv` |
| Lasso | geometry, channel, sensors | 5.21 | 18.5 | `Residuals/residuals_MLR_*.csv` |
| Random forest | geometry, channel, sensors | 4.74 | 18.3 | `Residuals/residuals_RF_*.csv` |
| XGBoost | geometry, channel, sensors | 4.74 | 18.4 | `Residuals/residuals_XGB_*.csv` |
| LightGBM | geometry, channel, sensors | 4.77 | 18.3 | `Residuals/residuals_LGBM_*.csv` |
| k-nearest neighbours | geometry, channel, sensors | 4.68 | 17.8 | `Residuals/residuals_KNN_*.csv` |
| ANN | geometry, channel, sensors | 4.52 | 18.5 | `Residuals/residuals_ANN_*.csv` |
| Previous-hour mean | the link's earlier packets | 2.48 | 10.6 | `CV_Results/history_benchmark.csv` |
| LightGBM, rung H1 | link history | 2.16 | 8.9 | `CV_Results/information_by_family.csv` |
| LightGBM, rung H4 | + same-channel history, clock, spreading factor | 1.77 | 7.7 | `CV_Results/information_by_family.csv` |
| LightGBM, rung H7 | + earlier SNR, sensors, link level, geometry | 1.77 | 7.7 | `CV_Results/information_by_family.csv` |

Every family on every rung is in `CV_Results/information_by_family.csv`; the paired intervals of the static models in `CV_Results/bootstrap_test_comparison.csv`; the margins by operating state and the estimators of adaptive data rate schemes in the result files of notebooks 16 and 17; the input groups with intervals in `CV_Results/static_feature_groups.csv` and `history_group_steps.csv`; the public datasets in `CV_Results/medellin_ladder.csv` and `uva_ladder.csv`.

![Held-out RMSE of the static models](docs/assets/held-out-rmse.png)

*Held-out RMSE of every static model with its 95 % interval from a paired circular moving-block bootstrap (1,000 replicates, blocks of 5,000 packets), on the 532,055 packets common to all models.*

![Hold-out RMSE and 99 % margin by rung and family](docs/assets/information-by-family.png)

*Hold-out RMSE (a) and 99 % margin (b) of the six families on the static inputs and on the rungs H1 to H7; dashed, the per-link mean; dotted, the previous-hour mean.*

![Coverage and mean margin of four margins by state and by test window](docs/assets/margin-calibration.png)

*Coverage (top) and mean margin above the anchor (bottom) of four margins at the 99 % target, LightGBM on rung H4: by working hours (a), recent spread (b) and spreading factor (c) on the hold-out, with 95 % intervals from resampling days, and in each test window, calibrated on the blocks before it (d).*

![The ladder on the office data and on the two public datasets](docs/assets/ladder-three-datasets.png)

*Hold-out RMSE (a) and 99 % margin (b) relative to the per-link mean of each dataset: the office, the Medellín data and the campus data, all with the same single-shift margin.*

## Citation

If you use the measurements, cite:

> N. Mokua Obiri and K. van Laerhoven, “A Comprehensive Data Description for LoRaWAN Path Loss Measurements in an Indoor Office Setting: Effects of Environmental Factors,” *IEEE Access*, vol. 13, pp. 83148–83170, 2025. https://doi.org/10.1109/ACCESS.2025.3569164

The dataset is CC BY 4.0. No separate licence has yet been declared for the analysis code.
