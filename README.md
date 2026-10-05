# Indoor LoRaWAN path loss modeling

**Temporally separated evaluation of analytical and empirical propagation models and machine learning regressors, with residual-tail calibration of fade margins as a function of what the predictor knows.**

[![Dataset](https://img.shields.io/badge/dataset-Zenodo-1682D4?logo=zenodo&logoColor=white)](https://doi.org/10.5281/zenodo.15349730)
[![Data paper](https://img.shields.io/badge/data%20paper-IEEE%20Access-00629B?logo=ieee&logoColor=white)](https://doi.org/10.1109/ACCESS.2025.3569164)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)

## Aim

This study compares analytical and empirical indoor propagation laws, linear models, tree ensembles, nearest-neighbour regression and neural networks on a later period of LoRaWAN path loss measurements. It then uses out-of-fold errors to estimate fade margins at specified residual-coverage targets, and asks how those margins depend on what the predictor knows: geometry and environmental sensors only, or also the link's own earlier packets.

RMSE and the upper residual tail are distinct objectives. With geometry and sensors alone the 99% fade margin is 18.9 to 19.3 dB for every model, although held-out RMSE ranges from 4.8 to 6.4 dB. With the link's earlier packets it falls to 9 to 12 dB, and to 4 to 5 dB when the margin follows the state of the link.

## Experimental design

The analysis uses 2,660,274 timestamped observations from six fixed indoor links collected from 1 October 2024 to 30 September 2025 in an office environment at the University of Siegen, at spreading factors 7 to 12. The static predictors are distance, carrier frequency, concrete- and wood-wall counts, CO₂, relative humidity, PM2.5, pressure and temperature; the response is path loss derived from the received signal strength at the project gateway. The SNR of the packet being predicted is not a predictor. The history rungs described below add inputs computed from the link's earlier packets only.

The evaluation protocol is deliberately chronological:

1. The earliest 80% of observations form the training period; the latest 20% are held out.
2. Hyperparameters are selected only within the training period, using five forward-chaining folds: each validation block is predicted by models trained on the blocks before it.
3. Out-of-fold residuals from those folds are used for residual-tail modelling and fade-margin calibration.
4. The untouched later period is used for final error and residual-coverage assessment.

This gives 2,128,219 training observations (1 October 2024 to 21 July 2025) and 532,055 test observations (21 July to 30 September 2025). The first block of 354,704 rows is only ever training data; the five following blocks of 354,703 rows give 1,773,515 out-of-fold residuals per model. Where scaling is used, it is fitted within each fold.

The end devices were restricted to spreading factors 7 to 10 in July 2025. The hold-out therefore contains SF11 or SF12 in only 33 of its rows, against 17 to 28% of the rows in the validation blocks, and margins calibrated on the validation blocks are conservative on it. The hold-out tests temporal transfer within the same building and six links; an unseen-link test fits on five links and predicts the sixth. Neither is evidence of generalisation to other buildings or link geometries.

Early stopping uses the latest part of each training set in time order: the latest training block for the ANN of notebook `07`, the latest 10% of the rows for the ANN and the GRU of the history notebooks.

## Main results

### Static models

![Held-out RMSE for the evaluated models](docs/assets/held-out-rmse.png)

*Held-out path loss error on the 532,055 rows common to all models. Points are RMSE; bars are 95% circular moving-block-bootstrap intervals (1,000 replicates; block length 5,000). Intervals quantify test-sampling uncertainty with fitted models held fixed.*

| Model | Out-of-fold RMSE | Held-out RMSE | Held-out R² | 95% margin | 99% margin |
|---|---|---|---|---|---|
| COST 231 multi-wall (calibrated) | 7.27 | 6.38 | 0.845 | 11.6 | 18.9 |
| Linear regression (Lasso) | 6.56 | 5.25 | 0.895 | 11.2 | 19.2 |
| Random forest | 6.32 | 4.93 | 0.907 | 11.0 | 19.2 |
| XGBoost | 6.36 | 4.83 | 0.911 | 11.3 | 19.2 |
| LightGBM | 6.34 | 4.87 | 0.910 | 10.7 | 19.2 |
| k-nearest neighbours | 6.68 | 5.01 | 0.904 | 11.3 | 18.9 |
| ANN (MLP) | 6.58 | 5.09 | 0.901 | 11.7 | 19.3 |

*RMSE in dB. Out-of-fold RMSE pools the 1,773,515 validation rows; R² is computed from the held-out residual files; margins in dB at the 95% and 99% residual-coverage targets.*

XGBoost leads at 4.83 dB (95% CI 4.59 to 5.07). The paired block bootstrap resolves it ahead of the random forest, kNN, the ANN and the linear model, by 0.10 to 0.42 dB, but not ahead of LightGBM; kNN, the ANN and the random forest are tied with one another. Every data-driven model beats the calibrated COST 231 multi-wall model by at least 1.1 dB and the fixed-parameter propagation models by at least 1.9 dB.

At a 99% residual-coverage target every static model needs 18.9 to 19.3 dB, at 98% 16.1 to 16.7 dB and at 95% 10.7 to 11.7 dB; the largest saving against the linear model at 99% is 0.29 dB. At the 98% and 99% targets calibration takes the conservative maximum of the empirical and three-component Gaussian-mixture tail estimates; the 95% target uses the empirical quantile. Held-out coverage at the 99% margin is 99.7 to 99.8%, and a fixed 10 dB margin reaches 96.0 to 98.2%. Empirical-quantile intervals use device-aware moving blocks; Gaussian-mixture tail intervals are parametric.

### Link history

![Hold-out RMSE and 99% fade margin by information rung and model family](docs/assets/information-by-family.png)

*Hold-out RMSE (left) and calibrated 99% fade margin (right) by information rung and model family. Dashed: each link's mean path loss; dotted: the previous-hour mean.*

From here the response is the deviation of path loss from the link's previous-hour mean, so the inputs are level-free and a model that learns nothing scores as the previous-hour mean. The rungs are cumulative: (1) the link's earlier path loss as lags and windows from five packets to seven days; (2) the history of the same channel, the clock and the spreading factor; (3) the SNR of earlier packets and the environmental sensors; (4) the link's level and its geometry. Every feature of a packet uses earlier packets of the same link only.

| Family | Geometry + sensors | + link history | + channel, clock, SF | + earlier SNR, sensors | + link level, geometry |
|---|---|---|---|---|---|
| Linear regression | 5.25 / 19.2 | 2.51 / 11.8 | 2.39 / 11.7 | 2.37 / 11.4 | 2.37 / 11.4 |
| Random forest | 4.93 / 19.2 | 2.35 / 11.0 | 1.98 / 10.2 | 1.87 / 10.5 | 1.87 / 10.4 |
| XGBoost | 4.83 / 19.2 | 2.34 / 10.9 | 1.95 / 10.0 | 1.82 / 10.2 | 1.82 / 10.1 |
| LightGBM | 4.87 / 19.2 | 2.34 / 10.8 | 1.95 / 10.0 | 1.82 / 10.2 | 1.82 / 10.1 |
| k-nearest neighbours | 5.01 / 18.9 | 2.46 / 11.2 | 2.16 / 10.7 | 2.16 / 11.0 | 2.19 / 11.1 |
| ANN (MLP) | 5.09 / 19.3 | 2.50 / 11.1 | 1.99 / 10.1 | 1.92 / 10.4 | 1.92 / 10.4 |
| GRU | not defined | 2.55 / 9.2 | 2.27 / 9.0 | 2.22 / 9.3 | 2.23 / 9.2 |

*Hold-out RMSE / 99% fade margin, both in dB. The first column is the static model of the table above. The GRU reads the last 30 packets of the link and is defined only where a history exists. Without a fitted model: link mean 5.04 dB / 19.0 dB; previous-hour mean 2.71 dB / 12.0 dB; previous packet 3.04 dB / 12.5 dB.*

Adding the link's earlier packets halves RMSE and brings the 99% margin from about 19 dB to between 9 and 12 dB in every family. The later rungs lower RMSE further, to 1.8 dB for the boosted trees, but leave the margin near 10 dB: channel, clock, spreading factor, earlier SNR, sensors and geometry do not reduce the deep-fade tail. The boosted trees, the random forest and the ANN end within 0.1 dB of one another at the last rung; kNN, the GRU and the linear model are 0.4 to 0.6 dB behind. The GRU, which reads the raw sequence, does not improve on the engineered windows.

Fitted on five links and tested on the sixth (rung 2), the tree ensembles reach 2.0 dB on average, 2.2 dB on the worst link and within 0.1 dB of the same models with the link in training; the linear model, kNN, the ANN and the GRU reach 2.1 to 2.4 dB. Coverage at the calibrated 99% margin is 99.7% on the unseen link (99.3% for the GRU).

### Margins that follow the link

Quantile models predict the margin itself from the same inputs (LightGBM and XGBoost with quantile objectives, the ANN with the pinball loss). Mean margin above the prediction at rung 2, with hold-out coverage:

| Target | One fixed margin above the previous-hour mean | LightGBM quantile | XGBoost quantile | ANN quantile |
|---|---|---|---|---|
| 90% | 3.8 dB (91.7%) | 2.2 dB (90.1%) | 2.2 dB (90.0%) | not fitted |
| 95% | 5.3 dB (96.7%) | 3.0 dB (95.3%) | 2.9 dB (95.3%) | not fitted |
| 99% | 10.6 dB (99.8%) | 4.7 dB (99.3%) | 4.7 dB (99.3%) | 4.9 dB (99.2%) |

The 99% margin of the LightGBM quantile model is 5.6 dB in working hours and 4.3 dB otherwise. At one uplink every 10 minutes LightGBM on rung 2 reaches 2.3 dB RMSE and a 99% margin of 10.4 dB, against 2.7 dB and 11.9 dB for the mean of the previous 20 packets; at one uplink per hour it reaches 2.6 dB and 11.8 dB, against 3.2 dB and 13.1 dB.

## Model scope

| Class | Models |
|---|---|
| Propagation baselines | Free-space path loss, fixed-point LDPLM, Motley–Keenan, COST 231 multi-wall (constant and wall losses calibrated by regression), ITU-R P.1238 |
| Linear | OLS, Ridge, Lasso, Elastic Net; the Lasso, selected by cross-validation, is the benchmark entry |
| Nonlinear classical ML | Random forest, XGBoost, LightGBM, k-nearest neighbours |
| Neural | Multilayer perceptron; GRU on the sequence of the link's earlier packets (history rungs only) |

Residual analyses test normality and tail shape, compare Gaussian-mixture representations, calibrate coverage-specific fade margins, inspect held-out margin exceedances, and use a paired block bootstrap for model comparisons.

## Reproduce the analysis

The reported model runs used Python 3.12. Download file 3 of the [Zenodo dataset, version 3](https://doi.org/10.5281/zenodo.23127026) and place it at the path expected by notebook `00`. The checksum is that of the file behind every result here:

```bash
mkdir -p Data_Files
curl -L https://zenodo.org/api/records/23127026/files/3.cleaned_dataset_per_device.csv/content -o Data_Files/cleaned_dataset_per_device.csv
printf '%s  %s\n' 2d69176011fb32e0ef5d664bf9285e98 Data_Files/cleaned_dataset_per_device.csv | md5sum --check
```

The file has 2,660,274 rows at spreading factors 7 to 12, with RSSI and SNR taken from the project gateway and pressure in hPa. The [data pipeline repository](https://github.com/nahshonmokua/LoRaWAN-Indoor-Path-Loss-Modelling-with-MultiWall-Environment-Factors) regenerates it from the raw export with notebooks 02 to 04.

Create the main environment from [`requirements.txt`](requirements.txt):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name lorawan-pathloss --display-name "Python (LoRaWAN path loss)"
jupyter lab
```

The GRU notebook trains on a GPU. Install TensorFlow with its CUDA libraries instead of the plain pin, for example `python -m pip install "tensorflow[and-cuda]==2.21.0"`.

Notebooks `06` and `14` require a separate CUDA 12/RAPIDS environment. The versions used for the reported kNN runs are recorded in [`requirements-rapids-cu12.txt`](requirements-rapids-cu12.txt):

```bash
python3.12 -m venv .venv-rapids
source .venv-rapids/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-rapids-cu12.txt
python -m ipykernel install --user --name lorawan-pathloss-rapids --display-name "Python (LoRaWAN path loss, RAPIDS)"
```

Select the RAPIDS kernel for `06_kNN.ipynb` and `14_History_kNN.ipynb`; use the main kernel for the other notebooks.

Run the numbered notebooks from the repository root. Their order is the dependency graph:

| Stage | Notebook(s) | Output |
|---|---|---|
| Split and folds | [`00_Data_Preparation.ipynb`](00_Data_Preparation.ipynb) | Chronological train/test data and forward-chaining fold indices |
| Baselines | [`01_Empirical_PLMs.ipynb`](01_Empirical_PLMs.ipynb), [`02_MLR.ipynb`](02_MLR.ipynb) | Empirical and linear-model fits and residuals |
| Nonlinear models | `03_RF.ipynb` through `07_ANN.ipynb` | Selected models and out-of-fold/test residuals |
| Residual tails | [`08_SHADOW FADING ANALYSIS.ipynb`](08_SHADOW%20FADING%20ANALYSIS.ipynb) | Distribution and Gaussian-mixture diagnostics |
| Margin calibration | [`09_FADE MARGIN ANALYSIS.ipynb`](09_FADE%20MARGIN%20ANALYSIS.ipynb) | Calibrated fade margins and held-out residual coverage |
| Synthesis | [`10_General_Comparisons.ipynb`](10_General_Comparisons.ipynb), [`11_Bootstrap_Model_Comparison.ipynb`](11_Bootstrap_Model_Comparison.ipynb) | Cross-model figures and paired uncertainty estimates |
| Link history | [`12_History_Features.ipynb`](12_History_Features.ipynb), [`history_features.py`](history_features.py) | Causal link-history inputs for the rungs |
| History models | `13_History_Models.ipynb` (linear, random forest, XGBoost, LightGBM, ANN), `14_History_kNN.ipynb`, `15_History_GRU.ipynb` | Residuals per family and rung, unseen-link test |
| Information tables | [`16_Information_by_Family.ipynb`](16_Information_by_Family.ipynb) | Rung by family tables and figure, quantile margins, sparse uplinks |

The full benchmark is compute-intensive. Approximate wall-clock times on a workstation with 32 CPU cores, 60 GB of RAM and one NVIDIA RTX 5090:

| Notebook | Time |
|---|---|
| `03_RF` | 5.2 h (Bayesian search, 60 candidates) |
| `04_XGBoost` | 8 min (CUDA) |
| `05_LightGBM` | 48 min |
| `06_kNN` | 13.7 h (400 candidates, RAPIDS GPU) |
| `07_ANN` | 4.0 h (CPU, deterministic operations; two-stage screen, 30-seed final ensemble) |
| `09_FADE MARGIN ANALYSIS` | 1.0 h |
| `12_History_Features` | 8 min |
| `13_History_Models` | 3.2 h |
| `14_History_kNN` | 20 min |
| `15_History_GRU` | 3.9 h (GPU) |
| `16_Information_by_Family` | 33 min |

The other notebooks take minutes. Executed result cells document the reported runs, but generated search tables, data, models, residuals, and bulk analysis figures are intentionally excluded from version control and must be regenerated for a clean reproduction; the curated figures above are retained. Several notebook figures request Times New Roman and fall back to an installed font when it is unavailable.

## Data and citation

The dataset is archived on [Zenodo](https://doi.org/10.5281/zenodo.23127026) under CC BY 4.0. These notebooks use version 3 (October 2026), which corrects the RSSI and SNR attribution and the pressure unit of versions 1 and 2, as described in the version history of the record. The concept DOI [`10.5281/zenodo.15349730`](https://doi.org/10.5281/zenodo.15349730) resolves to the latest release.

If you use the measurements, cite:

> N. Mokua Obiri and K. van Laerhoven, “A Comprehensive Data Description for LoRaWAN Path Loss Measurements in an Indoor Office Setting: Effects of Environmental Factors,” *IEEE Access*, vol. 13, pp. 83148–83170, 2025. https://doi.org/10.1109/ACCESS.2025.3569164

The dataset is CC BY 4.0. No separate licence has yet been declared for the analysis code.
