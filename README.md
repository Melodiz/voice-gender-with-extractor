# Voice Gender Recognition

## Overview

This project focuses on training a machine learning model to classify voices as male or female based on their acoustic properties. The dataset used for training consists of 36,168 voice samples collected from a Yandex contest. The best-performing model achieves an accuracy of 98% during cross-validation.

## Dataset Information

You can download the raw dataset, which includes `.wav` files and a corresponding `.csv` file with labels, from the following link: [Raw Dataset](https://disk.yandex.ru/d/IUUTPJFOfwn_OQ).

Alternatively, a pre-processed version of the dataset, containing 3.5k rows (due to hardware limitations), is available here: [Pre-processed Dataset](https://disk.yandex.ru/d/fDn4QOQxZKpOZg).

## Acoustic Properties

The following acoustic properties are measured for each voice sample:

- **meanfreq**: Mean frequency (in kHz)
- **sd**: Standard deviation of frequency
- **median**: Median frequency (in kHz)
- **Q25**: First quantile (in kHz)
- **Q75**: Third quantile (in kHz)
- **IQR**: Interquantile range (in kHz)
- **skew**: Skewness
- **kurt**: Kurtosis
- **sp.ent**: Spectral entropy
- **sfm**: Spectral flatness
- **mode**: Mode frequency
- **centroid**: Frequency centroid
- **peakf**: Peak frequency (frequency with highest energy)
- **meanfun**: Average fundamental frequency measured across the acoustic signal
- **minfun**: Minimum fundamental frequency measured across the acoustic signal
- **maxfun**: Maximum fundamental frequency measured across the acoustic signal
- **meandom**: Average dominant frequency measured across the acoustic signal
- **mindom**: Minimum dominant frequency measured across the acoustic signal
- **maxdom**: Maximum dominant frequency measured across the acoustic signal
- **dfrange**: Range of dominant frequency measured across the acoustic signal
- **modindx**: Modulation index, calculated as the accumulated absolute difference between adjacent measurements of fundamental frequencies divided by the frequency range


## Technology Stack

The project utilizes the following libraries and tools:

- **Pandas**: For data manipulation and analysis
- **NumPy**: For numerical operations
- **Seaborn** and **Matplotlib**: For data visualization
- **Scikit-learn**: For machine learning algorithms and model evaluation
- **XGBoost**: For gradient boosting algorithms
- **Librosa**: For audio processing and feature extraction
- **Concurrent**: For parallel processing
- **Logging**: For logging errors and warnings

## Environment Setup
1. **Create a Virtual Environment**:
   ```shell
   python -m venv env
   ```
2. ```shell
    source env/bin/activate
    ```
3. ```shell
    pip install -r requirements.txt
    ```
