# Motor Bearing Fault Detection

Bearing fault detection on the CWRU vibration dataset (40 accelerometer recordings across 0, 1, 2, and 3 hp motor loads).

## Files

- `raw/cwru/`: 40 `.mat` vibration files from the CWRU dataset
- `motor_fault_cwru.py`: main Python script for signal filtering, feature extraction, and model evaluation
- `motor_fault_cwru.ipynb`: Jupyter notebook version
- `reports/features.csv`: extracted time and FFT features
- `reports/model_comparison.csv`: accuracy and F1 scores across motor loads
- `reports/figures/`: generated plots (`fig1_signals_and_fft.png`, `fig2_feature_boxplots.png`, `fig3_confusion_matrix.png`)

## Usage

```bash
python motor_fault_cwru.py
```
