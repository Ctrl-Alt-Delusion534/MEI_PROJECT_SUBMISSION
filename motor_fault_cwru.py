import os, pathlib
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import decimate
from scipy.stats import kurtosis
import matplotlib
if not os.environ.get("DISPLAY") and "inline" not in matplotlib.get_backend():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

DATA_DIR = pathlib.Path("raw/cwru") if pathlib.Path("raw/cwru").exists() else pathlib.Path("data/raw/cwru")
OUT_DIR = pathlib.Path("reports")
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SAMPLING_RATE_HZ = 12000
WINDOW_SIZE = 4096
STEP_SIZE = 2048
CLASSES = ["Normal", "Inner_Race", "Ball", "Outer_Race"]

RECORDINGS = [
    ("normal_0", 97, "Normal", 0), ("normal_1", 98, "Normal", 1), ("normal_2", 99, "Normal", 2), ("normal_3", 100, "Normal", 3),
    ("IR007_0", 105, "Inner_Race", 0), ("IR007_1", 106, "Inner_Race", 1), ("IR007_2", 107, "Inner_Race", 2), ("IR007_3", 108, "Inner_Race", 3),
    ("IR014_0", 169, "Inner_Race", 0), ("IR014_1", 170, "Inner_Race", 1), ("IR014_2", 171, "Inner_Race", 2), ("IR014_3", 172, "Inner_Race", 3),
    ("IR021_0", 209, "Inner_Race", 0), ("IR021_1", 210, "Inner_Race", 1), ("IR021_2", 211, "Inner_Race", 2), ("IR021_3", 212, "Inner_Race", 3),
    ("B007_0", 118, "Ball", 0), ("B007_1", 119, "Ball", 1), ("B007_2", 120, "Ball", 2), ("B007_3", 121, "Ball", 3),
    ("B014_0", 185, "Ball", 0), ("B014_1", 186, "Ball", 1), ("B014_2", 187, "Ball", 2), ("B014_3", 188, "Ball", 3),
    ("B021_0", 222, "Ball", 0), ("B021_1", 223, "Ball", 1), ("B021_2", 224, "Ball", 2), ("B021_3", 225, "Ball", 3),
    ("OR007@6_0", 130, "Outer_Race", 0), ("OR007@6_1", 131, "Outer_Race", 1), ("OR007@6_2", 132, "Outer_Race", 2), ("OR007@6_3", 133, "Outer_Race", 3),
    ("OR014@6_0", 197, "Outer_Race", 0), ("OR014@6_1", 198, "Outer_Race", 1), ("OR014@6_2", 199, "Outer_Race", 2), ("OR014@6_3", 200, "Outer_Race", 3),
    ("OR021@6_0", 234, "Outer_Race", 0), ("OR021@6_1", 235, "Outer_Race", 1), ("OR021@6_2", 236, "Outer_Race", 2), ("OR021@6_3", 237, "Outer_Race", 3),
]

def load_accelerometer_signal(filename: str, file_id: int, fault_class: str) -> np.ndarray:
    path = DATA_DIR / f"{filename}.mat"
    if not path.exists():
        path = DATA_DIR / f"{filename.capitalize()}.mat"
    mat = sio.loadmat(path)
    key = f"X{file_id:03d}_DE_time"
    signal = np.asarray(mat[key], dtype=float).ravel()
    if fault_class == "Normal":
        signal = decimate(signal, 4, ftype="fir", zero_phase=True)
    return signal

def load_all_windows() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    windows_list, labels_list, loads_list = [], [], []
    for filename, file_id, fault_class, motor_load in RECORDINGS:
        sig = load_accelerometer_signal(filename, file_id, fault_class)
        num_wins = (len(sig) - WINDOW_SIZE) // STEP_SIZE + 1
        idx = np.arange(num_wins)[:, None] * STEP_SIZE + np.arange(WINDOW_SIZE)[None, :]
        wins = sig[idx]
        windows_list.append(wins)
        labels_list.extend([fault_class] * num_wins)
        loads_list.extend([motor_load] * num_wins)
    return np.vstack(windows_list), np.array(labels_list), np.array(loads_list)

WINDOWS, LABELS, LOADS = load_all_windows()
print(f"Loaded {len(RECORDINGS)} files -> {len(WINDOWS)} signal windows.")

def plot_time_and_fft() -> None:
    fig, axes = plt.subplots(4, 2, figsize=(10, 8), constrained_layout=True)
    examples = [("normal_0", 97, "Normal"), ("IR014_0", 169, "Inner_Race"), ("B014_0", 185, "Ball"), ("OR014@6_0", 197, "Outer_Race")]
    freqs_khz = np.fft.rfftfreq(WINDOW_SIZE, d=1.0 / SAMPLING_RATE_HZ) / 1000.0
    time_ms = np.arange(1200) / SAMPLING_RATE_HZ * 1000.0
    for row, (fname, fid, label) in enumerate(examples):
        sig = load_accelerometer_signal(fname, fid, label)[:WINDOW_SIZE]
        fft_mag = np.abs(np.fft.rfft(sig * np.hanning(WINDOW_SIZE))) / WINDOW_SIZE
        axes[row, 0].plot(time_ms, sig[:1200], lw=0.8)
        axes[row, 0].set_ylabel(f"{label}\nAmp")
        axes[row, 1].plot(freqs_khz, fft_mag, lw=0.8, color="tab:orange")
        axes[row, 1].set_ylabel("FFT Mag")
    axes[0, 0].set_title("Accelerometer Time Waveform (100 ms)")
    axes[0, 1].set_title("FFT Frequency Spectrum (0 - 6 kHz)")
    axes[3, 0].set_xlabel("Time (ms)")
    axes[3, 1].set_xlabel("Frequency (kHz)")
    fig.savefig(FIG_DIR / "fig1_signals_and_fft.png", dpi=200)
    plt.close(fig)

plot_time_and_fft()

def extract_basic_features(windows: np.ndarray) -> pd.DataFrame:
    abs_w = np.abs(windows)
    rms = np.sqrt(np.mean(windows ** 2, axis=1)) + 1e-12
    peak = np.max(abs_w, axis=1)
    peak_to_peak = np.ptp(windows, axis=1)
    crest_factor = peak / rms
    kurt = kurtosis(windows, axis=1, fisher=False)

    fft_power = np.abs(np.fft.rfft(windows * np.hanning(WINDOW_SIZE), axis=1)) ** 2
    freqs = np.fft.rfftfreq(WINDOW_SIZE, d=1.0 / SAMPLING_RATE_HZ)
    total_power = np.sum(fft_power, axis=1) + 1e-12
    mean_freq_hz = np.sum(fft_power * freqs[None, :], axis=1) / total_power
    low_band_ratio = np.sum(fft_power[:, freqs < 2000], axis=1) / total_power
    mid_band_ratio = np.sum(fft_power[:, (freqs >= 2000) & (freqs < 4000)], axis=1) / total_power
    high_band_ratio = np.sum(fft_power[:, freqs >= 4000], axis=1) / total_power

    return pd.DataFrame({
        "rms": rms,
        "peak": peak,
        "peak_to_peak": peak_to_peak,
        "crest_factor": crest_factor,
        "kurtosis": kurt,
        "mean_frequency_hz": mean_freq_hz,
        "low_band_0_2kHz": low_band_ratio,
        "mid_band_2_4kHz": mid_band_ratio,
        "high_band_4_6kHz": high_band_ratio,
    })

FEATURES = extract_basic_features(WINDOWS)
FEATURES.to_csv(OUT_DIR / "features.csv", index=False)

def plot_feature_boxplots(features: pd.DataFrame, labels: np.ndarray) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), constrained_layout=True)
    selected = ["rms", "peak_to_peak", "kurtosis", "mean_frequency_hz", "mid_band_2_4kHz", "high_band_4_6kHz"]
    for ax, col in zip(axes.ravel(), selected):
        groups = [features[col].to_numpy()[labels == c] for c in CLASSES]
        ax.boxplot(groups, showfliers=False)
        ax.set_xticklabels(CLASSES, rotation=15)
        ax.set_title(col)
    fig.savefig(FIG_DIR / "fig2_feature_boxplots.png", dpi=200)
    plt.close(fig)

plot_feature_boxplots(FEATURES, LABELS)

MODELS = {
    "Logistic_Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "k_Nearest_Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

def evaluate_models_across_loads(features: pd.DataFrame, labels: np.ndarray, loads: np.ndarray) -> pd.DataFrame:
    rows = []
    X_arr = features.to_numpy()
    best_cm = np.zeros((4, 4), dtype=int)
    for name, clf in MODELS.items():
        accs, f1s = [], []
        cm_total = np.zeros((4, 4), dtype=int)
        for test_load in range(4):
            train_mask = loads != test_load
            test_mask = loads == test_load
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
            pipe.fit(X_arr[train_mask], labels[train_mask])
            preds = pipe.predict(X_arr[test_mask])
            accs.append(accuracy_score(labels[test_mask], preds))
            f1s.append(f1_score(labels[test_mask], preds, average="macro"))
            cm_total += confusion_matrix(labels[test_mask], preds, labels=CLASSES)
        rows.append({"model": name, "accuracy_mean": np.mean(accs), "accuracy_std": np.std(accs), "macro_f1_mean": np.mean(f1s), "macro_f1_std": np.std(f1s)})
        if name == "Random_Forest":
            best_cm = cm_total

    fig, ax = plt.subplots(figsize=(5.5, 4.5), constrained_layout=True)
    im = ax.imshow(best_cm, cmap="Blues")
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(CLASSES, rotation=15); ax.set_yticklabels(CLASSES)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(best_cm[i, j]), ha="center", va="center", color="white" if best_cm[i, j] > 300 else "black")
    ax.set_xlabel("Predicted Class"); ax.set_ylabel("True Class")
    ax.set_title("Confusion Matrix across Unseen Motor Loads (Random Forest)")
    fig.colorbar(im, ax=ax)
    fig.savefig(FIG_DIR / "fig3_confusion_matrix.png", dpi=200)
    plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "model_comparison.csv", index=False)
    return df

MODEL_RESULTS = evaluate_models_across_loads(FEATURES, LABELS, LOADS)
print("\nModel Accuracy Across Unseen Motor Loads:")
print(MODEL_RESULTS.round(4).to_string(index=False))

