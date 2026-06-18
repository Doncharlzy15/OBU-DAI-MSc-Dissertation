

import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal

warnings.filterwarnings("ignore")

FS = 1259   # sEMG sampling frequency (Hz)
BP_LOW = 20     # bandpass lower bound (Hz)
BP_HIGH = 450    # bandpass upper bound (Hz)
BP_ORDER = 4      # Butterworth order
MVC_WIN_S = 1.0    # sliding window length for peak-MVC computation (s)

N_SUBJECTS = 13
N_TRIALS = 12

# Protocol: odd trials -> right side, even -> left side
RIGHT_TRIALS = {1, 3, 5, 7, 9, 11}
LEFT_TRIALS  = {2, 4, 6, 8, 10, 12}

# MVC muscle tags that appear in filenames and the channel name they match in raw CSVs
MVC_TAGS = {
    "R_BICEPS": "R BICEPS BRACHII: EMG 1 [V]",
    "R_DELTOID_ANTERIOR": "R DELTOID ANTERIOR: EMG 2 [V]",
    "R_DELTOID_MEDIUS": "R DELTOID MEDIUS: EMG 6 [V]",
    "R_DELTOID_POSTERIOR":"R DELTOID POSTERIOR: EMG 7 [V]",
    "L_BICEPS":  "L BICEPS BRACHII: EMG 4 [V]",
    "L_DELTOID_ANTERIOR": "L DELTOID ANTERIOR: EMG 3 [V]",
    "L_DELTOID_MEDIUS":"L DELTOID MEDIUS: EMG 8 [V]",
    "L_DELTOID_POSTERIOR":"L DELTOID POSTERIOR: EMG 5 [V]",
}

# path helpers
def subject_emg_dir(base: Path, subject: int) -> Path:
    """Return the sEMG data directory for a subject, handling the subject_8 nesting."""
    candidate = base / "sEMG_data" / f"subject_{subject}"
    # Subject 8 quirk: data lives one level deeper
    nested = candidate / f"subject_{subject}"
    return nested if nested.is_dir() else candidate


def trial_csv_path(emg_dir: Path, trial: int) -> Path:
    return emg_dir / f"trial_{trial}.csv"


def mvc_dir_path(emg_dir: Path) -> Path:
    return emg_dir / "MVC"


def spf_csv_path(base: Path, subject: int, trial: int) -> Path:
    """Return SPF label path, handling the capital-T quirk for subjects 1 & 2."""
    spf_dir = base / "self_perceived_fatigue_index" / f"subject_{subject}"
    # Subjects 1 and 2 saved with capital 'T'
    capital = spf_dir / f"Trial_{trial}.csv"
    lower = spf_dir / f"trial_{trial}.csv"
    return capital if capital.exists() else lower


# Filter design 

def build_bandpass(fs: int = FS, low: int = BP_LOW,
                   high: int = BP_HIGH, order: int = BP_ORDER):
    return scipy_signal.butter(order, [low, high], btype="bandpass", fs=fs)


def apply_filter(sig: np.ndarray, b, a) -> np.ndarray:
    """Zero-phase Butterworth bandpass; replaces NaN/Inf with 0 before filtering."""
    sig_clean = np.where(np.isfinite(sig), sig, 0.0)
    return scipy_signal.filtfilt(b, a, sig_clean)


# Raw data loaders 

def load_trial(path: Path):
    """
    Load one trial CSV.

    Returns
    time : (N,) array  — seconds
    channels : dict {full_channel_name: (N,) voltage array in V}
    """
    df = pd.read_csv(path)
    time_cols = [c for c in df.columns if "X [s]" in c]
    emg_cols = [c for c in df.columns if "[V]" in c]
    time = df[time_cols[0]].values
    channels = {col: df[col].values for col in emg_cols}
    return time, channels


def load_mvc_raw(mvc_file: Path):
    """
    Load one MVC CSV and return (time, raw_emg_V).
    Column 0: time, column 1: raw EMG (V), column 2: pre-computed sliding RMS.
    """
    df = pd.read_csv(mvc_file)
    time = df.iloc[:, 0].values
    raw_emg = df.iloc[:, 1].values
    return time, raw_emg


# MVC peak RMS

def compute_mvc_peak_rms_mv(raw_emg_v: np.ndarray, b, a,
                             fs: int = FS, win_s: float = MVC_WIN_S) -> float:
    """
    Bandpass-filter the MVC signal, slide a 1-second window and return
    the peak RMS converted to mV.  Mirrors the notebook exactly.
    """
    filt = apply_filter(raw_emg_v, b, a)
    win = int(win_s * fs)
    rms_vals = [
        np.sqrt(np.mean(filt[i : i + win] ** 2))
        for i in range(0, len(filt) - win, win // 2)
    ]
    return float(max(rms_vals) * 1000)  # V -> mV


# Core preprocessing routines

def preprocess_trial(path: Path, b, a) -> tuple[np.ndarray, dict]:
    """
    Load, filter, and convert one trial to mV.

    Returns
    -------
    time : (N,) seconds
    filt_channels : dict {full_channel_name: (N,) mV array}
    """
    time, channels = load_trial(path)
    filt_channels = {
        ch: apply_filter(sig, b, a) * 1000  # V -> mV
        for ch, sig in channels.items()
    }
    return time, filt_channels


def preprocess_mvc(mvc_dir: Path, subject: int, b, a) -> pd.DataFrame:
    """
    Process all 8 MVC files for a subject (R + L).

    Returns a DataFrame with columns:
      channel_full_name : the raw CSV channel name 
      channel_col : filtered CSV column name
      peak_rms_mV : peak 1-second RMS in mV
    """
    rows = []
    for tag, full_name in MVC_TAGS.items():
        mvc_file = mvc_dir / f"subject_{subject}_MVC_{tag}.csv"
        if not mvc_file.exists():
            print(f" [WARN] MVC file not found: {mvc_file.name} — skipping")
            continue
        _, raw_emg = load_mvc_raw(mvc_file)
        peak_mv = compute_mvc_peak_rms_mv(raw_emg, b, a)

        # Derive the column name used in the filtered trial CSV

        col_name = (full_name.split(":")[0]
                             .replace(" ", "_")
                             .replace("-", "_")) + "_filt"
        rows.append({
            "channel_full_name": full_name,
            "channel_col": col_name,
            "peak_rms_mV": peak_mv,
        })

    return pd.DataFrame(rows)


# Output writers

def save_filtered_trial(out_path: Path, time: np.ndarray,
                        filt_channels: dict) -> None:
    """Save filtered mV signals to CSV, mirroring the notebook's column naming."""
    df = pd.DataFrame({"time": time})
    for ch_name, sig in filt_channels.items():
        col = (ch_name.split(":")[0]
                      .replace(" ", "_")
                      .replace("-", "_")) + "_filt"
        df[col] = sig
    df.to_csv(out_path, index=False)


# Main pipeline

def run(base_dir: Path, subjects: list[int], trials: list[int],
        skip_existing: bool) -> None:

    out_dir = base_dir / "preprocessed_data"
    out_dir.mkdir(exist_ok=True)

    b, a = build_bandpass()
    manifest_rows = []

    for subj in subjects:
        emg_dir = subject_emg_dir(base_dir, subj)
        mvc_dir = mvc_dir_path(emg_dir)

        print(f"\n{'='*60}")
        print(f"Subject {subj:2d}  [{emg_dir}]")
        print(f"{'='*60}")

        # MVC (once per subject) 
        mvc_out = out_dir / f"subject_{subj}_mvc_rms.csv"
        if skip_existing and mvc_out.exists():
            print(f"  MVC  -> already exists, skipping")
        elif not mvc_dir.exists():
            print(f"[WARN] MVC directory missing: {mvc_dir} — skipping MVC")
        else:
            mvc_df = preprocess_mvc(mvc_dir, subj, b, a)
            mvc_df.to_csv(mvc_out, index=False)
            print(f"  MVC  -> {mvc_out.name}  ({len(mvc_df)} muscles)")

        # Trials
        for trial in trials:
            trial_path = trial_csv_path(emg_dir, trial)
            out_path = out_dir / f"subject_{subj}_trial{trial}_filtered.csv"
            spf_path = spf_csv_path(base_dir, subj, trial)

            row = {
                "subject": subj, "trial": trial,
                "status": "ok", "n_samples": None,
                "duration_s": None, "n_channels": None,
                "output_path": str(out_path),
            }

            if not trial_path.exists():
                print(f" T{trial:2d}  [SKIP] raw file not found: {trial_path.name}")
                row["status"] = "missing_raw"
                manifest_rows.append(row)
                continue

            if skip_existing and out_path.exists():
                print(f"T{trial:2d} [SKIP] already preprocessed")
                row["status"] = "skipped"
                manifest_rows.append(row)
                continue

            try:
                time, filt_ch = preprocess_trial(trial_path, b, a)
                save_filtered_trial(out_path, time, filt_ch)

                n_samp = len(time)
                dur_s = float(time[-1])
                n_ch = len(filt_ch)
                side = "R" if trial in RIGHT_TRIALS else "L"
                spf_ok = "SPF OK" if spf_path.exists() else "SPF MISSING"

                row.update({"n_samples": n_samp, "duration_s": round(dur_s, 2),
                             "n_channels": n_ch})
                print(f"T{trial:2d}  [{side}] {n_samp:,} samples  "
                      f"{dur_s:.1f}s  {n_ch}ch  {spf_ok}  "
                      f"-> {out_path.name}")

            except Exception as exc:
                print(f"  T{trial:2d}  [ERROR] {exc}")
                row["status"] = f"error: {exc}"

            manifest_rows.append(row)

    # Manifest 
    manifest_path = out_dir / "manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    print(f"\nManifest written -> {manifest_path}")


# CLI

def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch-preprocess Minho fatigue sEMG data")
    parser.add_argument(
        "--subjects", nargs="+", type=int,
        default=list(range(1, N_SUBJECTS + 1)),
        help="Subject IDs to process (default: all 1-13)")
    parser.add_argument(
        "--trials", nargs="+", type=int,
        default=list(range(1, N_TRIALS + 1)),
        help="Trial numbers to process (default: all 1-12)")
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip subjects/trials whose output files already exist")
    parser.add_argument(
        "--base-dir", type=Path,
        default=Path(__file__).parent,
        help="Root directory containing sEMG_data/, self_perceived_fatigue_index/, etc.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Base dir : {args.base_dir}")
    print(f"Subjects : {args.subjects}")
    print(f"Trials : {args.trials}")
    print(f"Skip existing: {args.skip_existing}")
    run(args.base_dir, args.subjects, args.trials, args.skip_existing)
