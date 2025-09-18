# models/tft/train_tft.py
import os
import pandas as pd, numpy as np
from pathlib import Path
import pickle
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from lightning.pytorch import Trainer, seed_everything  # <- lightning v2 import
import torch

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.models import TemporalFusionTransformer  # <- explicit models import

from rag.extract_events import extract_events_for_query

# -------------------
# Paths
# -------------------
RAW = Path("data/raw")
PROC = Path("data/processed")
MODEL_DIR = PROC / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# TFT-specific paths under models/tft/
TFT_DIR = Path(__file__).parent  # This is models/tft/
TFT_CHECKPOINTS = TFT_DIR / "checkpoints"
TFT_LOGS = TFT_DIR / "lightning_logs"
TFT_CHECKPOINTS.mkdir(parents=True, exist_ok=True)
TFT_LOGS.mkdir(parents=True, exist_ok=True)

# -------------------
# Data loader
# -------------------
def load_ons_vacancies() -> pd.DataFrame:
    """
    Load ONS CSV that looks like:
      "Title","UK Job Vacancies ratio per 100 emp jobs - Total"
      ...
      "2001 MAY","2.6"
    We skip the first 8 metadata rows and read (date, value).
    """
    p = RAW / "ons_vacancies_ratio_total.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Run scripts/download_uk_data.py first.")
    # Skip metadata rows - data starts from row 8 (0-indexed)
    df = pd.read_csv(p, skiprows=8, header=None, names=["date", "value"])
    # keep only numeric values
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"]).copy()
    # Parse "YYYY MON" into Timestamp (first day of month)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    df = df.dropna(subset=["date"]).sort_values("date")
    df["series_id"] = "uk_vacancies"
    return df

# -------------------
# Event features
# -------------------
def monthly_event_features(dates: pd.Series) -> pd.DataFrame:
    """
    Extract real policy events via OpenAI (from the GOV.UK RAG index)
    and aggregate to monthly features: event_flag, event_impact.
    """
    q = "Skilled Worker visa, sponsor licence, immigration rules changes, salary thresholds, enforcement"
    events = extract_events_for_query(q, k=8)

    rows = []
    for d in pd.to_datetime(dates).dt.to_period("M").dt.to_timestamp():
        month_events = []
        for e in events:
            ed = pd.to_datetime(e.get("date", None), errors="coerce")
            if pd.isna(ed):
                continue
            if ed.to_period("M") == d.to_period("M"):
                month_events.append(e)
        flag = 1.0 if month_events else 0.0
        impact_map = {"POSITIVE": 1, "NEGATIVE": -1, "UNCLEAR": 0}
        impact = float(np.sum([impact_map.get(e.get("impact","UNCLEAR"), 0) * float(e.get("confidence", 0.5))
                               for e in month_events])) if month_events else 0.0
        rows.append({"date": d, "event_flag": flag, "event_impact": impact})
    return pd.DataFrame(rows)

# -------------------
# Dataset prep
# -------------------
def prepare_dataset(df: pd.DataFrame, enc_len=24, pred_len=6):
    feats = monthly_event_features(df["date"])
    df = df.merge(feats, on="date", how="left").fillna({"event_flag": 0.0, "event_impact": 0.0})
    # monthly integer time index
    df["time_idx"] = (df["date"] - df["date"].min()).dt.days // 30
    cutoff = int(df["time_idx"].max() - pred_len)

    training = TimeSeriesDataSet(
        df[df.time_idx <= cutoff],
        time_idx="time_idx",
        target="value",
        group_ids=["series_id"],
        max_encoder_length=enc_len,
        max_prediction_length=pred_len,
        time_varying_known_reals=["time_idx", "event_flag", "event_impact"],
        time_varying_unknown_reals=["value"],
        add_relative_time_idx=True,
        add_encoder_length=True,
        target_normalizer=None,
        allow_missing_timesteps=True,
    )
    validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)
    return df, training, validation

# -------------------
# Main
# -------------------
if __name__ == "__main__":
    # ensure dotenv already loaded elsewhere or env variables exported
    seed_everything(11)

    df = load_ons_vacancies()
    df, train_ds, val_ds = prepare_dataset(df)

    train_dl = train_ds.to_dataloader(train=True, batch_size=64, num_workers=0)
    val_dl = val_ds.to_dataloader(train=False, batch_size=64, num_workers=0)

    trainer = Trainer(
        max_epochs=15,
        accelerator="auto",
        logger=False,
        enable_checkpointing=True,
        gradient_clip_val=0.1,
        default_root_dir=str(TFT_DIR),  # Use TFT directory for logs
    )

    # Create TFT LightningModule
    tft = TemporalFusionTransformer.from_dataset(
        train_ds,
        learning_rate=2e-3,
        hidden_size=32,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=16,
        loss=QuantileLoss(),
        optimizer="Adam"
    )

    # Fit using Lightning v2 signature
    trainer.fit(model=tft, train_dataloaders=train_dl, val_dataloaders=val_dl)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(tft.state_dict(), MODEL_DIR / "tft.pt")
    with open(MODEL_DIR / "tft_ds.pkl", "wb") as f:
        pickle.dump(train_ds, f)
    df.to_csv(MODEL_DIR / "training_frame.csv", index=False)
    print("[TRAIN] Done.")
