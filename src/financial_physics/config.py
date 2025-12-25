# src/financial_physics/config.py
from __future__ import annotations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# yfinance용 티커
ASSETS = {
    "KOSPI": "^KS11",
    "SP500": "^GSPC",
}

# 환율(USD/KRW)은 yfinance에서 대개 아래 티커가 동작함.
# (환경에 따라 다를 수 있어, 아래에서 실패하면 대체 티커로 자동 시도하도록 스크립트에 넣어둠)
FX_TICKERS_CANDIDATES = ["KRW=X", "USDKRW=X"]

START_DATE = "2005-01-01"
END_DATE = "2024-12-31"
