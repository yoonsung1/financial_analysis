# scripts/make_dataset.py
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

# 프로젝트 루트를 sys.path에 추가해서 src 패키지 import 가능하게
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from financial_physics.config import (  # noqa: E402
    ASSETS,
    FX_TICKERS_CANDIDATES,
    PROCESSED_DATA_DIR,
    START_DATE,
    END_DATE,
)
from financial_physics.preprocess import to_log_price, to_log_return  # noqa: E402


def _download_adj_close(ticker: str, start: str, end: str) -> pd.Series:
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise ValueError(f"다운로드 실패: ticker={ticker}")
    # yfinance 컬럼: Adj Close 또는 Close (환경에 따라 다름)
    col = "Adj Close" if "Adj Close" in df.columns else "Close"
    s = df[col].copy()
    s.name = ticker
    s.index = pd.to_datetime(s.index)
    return s


def _download_usdkrw(start: str, end: str) -> pd.Series:
    last_err = None
    for tk in FX_TICKERS_CANDIDATES:
        try:
            s = _download_adj_close(tk, start, end)
            s.name = "USDKRW"
            return s
        except Exception as e:
            last_err = e
    raise RuntimeError(f"USD/KRW 다운로드 실패. 시도 티커={FX_TICKERS_CANDIDATES}, 마지막 에러={last_err}")


def main() -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    series = {}

    # 환율
    series["USDKRW"] = _download_usdkrw(START_DATE, END_DATE)

    # 지수들
    for name, ticker in ASSETS.items():
        s = _download_adj_close(ticker, START_DATE, END_DATE)
        s.name = name
        series[name] = s

    # 하나의 DataFrame으로 병합 (날짜 인덱스 기준 outer join)
    prices = pd.concat(series.values(), axis=1).sort_index()

    # 결측 처리: 휴장/시차 등으로 생긴 NaN을 그대로 둔다(분석 단계에서 drop/ffill 여부 결정)
    log_prices = prices.apply(to_log_price)
    log_returns = prices.apply(to_log_return)

    # 저장
    prices_path = PROCESSED_DATA_DIR / "prices.parquet"
    lp_path = PROCESSED_DATA_DIR / "log_prices.parquet"
    lr_path = PROCESSED_DATA_DIR / "log_returns.parquet"

    prices.to_parquet(prices_path)
    log_prices.to_parquet(lp_path)
    log_returns.to_parquet(lr_path)

    print("[OK] 저장 완료")
    print(f"- {prices_path}")
    print(f"- {lp_path}")
    print(f"- {lr_path}")
    print("\n미리보기(마지막 5행):")
    print(prices.tail())


if __name__ == "__main__":
    main()
