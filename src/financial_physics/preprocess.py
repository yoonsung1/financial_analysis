# src/financial_physics/preprocess.py
from __future__ import annotations
import numpy as np
import pandas as pd


def to_log_price(price: pd.Series) -> pd.Series:
    """가격 시계열을 로그가격으로 변환."""
    s = price.astype(float).copy()
    s[s <= 0] = np.nan
    return np.log(s)


def to_log_return(price: pd.Series) -> pd.Series:
    """가격 시계열로부터 로그수익률 r_t = log(P_t/P_{t-1}) 계산."""
    lp = to_log_price(price)
    return lp.diff()
