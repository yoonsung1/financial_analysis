# src/financial_physics/metrics/msd.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Union

import numpy as np


ArrayLike = Union[np.ndarray, Iterable[float]]


@dataclass(frozen=True)
class MSDResult:
    """
    평균제곱변위(MSD) 계산 결과를 담는 컨테이너

    Attributes
    ----------
    lags : np.ndarray
        시간 지연(lag, τ) 값들
    msd : np.ndarray
        각 lag에 대한 MSD 값
    counts : np.ndarray
        각 lag에서 평균 계산에 사용된 샘플 개수
    """
    lags: np.ndarray
    msd: np.ndarray
    counts: np.ndarray


def _to_1d_array(x: ArrayLike) -> np.ndarray:
    """
    입력 데이터를 1차원 numpy 배열(float)로 변환하고 검증한다.

    Parameters
    ----------
    x : array-like
        1차원 시계열 데이터

    Returns
    -------
    np.ndarray
        1차원 float 배열

    Raises
    ------
    ValueError
        입력이 1차원이 아닐 경우
    """
    arr = np.asarray(list(x) if not isinstance(x, np.ndarray) else x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"입력은 1차원이어야 합니다. 현재 shape={arr.shape}")
    return arr


def msd(
    x: ArrayLike,
    max_lag: Optional[int] = None,
    min_lag: int = 1,
    use_nan_policy: str = "drop",
) -> MSDResult:
    """
    1차원 시계열에 대해 평균제곱변위(MSD)를 계산한다.

    MSD(τ) = ⟨ (x(t+τ) - x(t))² ⟩

    Parameters
    ----------
    x : array-like
        시계열 데이터 (예: 로그 가격 또는 누적 수익률)
    max_lag : int, optional
        계산할 최대 lag(τ)
        기본값: min(252, 데이터 길이의 절반)
    min_lag : int
        계산할 최소 lag (1 이상)
    use_nan_policy : {"drop", "raise"}
        - "drop": NaN이 포함된 쌍은 해당 lag에서 제외
        - "raise": NaN이 있으면 즉시 에러 발생

    Returns
    -------
    MSDResult
        lag, MSD 값, 사용된 샘플 수를 포함한 결과 객체

    Notes
    -----
    - MSD 계산 시 x(t)는 보통 로그 가격을 사용한다.
    - 수익률만 있는 경우 x(t) = 누적 수익률로 위치를 정의할 수 있다.
    """
    x_arr = _to_1d_array(x)

    if use_nan_policy not in {"drop", "raise"}:
        raise ValueError("use_nan_policy는 {'drop', 'raise'} 중 하나여야 합니다.")

    # NaN 처리 정책
    if np.isnan(x_arr).any():
        if use_nan_policy == "raise":
            raise ValueError("입력 데이터에 NaN이 포함되어 있습니다.")
        # drop인 경우: lag별 계산 시 NaN 쌍 제거

    n = len(x_arr)
    if n < 3:
        raise ValueError("입력 데이터 길이는 최소 3 이상이어야 합니다.")

    if min_lag < 1:
        raise ValueError("min_lag는 1 이상이어야 합니다.")

    # 최대 lag 기본값 설정
    if max_lag is None:
        # 252 ≈ 1년 거래일, n//2는 표본 수 감소 방지
        max_lag = min(252, n // 2)

    if max_lag <= min_lag:
        raise ValueError(
            f"max_lag는 min_lag보다 커야 합니다. "
            f"(max_lag={max_lag}, min_lag={min_lag})"
        )

    lags = np.arange(min_lag, max_lag + 1, dtype=int)
    msd_vals = np.empty_like(lags, dtype=float)
    counts = np.empty_like(lags, dtype=int)

    # 각 lag에 대해 MSD 계산
    # 계산 복잡도는 O(K*N)이지만 일별 데이터에서는 충분히 빠름
    for i, tau in enumerate(lags):
        # x(t+τ) - x(t)
        dx = x_arr[tau:] - x_arr[:-tau]

        if use_nan_policy == "drop":
            valid = ~np.isnan(dx)
            dx = dx[valid]

        counts[i] = dx.size
        msd_vals[i] = np.mean(dx * dx) if dx.size > 0 else np.nan

    return MSDResult(lags=lags, msd=msd_vals, counts=counts)


def estimate_alpha(
    lags: ArrayLike,
    msd_values: ArrayLike,
    fit_range: Optional[Tuple[int, int]] = None,
) -> Tuple[float, float]:
    """
    MSD ~ τ^α 관계에서 이상확산 지수 α를 추정한다.

    log(MSD) = log(C) + α * log(τ)

    Parameters
    ----------
    lags : array-like
        lag(τ) 값
    msd_values : array-like
        MSD(τ) 값
    fit_range : (start_tau, end_tau), optional
        회귀에 사용할 τ 범위
        (미시 구간 잡음/대규모 lag 불안정성 제거 목적)

    Returns
    -------
    alpha : float
        이상확산 지수 α
    intercept : float
        log(C) 절편
    """
    tau = _to_1d_array(lags)
    msd_v = _to_1d_array(msd_values)

    if tau.shape != msd_v.shape:
        raise ValueError("lags와 msd_values의 shape가 일치해야 합니다.")

    # 양수이면서 유한한 값만 사용
    mask = np.isfinite(tau) & np.isfinite(msd_v) & (tau > 0) & (msd_v > 0)

    if fit_range is not None:
        start_tau, end_tau = fit_range
        mask &= (tau >= start_tau) & (tau <= end_tau)

    tau = tau[mask]
    msd_v = msd_v[mask]

    if tau.size < 3:
        raise ValueError("회귀를 위한 데이터가 부족합니다 (최소 3개 필요).")

    X = np.log(tau)
    Y = np.log(msd_v)

    # 최소제곱법을 이용한 선형 회귀
    A = np.vstack([X, np.ones_like(X)]).T
    alpha, intercept = np.linalg.lstsq(A, Y, rcond=None)[0]

    return float(alpha), float(intercept)


def diffusion_coefficient_from_msd(alpha: float, intercept: float) -> float:
    """
    MSD 결과로부터 확산 계수 D를 추정한다.

    정상 확산(α ≈ 1)일 경우:
        MSD(τ) ≈ 2Dτ

    Parameters
    ----------
    alpha : float
        이상확산 지수
    intercept : float
        log(C)

    Returns
    -------
    float
        추정된 확산 계수 D

    Notes
    -----
    - α가 1에 가까울 때만 D를 상수로 해석하는 것이 타당하다.
    - α가 1에서 멀면 '이상 확산'으로 해석해야 한다.
    """
    C = float(np.exp(intercept))
    return C / 2.0


def rolling_alpha(
    x: ArrayLike,
    window: int = 252,
    step: int = 5,
    max_lag: Optional[int] = None,
    min_lag: int = 1,
    fit_range: Optional[Tuple[int, int]] = None,
    use_nan_policy: str = "drop",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    시계열 x(t)에 대해 rolling window로 MSD를 계산하고,
    각 구간에서 MSD ~ tau^alpha의 alpha를 추정하여 시간에 따른 alpha 변화를 반환한다.

    Parameters
    ----------
    x : array-like
        1차원 시계열 (예: 로그가격 또는 누적수익률)
    window : int
        rolling 윈도우 길이 (일별 데이터 기준 252≈1년)
    step : int
        윈도우를 얼마나 이동할지 (예: 5면 5일 단위로 alpha 추정)
    max_lag : int, optional
        MSD 계산 최대 lag. None이면 각 윈도우 길이에 따라 자동 결정(min(252, window//2))
    min_lag : int
        MSD 계산 최소 lag
    fit_range : (start_tau, end_tau), optional
        alpha 추정에 사용할 lag 범위. None이면 가능한 구간 전체 사용.
        (추천: (5, 60) 또는 (5, 120)처럼 너무 작고/큰 lag는 제외)
    use_nan_policy : {"drop", "raise"}
        NaN 처리 방식

    Returns
    -------
    centers : np.ndarray
        각 윈도우의 '중심 인덱스'(시간축 위치). (실제 날짜가 있으면 나중에 매핑)
    alphas : np.ndarray
        각 윈도우에서 추정된 alpha 값. (추정 불가한 경우 np.nan)

    Notes
    -----
    - alpha는 log-log 회귀의 기울기이며, 확산 특성을 나타냄:
        alpha ≈ 1 : 정상 확산(브라운 운동)
        alpha > 1 : super-diffusion (추세/가속된 확산 성향)
        alpha < 1 : sub-diffusion (제약/평균회귀 성향 가능)
    """
    x_arr = _to_1d_array(x)

    if window < 10:
        raise ValueError("window가 너무 작습니다. 최소 10 이상을 권장합니다.")
    if step < 1:
        raise ValueError("step은 1 이상이어야 합니다.")

    n = len(x_arr)
    if n < window:
        raise ValueError(f"데이터 길이(n={n})가 window({window})보다 짧습니다.")

    centers = []
    alphas = []

    # 윈도우 시작 인덱스: 0, step, 2*step, ...
    for start in range(0, n - window + 1, step):
        end = start + window
        seg = x_arr[start:end]

        # 이 윈도우에서 사용할 max_lag 결정
        seg_max_lag = max_lag if max_lag is not None else min(252, len(seg) // 2)

        # MSD 계산
        try:
            res = msd(
                seg,
                max_lag=seg_max_lag,
                min_lag=min_lag,
                use_nan_policy=use_nan_policy,
            )
        except Exception:
            # 해당 윈도우에서 MSD 계산 자체가 실패하면 alpha는 NaN
            centers.append(start + window // 2)
            alphas.append(np.nan)
            continue

        # alpha 추정
        try:
            alpha, _ = estimate_alpha(res.lags, res.msd, fit_range=fit_range)
        except Exception:
            alpha = np.nan

        centers.append(start + window // 2)
        alphas.append(alpha)

    return np.asarray(centers, dtype=int), np.asarray(alphas, dtype=float)
