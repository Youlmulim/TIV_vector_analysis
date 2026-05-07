"""
searchlight.py
--------------
Distance-based surface searchlight 분석 모듈.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    logger.warning("tqdm이 설치되지 않았습니다. 진행률 표시 없이 실행됩니다.")

try:
    from nilearn import datasets as nl_datasets
    NILEARN_AVAILABLE = True
except ImportError:
    NILEARN_AVAILABLE = False
    logger.warning("nilearn을 불러올 수 없습니다. surface coordinate 로드 시 nilearn이 필요합니다.")


def get_surface_coordinates(fsaverage_name: str, hemi: str) -> np.ndarray:
    """fsaverage surface mesh의 vertex 좌표를 반환한다."""
    if not NILEARN_AVAILABLE:
        raise ImportError(
            "nilearn이 설치되지 않았습니다. "
            "`pip install nilearn` 으로 설치 후 사용하세요."
        )
    if hemi not in ("L", "R"):
        raise ValueError(f"hemi는 'L' 또는 'R'이어야 합니다. 입력값: {hemi!r}")

    fsaverage = nl_datasets.fetch_surf_fsaverage(mesh=fsaverage_name)
    surf_path = fsaverage.infl_left if hemi == "L" else fsaverage.infl_right

    from nilearn.surface import load_surf_mesh

    coords, _ = load_surf_mesh(surf_path)
    logger.info(
        "fsaverage 좌표 로드 완료: mesh=%s, hemi=%s, n_vertices=%d",
        fsaverage_name, hemi, coords.shape[0],
    )
    return coords.astype(np.float32)


def build_surface_neighborhoods(coords: np.ndarray, radius_mm: float) -> list[np.ndarray]:
    """각 vertex 중심의 radius 기반 neighborhood를 구성한다."""
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords shape은 (n_vertices, 3)이어야 합니다. 입력값: {coords.shape}")

    tree = cKDTree(coords)
    logger.info(
        "neighborhood 구성 중: n_vertices=%d, radius=%.1f mm",
        coords.shape[0], radius_mm,
    )
    indices_list = tree.query_ball_point(coords, r=radius_mm, workers=-1)
    neighborhoods = [np.array(sorted(idx_list), dtype=np.int64) for idx_list in indices_list]

    sizes = np.array([len(neighbors) for neighbors in neighborhoods], dtype=np.int32)
    logger.info(
        "neighborhood 구성 완료: min=%d, max=%d, mean=%.1f vertices",
        int(sizes.min()), int(sizes.max()), float(sizes.mean()),
    )
    return neighborhoods


def compute_distance_searchlight_map(
    pattern_a: np.ndarray,
    pattern_b: np.ndarray,
    neighborhoods: list[np.ndarray],
    min_features: int = 5,
    distance_metric: str = "correlation",
) -> np.ndarray:
    """
    각 vertex neighborhood에서 두 local pattern 사이 distance를 계산한다.

    Parameters
    ----------
    pattern_a, pattern_b : np.ndarray
        shape (n_vertices,). run-across 평균된 condition beta pattern.
    neighborhoods : list[np.ndarray]
        각 중심 vertex의 local neighborhood 인덱스 목록.
    min_features : int
        local vertex 수가 이 값 미만이면 NaN.
    distance_metric : str
        "euclidean", "correlation", "cosine" 중 하나.
    """
    _validate_patterns(pattern_a, pattern_b, neighborhoods)

    n_vertices = pattern_a.shape[0]
    score_map = np.full(n_vertices, np.nan, dtype=np.float32)

    logger.info(
        "distance searchlight 시작: n_vertices=%d, min_features=%d, metric=%s",
        n_vertices, min_features, distance_metric,
    )

    iterator = range(n_vertices)
    if TQDM_AVAILABLE:
        iterator = tqdm(iterator, desc="searchlight", unit="vertex", ncols=80)

    for vertex_idx in iterator:
        neighbor_idx = neighborhoods[vertex_idx]
        if len(neighbor_idx) < min_features:
            continue

        local_a = pattern_a[neighbor_idx]
        local_b = pattern_b[neighbor_idx]
        score_map[vertex_idx] = _compute_local_distance(local_a, local_b, distance_metric)

    n_valid = int(np.sum(~np.isnan(score_map)))
    n_nan = int(np.sum(np.isnan(score_map)))
    logger.info(
        "distance searchlight 완료: valid=%d, NaN=%d, mean(valid)=%.6f, max=%.6f",
        n_valid, n_nan, float(np.nanmean(score_map)), float(np.nanmax(score_map)),
    )
    return score_map


def create_comparison_masks(
    x_scores: np.ndarray,
    y_scores: np.ndarray,
    z_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """세 score map에서 x>y, x>z, overlap mask를 생성한다."""
    if x_scores.shape != y_scores.shape or x_scores.shape != z_scores.shape:
        raise ValueError(
            f"score map shape이 일치해야 합니다. x={x_scores.shape}, y={y_scores.shape}, z={z_scores.shape}"
        )

    xy_mask = np.where(np.isnan(x_scores) | np.isnan(y_scores), False, x_scores > y_scores)
    xz_mask = np.where(np.isnan(x_scores) | np.isnan(z_scores), False, x_scores > z_scores)
    overlap = xy_mask & xz_mask
    return xy_mask.astype(bool), xz_mask.astype(bool), overlap.astype(bool)


def _validate_patterns(
    pattern_a: np.ndarray,
    pattern_b: np.ndarray,
    neighborhoods: list[np.ndarray],
) -> None:
    if pattern_a.ndim != 1 or pattern_b.ndim != 1:
        raise ValueError(
            f"pattern_a/pattern_b는 1D여야 합니다. 입력 shape: {pattern_a.shape}, {pattern_b.shape}"
        )
    if pattern_a.shape != pattern_b.shape:
        raise ValueError(
            f"pattern shape이 일치해야 합니다. 입력 shape: {pattern_a.shape}, {pattern_b.shape}"
        )
    if len(neighborhoods) != pattern_a.shape[0]:
        raise ValueError(
            f"neighborhood 개수({len(neighborhoods)})가 vertex 수({pattern_a.shape[0]})와 다릅니다."
        )


def _compute_local_distance(
    local_a: np.ndarray,
    local_b: np.ndarray,
    distance_metric: str,
) -> float:
    diff = local_a - local_b

    if distance_metric == "euclidean":
        return float(np.linalg.norm(diff))

    if distance_metric == "correlation":
        a_centered = local_a - np.mean(local_a)
        b_centered = local_b - np.mean(local_b)
        denom = np.linalg.norm(a_centered) * np.linalg.norm(b_centered)
        if denom == 0.0:
            return np.nan
        corr = float(np.dot(a_centered, b_centered) / denom)
        return 1.0 - corr

    if distance_metric == "cosine":
        denom = np.linalg.norm(local_a) * np.linalg.norm(local_b)
        if denom == 0.0:
            return np.nan
        cosine_sim = float(np.dot(local_a, local_b) / denom)
        return 1.0 - cosine_sim

    raise ValueError(
        f"distance_metric은 'euclidean', 'correlation', 'cosine' 중 하나여야 합니다. "
        f"입력값: {distance_metric!r}"
    )
