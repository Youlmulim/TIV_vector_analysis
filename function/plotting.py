"""
plotting.py
-----------
nilearn을 사용한 distance-based searchlight score map / comparison plot 저장 모듈.

저장 파일
---------
- searchlight_x_lateral.png   / searchlight_x_medial.png
- searchlight_y_lateral.png   / searchlight_y_medial.png
- searchlight_z_lateral.png   / searchlight_z_medial.png
- searchlight_comparison_lateral.png / searchlight_comparison_medial.png
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.colors import ListedColormap

logger = logging.getLogger(__name__)

try:
    from nilearn import datasets, plotting as nl_plotting
    from nilearn.surface import load_surf_mesh
    NILEARN_AVAILABLE = True
except ImportError:
    NILEARN_AVAILABLE = False
    logger.warning(
        "nilearn을 불러올 수 없습니다. --make-plots 옵션 사용 시 nilearn을 설치하세요."
    )


# ──────────────────────────────────────────────
# 공통 helper
# ──────────────────────────────────────────────

def _get_surf_mesh(fsaverage_name: str, hemi: str) -> Any:
    """
    fsaverage inflated surface mesh 경로를 반환한다.

    Parameters
    ----------
    fsaverage_name : str
    hemi : str  "L" 또는 "R"

    Returns
    -------
    str
        nilearn surface mesh 경로.
    """
    if not NILEARN_AVAILABLE:
        raise ImportError(
            "nilearn이 설치되지 않았습니다. `pip install nilearn` 으로 설치하세요."
        )
    fsaverage = datasets.fetch_surf_fsaverage(mesh=fsaverage_name)
    return fsaverage.infl_left if hemi == "L" else fsaverage.infl_right


def _close_figure(fig: Any) -> None:
    """matplotlib figure를 닫는다."""
    try:
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception:
        pass


def _save_surface_views(
    surf_mesh: Any,
    stat_map: np.ndarray,
    nilearn_hemi: str,
    out_dir: Path,
    base_name: str,
    title_base: str,
    plot_kwargs: dict[str, Any],
    is_roi: bool = False,
) -> None:
    """
    lateral / medial view 두 장을 저장하는 내부 helper.

    Parameters
    ----------
    surf_mesh : str
        nilearn surface mesh 경로.
    stat_map : np.ndarray
        shape (n_vertices,).
    nilearn_hemi : str
        "left" 또는 "right".
    out_dir : Path
        저장 디렉토리.
    base_name : str
        파일명 prefix (예: "searchlight_x").
    title_base : str
        plot title prefix.
    plot_kwargs : dict
        nilearn plot_surf_stat_map / plot_surf_roi에 전달할 kwargs.
    is_roi : bool
        True이면 plot_surf_roi, False이면 plot_surf_stat_map 사용.
    """
    views = [("lateral", "lateral"), ("medial", "medial")]

    for view, suffix in views:
        title = f"{title_base} ({view})"
        filename = f"{base_name}_{suffix}.png"

        if is_roi:
            fig = nl_plotting.plot_surf_roi(
                surf_mesh=surf_mesh,
                roi_map=stat_map,
                hemi=nilearn_hemi,
                view=view,
                title=title,
                **plot_kwargs,
            )
        else:
            fig = nl_plotting.plot_surf_stat_map(
                surf_mesh=surf_mesh,
                stat_map=stat_map,
                hemi=nilearn_hemi,
                view=view,
                title=title,
                **plot_kwargs,
            )

        save_path = out_dir / filename
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        _close_figure(fig)
        logger.info("  plot 저장: %s", save_path)


# ──────────────────────────────────────────────
# searchlight score map plot
# ──────────────────────────────────────────────

def plot_searchlight_surface(
    scores: np.ndarray,
    out_dir: Path,
    subject: str,
    session: str,
    set_id: str,
    hemi: str,
    target: str,
    fsaverage_name: str = "fsaverage",
) -> None:
    """
    단일 target의 distance searchlight score map을 surface에 시각화하고
    lateral / medial view PNG로 저장한다.

    파일명: searchlight_{target}_lateral.png, searchlight_{target}_medial.png

    Parameters
    ----------
    scores : np.ndarray
        shape (n_vertices,). NaN 포함 가능.
    out_dir : Path
        PNG 저장 디렉토리.
    subject, session, set_id, hemi : str
    target : str
        "x", "y", "z" 중 하나.
    fsaverage_name : str

    Raises
    ------
    ImportError
        nilearn이 없는 경우.
    ValueError
        hemi가 "L" 또는 "R"이 아닌 경우.
    """
    if not NILEARN_AVAILABLE:
        raise ImportError(
            "nilearn이 설치되지 않았습니다. `pip install nilearn` 으로 설치하세요."
        )
    if hemi not in ("L", "R"):
        raise ValueError(f"hemi는 'L' 또는 'R'이어야 합니다. 입력값: {hemi!r}")

    surf_mesh = _get_surf_mesh(fsaverage_name, hemi)
    nilearn_hemi = "left" if hemi == "L" else "right"

    target_labels = {
        "x": "dist(A_AB, A_AD)",
        "y": "dist(B_AB, B_BD)",
        "z": "dist(D_AD, D_BD)",
    }
    label = target_labels.get(target, target)
    title_base = f"{subject} | {session} | {set_id} | hemi-{hemi} | {label}"

    plot_kwargs = {
        "colorbar": True,
        "cmap": "hot_r",
        "bg_on_data": True,
        "darkness": 0.7,
    }

    logger.info(
        "[%s | %s | %s | hemi-%s] distance searchlight score map plot (target=%s)",
        subject, session, set_id, hemi, target,
    )

    _save_surface_views(
        surf_mesh=surf_mesh,
        stat_map=scores,
        nilearn_hemi=nilearn_hemi,
        out_dir=out_dir,
        base_name=f"searchlight_{target}",
        title_base=title_base,
        plot_kwargs=plot_kwargs,
        is_roi=False,
    )


# ──────────────────────────────────────────────
# comparison mask plot
# ──────────────────────────────────────────────

def plot_searchlight_comparison_surface(
    xy_mask: np.ndarray,
    xz_mask: np.ndarray,
    out_dir: Path,
    subject: str,
    session: str,
    set_id: str,
    hemi: str,
    fsaverage_name: str = "fsaverage",
) -> None:
    """
    x/y/z score 비교 mask를
    surface에 시각화하고 lateral / medial view PNG로 저장한다.

    색상 코드:
    - 빨강  (1): x > y 만 만족
    - 파랑  (2): x > z 만 만족
    - 노랑  (3): 둘 다 만족 (overlap)

    파일명: searchlight_comparison_lateral.png,
            searchlight_comparison_medial.png

    Parameters
    ----------
    xy_mask, xz_mask : np.ndarray
        shape (n_vertices,) bool mask.
    out_dir : Path
    subject, session, set_id, hemi : str
    fsaverage_name : str

    Raises
    ------
    ImportError
        nilearn이 없는 경우.
    ValueError
        hemi가 "L" 또는 "R"이 아닌 경우.
    """
    if not NILEARN_AVAILABLE:
        raise ImportError(
            "nilearn이 설치되지 않았습니다. `pip install nilearn` 으로 설치하세요."
        )
    if hemi not in ("L", "R"):
        raise ValueError(f"hemi는 'L' 또는 'R'이어야 합니다. 입력값: {hemi!r}")

    surf_mesh = _get_surf_mesh(fsaverage_name, hemi)
    nilearn_hemi = "left" if hemi == "L" else "right"

    # label map 구성
    roi_map = np.zeros(xy_mask.shape, dtype=np.int8)
    roi_map[xy_mask & ~xz_mask] = 1
    roi_map[xz_mask & ~xy_mask] = 2
    roi_map[xy_mask & xz_mask] = 3

    cmap = ListedColormap(["#d62728", "#1f77b4", "#ffbf00"])
    title_base = f"{subject} | {session} | {set_id} | hemi-{hemi} | x>y/x>z comparison"

    plot_kwargs = {
        "colorbar": False,
        "cmap": cmap,
        "bg_on_data": True,
    }

    logger.info(
        "[%s | %s | %s | hemi-%s] comparison mask plot",
        subject, session, set_id, hemi,
    )

    _save_surface_views(
        surf_mesh=surf_mesh,
        stat_map=roi_map,
        nilearn_hemi=nilearn_hemi,
        out_dir=out_dir,
        base_name="searchlight_comparison",
        title_base=title_base,
        plot_kwargs=plot_kwargs,
        is_roi=True,
    )
