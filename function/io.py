"""
io.py
-----
Distance-based searchlight용 beta 로딩/결과 저장 모듈.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# (context, condition) -> key
CONDITION_SPECS: dict[str, tuple[str, str]] = {
    "A_AB": ("AB", "A_pure"),
    "A_AD": ("AD", "A_pure"),
    "B_AB": ("AB", "B_pure"),
    "B_BD": ("BD", "B_pure"),
    "D_AD": ("AD", "D_pure"),
    "D_BD": ("BD", "D_pure"),
}

# target -> (condition_key_0, condition_key_1)
TARGET_SPECS: dict[str, tuple[str, str]] = {
    "x": ("A_AB", "A_AD"),
    "y": ("B_AB", "B_BD"),
    "z": ("D_AD", "D_BD"),
}


def load_selection_session_mapping(json_path: str | Path) -> dict[str, str]:
    """subject -> selection session 매핑 JSON을 로드한다."""
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Selection session mapping JSON을 찾을 수 없습니다: {path}")
    print(f"[load] {path}")
    with path.open("r", encoding="utf-8") as f:
        mapping: dict[str, str] = json.load(f)
    logger.info("Selection session mapping 로드 완료 (%d subjects)", len(mapping))
    return mapping


def find_beta_files(
    glm_root: Path,
    subject: str,
    session: str,
    set_id: str,
    context: str,
    condition: str,
    hemi: str,
) -> list[Path]:
    """주어진 context/condition/hemi에 해당하는 모든 run beta 파일을 찾는다."""
    beta_dir = glm_root / subject / session / set_id / "betas"
    pattern = (
        f"{subject}_{session}_task-TIV{set_id}{context}"
        f"_run-*_condition-{condition}_hemi-{hemi}_beta.npy"
    )
    files = sorted(beta_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"[{subject} | {session} | {set_id} | hemi-{hemi}] "
            f"beta 파일을 찾을 수 없습니다.\n"
            f"  탐색 경로: {beta_dir}\n"
            f"  패턴: {pattern}"
        )

    logger.debug(
        "[%s | %s | %s | hemi-%s] context=%s condition=%s -> %d개 파일 발견",
        subject, session, set_id, hemi, context, condition, len(files),
    )
    return files


def load_and_average_beta(files: list[Path]) -> np.ndarray:
    """여러 run beta를 로드해 run-across 평균 pattern을 반환한다."""
    arrays: list[np.ndarray] = []

    for path in files:
        print(f"[load] {path}")
        arr = np.load(path)
        arrays.append(arr)

    shapes = [arr.shape for arr in arrays]
    if len(set(shapes)) > 1:
        raise ValueError(
            f"Run beta 파일들 간 shape이 일치하지 않습니다: {shapes}\n"
            f"  파일 목록: {[str(path) for path in files]}"
        )

    stacked = np.stack(arrays, axis=0)
    averaged = stacked.mean(axis=0)
    logger.debug("%d개 run 평균 완료 -> shape %s", len(files), averaged.shape)
    return averaged


def load_condition_betas(
    glm_root: Path,
    subject: str,
    session: str,
    set_id: str,
    hemi: str,
) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    """6개 condition beta를 모두 로드하고 condition별 run-across 평균을 반환한다."""
    beta_dict: dict[str, np.ndarray] = {}
    files_used: dict[str, list[str]] = {}

    for key, (context, condition) in CONDITION_SPECS.items():
        logger.info(
            "[%s | %s | %s | hemi-%s] run-average beta 로딩 중: %s",
            subject, session, set_id, hemi, key,
        )
        found_files = find_beta_files(
            glm_root=glm_root,
            subject=subject,
            session=session,
            set_id=set_id,
            context=context,
            condition=condition,
            hemi=hemi,
        )
        beta_dict[key] = load_and_average_beta(found_files)
        files_used[key] = [str(path) for path in found_files]

    return beta_dict, files_used


def save_subject_score_maps(
    out_root: Path,
    subject: str,
    session: str,
    set_id: str,
    hemi: str,
    x_scores: np.ndarray,
    y_scores: np.ndarray,
    z_scores: np.ndarray,
    files_used: dict[str, list[str]],
    metadata: dict[str, Any],
) -> Path:
    """subject/session/set/hemi 단위 distance searchlight score map을 저장한다."""
    out_dir = out_root / subject / session / set_id / f"hemi-{hemi}"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "searchlight_scores_x.npy", x_scores)
    np.save(out_dir / "searchlight_scores_y.npy", y_scores)
    np.save(out_dir / "searchlight_scores_z.npy", z_scores)

    summary: dict[str, Any] = {
        "subject": subject,
        "session": session,
        "set_id": set_id,
        "hemisphere": hemi,
        "input_files_used": files_used,
        "n_files_per_condition": {key: len(paths) for key, paths in files_used.items()},
        "score_interpretation": {
            "x_score": "distance(A_AB local pattern, A_AD local pattern)",
            "y_score": "distance(B_AB local pattern, B_BD local pattern)",
            "z_score": "distance(D_AD local pattern, D_BD local pattern)",
        },
        **_score_stats_dict(x_scores, "x"),
        **_score_stats_dict(y_scores, "y"),
        **_score_stats_dict(z_scores, "z"),
        **metadata,
    }

    with (out_dir / "searchlight_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(
        "[%s | %s | %s | hemi-%s] subject score map 저장 완료: %s",
        subject, session, set_id, hemi, out_dir,
    )
    return out_dir


def load_saved_score_maps(score_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """저장된 x/y/z searchlight score map을 로드한다."""
    x_path = score_dir / "searchlight_scores_x.npy"
    y_path = score_dir / "searchlight_scores_y.npy"
    z_path = score_dir / "searchlight_scores_z.npy"

    for path in (x_path, y_path, z_path):
        if not path.exists():
            raise FileNotFoundError(f"score map 파일이 없습니다: {path}")

    print(f"[load] {x_path}")
    x_scores = np.load(x_path)
    print(f"[load] {y_path}")
    y_scores = np.load(y_path)
    print(f"[load] {z_path}")
    z_scores = np.load(z_path)
    return x_scores, y_scores, z_scores


def save_group_mean_outputs(
    out_root: Path,
    set_id: str,
    hemi: str,
    x_mean: np.ndarray,
    y_mean: np.ndarray,
    z_mean: np.ndarray,
    xy_mask: np.ndarray,
    xz_mask: np.ndarray,
    subject_dirs: list[str],
    metadata: dict[str, Any],
) -> Path:
    """across-subject mean score map과 최종 비교 mask를 저장한다."""
    out_dir = out_root / "combine" / set_id / f"hemi-{hemi}"
    out_dir.mkdir(parents=True, exist_ok=True)

    overlap = xy_mask & xz_mask
    label_map = np.zeros(xy_mask.shape, dtype=np.int8)
    label_map[xy_mask & ~xz_mask] = 1
    label_map[xz_mask & ~xy_mask] = 2
    label_map[overlap] = 3

    np.save(out_dir / "searchlight_scores_x_mean.npy", x_mean)
    np.save(out_dir / "searchlight_scores_y_mean.npy", y_mean)
    np.save(out_dir / "searchlight_scores_z_mean.npy", z_mean)
    np.save(out_dir / "searchlight_mask_x_gt_y.npy", xy_mask)
    np.save(out_dir / "searchlight_mask_x_gt_z.npy", xz_mask)
    np.save(out_dir / "searchlight_overlap_mask.npy", overlap)
    np.save(out_dir / "searchlight_label_map.npy", label_map)

    n_vertices = int(xy_mask.size)
    summary: dict[str, Any] = {
        "combine_type": "across_subject_mean",
        "set_id": set_id,
        "hemisphere": hemi,
        "subject_score_dirs": subject_dirs,
        "n_subjects": len(subject_dirs),
        "n_vertices": n_vertices,
        "n_vertices_x_gt_y": int(xy_mask.sum()),
        "n_vertices_x_gt_z": int(xz_mask.sum()),
        "n_vertices_overlap": int(overlap.sum()),
        "percent_x_gt_y": round(100.0 * float(xy_mask.sum()) / n_vertices, 4) if n_vertices > 0 else 0.0,
        "percent_x_gt_z": round(100.0 * float(xz_mask.sum()) / n_vertices, 4) if n_vertices > 0 else 0.0,
        "percent_overlap": round(100.0 * float(overlap.sum()) / n_vertices, 4) if n_vertices > 0 else 0.0,
        "selection_rule": {
            "xy_mask": "x_mean > y_mean",
            "xz_mask": "x_mean > z_mean",
            "overlap": "(x_mean > y_mean) & (x_mean > z_mean)",
        },
        "score_interpretation": {
            "x_mean": "mean(subject x_score maps)",
            "y_mean": "mean(subject y_score maps)",
            "z_mean": "mean(subject z_score maps)",
        },
        **_score_stats_dict(x_mean, "x_mean"),
        **_score_stats_dict(y_mean, "y_mean"),
        **_score_stats_dict(z_mean, "z_mean"),
        **metadata,
    }

    with (out_dir / "searchlight_group_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("[combine | %s | hemi-%s] group mean 저장 완료: %s", set_id, hemi, out_dir)
    return out_dir


def _score_stats_dict(arr: np.ndarray, name: str) -> dict[str, Any]:
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return {
            f"score_{name}_mean": None,
            f"score_{name}_std": None,
            f"score_{name}_min": None,
            f"score_{name}_max": None,
            f"score_{name}_n_valid": 0,
            f"score_{name}_n_nan": int(np.isnan(arr).sum()),
        }

    return {
        f"score_{name}_mean": round(float(np.mean(valid)), 6),
        f"score_{name}_std": round(float(np.std(valid)), 6),
        f"score_{name}_min": round(float(np.min(valid)), 6),
        f"score_{name}_max": round(float(np.max(valid)), 6),
        f"score_{name}_n_valid": int(len(valid)),
        f"score_{name}_n_nan": int(np.isnan(arr).sum()),
    }
