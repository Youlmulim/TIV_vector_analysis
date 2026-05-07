"""
main.py
-------
Distance-based surface searchlight 파이프라인 진입점.

분석 개요
----------
1. 각 subject/session/set/hemi에서 condition별 run-across 평균 beta를 만든다.
2. 각 vertex neighborhood에서 local pattern distance를 계산한다.
3. subject별 x/y/z score map을 저장한다.
4. set/hemi별 across-subject mean map을 계산한다.
5. group mean map에서 x_mean > y_mean, x_mean > z_mean, overlap을 계산한다.

python /home/youlim/TIV/vector_analysis/step1-3_searchlight/main.py \
  --subjects sub-001 sub-002 sub-003 \
  --sets Q1 Q2 \
  --hemis L R \
  --selection-session-json /home/youlim/TIV/vector_analysis/step1-3_searchlight/mapping.json \
  --glm-root /home/youlim/TIV/vector_analysis/step1_GLM/results \
  --out-root /home/youlim/TIV/vector_analysis/step1-3_searchlight/results \
  --radius-mm 6 \
  --min-features 5 \
  --distance-metric correlation \
  --make-plots

"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np

from functions.io import (
    TARGET_SPECS,
    load_condition_betas,
    load_saved_score_maps,
    load_selection_session_mapping,
    save_group_mean_outputs,
    save_subject_score_maps,
)
from functions.plotting import (
    plot_searchlight_comparison_surface,
    plot_searchlight_surface,
)
from functions.searchlight import (
    build_surface_neighborhoods,
    compute_distance_searchlight_map,
    create_comparison_masks,
    get_surface_coordinates,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Distance-based surface searchlight pipeline: "
            "각 vertex neighborhood의 local pattern distance를 계산하고 "
            "across-subject mean 비교 map을 생성한다."
        )
    )
    parser.add_argument(
        "--glm-root",
        type=Path,
        default=Path("/home/youlim/TIV/vector_analysis/step1_GLM/results"),
        help="GLM 결과 루트 디렉토리 (기본값: %(default)s)",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("/home/youlim/TIV/vector_analysis/step1-3_searchlight/results"),
        help="출력 루트 디렉토리 (기본값: %(default)s)",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        required=True,
        metavar="SUBJECT",
        help="처리할 subject ID 목록",
    )
    parser.add_argument(
        "--sets",
        nargs="+",
        default=["Q1", "Q2"],
        metavar="SET_ID",
        help="처리할 set ID 목록 (기본값: %(default)s)",
    )
    parser.add_argument(
        "--hemis",
        nargs="+",
        default=["L", "R"],
        choices=["L", "R"],
        metavar="HEMI",
        help="처리할 hemisphere 목록 (기본값: %(default)s)",
    )
    parser.add_argument(
        "--selection-session-json",
        type=Path,
        required=True,
        metavar="JSON_PATH",
        help="subject -> selection session 매핑 JSON",
    )
    parser.add_argument(
        "--radius-mm",
        type=float,
        default=6.0,
        metavar="RADIUS",
        help="searchlight neighborhood 반경 (mm, 기본값: %(default)s)",
    )
    parser.add_argument(
        "--min-features",
        type=int,
        default=5,
        metavar="N",
        help="neighborhood 최소 vertex 수. 미만이면 NaN (기본값: %(default)s)",
    )
    parser.add_argument(
        "--distance-metric",
        type=str,
        default="correlation",
        choices=["euclidean", "correlation", "cosine"],
        help="local pattern distance metric (기본값: %(default)s)",
    )
    parser.add_argument(
        "--make-plots",
        action="store_true",
        help="subject 및 group surface plot을 PNG로 저장할지 여부",
    )
    parser.add_argument(
        "--fsaverage",
        type=str,
        default="fsaverage",
        help="nilearn.datasets.fetch_surf_fsaverage mesh 이름 (기본값: %(default)s)",
    )
    return parser.parse_args()


def process_one_searchlight(
    glm_root: Path,
    out_root: Path,
    subject: str,
    session: str,
    set_id: str,
    hemi: str,
    neighborhoods: list[np.ndarray],
    radius_mm: float,
    distance_metric: str,
    min_features: int,
    make_plots: bool,
    fsaverage_name: str,
) -> tuple[bool, Path | None]:
    """subject/session/set/hemi 단위 distance searchlight를 실행한다."""
    tag = f"[{subject} | {session} | {set_id} | hemi-{hemi}]"
    logger.info("─" * 60)
    logger.info("%s distance searchlight 시작", tag)

    try:
        beta_dict, files_used = load_condition_betas(
            glm_root=glm_root,
            subject=subject,
            session=session,
            set_id=set_id,
            hemi=hemi,
        )

        x_scores = compute_distance_searchlight_map(
            pattern_a=beta_dict[TARGET_SPECS["x"][0]],
            pattern_b=beta_dict[TARGET_SPECS["x"][1]],
            neighborhoods=neighborhoods,
            min_features=min_features,
            distance_metric=distance_metric,
        )
        y_scores = compute_distance_searchlight_map(
            pattern_a=beta_dict[TARGET_SPECS["y"][0]],
            pattern_b=beta_dict[TARGET_SPECS["y"][1]],
            neighborhoods=neighborhoods,
            min_features=min_features,
            distance_metric=distance_metric,
        )
        z_scores = compute_distance_searchlight_map(
            pattern_a=beta_dict[TARGET_SPECS["z"][0]],
            pattern_b=beta_dict[TARGET_SPECS["z"][1]],
            neighborhoods=neighborhoods,
            min_features=min_features,
            distance_metric=distance_metric,
        )

        out_dir = save_subject_score_maps(
            out_root=out_root,
            subject=subject,
            session=session,
            set_id=set_id,
            hemi=hemi,
            x_scores=x_scores,
            y_scores=y_scores,
            z_scores=z_scores,
            files_used=files_used,
            metadata={
                "radius_mm": radius_mm,
                "min_features": min_features,
                "distance_metric": distance_metric,
            },
        )

        if make_plots:
            for target_name, scores in (("x", x_scores), ("y", y_scores), ("z", z_scores)):
                plot_searchlight_surface(
                    scores=scores,
                    out_dir=out_dir,
                    subject=subject,
                    session=session,
                    set_id=set_id,
                    hemi=hemi,
                    target=target_name,
                    fsaverage_name=fsaverage_name,
                )

        logger.info("%s 처리 완료", tag)
        return True, out_dir

    except FileNotFoundError as e:
        logger.error("%s 파일 없음 오류:\n  %s", tag, e)
        return False, None
    except (ValueError, KeyError) as e:
        logger.error("%s 데이터 오류:\n  %s", tag, e)
        return False, None
    except ImportError as e:
        logger.error("%s import 오류:\n  %s", tag, e)
        return False, None
    except Exception as e:
        logger.exception("%s 예상치 못한 오류: %s", tag, e)
        return False, None


def aggregate_across_subjects(
    out_root: Path,
    session_mapping: dict[str, str],
    subjects: list[str],
    sets: list[str],
    hemis: list[str],
    make_plots: bool,
    fsaverage_name: str,
    distance_metric: str,
    radius_mm: float,
    min_features: int,
) -> list[tuple[str, bool]]:
    """set/hemi별 across-subject mean score map과 최종 comparison plot을 생성한다."""
    results: list[tuple[str, bool]] = []

    for set_id in sets:
        for hemi in hemis:
            tag = f"[combine | {set_id} | hemi-{hemi}]"
            logger.info("─" * 60)
            logger.info("%s across-subject mean 시작", tag)

            try:
                x_maps: list[np.ndarray] = []
                y_maps: list[np.ndarray] = []
                z_maps: list[np.ndarray] = []
                subject_dirs: list[str] = []

                for subject in subjects:
                    if subject not in session_mapping:
                        continue
                    session = session_mapping[subject]
                    score_dir = out_root / subject / session / set_id / f"hemi-{hemi}"
                    x_scores, y_scores, z_scores = load_saved_score_maps(score_dir)
                    x_maps.append(x_scores)
                    y_maps.append(y_scores)
                    z_maps.append(z_scores)
                    subject_dirs.append(str(score_dir))

                if not subject_dirs:
                    raise FileNotFoundError(f"{tag} group mean에 사용할 subject score map이 없습니다.")

                x_mean = nanmean_stack(x_maps)
                y_mean = nanmean_stack(y_maps)
                z_mean = nanmean_stack(z_maps)
                xy_mask, xz_mask, _ = create_comparison_masks(x_mean, y_mean, z_mean)

                out_dir = save_group_mean_outputs(
                    out_root=out_root,
                    set_id=set_id,
                    hemi=hemi,
                    x_mean=x_mean,
                    y_mean=y_mean,
                    z_mean=z_mean,
                    xy_mask=xy_mask,
                    xz_mask=xz_mask,
                    subject_dirs=subject_dirs,
                    metadata={
                        "distance_metric": distance_metric,
                        "radius_mm": radius_mm,
                        "min_features": min_features,
                    },
                )

                if make_plots:
                    for target_name, scores in (
                        ("x", x_mean),
                        ("y", y_mean),
                        ("z", z_mean),
                    ):
                        plot_searchlight_surface(
                            scores=scores,
                            out_dir=out_dir,
                            subject="combine",
                            session="across-subjects",
                            set_id=set_id,
                            hemi=hemi,
                            target=target_name,
                            fsaverage_name=fsaverage_name,
                        )

                    plot_searchlight_comparison_surface(
                        xy_mask=xy_mask,
                        xz_mask=xz_mask,
                        out_dir=out_dir,
                        subject="combine",
                        session="across-subjects",
                        set_id=set_id,
                        hemi=hemi,
                        fsaverage_name=fsaverage_name,
                    )

                logger.info("%s across-subject mean 완료", tag)
                results.append((tag, True))
            except Exception as e:
                logger.exception("%s across-subject mean 오류: %s", tag, e)
                results.append((tag, False))

    return results


def nanmean_stack(arrays: list[np.ndarray]) -> np.ndarray:
    """shape이 같은 배열들의 NaN-safe mean을 계산한다."""
    shapes = {arr.shape for arr in arrays}
    if len(shapes) != 1:
        raise ValueError(f"across-subject mean을 위해서는 shape이 같아야 합니다. 입력 shape: {sorted(shapes)}")

    stacked = np.stack(arrays, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_map = np.nanmean(stacked, axis=0)
    return mean_map.astype(np.float32)


def build_neighborhood_cache(
    fsaverage_name: str,
    hemis: list[str],
    radius_mm: float,
) -> dict[str, list[np.ndarray]]:
    """hemi별 neighborhood를 한 번만 계산해 캐시한다."""
    cache: dict[str, list[np.ndarray]] = {}
    for hemi in hemis:
        coords = get_surface_coordinates(fsaverage_name=fsaverage_name, hemi=hemi)
        cache[hemi] = build_surface_neighborhoods(coords=coords, radius_mm=radius_mm)
    return cache


def main() -> None:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Distance-Based Surface Searchlight Pipeline 시작")
    logger.info("  GLM root       : %s", args.glm_root)
    logger.info("  Out root       : %s", args.out_root)
    logger.info("  Subjects       : %s", args.subjects)
    logger.info("  Sets           : %s", args.sets)
    logger.info("  Hemis          : %s", args.hemis)
    logger.info("  radius_mm      : %.1f", args.radius_mm)
    logger.info("  min_features   : %d", args.min_features)
    logger.info("  distance_metric: %s", args.distance_metric)
    logger.info("  make_plots     : %s", args.make_plots)
    logger.info("  fsaverage      : %s", args.fsaverage)
    logger.info("=" * 60)

    session_mapping = load_selection_session_mapping(args.selection_session_json)
    neighborhoods_by_hemi = build_neighborhood_cache(
        fsaverage_name=args.fsaverage,
        hemis=args.hemis,
        radius_mm=args.radius_mm,
    )

    valid_subjects = [subject for subject in args.subjects if subject in session_mapping]
    results: list[tuple[str, str, str, str, bool]] = []

    for subject in args.subjects:
        if subject not in session_mapping:
            logger.error(
                "[%s] selection session mapping에 해당 subject가 없습니다. mapping keys: %s",
                subject, list(session_mapping.keys()),
            )
            for set_id in args.sets:
                for hemi in args.hemis:
                    results.append((subject, "N/A", set_id, hemi, False))
            continue

        session = session_mapping[subject]
        for set_id in args.sets:
            for hemi in args.hemis:
                success, _ = process_one_searchlight(
                    glm_root=args.glm_root,
                    out_root=args.out_root,
                    subject=subject,
                    session=session,
                    set_id=set_id,
                    hemi=hemi,
                    neighborhoods=neighborhoods_by_hemi[hemi],
                    radius_mm=args.radius_mm,
                    distance_metric=args.distance_metric,
                    min_features=args.min_features,
                    make_plots=args.make_plots,
                    fsaverage_name=args.fsaverage,
                )
                results.append((subject, session, set_id, hemi, success))

    aggregate_results = aggregate_across_subjects(
        out_root=args.out_root,
        session_mapping=session_mapping,
        subjects=valid_subjects,
        sets=args.sets,
        hemis=args.hemis,
        make_plots=args.make_plots,
        fsaverage_name=args.fsaverage,
        distance_metric=args.distance_metric,
        radius_mm=args.radius_mm,
        min_features=args.min_features,
    )

    logger.info("=" * 60)
    logger.info("Pipeline 완료 — 처리 결과 요약")
    n_ok = sum(1 for *_, ok in results if ok)
    n_fail = len(results) - n_ok
    n_agg_ok = sum(1 for _, ok in aggregate_results if ok)
    n_agg_fail = len(aggregate_results) - n_agg_ok
    logger.info("  subject-level 성공: %d / %d", n_ok, len(results))
    logger.info("  across-subject 성공: %d / %d", n_agg_ok, len(aggregate_results))

    if n_fail > 0:
        logger.warning("  subject-level 실패: %d 건", n_fail)
        for sub, ses, sid, hm, ok in results:
            if not ok:
                logger.warning("    ✗ [%s | %s | %s | hemi-%s]", sub, ses, sid, hm)

    if n_agg_fail > 0:
        logger.warning("  across-subject 실패: %d 건", n_agg_fail)
        for tag, ok in aggregate_results:
            if not ok:
                logger.warning("    ✗ %s", tag)

    logger.info("=" * 60)

    if n_fail > 0 or n_agg_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
