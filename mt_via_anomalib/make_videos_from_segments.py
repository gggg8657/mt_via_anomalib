import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Optional

import cv2


def _normalize_path(path: str, base_root: Optional[str]) -> str:
    if os.path.isabs(path):
        return path
    if base_root:
        return os.path.normpath(os.path.join(base_root, path))
    return os.path.normpath(path)


def _flatten(list_of_lists: List[List[str]]) -> List[str]:
    return [item for sub in list_of_lists for item in sub]


def _ensure_existing(paths: List[str]) -> List[str]:
    return [p for p in paths if os.path.isfile(p)]


def _crop_image(img, x: int, y: int, w: int, h: int):
    if img is None:
        return None
    ih, iw = img.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(iw, x0 + max(0, w))
    y1 = min(ih, y0 + max(0, h))
    if x1 <= x0 or y1 <= y0:
        return None
    return img[y0:y1, x0:x1]


def _detect_frame_size(first_image_path: str, crop_rect: Optional[Tuple[int, int, int, int]] = None) -> Tuple[int, int]:
    img = cv2.imread(first_image_path)
    if img is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {first_image_path}")
    if crop_rect is not None:
        cx, cy, cw, ch = crop_rect
        img = _crop_image(img, cx, cy, cw, ch)
        if img is None:
            raise ValueError(f"크롭 결과가 비어 있습니다. 경로: {first_image_path}, 크롭: {crop_rect}")
    h, w = img.shape[:2]
    return w, h


def _write_videos_for_category(
    category: str,
    image_paths: List[str],
    output_dir: str,
    fps: int,
    chunk_size: int,
    fourcc_str: str,
    resize_to: Optional[Tuple[int, int]] = None,
    start_index: int = 0,
    crop_rect: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[List[str], int]:
    written_files: List[str] = []
    if not image_paths:
        return written_files, start_index

    os.makedirs(output_dir, exist_ok=True)

    # 결정적 프레임 크기
    if resize_to is None:
        frame_size = _detect_frame_size(image_paths[0], crop_rect)  # (w, h)
    else:
        frame_size = resize_to

    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)

    # 청크 단위로 분할하여 normal_0, normal_1 ... 형식으로 저장
    num_chunks = (len(image_paths) + chunk_size - 1) // chunk_size
    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = min((chunk_idx + 1) * chunk_size, len(image_paths))
        chunk_images = image_paths[start:end]

        out_name = f"{category}_{start_index + chunk_idx}.avi"
        out_path = os.path.join(output_dir, out_name)

        writer = cv2.VideoWriter(out_path, fourcc, fps, frame_size)
        if not writer.isOpened():
            raise RuntimeError(f"비디오 파일을 열 수 없습니다: {out_path}")

        for img_path in chunk_images:
            img = cv2.imread(img_path)
            if img is None:
                # 손상/누락 프레임은 건너뛰기
                continue
            if crop_rect is not None:
                cx, cy, cw, ch = crop_rect
                img = _crop_image(img, cx, cy, cw, ch)
                if img is None:
                    continue
            if (img.shape[1], img.shape[0]) != frame_size:
                img = cv2.resize(img, frame_size, interpolation=cv2.INTER_AREA)
            writer.write(img)

        writer.release()
        written_files.append(out_path)

    return written_files, start_index + num_chunks


def _extract_category_to_images(data: Any, base_root: Optional[str]) -> Dict[str, List[List[str]]]:
    """
    image_segments.json의 다양한 스키마를 너그럽게 지원하는 추출기.

    지원 형태 예시:
    1) [{"category": "normal", "images": ["a.jpg", "b.jpg"]}, ...]
    2) [{"category": "normal", "paths": [...]}, ...] 또는 "frames" 키
    3) {"normal": ["a.jpg", "b.jpg"], "abnormal": ["..."]}
    4) [{"category": "normal", "items": [{"path": "a.jpg"}, {"path": "b.jpg"}]}]
    5) 각 항목이 단일 이미지 레코드: [{"category": "normal", "path": "a.jpg"}, ...]
    """

    category_to_lists: Dict[str, List[List[str]]] = defaultdict(list)

    def normalize_list(paths: List[str]) -> List[str]:
        return [_normalize_path(p, base_root) for p in paths]

    if isinstance(data, dict):
        # 형태 3) 가정: 카테고리 키 -> 이미지 경로 리스트
        for cat, paths in data.items():
            if isinstance(paths, list) and paths and isinstance(paths[0], str):
                category_to_lists[cat].append(normalize_list(paths))
            elif isinstance(paths, list) and paths and isinstance(paths[0], dict):
                # 하위 dict에서 path 키 추출
                extracted = [item.get("path") for item in paths if isinstance(item, dict) and item.get("path")]
                category_to_lists[cat].append(normalize_list(extracted))
        # 각 그룹별 파일 존재 여부 필터링 유지
        return {k: [
            _ensure_existing(group)
        ] if isinstance(group, list) else [] for k, group in category_to_lists.items()}  # type: ignore[dict-item]

    if isinstance(data, list):
        # 리스트 항목 형태별 처리
        for item in data:
            if not isinstance(item, dict):
                continue
            cat = item.get("category") or item.get("label") or item.get("class")

            # 1) images/paths/frames 키 우선
            for key in ("images", "paths", "frames"):
                if key in item and isinstance(item[key], list) and (not item[key] or isinstance(item[key][0], str)):
                    if cat:
                        category_to_lists[cat].append(normalize_list(item[key]))
                    break
            else:
                # 4) items: [{path: ...}, ...]
                if "items" in item and isinstance(item["items"], list):
                    extracted = [sub.get("path") for sub in item["items"] if isinstance(sub, dict) and sub.get("path")]
                    if extracted and cat:
                        category_to_lists[cat].append(normalize_list(extracted))
                    continue

                # 5) 단일 이미지 레코드
                if "path" in item and isinstance(item["path"], str) and cat:
                    category_to_lists[cat].append(normalize_list([item["path"]]))
        # 그룹별 존재 파일 필터링(빈 그룹 제거)
        cleaned: Dict[str, List[List[str]]] = {}
        for k, groups in category_to_lists.items():
            filtered_groups = []
            for group in groups:
                existing = _ensure_existing(group)
                if existing:
                    filtered_groups.append(existing)
            if filtered_groups:
                cleaned[k] = filtered_groups
        return cleaned

    raise ValueError("지원되지 않는 JSON 구조입니다. dict 또는 list 형태여야 합니다.")


def make_videos(
    json_path: str,
    output_dir: str,
    base_image_root: Optional[str] = None,
    fps: int = 25,
    max_frames_per_video: int = 1000,
    fourcc: str = "XVID",
    resize_width: Optional[int] = None,
    resize_height: Optional[int] = None,
    crop_x: Optional[int] = None,
    crop_y: Optional[int] = None,
    crop_w: Optional[int] = None,
    crop_h: Optional[int] = None,
) -> Dict[str, List[str]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    category_to_images = _extract_category_to_images(data, base_image_root)

    # 프레임 크기 강제 지정 여부
    resize_to = None
    if resize_width is not None and resize_height is not None:
        resize_to = (resize_width, resize_height)

    crop_rect = None
    if (
        crop_x is not None and crop_y is not None and crop_w is not None and crop_h is not None
    ):
        crop_rect = (int(crop_x), int(crop_y), int(crop_w), int(crop_h))

    results: Dict[str, List[str]] = {}
    for category, groups in category_to_images.items():
        if not groups:
            continue
        current_index = 0
        written_all: List[str] = []
        for group in groups:
            written, current_index = _write_videos_for_category(
                category=category,
                image_paths=group,
                output_dir=output_dir,
                fps=fps,
                chunk_size=max_frames_per_video,
                fourcc_str=fourcc,
                resize_to=resize_to,
                start_index=current_index,
                crop_rect=crop_rect,
            )
            written_all.extend(written)
        results[category] = written_all

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="image_segments.json 기반 카테고리별 영상 생성기")
    parser.add_argument("--json", required=True, help="image_segments.json 경로")
    parser.add_argument("--out", required=True, help="출력 비디오 디렉터리")
    parser.add_argument("--base", default=None, help="이미지 루트 디렉터리 (상대경로를 이 루트에 결합)")
    parser.add_argument("--fps", type=int, default=25, help="프레임레이트")
    parser.add_argument("--chunk", type=int, default=1000, help="비디오당 최대 프레임 수 (청크 분할)")
    parser.add_argument("--fourcc", default="XVID", help="코덱 FourCC (예: XVID, mp4v)")
    parser.add_argument("--width", type=int, default=None, help="강제 리사이즈 가로")
    parser.add_argument("--height", type=int, default=None, help="강제 리사이즈 세로")
    parser.add_argument("--crop_x", type=int, default=None, help="크롭 시작 x (좌상단 기준)")
    parser.add_argument("--crop_y", type=int, default=None, help="크롭 시작 y (좌상단 기준)")
    parser.add_argument("--crop_w", type=int, default=None, help="크롭 너비")
    parser.add_argument("--crop_h", type=int, default=None, help="크롭 높이")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    outputs = make_videos(
        json_path=args.json,
        output_dir=args.out,
        base_image_root=args.base,
        fps=args.fps,
        max_frames_per_video=args.chunk,
        fourcc=args.fourcc,
        resize_width=args.width,
        resize_height=args.height,
        crop_x=args.crop_x,
        crop_y=args.crop_y,
        crop_w=args.crop_w,
        crop_h=args.crop_h,
    )
    for cat, files in outputs.items():
        print(f"[{cat}] 생성 파일:")
        for fp in files:
            print(f"  - {fp}")


