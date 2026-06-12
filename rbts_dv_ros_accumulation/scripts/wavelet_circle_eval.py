#!/usr/bin/env python3
"""Offline wavelet-vs-baseline Hough circle evaluation for recorded event frames.

The tool reads `/motion_compensator/image` frames from a debug rosbag and runs
two Hough pipelines:

1. Baseline preprocessing that mirrors `hole_detector.cpp`.
2. Haar wavelet shrinkage denoising inspired by event-frame preprocessing papers.

It writes side-by-side montages and summary statistics so the wavelet step can
be evaluated before adding latency to the live C++ detector.
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
PRESET_DIR = REPO_ROOT / "flap_roller_viz" / "launch" / "presets"
_PYWT = None


@dataclass
class DetectorParams:
    detector_mode: str = "hough_preproc"
    roi_top: int = 50
    roi_bottom: int = 590
    roi_left: int = 0
    roi_right: int = 0
    min_radius: int = 95
    max_radius: int = 120
    dp: float = 2.0
    min_dist: float = -1.0
    param1: int = 95
    param2: int = 40
    bilateral_enable: bool = False
    bilateral_d: int = 5
    bilateral_sigma_color: float = 50.0
    bilateral_sigma_space: float = 50.0
    clahe_enable: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_x: int = 8
    clahe_tile_grid_y: int = 8
    threshold_mode: str = "otsu"
    threshold_value: int = 80
    morph_open_enable: bool = False
    morph_open_kernel: int = 3
    morph_close_enable: bool = True
    morph_close_kernel: int = 3

    @classmethod
    def from_mapping(cls, values: Dict[str, object]) -> "DetectorParams":
        params = cls()
        key_map = {
            "minRadius": "min_radius",
            "maxRadius": "max_radius",
            "minDist": "min_dist",
        }
        for key, value in values.items():
            attr = key_map.get(key, key)
            if not hasattr(params, attr):
                continue
            current = getattr(params, attr)
            if isinstance(current, bool):
                setattr(params, attr, bool(value))
            elif isinstance(current, int) and not isinstance(current, bool):
                setattr(params, attr, int(value))
            elif isinstance(current, float):
                setattr(params, attr, float(value))
            else:
                setattr(params, attr, str(value))
        params.normalize()
        return params

    def normalize(self) -> None:
        if self.roi_bottom < self.roi_top:
            self.roi_top, self.roi_bottom = self.roi_bottom, self.roi_top
        self.min_radius = max(1, self.min_radius)
        self.max_radius = max(self.min_radius, self.max_radius)
        self.dp = max(0.1, self.dp)
        self.param1 = max(1, self.param1)
        self.param2 = max(1, self.param2)
        self.bilateral_d = make_odd_at_least(self.bilateral_d, 1)
        self.clahe_tile_grid_x = max(1, self.clahe_tile_grid_x)
        self.clahe_tile_grid_y = max(1, self.clahe_tile_grid_y)
        self.threshold_value = int(np.clip(self.threshold_value, 0, 255))
        self.morph_open_kernel = make_odd_at_least(self.morph_open_kernel, 1)
        self.morph_close_kernel = make_odd_at_least(self.morph_close_kernel, 1)


@dataclass
class PipelineStats:
    name: str
    total_frames: int = 0
    detected_frames: int = 0
    total_circles: int = 0
    selected_radii: Optional[List[float]] = None
    selected_centers_y: Optional[List[float]] = None

    def __post_init__(self) -> None:
        self.selected_radii = []
        self.selected_centers_y = []

    def update(self, circles: np.ndarray, image_center_x: float) -> None:
        self.total_frames += 1
        self.total_circles += int(len(circles))
        if len(circles) == 0:
            return
        self.detected_frames += 1
        selected = min(circles, key=lambda c: abs(float(c[0]) - image_center_x))
        self.selected_radii.append(float(selected[2]))
        self.selected_centers_y.append(float(selected[1]))

    def summary(self) -> Dict[str, float]:
        radii = np.asarray(self.selected_radii or [], dtype=np.float64)
        centers_y = np.asarray(self.selected_centers_y or [], dtype=np.float64)
        return {
            "frames": float(self.total_frames),
            "detected_frames": float(self.detected_frames),
            "detection_rate": safe_div(self.detected_frames, self.total_frames),
            "mean_circles_per_frame": safe_div(self.total_circles, self.total_frames),
            "mean_selected_radius_px": float(radii.mean()) if radii.size else 0.0,
            "selected_radius_std_px": float(radii.std()) if radii.size else 0.0,
            "mean_selected_center_y_px": float(centers_y.mean()) if centers_y.size else 0.0,
            "selected_center_y_jitter_px": float(centers_y.std()) if centers_y.size else 0.0,
        }


def make_odd_at_least(value: int, minimum: int) -> int:
    value = max(int(value), minimum)
    return value + 1 if value % 2 == 0 else value


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def get_pywt():
    global _PYWT
    if _PYWT is None:
        try:
            import pywt  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised on user machine
            raise SystemExit(
                "PyWavelets is required. Install it with: pip install PyWavelets"
            ) from exc
        _PYWT = pywt
    return _PYWT


def parse_scalar(raw: str) -> object:
    value = raw.strip()
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("\"'")


def load_simple_yaml(path: Path) -> Dict[str, object]:
    values: Dict[str, object] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = parse_scalar(value)
    return values


def load_preset(preset: str) -> Tuple[DetectorParams, Path]:
    candidate_paths = [Path(preset).expanduser(), PRESET_DIR / f"{preset}.yaml"]
    try:
        import rospkg  # type: ignore

        flap_path = Path(rospkg.RosPack().get_path("flap_roller_viz"))
        candidate_paths.append(flap_path / "launch" / "presets" / f"{preset}.yaml")
    except Exception:
        pass

    preset_path = next((path for path in candidate_paths if path.exists()), None)
    if preset_path is None:
        candidates = ", ".join(str(path) for path in candidate_paths)
        raise FileNotFoundError(f"Could not find preset '{preset}'. Checked: {candidates}")

    try:
        import yaml  # type: ignore

        with preset_path.open("r") as handle:
            raw = yaml.safe_load(handle) or {}
    except ImportError:
        raw = load_simple_yaml(preset_path)

    if not isinstance(raw, dict):
        raise ValueError(f"Preset {preset_path} did not contain a mapping")
    return DetectorParams.from_mapping(raw), preset_path


def effective_roi(gray: np.ndarray, params: DetectorParams) -> Tuple[int, int, int, int]:
    rows, cols = gray.shape[:2]
    top = int(np.clip(params.roi_top, 0, rows))
    bottom = int(np.clip(params.roi_bottom, 0, rows))
    if bottom <= top:
        top, bottom = 0, rows

    left = int(np.clip(params.roi_left, 0, cols)) if params.roi_left > 0 else 0
    right = int(np.clip(params.roi_right, 0, cols)) if params.roi_right > 0 else cols
    if right <= left:
        left, right = 0, cols
    return left, top, right - left, bottom - top


def paste_roi(shape: Tuple[int, int], roi: Tuple[int, int, int, int], roi_image: np.ndarray) -> np.ndarray:
    full = np.zeros(shape, dtype=np.uint8)
    x, y, w, h = roi
    if w > 0 and h > 0:
        full[y : y + h, x : x + w] = roi_image
    return full


def preprocess_baseline(
    gray: np.ndarray, params: DetectorParams
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    roi = effective_roi(gray, params)
    x, y, w, h = roi
    work = gray[y : y + h, x : x + w].copy()

    if params.bilateral_enable:
        work = cv2.bilateralFilter(
            work,
            params.bilateral_d,
            params.bilateral_sigma_color,
            params.bilateral_sigma_space,
        )
    if params.clahe_enable:
        clahe = cv2.createCLAHE(
            clipLimit=params.clahe_clip_limit,
            tileGridSize=(params.clahe_tile_grid_x, params.clahe_tile_grid_y),
        )
        work = clahe.apply(work)

    if params.threshold_mode == "otsu":
        _, cleaned = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    elif params.threshold_mode == "fixed":
        _, cleaned = cv2.threshold(work, params.threshold_value, 255, cv2.THRESH_BINARY)
    else:
        cleaned = work.copy()

    if params.morph_open_enable:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (params.morph_open_kernel, params.morph_open_kernel)
        )
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    if params.morph_close_enable:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (params.morph_close_kernel, params.morph_close_kernel)
        )
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    return cleaned, paste_roi(gray.shape, roi, cleaned), roi


def normalize_to_unit(work: np.ndarray) -> np.ndarray:
    as_float = work.astype(np.float32)
    min_value = float(as_float.min()) if as_float.size else 0.0
    max_value = float(as_float.max()) if as_float.size else 0.0
    if max_value <= min_value:
        return np.zeros_like(as_float, dtype=np.float32)
    return (as_float - min_value) / (max_value - min_value)


def estimate_sigma_from_hh(coeffs: List[object]) -> float:
    if len(coeffs) < 2:
        return 0.0
    finest_details = coeffs[-1]
    if not isinstance(finest_details, tuple) or len(finest_details) != 3:
        return 0.0
    hh = np.asarray(finest_details[2], dtype=np.float32)
    if hh.size == 0:
        return 0.0
    return float(np.median(np.abs(hh))) / 0.6745


def preprocess_wavelet(
    gray: np.ndarray,
    params: DetectorParams,
    wavelet_name: str,
    levels: int,
    low_energy_frac: float,
    threshold_mode: str,
    threshold_scale: float,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    pywt = get_pywt()
    roi = effective_roi(gray, params)
    x, y, w, h = roi
    unit = normalize_to_unit(gray[y : y + h, x : x + w])
    if low_energy_frac > 0.0:
        unit = unit.copy()
        unit[unit < low_energy_frac] = 0.0

    max_level = pywt.dwtn_max_level(unit.shape, wavelet_name)
    use_level = max(1, min(int(levels), max_level if max_level > 0 else 1))
    coeffs = pywt.wavedec2(unit, wavelet=wavelet_name, level=use_level, mode="periodization")

    sigma = estimate_sigma_from_hh(coeffs)
    threshold = threshold_scale * sigma * math.sqrt(2.0 * math.log(max(unit.size, 2)))

    if threshold_mode != "none" and threshold > 0.0:
        denoised_coeffs: List[object] = [coeffs[0]]
        for detail in coeffs[1:]:
            denoised_coeffs.append(
                tuple(pywt.threshold(band, threshold, mode=threshold_mode) for band in detail)
            )
        coeffs = denoised_coeffs

    reconstructed = pywt.waverec2(coeffs, wavelet=wavelet_name, mode="periodization")
    reconstructed = reconstructed[:h, :w]
    wavelet_roi = np.uint8(np.round(np.clip(reconstructed, 0.0, 1.0) * 255.0))
    return wavelet_roi, paste_roi(gray.shape, roi, wavelet_roi), roi


def hough_circles(
    roi_image: np.ndarray, roi: Tuple[int, int, int, int], params: DetectorParams
) -> np.ndarray:
    x, y, _w, h = roi
    if roi_image.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    min_dist = params.min_dist if params.min_dist > 0.0 else float(h)
    circles = cv2.HoughCircles(
        roi_image,
        cv2.HOUGH_GRADIENT,
        params.dp,
        min_dist,
        param1=params.param1,
        param2=params.param2,
        minRadius=params.min_radius,
        maxRadius=params.max_radius,
    )
    if circles is None:
        return np.empty((0, 3), dtype=np.float32)
    circles = np.asarray(circles, dtype=np.float32).reshape(-1, 3)
    circles[:, 0] += float(x)
    circles[:, 1] += float(y)
    return circles


def to_bgr(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def draw_overlay(
    source: np.ndarray,
    circles: np.ndarray,
    roi: Tuple[int, int, int, int],
    title: str,
    frame_index: int,
) -> np.ndarray:
    out = to_bgr(source)
    x, y, w, h = roi
    cv2.rectangle(out, (x, y), (x + w - 1, y + h - 1), (255, 255, 0), 1)
    for circle in circles:
        cx, cy, radius = circle
        center = (int(round(cx)), int(round(cy)))
        cv2.circle(out, center, int(round(radius)), (0, 255, 0), 2)
        cv2.circle(out, center, 2, (0, 0, 255), 2)
    label = f"{title} | frame {frame_index} | circles={len(circles)}"
    cv2.putText(out, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(out, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    return out


def make_montage(
    gray: np.ndarray,
    baseline_full: np.ndarray,
    wavelet_full: np.ndarray,
    baseline_circles: np.ndarray,
    wavelet_circles: np.ndarray,
    roi: Tuple[int, int, int, int],
    frame_index: int,
) -> np.ndarray:
    raw = draw_overlay(gray, np.empty((0, 3), dtype=np.float32), roi, "raw", frame_index)
    baseline = draw_overlay(baseline_full, baseline_circles, roi, "baseline", frame_index)
    wavelet = draw_overlay(wavelet_full, wavelet_circles, roi, "wavelet", frame_index)
    return cv2.hconcat([raw, baseline, wavelet])


def decode_image(bridge: object, msg: object) -> np.ndarray:
    try:
        image = bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
    except Exception:
        if not all(hasattr(msg, attr) for attr in ("height", "width", "step", "data")):
            raise
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        image = raw.reshape(int(msg.height), int(msg.step))[:, : int(msg.width)]
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.ascontiguousarray(image, dtype=np.uint8)


def iter_bag_images(
    bag_path: Path, image_topic: str, frame_stride: int
) -> Iterable[Tuple[int, object]]:
    try:
        import rosbag  # type: ignore
        from cv_bridge import CvBridge  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised on user machine
        raise SystemExit(
            "Run this script from a sourced ROS environment so rosbag and cv_bridge are available."
        ) from exc

    bridge = CvBridge()
    with rosbag.Bag(str(bag_path), "r") as bag:
        raw_index = -1
        for _topic, msg, _stamp in bag.read_messages(topics=[image_topic]):
            raw_index += 1
            if raw_index % frame_stride != 0:
                continue
            yield raw_index, decode_image(bridge, msg)


def print_summary(summary: Dict[str, Dict[str, float]]) -> None:
    print("\nSummary")
    print("-------")
    for name, values in summary.items():
        print(
            f"{name}: frames={int(values['frames'])} "
            f"detected={int(values['detected_frames'])} "
            f"rate={values['detection_rate']:.3f} "
            f"mean_circles/frame={values['mean_circles_per_frame']:.3f} "
            f"radius_std={values['selected_radius_std_px']:.3f}px "
            f"center_y_jitter={values['selected_center_y_jitter_px']:.3f}px"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare baseline and Haar-wavelet Hough circle detection on a debug rosbag.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--bag", required=True, type=Path, help="Input rosbag containing event frames.")
    parser.add_argument("--image-topic", default="/motion_compensator/image")
    parser.add_argument("--preset", default="balanced", help="Preset name or YAML path.")
    parser.add_argument("--out", default="wavelet_eval_out", type=Path, help="Output directory.")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means process all frames.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Process every Nth frame.")
    parser.add_argument("--no-images", action="store_true", help="Do not write montage PNGs.")
    parser.add_argument("--video", action="store_true", help="Write an MP4 montage video.")
    parser.add_argument("--video-fps", type=float, default=20.0)

    parser.add_argument("--wavelet", default="haar", help="PyWavelets wavelet name; haar/db1 is fastest.")
    parser.add_argument("--levels", type=int, default=1, help="Wavelet decomposition levels.")
    parser.add_argument("--low-energy-frac", type=float, default=0.3, help="Zero normalized pixels below this.")
    parser.add_argument(
        "--threshold-mode",
        choices=("soft", "hard", "none"),
        default="soft",
        help="Wavelet detail coefficient thresholding mode.",
    )
    parser.add_argument("--threshold-scale", type=float, default=1.0, help="Scale applied to VisuShrink threshold.")

    parser.add_argument("--dp", type=float, help="Override Hough dp.")
    parser.add_argument("--param1", type=int, help="Override Hough param1.")
    parser.add_argument("--param2", type=int, help="Override Hough param2.")
    parser.add_argument("--min-radius", type=int, help="Override Hough min radius.")
    parser.add_argument("--max-radius", type=int, help="Override Hough max radius.")
    return parser


def apply_overrides(params: DetectorParams, args: argparse.Namespace) -> None:
    for arg_name, attr in (
        ("dp", "dp"),
        ("param1", "param1"),
        ("param2", "param2"),
        ("min_radius", "min_radius"),
        ("max_radius", "max_radius"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            setattr(params, attr, value)
    params.normalize()


def main() -> int:
    args = build_arg_parser().parse_args()
    args.bag = args.bag.expanduser()
    if not args.bag.exists():
        raise SystemExit(f"Bag does not exist: {args.bag}")
    if args.frame_stride < 1:
        raise SystemExit("--frame-stride must be >= 1")

    params, preset_path = load_preset(args.preset)
    apply_overrides(params, args)

    output_dir = args.out.expanduser()
    frames_dir = output_dir / "frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_images:
        frames_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded preset: {preset_path}")
    print(f"Output directory: {output_dir}")
    print(
        "Hough params: "
        f"roi=({params.roi_left},{params.roi_top})-({params.roi_right},{params.roi_bottom}) "
        f"r=[{params.min_radius},{params.max_radius}] dp={params.dp} "
        f"p1={params.param1} p2={params.param2}"
    )
    print(
        "Wavelet params: "
        f"wavelet={args.wavelet} levels={args.levels} "
        f"low_energy_frac={args.low_energy_frac} "
        f"threshold={args.threshold_mode} scale={args.threshold_scale}"
    )

    baseline_stats = PipelineStats("baseline")
    wavelet_stats = PipelineStats("wavelet")
    video_writer = None
    processed = 0

    try:
        for raw_index, gray in iter_bag_images(args.bag, args.image_topic, args.frame_stride):
            baseline_roi, baseline_full, roi = preprocess_baseline(gray, params)
            baseline_circles = hough_circles(baseline_roi, roi, params)

            wavelet_roi, wavelet_full, _ = preprocess_wavelet(
                gray,
                params,
                args.wavelet,
                args.levels,
                args.low_energy_frac,
                args.threshold_mode,
                args.threshold_scale,
            )
            wavelet_circles = hough_circles(wavelet_roi, roi, params)

            image_center_x = float(gray.shape[1]) / 2.0
            baseline_stats.update(baseline_circles, image_center_x)
            wavelet_stats.update(wavelet_circles, image_center_x)

            montage = make_montage(
                gray,
                baseline_full,
                wavelet_full,
                baseline_circles,
                wavelet_circles,
                roi,
                raw_index,
            )
            if not args.no_images:
                cv2.imwrite(str(frames_dir / f"frame_{raw_index:06d}.png"), montage)
            if args.video:
                if video_writer is None:
                    height, width = montage.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(
                        str(output_dir / "wavelet_hough_eval.mp4"),
                        fourcc,
                        args.video_fps,
                        (width, height),
                    )
                video_writer.write(montage)

            processed += 1
            if processed % 100 == 0:
                print(f"Processed {processed} frames...")
            if args.max_frames > 0 and processed >= args.max_frames:
                break
    finally:
        if video_writer is not None:
            video_writer.release()

    if processed == 0:
        raise SystemExit(
            f"No frames found on topic {args.image_topic}. "
            "Record with bag_mode:=debug or pass the topic that contains sensor_msgs/Image frames."
        )

    summary = {
        "baseline": baseline_stats.summary(),
        "wavelet": wavelet_stats.summary(),
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print_summary(summary)
    print(f"\nWrote summary: {output_dir / 'summary.json'}")
    if not args.no_images:
        print(f"Wrote montage frames: {frames_dir}")
    if args.video:
        print(f"Wrote video: {output_dir / 'wavelet_hough_eval.mp4'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
