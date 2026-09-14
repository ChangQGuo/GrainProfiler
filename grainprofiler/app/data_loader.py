"""Read and index all pipeline result files from a directory."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from grainprofiler.app.models import (
    AxisData,
    ContourData,
    LoadResult,
    YoloData,
    YoloDetection,
)

_KERNEL_NAME_RE = re.compile(r"^(.+)_kernel_(\d+)(\.jpg)?$")


class DataLoader:
    """Loads and indexes a pipeline output directory.

    Public read-only attributes populated after a successful load() call.
    """

    def __init__(self) -> None:
        self.result_dir: str = ""
        self.image_names: list[str] = []
        self.metadata: pd.DataFrame = pd.DataFrame()
        self.detect_results: pd.DataFrame = pd.DataFrame()
        self.measurements: pd.DataFrame = pd.DataFrame()
        self.final_individual: pd.DataFrame = pd.DataFrame()
        self.plant_medians: pd.DataFrame = pd.DataFrame()
        self.contours: dict[str, ContourData] = {}
        self._contour_files: dict[str, Path] = {}     # image_stem → per-image JSON path
        self._contour_cache: dict[str, dict[str, list[list[float]]]] = {}  # LRU cache
        self._contour_cache_order: list[str] = []
        self._contour_cache_max: int = 5
        self.axis_results: dict[str, AxisData] = {}
        self.bounding_boxes: dict[str, YoloData] = {}
        self.kernel_to_image: dict[str, str] = {}
        self.median_profiles: dict[str, list[float]] = {}  # plant_name → 100 width values
        self.kernel_width_profiles: dict[str, list[float]] = {}  # kernel_name → 100 width values
        self.image_to_plant: dict[str, str] = {}           # image_name → plant_name
        self.latent_traits: pd.DataFrame = pd.DataFrame()   # plant_id + latent_1..latent_5

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, result_dir: str) -> LoadResult:
        """Read all files from *result_dir*.

        Returns a LoadResult.  On ``ok=True`` the DataLoader attributes
        are fully populated; on ``ok=False`` the caller should inspect
        ``errors`` and not proceed.
        """
        self.result_dir = result_dir
        errors: list[str] = []
        warnings: list[str] = []

        rp = Path(result_dir)

        # --- essential files ---
        meas = self._read_parquet_or_csv(rp, "measurements")
        if meas is None:
            errors.append("measurements.csv (or .parquet) missing or unreadable")
            return LoadResult(ok=False, errors=errors)
        self.measurements = meas

        meta = self._read_csv(rp / "metadata.csv")
        if meta is not None:
            self.metadata = meta
        else:
            warnings.append("metadata.csv not found")

        det = self._read_csv(rp / "detect_results.csv")
        if det is not None:
            self.detect_results = det
        else:
            warnings.append("detect_results.csv not found")

        final = self._read_csv(rp / "final_output_individual.csv")
        if final is not None:
            self.final_individual = final

        pm = self._read_csv(rp / "final_output_plant_median.csv")
        if pm is not None:
            self.plant_medians = pm

        lt = self._read_csv(rp / "latent_traits.csv")
        if lt is not None:
            self.latent_traits = lt

        # --- optional JSON files ---
        contours_dir = rp / "contours"
        if contours_dir.is_dir():
            self._index_contour_files(contours_dir)
        else:
            # Fallback: single legacy kernel_contours.json
            contours_raw = self._read_json(rp / "kernel_contours.json")
            if contours_raw is not None:
                self._index_contours_legacy(contours_raw)
            else:
                warnings.append("contours/ not found — contour overlays disabled")

        axis_raw = self._read_json(rp / "axis_results.json")
        if axis_raw is not None:
            self._index_axis(axis_raw)
        else:
            warnings.append("axis_results.json not found — axis overlays disabled")

        bbox_raw = self._read_json(rp / "yolo_bounding_boxes.json")
        if bbox_raw is not None:
            self._index_bboxes(bbox_raw)
        else:
            warnings.append("yolo_bounding_boxes.json not found — overlay positioning degraded")

        # --- median width profiles (optional) ---
        profiles_raw = self._read_txt(rp / "rep_width_profiles.txt")
        if profiles_raw is not None:
            self._index_median_profiles(profiles_raw)

        # --- per-kernel width profiles (optional, for interactive median chart) ---
        kernel_profiles_raw = self._read_txt(rp / "kernel_width_profiles.txt")
        if kernel_profiles_raw is not None:
            self._index_kernel_width_profiles(kernel_profiles_raw)

        # --- derive image list ---
        self._build_image_list()
        self._build_kernel_to_image_map()
        self._build_image_to_plant_map()

        total = len(self.measurements)
        return LoadResult(
            ok=True,
            errors=errors,
            warnings=warnings,
            image_names=self.image_names,
            total_kernels=total,
        )

    def get_sample_ids(self) -> list[str]:
        """Ordered list of image names."""
        return self.image_names

    def get_kernels_for_sample(self, image_name: str) -> list[str]:
        """Return kernel_names belonging to *image_name*."""
        prefix = Path(image_name).stem
        return [k for k in self.kernel_to_image if self.kernel_to_image[k] == image_name]

    def get_kernel_contour(self, kernel_name: str) -> ContourData | None:
        # Fast path: already in legacy full-load dict
        if kernel_name in self.contours:
            return self.contours[kernel_name]
        # Lazy path: look up image_stem, load per-image file if needed
        image_name = self.kernel_to_image.get(kernel_name)
        if image_name is None:
            return None
        stem = Path(image_name).stem
        file_path = self._contour_files.get(stem)
        if file_path is None:
            return None
        # Load into LRU cache
        if stem not in self._contour_cache:
            try:
                with open(file_path, encoding='utf-8') as f:
                    raw = json.load(f)
            except Exception:
                return None
            self._contour_cache[stem] = raw
            self._contour_cache_order.append(stem)
            # Evict oldest
            while len(self._contour_cache_order) > self._contour_cache_max:
                old = self._contour_cache_order.pop(0)
                del self._contour_cache[old]
        # Populate ContourData
        pts = self._contour_cache[stem].get(kernel_name)
        if pts is None:
            return None
        cd = ContourData(kernel_name=kernel_name, contour=pts)
        self.contours[kernel_name] = cd
        return cd

    def get_kernel_axis(self, kernel_name: str) -> AxisData | None:
        return self.axis_results.get(kernel_name)

    def get_kernel_box(self, kernel_name: str) -> tuple[int, int, int, int] | None:
        """Return the YOLO detection box [x1,y1,x2,y2] for *kernel_name* in full-image coords."""
        image_name = self.kernel_to_image.get(kernel_name)
        if image_name is None:
            return None
        yolo = self.bounding_boxes.get(image_name)
        if yolo is None:
            return None

        m = _KERNEL_NAME_RE.match(Path(kernel_name).stem)
        if m is None:
            return None
        kid = int(m.group(2))
        det = yolo.accepted_boxes.get(kid)
        if det is None:
            return None
        return det.box

    def get_kernel_measurement(self, kernel_name: str) -> pd.Series | None:
        df = self.measurements
        if df.empty:
            return None
        df = df.set_index("kernel_name", drop=False)
        try:
            return df.loc[kernel_name]
        except KeyError:
            return None

    def get_image_metadata(self, image_name: str) -> pd.Series | None:
        df = self.metadata
        if df.empty:
            return None
        df = df.set_index("image_name", drop=False)
        try:
            return df.loc[image_name]
        except KeyError:
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_parquet_or_csv(base: Path, stem: str) -> pd.DataFrame | None:
        pqt = base / f"{stem}.parquet"
        if pqt.exists():
            try:
                return pd.read_parquet(pqt)
            except Exception:
                pass
        csv = base / f"{stem}.csv"
        try:
            return pd.read_csv(csv)
        except Exception:
            return None

    @staticmethod
    def _read_csv(path: Path) -> pd.DataFrame | None:
        try:
            return pd.read_csv(path)
        except Exception:
            return None

    @staticmethod
    def _read_json(path: Path) -> Any | None:
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return None

    @staticmethod
    def _read_txt(path: Path) -> list[str] | None:
        try:
            with open(path, encoding="utf-8") as fh:
                return [line.strip() for line in fh if line.strip()]
        except Exception:
            return None

    def _index_median_profiles(self, lines: list[str]) -> None:
        for line in lines:
            parts = line.split()
            if len(parts) >= 20:
                plant = parts[0]
                try:
                    values = [float(v) for v in parts[1:]]
                except ValueError:
                    continue  # skip malformed lines instead of aborting the load
                self.median_profiles[plant] = values

    def _index_kernel_width_profiles(self, lines: list[str]) -> None:
        for line in lines:
            parts = line.split()
            if len(parts) >= 20:
                kernel = parts[0]
                try:
                    values = [float(v) for v in parts[1:]]
                except ValueError:
                    continue  # skip malformed lines instead of aborting the load
                self.kernel_width_profiles[kernel] = values

    def get_kernel_width_profiles_for_image(self, image_name: str) -> dict[str, list[float]]:
        """Return {kernel_name: [100 width values]} for all kernels in *image_name*."""
        result = {}
        prefix = Path(image_name).stem + "_kernel_"
        for kn in self.kernel_width_profiles:
            if kn.startswith(prefix):
                result[kn] = self.kernel_width_profiles[kn]
        return result

    def _index_contours_legacy(self, raw: list[dict]) -> None:
        """Load ALL contours from legacy single kernel_contours.json (eager)."""
        for item in raw:
            kn = item.get("kernel_name", "")
            pts = item.get("contour", [])
            self.contours[kn] = ContourData(kernel_name=kn, contour=pts)

    def _index_contour_files(self, contours_dir: Path) -> None:
        """Scan per-image contour files — lazily loaded on demand."""
        for f in contours_dir.glob("*_contours.json"):
            # filename format: <image_name>.jpg_contours.json
            stem = f.name.replace("_contours.json", "").replace(".jpg", "")
            self._contour_files[stem] = f

    def _index_axis(self, raw: list[dict]) -> None:
        for item in raw:
            try:
                kn = item.get("kernel_name", "")
                self.axis_results[kn] = AxisData(
                    kernel_name=kn,
                    bottom=tuple(item.get("bottom", [0.0, 0.0])),
                    top=tuple(item.get("top", [0.0, 0.0])),
                    shape_label=item.get("shape_label", ""),
                    axis_length=float(item.get("axis_length") or 0),
                    max_width=float(item.get("max_width") or 0),
                    area=float(item.get("area") or 0),
                    perimeter=float(item.get("perimeter") or 0),
                    circularity=float(item.get("circularity") or 0),
                )
            except (TypeError, ValueError):
                continue

    def _index_bboxes(self, raw: dict) -> None:
        for image_name, data in raw.items():
            dets: dict[int, YoloDetection] = {}
            accepted = data.get("accepted_detections") or data.get("detections") or []
            for d in accepted:
                try:
                    kid = int(d.get("kernel_id") or 0)
                    box = d.get("box", [0, 0, 0, 0])
                    conf = float(d.get("confidence") or 0.0)
                    dets[kid] = YoloDetection(
                        kernel_id=kid,
                        box=tuple(int(v) for v in box),
                        confidence=conf,
                    )
                except (TypeError, ValueError):
                    continue
            img_w = 0
            img_h = 0
            img_size = data.get("image_size")
            if img_size:
                try:
                    img_w = int(img_size.get("width") or 0)
                    img_h = int(img_size.get("height") or 0)
                except (TypeError, ValueError):
                    pass
            self.bounding_boxes[image_name] = YoloData(
                image_name=image_name,
                image_size=(img_w, img_h),
                accepted_boxes=dets,
            )

    def _build_image_list(self) -> None:
        """Derive the ordered list of unique image names from measurements."""
        seen: set[str] = set()
        for kn in self.measurements.get("kernel_name", []):
            m = _KERNEL_NAME_RE.match(Path(str(kn)).stem)
            if m:
                image_name = m.group(1) + ".jpg"
                if image_name not in seen:
                    seen.add(image_name)
                    self.image_names.append(image_name)

        # Also pull from metadata if available
        if not self.image_names and not self.metadata.empty:
            self.image_names = list(self.metadata["image_name"].astype(str))

    def _build_kernel_to_image_map(self) -> None:
        """Map kernel_name -> image_name for fast lookup."""
        for kn in self.measurements.get("kernel_name", []):
            m = _KERNEL_NAME_RE.match(Path(str(kn)).stem)
            if m:
                self.kernel_to_image[str(kn)] = m.group(1) + ".jpg"

    def _build_image_to_plant_map(self) -> None:
        """Map image_name -> plant_name from metadata."""
        if self.metadata.empty:
            return
        for _, row in self.metadata.iterrows():
            img = str(row.get("image_name", ""))
            plant = str(row.get("plant_name", ""))
            if img and plant:
                self.image_to_plant[img] = plant

    def get_median_profile(self, image_name: str) -> list[float] | None:
        """Return the 100-point median width profile for an image's plant."""
        plant = self.image_to_plant.get(image_name, "")
        if not plant:
            # Try fuzzy match
            stem = Path(image_name).stem
            for p in self.median_profiles:
                if stem in p or p in stem:
                    return self.median_profiles[p]
            return None
        return self.median_profiles.get(plant)
