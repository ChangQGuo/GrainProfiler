"""Locate tray photos from result directories and siblings."""

from pathlib import Path

_CANDIDATES = [
    "test_image_min", "test_image", "test_image_best",
    "image_data", "sample_image", "test_image_results",
]


def find_photo(image_name: str, result_dir: str) -> str | None:
    """Find the best tray photo for *image_name*.

    Priority:
      1. measurement_overlay inside result_dir (has pre-drawn axes, ideal base)
      2. Original in result_dir
      3. Sibling directories (test_image_min, etc.)
      4. Project-root-level directories
      5. Any .jpg deep search
    """
    result_path = Path(result_dir).resolve()

    # 1. measurement_overlay — preferred base
    overlay = result_path / "measurement_overlay" / f"measurement_overlay_{image_name}"
    if overlay.exists():
        return str(overlay)

    # 2. Original directly in result_dir
    direct = result_path / image_name
    if direct.exists():
        return str(direct)

    # 3. Sibling directories
    parent = result_path.parent
    for cand in _CANDIDATES:
        p = parent / cand / image_name
        if p.exists():
            return str(p)

    # 4. Grandparent level
    grandparent = parent.parent
    if grandparent != parent:
        for cand in _CANDIDATES:
            p = grandparent / cand / image_name
            if p.exists():
                return str(p)

    # 5. Deep search
    for p in parent.rglob(image_name):
        return str(p)

    return None
