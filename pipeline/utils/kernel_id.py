"""
kernel_id.py — canonical maize-kernel identity parsing.

The subimage naming convention ``{image_stem}_kernel_{id:03d}.jpg`` is the single
source of truth that links every kernel back to its tray image (and therefore to
OCR metadata, plant name, and physical calibration).  Both the pipeline and the
GUI must recover this identity consistently, so the logic lives here and every
other module delegates to it instead of re-implementing the same regex/rsplit.

The two public helpers have slightly different semantics and are kept faithful
to their historical behaviour:

  * ``parse_kernel_id_from_name``  — extract the integer kernel id (basename only).
  * ``get_original_image_name``    — recover the tray-image filename (keeps path
    and extension, returns the input unchanged when the name does not match).
"""

import os
import re

# Matches the trailing ``_kernel_<digits>`` of a subimage stem.
_KERNEL_ID_RE = re.compile(r"_kernel_(\d+)$")


def split_kernel_name(kernel_name):
    """Split ``{stem}_kernel_{id}`` into ``(stem, kernel_id)``.

    ``kernel_name`` may be a bare filename or a full path; the basename is used
    for matching.  Returns ``(stem, kernel_id)`` or ``None`` when the name does
    not follow the ``_kernel_<digits>`` convention.
    """
    stem = os.path.splitext(os.path.basename(str(kernel_name)))[0]
    match = _KERNEL_ID_RE.search(stem)
    if not match:
        return None
    return stem[: match.start()], int(match.group(1))


def parse_kernel_id_from_name(kernel_name):
    """Return the integer kernel id, or ``None`` if the name is malformed."""
    parts = split_kernel_name(kernel_name)
    return parts[1] if parts else None


def get_original_image_name(kernel_name):
    """Recover the original full-image filename from a subimage filename.

    e.g. 'IMG_14_kernel_001.jpg' -> 'IMG_14.jpg'
    """
    stem = os.path.splitext(kernel_name)[0]
    parts = stem.rsplit("_kernel_", 1)
    ext = os.path.splitext(kernel_name)[1]
    if len(parts) == 2:
        return parts[0] + ext
    return kernel_name
