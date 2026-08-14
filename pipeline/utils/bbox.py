"""bbox.py - canonical YOLO bounding-box JSON parsing.

``yolo_bounding_boxes.json`` has accumulated several historical schemas (a bare
list of boxes, ``accepted_detections``, ``detections`` with a ``deleted`` flag,
and ``accepted_boxes``).  Every consumer must interpret these identically, so
the single implementation lives here and both the segmentation stage and the QC
overlay renderer delegate to it.
"""


def get_accepted_detections(image_entry):
    """Return the final kept detections for one image, in kernel_id order.

    Accepts every historical schema and always returns::

        [{'kernel_id': int, 'box': [x1, y1, x2, y2]}, ...]

    ``deleted`` detections are skipped in every schema that carries the flag.
    """
    accepted = []

    if isinstance(image_entry, list):
        for kernel_id, box in enumerate(image_entry, start=1):
            accepted.append({'kernel_id': kernel_id, 'box': [int(v) for v in box]})
        return accepted

    if not isinstance(image_entry, dict):
        return accepted

    accepted_detections = image_entry.get('accepted_detections', [])
    if accepted_detections:
        for det in sorted(accepted_detections, key=lambda item: item.get('kernel_id', 0)):
            if det.get('deleted', False):
                continue
            kernel_id = det.get('kernel_id')
            box = det.get('box')
            if kernel_id is None or box is None:
                continue
            accepted.append({'kernel_id': int(kernel_id), 'box': [int(v) for v in box]})
        return accepted

    detections = image_entry.get('detections', [])
    if detections:
        ordered = sorted(
            detections,
            key=lambda item: (item.get('kernel_id') is None, item.get('kernel_id') or 0),
        )
        for det in ordered:
            if det.get('deleted', False):
                continue
            kernel_id = det.get('kernel_id')
            box = det.get('box')
            if kernel_id is None or box is None:
                continue
            accepted.append({'kernel_id': int(kernel_id), 'box': [int(v) for v in box]})
        return accepted

    for kernel_id, box in enumerate(image_entry.get('accepted_boxes', []), start=1):
        accepted.append({'kernel_id': kernel_id, 'box': [int(v) for v in box]})

    return accepted
