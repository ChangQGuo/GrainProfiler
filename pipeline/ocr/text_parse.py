"""text_parse.py - pure OCR text normalization and candidate parsing.

Extracted from metadata_extraction.py.  Depends only on re / numpy (for bbox
array access); no cv2, no OCR engine.  These are independently unit-testable.
"""

import re

import numpy as np


def normalize_weight_text(text):
    cleaned = text.strip()
    cleaned = cleaned.translate(
        str.maketrans(
            {
                "O": "0",
                "o": "0",
                "I": "1",
                "l": "1",
                "|": "1",
                "B": "8",
                "S": "5",
                ",": ".",
                ":": ".",
                ";": ".",
            }
        )
    )
    cleaned = re.sub(r"[^0-9.\-]", "", cleaned)
    # Keep a single leading minus (negative tare/scale-drift readings), drop any
    # stray internal minus so the value stays parseable.
    if "-" in cleaned:
        cleaned = "-" + cleaned.replace("-", "")
    if cleaned.count(".") > 1:
        first = cleaned.find(".")
        cleaned = cleaned[: first + 1] + cleaned[first + 1 :].replace(".", "")
    return cleaned




def normalize_label_text(text):
    """Normalize OCR label text by collapsing separator runs into single hyphens.

    Character substitutions prevent common OCR confusion on plant-breeding labels:
      O, o → 0    (e.g. '24-05-SZ-CG-OO77-O3' → '24-05-SZ-CG-0077-03')
      C, c → 0 only when followed by a digit, preserving letter codes
                  (e.g. '24-05-SZ-CG-C077-03' → '24-05-SZ-CG-0077-03',
                   'CG' is preserved)
    """
    cleaned = text.strip().upper()
    # O is almost always a misread 0 in digit slots → map to 0
    cleaned = cleaned.replace("O", "0")
    # C → 0 only when it starts a digit run (C077 → 0077), never in letter codes (CG)
    cleaned = re.sub(r"C(?=[0-9])", "0", cleaned)
    cleaned = cleaned.replace("_", "-")
    cleaned = re.sub(r"[^A-Z0-9]+", "-", cleaned)
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-")
    return cleaned




def choose_best_label_candidate(label_texts):
    """Pick the most label-like OCR result and keep its source item for QC."""
    label_pattern = re.compile(r"[A-Z0-9]+(?:-[A-Z0-9]+){2,}")
    best_candidate = None
    best_score = -1e9

    normalized_items = []
    for item in label_texts:
        normalized = normalize_label_text(item["text"])
        if not normalized:
            continue

        normalized_items.append((normalized, item))
        hyphen_count = normalized.count("-")
        alnum_len = len(re.sub(r"[^A-Z0-9]", "", normalized))
        if hyphen_count < 2 or alnum_len < 8:
            continue

        score = item["confidence"] * 100.0 + hyphen_count * 12.0 + alnum_len
        if label_pattern.fullmatch(normalized):
            score += 20.0

        if score > best_score:
            best_score = score
            best_candidate = {
                "value": normalized,
                "score": round(float(score), 3),
                "item": item,
            }

    if best_candidate is not None:
        return best_candidate

    ordered = sorted(
        normalized_items,
        key=lambda item: (item[1]["bbox"][:, 1].mean(), item[1]["bbox"][:, 0].min()),
    )
    joined = "".join(item[0] for item in ordered)
    match = label_pattern.search(joined)
    if not match:
        return None

    target = match.group(0)
    for normalized, item in ordered:
        if normalized == target or normalized in target or target in normalized:
            return {
                "value": target,
                "score": round(float(item["confidence"] * 100.0), 3),
                "item": item,
            }

    fallback_item = ordered[0][1] if ordered else None
    return {
        "value": target,
        "score": None,
        "item": fallback_item,
    }




def extract_weight_candidates(normalized_text, min_weight, max_weight):
    decimal_matches = re.findall(r"\d{1,3}\.\d{1,2}", normalized_text)
    candidates = []

    for candidate in decimal_matches:
        try:
            value = float(candidate)
        except ValueError:
            continue
        if min_weight <= value <= max_weight:
            candidates.append(
                {
                    "value": f"{value:.2f}",
                    "source_kind": "explicit_decimal",
                }
            )

    digits_only = re.sub(r"[^0-9]", "", normalized_text)
    if 3 <= len(digits_only) <= 5:
        inferred = digits_only[:-2] + "." + digits_only[-2:]
        try:
            value = float(inferred)
        except ValueError:
            value = None
        if value is not None and min_weight <= value <= max_weight:
            candidates.append(
                {
                    "value": f"{value:.2f}",
                    "source_kind": "inferred_decimal",
                }
            )

    return candidates




def summarize_texts(texts):
    summary = []
    for item in texts:
        summary.append(f"{item['source']}:{item['text']}({item['confidence']:.2f})")
    return " | ".join(summary)




def choose_best_weight_candidate(weight_texts, ocr_cfg, image_shape=None):
    """Score all parsed numeric candidates and keep the most plausible weight."""
    min_weight = float(ocr_cfg.get("weight_min_value", 0.05))
    max_weight = float(ocr_cfg.get("weight_max_value", 999.99))
    height = image_shape[0] if image_shape is not None else None
    width = image_shape[1] if image_shape is not None else None

    all_candidates = []
    for item in weight_texts:
        normalized = normalize_weight_text(item["text"])
        for candidate in extract_weight_candidates(normalized, min_weight, max_weight):
            all_candidates.append(
                {
                    "value": candidate["value"],
                    "source_kind": candidate["source_kind"],
                    "item": item,
                }
            )

    if not all_candidates:
        return None

    explicit_candidates = [cand for cand in all_candidates if cand["source_kind"] == "explicit_decimal"]
    candidates_to_score = explicit_candidates if explicit_candidates else all_candidates

    best_candidate = None
    best_score = -1e9
    for candidate in candidates_to_score:
        item = candidate["item"]
        score = item["confidence"] * 100.0
        score += 25.0 if candidate["source_kind"] == "explicit_decimal" else 8.0

        if item["source"].endswith("_binary"):
            score += 5.0
        elif item["source"].endswith("_scaled"):
            score += 2.5

        score += 10.0 * float(item.get("region_priority", 0.0))

        if width is not None and height is not None and "bbox" in item:
            center_x = float(item["bbox"][:, 0].mean())
            center_y = float(item["bbox"][:, 1].mean())
            center_x_bonus = 1.0 - abs(center_x - width / 2.0) / max(width / 2.0, 1.0)
            lower_bonus = center_y / max(float(height), 1.0)
            score += 8.0 * max(center_x_bonus, -1.0)
            score += 6.0 * lower_bonus

        if score > best_score:
            best_score = score
            best_candidate = {
                "value": candidate["value"],
                "source_kind": candidate["source_kind"],
                "score": round(float(score), 3),
                "item": item,
            }

    return best_candidate



