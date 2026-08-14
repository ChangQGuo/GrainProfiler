"""
Shared device helpers for pipeline stages.

Most stages use the same GPU. A stage-specific config value can override the
global runtime.gpu_device value when needed.
"""


def _is_empty_device(value):
    if value is None:
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"auto", "default", "global"}


def resolve_stage_device(config, section_name, key="device", default="cuda:0"):
    """Return the stage device, falling back to runtime.gpu_device."""
    section = config.get(section_name, {}) if isinstance(config, dict) else {}
    value = section.get(key)
    if not _is_empty_device(value):
        return str(value).strip()

    runtime = config.get("runtime", {}) if isinstance(config, dict) else {}
    runtime_device = runtime.get("gpu_device")
    if not _is_empty_device(runtime_device):
        return str(runtime_device).strip()

    return str(default)


def parse_cuda_device(device):
    """
    Convert a device string into PaddleOCR-friendly GPU options.

    Returns:
        (use_gpu, gpu_id)
    """
    text = str(device or "").strip().lower()
    if text in {"cpu", "none", "false", "off"}:
        return False, None

    if text.startswith("cuda"):
        if ":" not in text:
            return True, 0
        try:
            return True, int(text.split(":", 1)[1])
        except ValueError:
            return True, 0

    if text.isdigit():
        return True, int(text)

    return True, None


def paddleocr_device_kwargs(device):
    """Build PaddleOCR constructor kwargs from a shared device string."""
    use_gpu, gpu_id = parse_cuda_device(device)
    kwargs = {"use_gpu": bool(use_gpu)}
    if use_gpu and gpu_id is not None:
        kwargs["gpu_id"] = int(gpu_id)
    return kwargs
