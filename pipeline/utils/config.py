"""config.py - shared pipeline config loading with an optional local override.

Machine-specific settings (conda python paths, GPU device, input directory,
model checkpoints) can live in ``env.local.yaml`` next to ``config.yaml`` and
are merged RECURSIVELY over ``config.yaml``.  This keeps scientific settings in
``config.yaml`` while letting each machine override only the paths it differs
on, without editing the shared file.

``env.local.yaml`` is git-ignored; ``env.local.yaml.example`` documents the keys.
"""

import os

import yaml


def _deep_merge(base, override):
    """Recursively overlay ``override`` onto ``base`` and return a new dict."""
    result = dict(base or {})
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path):
    """Load ``config.yaml``, then overlay ``env.local.yaml`` when present.

    ``config_path`` may be relative or absolute.  ``env.local.yaml`` is looked
    up in the same directory as ``config_path``.  Behaviour is identical to
    ``yaml.safe_load(config_path)`` when no override file exists.
    """
    config_path = os.path.abspath(str(config_path))
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    env_path = os.path.join(os.path.dirname(config_path), "env.local.yaml")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            env = yaml.safe_load(f) or {}
        config = _deep_merge(config, env)

    return config
