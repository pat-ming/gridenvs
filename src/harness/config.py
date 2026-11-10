"""
Configuration loader with CLI override support.

Usage:
    from harness.config import load_config
    cfg = load_config("configs/smoke.yaml")
    # Override from CLI:  python train.py --config configs/smoke.yaml --set training.epochs=50
"""

import copy
import sys
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict:
    """Load a YAML config file and return as a nested dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def override_config(cfg: dict, overrides: list[str]) -> dict:
    """
    Apply dot-separated overrides to a config dict.

    Existing keys only, so typos fail loudly; prefix with "+" to add a new key.

    Example:
        override_config(cfg, ["training.epochs=50", "+grpo.beta=0.04"])
    """
    cfg = copy.deepcopy(cfg)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must be key=value, got: {override}")
        key_path, value_str = override.split("=", 1)
        add = key_path.startswith("+")
        keys = key_path.lstrip("+").split(".")
        d = cfg
        for k in keys[:-1]:
            if add:
                d = d.setdefault(k, {})
            elif k not in d or not isinstance(d[k], dict):
                raise KeyError(f"Config path not found: {key_path}")
            else:
                d = d[k]
        final_key = keys[-1]
        if final_key not in d and not add:
            raise KeyError(f"Config key not found: {key_path} (use +{key_path})")
        d[final_key] = _cast_value(value_str, d.get(final_key))
    return cfg


def _cast_value(value_str: str, existing: Any) -> Any:
    """Cast a string value to match the type of the existing config value.
    Values starting with [ or { are parsed as YAML whatever the existing type."""
    if value_str.lstrip()[:1] in ("[", "{"):
        return yaml.safe_load(value_str)
    if existing is None:
        # Try int, then float, then keep as string
        for caster in (int, float):
            try:
                return caster(value_str)
            except ValueError:
                continue
        if value_str.lower() in ("true", "false"):
            return value_str.lower() == "true"
        return value_str
    if isinstance(existing, bool):
        return value_str.lower() in ("true", "1", "yes")
    if isinstance(existing, int):
        return int(value_str)
    if isinstance(existing, float):
        return float(value_str)
    if isinstance(existing, list):
        # Try JSON first (handles '["a", "b"]' syntax from CLI)
        import json as _json

        try:
            parsed = _json.loads(value_str)
            if isinstance(parsed, list):
                return parsed
        except (_json.JSONDecodeError, ValueError):
            pass
        # Fall back to comma-separated
        return [v.strip() for v in value_str.split(",")]
    return value_str


def merge_configs(base: dict, other: dict) -> dict:
    """Deep merge; values in `other` win, nested dicts merge key by key."""
    out = copy.deepcopy(base)
    for k, v in other.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_configs(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def parse_cli_overrides() -> tuple[list[str], list[str]]:
    """
    Parse sys.argv for --config and --set arguments.
    Returns (config_paths, overrides). --config can repeat; later files merge
    over earlier ones (e.g. a base config, then a model preset).
    """
    args = sys.argv[1:]
    config_paths = []
    overrides = []
    i = 0
    while i < len(args):
        if args[i] == "--config" and i + 1 < len(args):
            config_paths.append(args[i + 1])
            i += 2
        elif args[i] == "--set" and i + 1 < len(args):
            overrides.append(args[i + 1])
            i += 2
        else:
            i += 1
    return config_paths, overrides


def get_config(default_config: str = "configs/smoke.yaml") -> dict:
    """Load the --config files (default_config if none), then apply --set overrides."""
    config_paths, overrides = parse_cli_overrides()
    cfg = {}
    for path in config_paths or [default_config]:
        cfg = merge_configs(cfg, load_config(path))
    if overrides:
        cfg = override_config(cfg, overrides)
    return cfg
