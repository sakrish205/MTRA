"""Configuration loading.

Configuration lives outside the package on purpose: the parameter catalogue and
the robot grammars are evidence-bearing artefacts that a reviewing engineer may
need to edit (or audit) without touching Python. This module finds them, parses
them once, and hands out immutable views.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from mtra.core.errors import InputError

REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR_ENV = "MTRA_CONFIG_DIR"


def config_dir() -> Path:
    """Locate the config directory: env override, then repo layout."""
    override = os.environ.get(_CONFIG_DIR_ENV)
    if override:
        path = Path(override)
        if not path.is_dir():
            raise InputError(f"{_CONFIG_DIR_ENV} points at a non-directory: {path}")
        return path
    path = REPO_ROOT / "config"
    if not path.is_dir():
        raise InputError(
            f"config directory not found at {path}; set {_CONFIG_DIR_ENV} to override"
        )
    return path


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise InputError(f"configuration file missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise InputError(f"{path} must contain a YAML mapping")
    return loaded


@dataclass(frozen=True)
class Settings:
    """Analysis defaults with dotted-path lookup."""

    data: dict[str, Any] = field(default_factory=dict)

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted: str) -> Any:
        sentinel = object()
        value = self.get(dotted, sentinel)
        if value is sentinel:
            raise InputError(f"required configuration key missing: {dotted}")
        return value


@dataclass(frozen=True)
class ParameterSpec:
    id: str
    group: str
    name: str
    unit: str
    phase: int


@dataclass(frozen=True)
class Catalogue:
    groups: dict[str, str]
    parameters: tuple[ParameterSpec, ...]

    def __len__(self) -> int:
        return len(self.parameters)

    def by_id(self, parameter_id: str) -> ParameterSpec:
        for spec in self.parameters:
            if spec.id == parameter_id:
                return spec
        raise KeyError(parameter_id)

    def group_name(self, group: str) -> str:
        return self.groups.get(group, group)

    def group_members(self, group: str) -> tuple[ParameterSpec, ...]:
        return tuple(p for p in self.parameters if p.group == group)


@dataclass(frozen=True)
class RobotFormat:
    """One controller grammar, as configured."""

    key: str
    display_name: str
    provisional: bool
    supported: bool
    note: str
    extensions: tuple[str, ...]
    raw: dict[str, Any]

    def pattern(self, dotted: str) -> str:
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                raise InputError(f"grammar {self.key}: missing key {dotted}")
            node = node[part]
        if not isinstance(node, str):
            raise InputError(f"grammar {self.key}: {dotted} is not a pattern string")
        return node


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    return Settings(data=_read_yaml(config_dir() / "defaults.yaml"))


@lru_cache(maxsize=1)
def load_catalogue() -> Catalogue:
    raw = _read_yaml(config_dir() / "parameters.yaml")
    groups = {str(k): str(v) for k, v in raw.get("groups", {}).items()}
    entries = raw.get("parameters") or []
    if not entries:
        raise InputError("parameters.yaml contains no parameters")
    specs = tuple(
        ParameterSpec(
            id=str(entry["id"]),
            group=str(entry["group"]),
            name=str(entry["name"]),
            unit=str(entry.get("unit", "")),
            phase=int(entry.get("phase", 1)),
        )
        for entry in entries
    )
    seen: set[str] = set()
    for spec in specs:
        if spec.id in seen:
            raise InputError(f"duplicate parameter id in catalogue: {spec.id}")
        seen.add(spec.id)
    return Catalogue(groups=groups, parameters=specs)


@lru_cache(maxsize=1)
def _robot_formats() -> dict[str, RobotFormat]:
    raw = _read_yaml(config_dir() / "robot_formats.yaml")
    formats: dict[str, RobotFormat] = {}
    for key, node in (raw.get("formats") or {}).items():
        formats[str(key)] = RobotFormat(
            key=str(key),
            display_name=str(node.get("display_name", key)),
            provisional=bool(node.get("provisional", True)),
            supported=bool(node.get("supported", True)),
            note=str(node.get("note", "")),
            extensions=tuple(str(e).lower() for e in node.get("extensions", [])),
            raw=node,
        )
    return formats


def load_robot_format(key: str) -> RobotFormat:
    formats = _robot_formats()
    normalised = key.strip().lower()
    if normalised not in formats:
        known = ", ".join(sorted(formats))
        raise InputError(f"unknown robot format {key!r}; configured formats: {known}")
    return formats[normalised]


def known_robot_formats() -> tuple[str, ...]:
    return tuple(sorted(_robot_formats()))
