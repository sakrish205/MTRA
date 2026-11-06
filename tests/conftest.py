"""Test harness: run from the repo without installing the package.

The fixtures are session-scoped because building the plate mesh, parsing the
program and running the full measurement sweep are the expensive parts, and every
assertion in the suite is about a value that must not change between tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT / "src", ROOT):
    text = str(candidate)
    if text not in sys.path:
        sys.path.insert(0, text)

from tests.fixtures.synthetic import (  # noqa: E402
    build_cylinder_project,
    build_project,
)
from mtra.pipeline.run import analyse_project  # noqa: E402


@pytest.fixture(scope="session")
def plate_project(tmp_path_factory):
    return build_project(tmp_path_factory.mktemp("plate"))


@pytest.fixture(scope="session")
def plate_result(plate_project):
    manifest, _truth = plate_project
    return analyse_project(manifest)


@pytest.fixture(scope="session")
def cylinder_project(tmp_path_factory):
    return build_cylinder_project(tmp_path_factory.mktemp("cylinder"))


@pytest.fixture(scope="session")
def cylinder_result(cylinder_project):
    manifest, _truth = cylinder_project
    return analyse_project(manifest)
