"""Unit tests for sidebar_order utilities."""

import json
import tempfile
from pathlib import Path

import pytest

from app.utils.sidebar_order import (
    apply_order_with_fallback,
    get_sidebar_order_path,
    load_sidebar_order,
    save_sidebar_order,
)


class TestGetSidebarOrderPath:
    """Tests for get_sidebar_order_path()."""

    def test_returns_correct_path(self):
        """Should return workspaces_root/.xima/sidebar_order.json."""
        ws_root = Path("/test/workspaces")
        path = get_sidebar_order_path(ws_root)
        assert path == Path("/test/workspaces/.xima/sidebar_order.json").resolve()

    def test_resolves_path(self):
        """Should resolve the path to absolute."""
        ws_root = Path("./workspaces")
        path = get_sidebar_order_path(ws_root)
        assert path.is_absolute()


class TestLoadSidebarOrder:
    """Tests for load_sidebar_order()."""

    def test_returns_default_when_file_not_exists(self):
        """Should return default structure when file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws_root = Path(tmpdir)
            result = load_sidebar_order(ws_root)
            assert result == {
                "version": 1,
                "workspaces": [],
                "experiments": {},
                "updated_at": None,
            }

    def test_loads_valid_json(self):
        """Should load valid JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws_root = Path(tmpdir)
            xima_dir = ws_root / ".xima"
            xima_dir.mkdir()

            data = {
                "version": 1,
                "workspaces": ["ws1", "ws2"],
                "experiments": {"ws1": ["exp1", "exp2"]},
                "updated_at": "2026-01-27T12:00:00Z",
            }
            (xima_dir / "sidebar_order.json").write_text(json.dumps(data))

            result = load_sidebar_order(ws_root)
            assert result == data

    def test_returns_default_on_invalid_json(self):
        """Should return default structure when JSON is invalid."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws_root = Path(tmpdir)
            xima_dir = ws_root / ".xima"
            xima_dir.mkdir()

            (xima_dir / "sidebar_order.json").write_text("invalid json")

            result = load_sidebar_order(ws_root)
            assert result == {
                "version": 1,
                "workspaces": [],
                "experiments": {},
                "updated_at": None,
            }

    def test_handles_missing_fields(self):
        """Should provide defaults for missing fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws_root = Path(tmpdir)
            xima_dir = ws_root / ".xima"
            xima_dir.mkdir()

            data = {"version": 2}  # Missing workspaces, experiments
            (xima_dir / "sidebar_order.json").write_text(json.dumps(data))

            result = load_sidebar_order(ws_root)
            assert result["version"] == 2
            assert result["workspaces"] == []
            assert result["experiments"] == {}
            assert result["updated_at"] is None

    def test_handles_wrong_field_types(self):
        """Should use defaults when field types are wrong."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws_root = Path(tmpdir)
            xima_dir = ws_root / ".xima"
            xima_dir.mkdir()

            data = {
                "version": 1,
                "workspaces": "not a list",  # Wrong type
                "experiments": ["not", "a", "dict"],  # Wrong type
            }
            (xima_dir / "sidebar_order.json").write_text(json.dumps(data))

            result = load_sidebar_order(ws_root)
            assert result["workspaces"] == []
            assert result["experiments"] == {}


class TestSaveSidebarOrder:
    """Tests for save_sidebar_order()."""

    def test_creates_xima_directory(self):
        """Should create .xima directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws_root = Path(tmpdir)
            order = {"workspaces": ["ws1"], "experiments": {}}

            save_sidebar_order(ws_root, order=order)

            assert (ws_root / ".xima").exists()
            assert (ws_root / ".xima" / "sidebar_order.json").exists()

    def test_saves_valid_json(self):
        """Should save valid JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws_root = Path(tmpdir)
            order = {
                "workspaces": ["ws1", "ws2"],
                "experiments": {"ws1": ["exp1"]},
            }

            result = save_sidebar_order(ws_root, order=order)

            # Check returned payload
            assert result["version"] == 1
            assert result["workspaces"] == ["ws1", "ws2"]
            assert result["experiments"] == {"ws1": ["exp1"]}
            assert "updated_at" in result

            # Check file contents
            path = get_sidebar_order_path(ws_root)
            saved = json.loads(path.read_text())
            assert saved == result

    def test_adds_version_and_timestamp(self):
        """Should add version and updated_at fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws_root = Path(tmpdir)
            order = {}

            result = save_sidebar_order(ws_root, order=order)

            assert result["version"] == 1
            assert result["updated_at"] is not None
            assert "T" in result["updated_at"]  # ISO format

    def test_handles_missing_fields_in_order(self):
        """Should provide empty arrays/objects for missing fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws_root = Path(tmpdir)
            order = {}  # No workspaces or experiments

            result = save_sidebar_order(ws_root, order=order)

            assert result["workspaces"] == []
            assert result["experiments"] == {}

    def test_atomic_write_with_tmp_file(self):
        """Should use atomic write (tmp + replace) to prevent corruption."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws_root = Path(tmpdir)
            order = {"workspaces": ["ws1"]}

            save_sidebar_order(ws_root, order=order)

            path = get_sidebar_order_path(ws_root)
            tmp_path = path.with_suffix(path.suffix + ".tmp")

            # Tmp file should be cleaned up after replace
            assert not tmp_path.exists()
            assert path.exists()


class TestApplyOrderWithFallback:
    """Tests for apply_order_with_fallback()."""

    def test_returns_original_when_desired_is_none(self):
        """Should return original list when desired order is None."""
        ids = ["a", "b", "c"]
        result = apply_order_with_fallback(ids, None)
        assert result == ["a", "b", "c"]

    def test_returns_original_when_desired_is_empty(self):
        """Should return original list when desired order is empty."""
        ids = ["a", "b", "c"]
        result = apply_order_with_fallback(ids, [])
        assert result == ["a", "b", "c"]

    def test_applies_complete_desired_order(self):
        """Should apply desired order when all IDs are present."""
        ids = ["a", "b", "c"]
        desired = ["c", "a", "b"]
        result = apply_order_with_fallback(ids, desired)
        assert result == ["c", "a", "b"]

    def test_appends_missing_ids(self):
        """Should append IDs not in desired order at the end."""
        ids = ["a", "b", "c", "d"]
        desired = ["d", "a"]
        result = apply_order_with_fallback(ids, desired)
        assert result == ["d", "a", "b", "c"]

    def test_ignores_unknown_ids(self):
        """Should ignore IDs in desired order that don't exist."""
        ids = ["a", "b", "c"]
        desired = ["x", "c", "y", "a"]
        result = apply_order_with_fallback(ids, desired)
        assert result == ["c", "a", "b"]

    def test_removes_duplicates(self):
        """Should remove duplicate IDs in desired order."""
        ids = ["a", "b", "c"]
        desired = ["c", "a", "c", "b"]
        result = apply_order_with_fallback(ids, desired)
        assert result == ["c", "a", "b"]

    def test_handles_case_insensitive(self):
        """Should handle IDs case-insensitively (lowercase)."""
        ids = ["a", "b", "c"]
        desired = ["C", "A"]
        result = apply_order_with_fallback(ids, desired)
        assert result == ["c", "a", "b"]

    def test_filters_non_string_values(self):
        """Should filter out non-string values in desired order."""
        ids = ["a", "b", "c"]
        desired = ["c", None, 123, "a", {"key": "val"}]
        result = apply_order_with_fallback(ids, desired)
        assert result == ["c", "a", "b"]

    def test_strips_whitespace(self):
        """Should strip whitespace from IDs in desired order."""
        ids = ["a", "b", "c"]
        desired = ["  c  ", " a "]
        result = apply_order_with_fallback(ids, desired)
        assert result == ["c", "a", "b"]

    def test_empty_ids_list(self):
        """Should handle empty IDs list."""
        ids = []
        desired = ["a", "b"]
        result = apply_order_with_fallback(ids, desired)
        assert result == []

    def test_complex_scenario(self):
        """Should handle complex scenario with all edge cases."""
        ids = ["ws1", "ws2", "ws3", "ws4"]
        desired = ["WS3", "  ws1  ", None, "unknown", "ws3", 123, "ws2"]
        result = apply_order_with_fallback(ids, desired)
        # Expected: ws3 (first occurrence), ws1, ws2, ws4 (missing)
        assert result == ["ws3", "ws1", "ws2", "ws4"]
