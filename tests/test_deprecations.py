from __future__ import annotations

import importlib
import warnings


def _assert_legacy_attribute_access_emits_deprecation(module_name: str, attribute_name: str) -> None:
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", DeprecationWarning)
        module = importlib.import_module(module_name)
        getattr(module, attribute_name)
    assert any(item.category is DeprecationWarning for item in captured), f"expected DeprecationWarning for {module_name}"


def test_legacy_import_roots_emit_deprecation_warnings() -> None:
    _assert_legacy_attribute_access_emits_deprecation("core", "array_factor_linear")
    _assert_legacy_attribute_access_emits_deprecation("algorithms", "mvdr_weights")
    _assert_legacy_attribute_access_emits_deprecation("data", "simulate_array_iq")
    _assert_legacy_attribute_access_emits_deprecation("simulations", "load_scenario_config")
    _assert_legacy_attribute_access_emits_deprecation("visualize", "build_heatmap")
