import re
import shutil
import subprocess
from pathlib import Path

import pytest


def test_dashboard_shap_formatter_rejects_non_finite_values():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to execute the dashboard SHAP helper.")

    source = Path("app/dashboard.py").read_text(encoding="utf-8")
    match = re.search(
        r"function formatShapContribution\(value\) \{.*?\n\}",
        source,
        flags=re.DOTALL,
    )
    assert match, "Safe SHAP contribution formatter was not found in dashboard source."

    script = match.group(0) + """
const assert = require('node:assert/strict');
assert.equal(formatShapContribution(0.12567), '+0.1257');
assert.equal(formatShapContribution(-0.12567), '-0.1257');
assert.equal(formatShapContribution(0), '0.0000');
assert.equal(formatShapContribution('not-a-number'), null);
assert.equal(formatShapContribution(NaN), null);
assert.equal(formatShapContribution(Infinity), null);
assert.equal(formatShapContribution(-Infinity), null);
"""

    result = subprocess.run(
        [node, "-e", script],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


def test_dashboard_shap_rendering_reports_when_no_valid_contributions_remain():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to execute the dashboard SHAP helper.")

    source = Path("app/dashboard.py").read_text(encoding="utf-8")
    match = re.search(
        r"function hasValidShapContributions\(drivers\) \{.*?\n\}",
        source,
        flags=re.DOTALL,
    )
    assert match, "SHAP contribution availability helper was not found in dashboard source."

    script = match.group(0) + """
const assert = require('node:assert/strict');
assert.equal(hasValidShapContributions([{shap_value: 0.2}]), true);
assert.equal(hasValidShapContributions([{shap_value: 0}]), true);
assert.equal(hasValidShapContributions([{shap_value: NaN}]), false);
assert.equal(hasValidShapContributions([{shap_value: Infinity}]), false);
assert.equal(hasValidShapContributions([{shap_value: 'invalid'}]), false);
assert.equal(hasValidShapContributions([]), false);
assert.equal(hasValidShapContributions(null), false);
"""

    result = subprocess.run(
        [node, "-e", script],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


def test_dashboard_explanation_status_uses_valid_contributions():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to execute the dashboard SHAP helper.")

    source = Path("app/dashboard.py").read_text(encoding="utf-8")
    match = re.search(
        r"function hasValidShapContributions\(drivers\) \{.*?\n\}",
        source,
        flags=re.DOTALL,
    )
    assert match, "SHAP contribution availability helper was not found in dashboard source."

    script = match.group(0) + """
const assert = require('node:assert/strict');

function explanationStatus(status, drivers) {
  const hasValidContributions = hasValidShapContributions(drivers);
  if (status === 'unavailable') return 'unavailable';
  if (status === 'not_requested') return 'not_requested';
  if (!hasValidContributions) return 'no_valid_contributions';
  return 'available';
}

assert.equal(explanationStatus('available', [{shap_value: 0.25}]), 'available');
assert.equal(explanationStatus('available', [{shap_value: NaN}]), 'no_valid_contributions');
assert.equal(explanationStatus('available', [{shap_value: Infinity}]), 'no_valid_contributions');
assert.equal(explanationStatus('available', []), 'no_valid_contributions');
assert.equal(explanationStatus('unavailable', [{shap_value: 0.25}]), 'unavailable');
assert.equal(explanationStatus('not_requested', []), 'not_requested');
"""

    result = subprocess.run(
        [node, "-e", script],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
