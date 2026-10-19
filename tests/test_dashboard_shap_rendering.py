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
