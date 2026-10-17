import re
import shutil
import subprocess
from pathlib import Path

import pytest


def test_ieee_csv_export_escapes_formula_like_text():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to execute the dashboard CSV helper.")

    source = Path("app/dashboard.py").read_text()
    match = re.search(
        r"const q = v => \{.*?\n[ \t\u00a0]*\};",
        source,
        flags=re.DOTALL,
    )
    assert match, "CSV export escaping helper was not found in dashboard source."

    script = match.group(0) + """
const assert = require('node:assert/strict');
const cases = [
  ['=1+1', "\\"' =1+1\\"".replace("' ", "'")],
  ['+SUM(A1:A2)', "\\"'+SUM(A1:A2)\\""],
  ['-cmd', "\\"'-cmd\\""],
  ['@SUM(A1)', "\\"'@SUM(A1)\\""],
  ['  =1+1', "\\"'  =1+1\\""],
  ['\\t=1+1', "\\"'\\t=1+1\\""],
  ['normal transaction', "\\"normal transaction\\""],
  ['merchant "quoted"', "\\"merchant \\"\\"quoted\\"\\"\\""],
  [-12.5, "\\"-12.5\\""],
  [0.42, "\\"0.42\\""]
];
for (const [input, expected] of cases) {
  assert.equal(q(input), expected);
}
"""

    result = subprocess.run(
        [node, "-e", script],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
