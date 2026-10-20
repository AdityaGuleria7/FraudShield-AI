import re
import shutil
import subprocess
from pathlib import Path

import pytest


def test_ieee_csv_export_escapes_formula_like_text():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to execute the dashboard CSV helper.")

    source = (Path("app/dashboard.py").read_text() + Path("app/templates/dashboard.html").read_text(encoding="utf-8"))
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


def test_ieee_csv_export_handles_unicode_formula_prefixes():
    import json

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to execute the dashboard CSV helper.")

    source = (Path("app/dashboard.py").read_text(encoding="utf-8") + Path("app/templates/dashboard.html").read_text(encoding="utf-8"))
    match = re.search(
        r"const q = v => \{.*?\n[ \t\u00a0]*\};",
        source,
        flags=re.DOTALL,
    )
    assert match, "CSV export escaping helper was not found in dashboard source."

    cases = [
        ("=1+1", True),
        ("  =1+1", True),
        ("\t=1+1", True),
        ("\n=1+1", True),
        ("\r=1+1", True),
        ("\u00a0=1+1", True),
        ("\u2003=1+1", True),
        ("\u202f=1+1", True),
        ("\u200b=1+1", True),
        ("\ufeff=1+1", True),
        ("ordinary text", False),
        (-12.5, False),
    ]
    cases_json = json.dumps(cases, ensure_ascii=True)
    script = match.group(0) + "\n" + """
const assert = require('node:assert/strict');
const cases = CASES_JSON;
for (const [input, shouldEscape] of cases) {
  const output = q(input);
  const escaped = output.charCodeAt(0) === 34 && output.charCodeAt(1) === 39;
  assert.equal(escaped, shouldEscape, JSON.stringify(input) + ': ' + output);
}
""".replace("CASES_JSON", cases_json)

    result = subprocess.run(
        [node, "-e", script],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
