import re
import shutil
import subprocess
from pathlib import Path

import pytest


def test_finding_cards_use_neutral_badges_for_all_patterns():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to execute the dashboard rendering helper.")

    source = Path("app/dashboard.py").read_text(encoding="utf-8")
    match = re.search(
        r"function renderFindingCard\(finding\) \{.*?^\}",
        source,
        flags=re.DOTALL | re.MULTILINE,
    )
    assert match, "Finding-card renderer was not found."

    script = """
const assert = require('node:assert/strict');
const esc = value => String(value ?? '—');
""" + match.group(0) + """
for (const pattern of [
  'ACCOUNT_TAKEOVER',
  'VELOCITY_ATTACK',
  'NEW_DEVICE_HIGH_VALUE',
  'IMPOSSIBLE_TRAVEL',
  'UNRECOGNIZED_PATTERN'
]) {
  const html = renderFindingCard({
    pattern,
    explanation: {
      headline: 'Example finding',
      plain_summary: 'Evidence for analyst review'
    }
  });
  assert.ok(html.includes('PATTERN DETECTED'), pattern);
  assert.ok(html.includes('risk-PATTERN'), pattern);
  assert.ok(!/risk-(CRITICAL|HIGH|MEDIUM|LOW)/.test(html), pattern);
}
assert.ok(!/patternWeight/.test(renderFindingCard.toString()));
"""

    result = subprocess.run(
        [node, "-e", script],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
