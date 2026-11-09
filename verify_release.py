"""Verify the SHA-256 manifest of an extracted release using only Python."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
failures = []
lines = (ROOT/'SHA256SUMS.txt').read_text().splitlines()
for line in lines:
    expected, name = line.split('  ', 1)
    path = ROOT/name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        failures.append(name)
if failures:
    raise SystemExit('Checksum mismatch: '+', '.join(failures))
print(f'PASS: {len(lines)} release files match SHA-256 checksums.')
