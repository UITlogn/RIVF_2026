"""Verify the distributed research artifact's SHA-256 coverage and bytes."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    listed = {}
    for line in (ROOT / 'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        expected, name = line.split('  ', 1)
        if name in listed or Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError(f'invalid or duplicate checksum path: {name}')
        listed[name] = expected
    errors = [name for name,h in listed.items()
              if not (ROOT/name).is_file() or digest(ROOT/name) != h]
    manifest = json.loads((ROOT/'MANIFEST.json').read_text(encoding='utf-8'))
    records = manifest['files']
    expected_files = set(listed) - {'MANIFEST.json'}
    if {r['path'] for r in records} != expected_files:
        errors.append('manifest coverage')
    for record in records:
        p = ROOT/record['path']
        if not p.is_file() or digest(p) != record['sha256'] or p.stat().st_size != record['bytes']:
            errors.append('manifest: '+record['path'])
    # A source ZIP has no Git metadata; in a checkout also detect omissions.
    if (ROOT/'.git').exists():
        tracked = subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
        omitted = set(filter(None,tracked)) - set(listed) - {'SHA256SUMS.txt'}
        errors.extend('unhashed tracked file: '+n for n in sorted(omitted))
    if errors:
        raise SystemExit('FAIL: '+', '.join(errors))
    print(f'PASS: {len(listed)} file hashes and manifest coverage verified.')

if __name__ == '__main__':
    main()
