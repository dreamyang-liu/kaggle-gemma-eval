"""Package an agent dir as a Kaggle submission.zip, with the official checks from the getting-started notebook
(no symlinks, adk_submission.validate_directory, size and extension limits). Entries are written in sorted order
with fixed timestamps, so the archive is reproducible.
usage: docker run --rm -v $PWD:/w gemma4-kaggle:v170 python -I /w/scripts/package_submission.py /w/agents/X /w/X.zip"""
import os
import sys
import zipfile
from pathlib import Path

from adk_submission import validate_directory
from swegemma.config import ALLOWED_SUBMISSION_EXTENSIONS, MAX_SUBMISSION_SIZE_BYTES, build_submission_limits

agent_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
assert (agent_dir / 'agent.yaml').is_file(), 'agent.yaml must be at the root'
for dirpath, dirnames, filenames in os.walk(agent_dir, followlinks=False):
    for name in (*dirnames, *filenames):
        assert not (Path(dirpath) / name).is_symlink(), f'symlink: {name}'
limits, _ = build_submission_limits()
validate_directory(agent_dir, limits=limits)

files = sorted(p for p in agent_dir.rglob('*') if p.is_file())
with zipfile.ZipFile(out, 'w') as zf:
    for p in files:
        e = zipfile.ZipInfo(p.relative_to(agent_dir).as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
        e.compress_type = zipfile.ZIP_DEFLATED
        e.external_attr = 0o100644 << 16
        zf.writestr(e, p.read_bytes())
with zipfile.ZipFile(out) as zf:
    assert zf.testzip() is None
    infos = zf.infolist()
    assert sum(i.file_size for i in infos) <= MAX_SUBMISSION_SIZE_BYTES
    for i in infos:
        assert Path(i.filename).suffix.lower() in ALLOWED_SUBMISSION_EXTENSIONS, f'disallowed extension: {i.filename}'
        assert zf.read(i.filename) == (agent_dir / i.filename).read_bytes()
    print(f'{out}: {len(infos)} files, {out.stat().st_size} bytes')
    for i in infos:
        print(f'  {i.filename} ({i.file_size} B)')
