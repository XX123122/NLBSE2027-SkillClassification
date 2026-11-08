"""Run provenance and server guards shared by experiment entrypoints."""

import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def require_server():
    if platform.system() != "Linux" or not (ROOT / '.skillcomp-server').is_file():
        raise RuntimeError("Real-data computation requires Linux and an explicitly initialized .skillcomp-server marker")
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible and (ROOT/'configs/gpu_resource_policy.json').exists():
        from skillcomp.gpu_policy import snapshot,assert_visible_devices
        gpus,_,_=snapshot();assert_visible_devices(visible,gpus)


def encoder_python():
    """Explicit external GPU runtime or this project's current interpreter."""
    config=ROOT/'configs/local_runtime.json'
    settings=json.loads(config.read_text()) if config.exists() else {}
    return os.environ.get('SKILLCOMP_ENCODER_PYTHON') or settings.get('encoder_python') or sys.executable


def official_module():
    spec = importlib.util.spec_from_file_location("official", ROOT / "upstream/skill-classification/competition_baseline/main.py")
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def json_write(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def provenance(out, config):
    json_write(out / "config.json", config)
    # JSON is valid YAML 1.2, retaining an exact machine-readable config.yaml.
    json_write(out / "config.yaml", config)
    manifest = json.loads((ROOT / "reports/local_manifest.json").read_text())
    db = ROOT / "data/raw/skillscope_data.db"
    with db.open("rb") as f:
        actual = hashlib.file_digest(f, "sha256").hexdigest()
    assert actual == manifest["database"]["sha256"], "Database changed"
    json_write(out / "data_manifest.json", manifest)
    files = [ROOT / "pyproject.toml"]
    for folder in ("src", "scripts", "configs", "tests"):
        files.extend(p for p in (ROOT / folder).rglob("*") if p.is_file()
                     and "__pycache__" not in p.parts and not any(s.endswith('.egg-info') for s in p.parts))
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    source_hash = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    json_write(out / "code_manifest.json", {"source_tree_sha256": source_hash, "files": hashes})
    for source in files:
        target = out / "source" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    (out / "run.log").symlink_to(ROOT / "logs" / (out.name + ".log"))
    (out / "git_commit.txt").write_text("upstream=" + manifest["upstream_commit"] + "\nproject_source_sha256=" + source_hash + "\n")
    commands = [[str(ROOT / ".venv/bin/python"), "-m", "pip", "freeze"], ["uname", "-a"], ["free", "-h"],
                ["nvidia-smi", "--query-gpu=index,name,memory.free", "--format=csv"]]
    environment = "Python " + platform.python_version() + "\n"
    for cmd in commands:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        environment += "$ " + " ".join(cmd) + "\n" + result.stdout + result.stderr + "\n"
    (out / "environment.txt").write_text(environment)
    return source_hash


def status(out, state, **kwargs):
    data = {"state": state, "updated_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(), **kwargs}
    json_write(out / "status.json", data)
    print(json.dumps(data), flush=True)
