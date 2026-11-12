"""End to end, local mode: an agent's `evo` call (bin/evo) goes through the queue in its worktree, the orchestrator's
JobExecutor mirrors the worktree to the GPU-host copy and runs the framework's `evo` there with the episode's settings.
A stand-in framework (fake evo.cli) replaces the real one, so no GPU is needed."""
import os
import shutil
import subprocess

import pytest

import orchestrator as O

FAKE_CLI = r'''
import base64, os, sys
argv = [x for x in base64.b64decode(os.environ["EVO_ARGV_B64"]).decode().split("\0") if x]
wt = os.environ["EVO_WORKTREE"]
print("argv", argv)
print("gpu", os.environ["EVO_GPU"], "evals", os.environ["EVO_MAX_EVALS"], "agent", os.environ["EVO_AGENT_ID"])
print("worktree has", sorted(os.listdir(wt)))
sys.exit(3 if argv[:1] == ["fail"] else 0)
'''


@pytest.mark.skipif(not shutil.which("rsync"), reason="needs rsync")
def test_evo_call_roundtrip(tmp_path, monkeypatch):
    home = tmp_path / "home"
    fw = home / "framework" / "evo"; fw.mkdir(parents=True)
    (fw / "__init__.py").write_text(""); (fw / "cli.py").write_text(FAKE_CLI)
    monkeypatch.setattr(O, "REMOTE", str(home)); monkeypatch.setattr(O, "MODE", "local")
    monkeypatch.setattr(O, "PY", shutil.which("python3"))
    local = tmp_path / "runs" / "ep1" / "engine"; local.mkdir(parents=True)
    (local / "nanovllm.py").write_text("# candidate engine\n")
    (local / "AGENTS.md").write_text("agent-side only\n")
    env = dict(os.environ, EVO_LOCAL_WORKTREE=str(local))
    with O.JobExecutor("ep1", str(local), {"EVO_GPU": 5, "EVO_MAX_EVALS": 8}):
        p = subprocess.run([os.path.join(O.HERE, "bin", "evo"), "run", "--", "python", "-c", "print('a b')"],
                           env=env, capture_output=True, text=True, timeout=60)
        assert p.returncode == 0, p.stderr
        assert "argv ['run', '--', 'python', '-c', \"print('a b')\"]" in p.stdout     # arguments survive without quoting
        assert "gpu 5 evals 8 agent ep1" in p.stdout                                    # settings come from the executor
        assert "worktree has ['nanovllm.py']" in p.stdout                                # agent-side files stay local
        q = subprocess.run([os.path.join(O.HERE, "bin", "evo"), "fail"], env=env, capture_output=True, text=True, timeout=60)
        assert q.returncode == 3                                                         # exit codes pass through
    assert not any(f.endswith((".req", ".taken", ".out", ".rc")) for f in os.listdir(local / ".evo" / "jobs"))
