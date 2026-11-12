import subprocess

from evo.diffsyms import symbols

BASE = '''import os


class Runner:
    def run(self):
        return 1

    def stop(self):
        return 2


def helper():
    return 3
'''


def git(repo, *a):
    return subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True, text=True).stdout.strip()


def test_symbols(tmp_path):
    git(tmp_path, "init", "-q")
    (tmp_path / "m.py").write_text(BASE)
    git(tmp_path, "add", "-A"); git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "base")
    base = git(tmp_path, "rev-parse", "HEAD")
    (tmp_path / "m.py").write_text(BASE.replace("import os", "import os, sys").replace("return 1", "return 10"))
    (tmp_path / "new.py").write_text("x = 1\n")
    git(tmp_path, "add", "-A"); git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "change")
    head = git(tmp_path, "rev-parse", "HEAD")
    # the import change is ignored, the method is named precisely, a new file is claimed whole
    assert symbols(str(tmp_path), base, head) == ["m.py:Runner.run", "new.py"]
