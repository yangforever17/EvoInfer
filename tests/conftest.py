"""The tests need no GPU, no model and no agent CLI. They run against a throw-away EvoInfer home and the default config."""
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["EVO_HOME"] = tempfile.mkdtemp(prefix="evoinfer-test-home-")
os.environ["EVO_CONFIG"] = os.path.join(REPO, "tests", "no-such-config.toml")   # built-in defaults
sys.path[:0] = [REPO, os.path.join(REPO, "framework")]
