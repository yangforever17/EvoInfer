"""DeepSeek Harness as an agent CLI: the key-adding forwarder and the route the orchestrator gives dsh.

The forwarder tests relay to a fake upstream on loopback; nothing here needs dsh, a model or a network. (An end-to-end
run of dsh 0.2.2 against fake endpoints - skills found, `evo` through the job queue, writes outside the worktree
refused, only the forwarder holding the key - is described in docs/extending.md.)"""
import http.server
import json
import threading
import urllib.error
import urllib.request

import pytest

from evo.llm_forward import Forwarder


class _Upstream(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    seen = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        _Upstream.seen.append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}, "body": body})
        if self.path.endswith("/messages"):        # Anthropic stream
            ev = [{"type": "message_start", "message": {"usage": {"input_tokens": 10, "cache_read_input_tokens": 90, "output_tokens": 1}}},
                  {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hi"}},
                  {"type": "message_delta", "usage": {"output_tokens": 7}}]
            self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
            for e in ev:
                self.wfile.write(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode())
        else:                                      # OpenAI, not streamed
            out = json.dumps({"choices": [], "usage": {"prompt_tokens": 50, "completion_tokens": 5,
                                                       "prompt_tokens_details": {"cached_tokens": 20}}}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(out)


@pytest.fixture(autouse=True)
def no_proxy(monkeypatch):
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture()
def upstream():
    _Upstream.seen = []
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown(); srv.server_close()


def _post(url, headers, body=b"{}"):
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
            urllib.request.Request(url, data=body, headers=headers, method="POST"), timeout=10) as r:
        return r.read()


def test_openai_route_replaces_the_key_and_counts_tokens(upstream):
    with Forwarder(upstream + "/v1", "REAL", auth="bearer", strip_prefix="/v1", proxy=None) as fw:
        port = fw.server.server_address[1]
        _post(f"http://127.0.0.1:{port}/v1/chat/completions",
              {"Authorization": "Bearer local-forwarder", "Content-Type": "application/json", "X-Extra": "kept"})
    s = _Upstream.seen[-1]
    assert s["path"] == "/v1/chat/completions"
    assert s["headers"]["authorization"] == "Bearer REAL" and s["headers"]["x-extra"] == "kept"
    assert fw.usage == {"input_tokens": 50, "cached_input_tokens": 20, "output_tokens": 5, "requests": 1, "errors": 0}


def test_anthropic_route_uses_x_api_key_and_counts_stream_usage(upstream):
    with Forwarder(upstream + "/anthropic", "REAL", auth="x-api-key") as fw:
        port = fw.server.server_address[1]
        out = _post(f"http://127.0.0.1:{port}/v1/messages",
                    {"x-api-key": "local-forwarder", "anthropic-version": "2023-06-01", "Content-Type": "application/json"})
    s = _Upstream.seen[-1]
    assert s["path"] == "/anthropic/v1/messages"
    assert s["headers"]["x-api-key"] == "REAL" and "authorization" not in s["headers"]
    assert s["headers"]["anthropic-version"] == "2023-06-01"
    assert b"text_delta" in out                                    # the stream reaches the client unchanged
    # input = uncached + cache reads; output from message_delta only
    assert fw.usage["input_tokens"] == 100 and fw.usage["cached_input_tokens"] == 90 and fw.usage["output_tokens"] == 7


def test_unreachable_upstream_is_a_502_not_a_hang():
    with Forwarder("http://127.0.0.1:9", "REAL") as fw:
        port = fw.server.server_address[1]
        with pytest.raises(urllib.error.HTTPError) as e:
            _post(f"http://127.0.0.1:{port}/v1/chat/completions", {"Content-Type": "application/json"})
    assert e.value.code == 502 and fw.usage["errors"] == 1


@pytest.mark.parametrize("api", ["deepseek", "openai"])
def test_dsh_route_keeps_the_key_out_of_dsh(api, monkeypatch):
    import orchestrator as O
    monkeypatch.setenv("SOME_KEY", "REAL-SECRET")
    pol = {"cli": "dsh", "api": api, "api_key_env": "SOME_KEY", "base_url": "https://gw.example/v1"}
    fw, env, patch = O.dsh_route(pol, "m1")
    fw.stop()
    assert "REAL-SECRET" not in patch and "REAL-SECRET" not in json.dumps(env)
    assert "127.0.0.1" in json.dumps(env) + patch                    # dsh only ever talks to the forwarder
    assert "id: tool-web\n  disabled: true" in patch and f"timeoutMs: {O.DSH_CMD_TIMEOUT_S * 1000}" in patch
    assert ("provider: deepseek-official" if api == "deepseek" else "provider: evo-gw") in patch
    cmd = O.dsh_command("/x/p.yml")
    assert cmd[cmd.index("--permission-mode") + 1] == "workspace-write" and cmd[cmd.index("--approval-policy") + 1] == "never"


def test_dsh_route_needs_a_key(monkeypatch):
    import orchestrator as O
    monkeypatch.delenv("NO_SUCH_KEY", raising=False)
    with pytest.raises(SystemExit):
        O.dsh_route({"api": "deepseek", "api_key_env": "NO_SUCH_KEY"}, "m")


def test_dsh_worktree_layout(tmp_path, monkeypatch):
    """Skills go where dsh discovers them (<project>/.dsh/skills); .dsh never reaches the GPU host."""
    import orchestrator as O
    assert O.agent_dir("dsh") == ".dsh" and ".dsh" in O.LOCAL_ONLY and "dsh" in O.AGENT_KINDS
    monkeypatch.setattr(O, "RUNS", str(tmp_path))
    monkeypatch.setattr(O, "rsync", lambda *a, **k: None)
    local = O.make_local("e1", "dsh", "/remote/wt", "v0")
    assert (tmp_path / "e1" / "engine" / ".dsh" / "skills" / "evo-workflow" / "SKILL.md").exists()
    assert ".dsh/skills/evo-workflow/SKILL.md" in open(f"{local}/AGENTS.md").read()
