"""A key-adding forwarder to a model endpoint, for agent CLIs whose key must stay out of the agent's environment.

The agent CLI (DeepSeek Harness) talks to http://127.0.0.1:<port> with a placeholder credential; the forwarder runs in
the orchestrator's process, outside the agent's sandbox, replaces the credential with the real key and relays the
request and the (streamed) response unchanged. It also counts tokens from the responses, in both wire formats:

  auth="bearer"     OpenAI-compatible chat completions (Authorization: Bearer <key>)
  auth="x-api-key"  Anthropic-compatible Messages, e.g. DeepSeek's official endpoint (x-api-key: <key>)

Standard library only (it runs on the agent side). Used by orchestrator.py (--agent dsh) and
experiments/no_harness/dsh.py.
"""
import http.server
import json
import threading
import urllib.error
import urllib.request

# credentials the client may send (placeholders) and hop-by-hop headers: never relayed
_DROP = {"host", "content-length", "authorization", "x-api-key", "x-dsh-auth-token", "connection", "accept-encoding",
         "proxy-authorization", "keep-alive", "transfer-encoding", "te", "upgrade"}


class Forwarder:
    def __init__(self, upstream, key, auth="bearer", strip_prefix="", proxy=None):
        """upstream: endpoint root the request path is appended to (after removing strip_prefix from it)."""
        if auth not in ("bearer", "x-api-key"):
            raise ValueError(f"auth must be 'bearer' or 'x-api-key', not {auth!r}")
        self.upstream, self.key, self.auth, self.strip = upstream.rstrip("/"), key, auth, strip_prefix
        handlers = [urllib.request.ProxyHandler({"http": proxy, "https": proxy})] if proxy else []
        self.opener = urllib.request.build_opener(*handlers)   # without a proxy: this process's $HTTPS_PROXY
        self.usage = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0, "requests": 0, "errors": 0}
        self.lock = threading.Lock()
        self.server = None

    # ------------------------------------------------------------------ token accounting
    def count(self, obj):
        """Add the usage in one response object or stream event (OpenAI or Anthropic shape) to the totals."""
        if not isinstance(obj, dict):
            return
        u = obj.get("usage")
        if obj.get("type") == "message_start":          # Anthropic stream: input side arrives first
            u = (obj.get("message") or {}).get("usage")
            if isinstance(u, dict):
                u = dict(u, output_tokens=0)          # the output count comes with message_delta
        if not isinstance(u, dict):
            return
        with self.lock:
            if "prompt_tokens" in u or "completion_tokens" in u:          # OpenAI
                self.usage["input_tokens"] += u.get("prompt_tokens") or 0
                self.usage["cached_input_tokens"] += (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
                self.usage["output_tokens"] += u.get("completion_tokens") or 0
            else:                                                          # Anthropic: input_tokens excludes cache
                cached = u.get("cache_read_input_tokens") or 0
                self.usage["input_tokens"] += (u.get("input_tokens") or 0) + cached + (u.get("cache_creation_input_tokens") or 0)
                self.usage["cached_input_tokens"] += cached
                self.usage["output_tokens"] += u.get("output_tokens") or 0

    def _bump(self, k):
        with self.lock:
            self.usage[k] += 1

    # ------------------------------------------------------------------ server
    def _handler(self):
        fw = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _reply(self, code, ctype, data):
                self.send_response(code); self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

            def _relay(self, method):
                n = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(n) if n else None
                path = self.path[len(fw.strip):] if fw.strip and self.path.startswith(fw.strip) else self.path
                hdrs = {k: v for k, v in self.headers.items() if k.lower() not in _DROP}
                hdrs.update({"Authorization": "Bearer " + fw.key} if fw.auth == "bearer" else {"x-api-key": fw.key})
                req = urllib.request.Request(fw.upstream + path, data=body, method=method, headers=hdrs)
                fw._bump("requests")
                try:
                    r = fw.opener.open(req, timeout=1200)
                except urllib.error.HTTPError as e:
                    fw._bump("errors")
                    self._reply(e.code, e.headers.get("Content-Type") or "application/json", e.read())
                    return
                except Exception as e:   # tunnel or gateway down: a 502 lets the client's own retry policy handle it
                    fw._bump("errors")
                    self._reply(502, "application/json", json.dumps({"error": {"message": f"forwarder: {type(e).__name__}"}}).encode())
                    return
                self.send_response(r.status)
                for k, v in r.headers.items():
                    if k.lower() in ("content-type", "request-id", "x-request-id", "anthropic-ratelimit-requests-remaining"):
                        self.send_header(k, v)
                self.end_headers()                      # HTTP/1.0: the body ends when the connection closes
                stream = "event-stream" in (r.headers.get("Content-Type") or "")
                buf, whole = b"", []
                try:
                    while chunk := r.read1(65536):
                        self.wfile.write(chunk); self.wfile.flush()
                        if stream:
                            buf += chunk
                            *lines, buf = buf.split(b"\n")
                            for ln in lines:
                                if ln.startswith(b"data:") and b'"usage"' in ln:
                                    try:
                                        fw.count(json.loads(ln[5:]))
                                    except ValueError:
                                        pass
                        else:
                            whole.append(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                if whole:
                    try:
                        fw.count(json.loads(b"".join(whole)))
                    except ValueError:
                        pass

            def do_POST(self):
                self._relay("POST")

            def do_GET(self):
                self._relay("GET")

        return H

    def start(self):
        """Listen on a free loopback port; returns the port."""
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self.server.server_address[1]

    def stop(self):
        if self.server:
            self.server.shutdown(); self.server.server_close(); self.server = None

    def __enter__(self):
        self.start(); return self

    def __exit__(self, *exc):
        self.stop()
