"""Static server with HTTP Range support, so the video can be scrubbed.

`python -m http.server` answers every request with the whole file, which means
the browser has to download the entire clip before it plays and the scrubber
does nothing — and scrubbing is the whole demo. Stdlib only, so this still runs
on a laptop with no network.

    python pipeline/serve.py            # serves web/ on 8765
    python pipeline/serve.py 9000 web
"""

import os
import re
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


class RangeHandler(SimpleHTTPRequestHandler):
    # Keep-alive matters here: the media element issues a stream of small range
    # requests, and a fresh TCP connection for each one is most of the latency.
    protocol_version = "HTTP/1.1"

    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def copyfile(self, source, outputfile):
        try:
            super().copyfile(source, outputfile)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # the browser moved on mid-stream — normal when scrubbing

    def send_head(self):
        header = self.headers.get("Range")
        if not header:
            return super().send_head()

        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return super().send_head()
        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found")
            return None

        size = os.fstat(f.fileno()).st_size
        m = RANGE_RE.match(header.strip())
        if not m:
            f.close()
            self.send_error(400, "Malformed Range header")
            return None

        first, last = m.group(1), m.group(2)
        if first == "":  # suffix form: bytes=-500 → the last 500 bytes
            length = int(last or 0)
            start, end = max(0, size - length), size - 1
        else:
            start = int(first)
            end = int(last) if last else size - 1

        if start >= size or start > end:
            f.close()
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")   # HTTP/1.1 needs the framing
            self.end_headers()
            return None

        end = min(end, size - 1)
        f.seek(start)

        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        return _Slice(f, end - start + 1)


class _Slice:
    """Wraps the file so copyfile() stops at the end of the requested range."""

    def __init__(self, f, remaining):
        self.f, self.remaining = f, remaining

    def read(self, n=-1):
        if self.remaining <= 0:
            return b""
        if n < 0 or n > self.remaining:
            n = self.remaining
        chunk = self.f.read(n)
        self.remaining -= len(chunk)
        return chunk

    def close(self):
        self.f.close()


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    root = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), "..", "web")
    root = os.path.abspath(root)
    handler = partial(RangeHandler, directory=root)
    print(f"serving {root} on http://localhost:{port}  (Range supported)")
    # Threaded, because a single-threaded server spends the whole clip transfer
    # unable to answer anything else — the page's own data request included.
    ThreadingHTTPServer(("", port), handler).serve_forever()


if __name__ == "__main__":
    main()
