import numpy as np
import os
import logging
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from src.utils.logAndPrint import logAndPrint


class PreviewHTTPHandler(SimpleHTTPRequestHandler):
    previewPath = None
    autoRefresh = False
    refreshInterval = 1000

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.end_headers()

            autoRefreshEnabled = "true" if self.autoRefresh else "false"

            htmlContent = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>TAS Preview</title>
                <style>
                    body {{
                        margin: 0;
                        padding: 0;
                        background-color: #000000;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        min-height: 100vh;
                        overflow: hidden;
                    }}
                    img {{
                        max-width: 100vw;
                        max-height: 100vh;
                        width: auto;
                        height: auto;
                        display: block;
                    }}
                    .toggleContainer {{
                        position: fixed;
                        top: 20px;
                        right: 20px;
                        z-index: 1000;
                    }}
                    .switch {{
                        position: relative;
                        display: inline-block;
                        width: 50px;
                        height: 28px;
                    }}
                    .switch input {{
                        opacity: 0;
                        width: 0;
                        height: 0;
                    }}
                    .slider {{
                        position: absolute;
                        cursor: pointer;
                        top: 0;
                        left: 0;
                        right: 0;
                        bottom: 0;
                        background-color: #ccc;
                        transition: .4s;
                        border-radius: 28px;
                    }}
                    .slider:before {{
                        position: absolute;
                        content: "";
                        height: 20px;
                        width: 20px;
                        left: 4px;
                        bottom: 4px;
                        background-color: white;
                        transition: .4s;
                        border-radius: 50%;
                    }}
                    input:checked + .slider {{
                        background-color: #2196F3;
                    }}
                    input:checked + .slider:before {{
                        transform: translateX(22px);
                    }}
                </style>
            </head>
            <body>
                <div class="toggleContainer">
                    <label class="switch">
                        <input type="checkbox" id="autoRefreshToggle" checked="{autoRefreshEnabled}">
                        <span class="slider"></span>
                    </label>
                </div>
                <img id="preview" src="/image" alt="Preview">
                <script>
                    let autoRefresh = {autoRefreshEnabled};
                    let refreshInterval = {self.refreshInterval};
                    let intervalId = null;

                    function startAutoRefresh() {{
                        if (intervalId) clearInterval(intervalId);
                        intervalId = setInterval(function() {{
                            document.getElementById('preview').src = '/image?' + new Date().getTime();
                        }}, refreshInterval);
                    }}

                    function stopAutoRefresh() {{
                        if (intervalId) {{
                            clearInterval(intervalId);
                            intervalId = null;
                        }}
                    }}

                    document.getElementById('autoRefreshToggle').addEventListener('change', function() {{
                        autoRefresh = this.checked;
                        if (autoRefresh) {{
                            startAutoRefresh();
                        }} else {{
                            stopAutoRefresh();
                        }}
                    }});

                    if (autoRefresh) {{
                        startAutoRefresh();
                    }}
                </script>
            </body>
            </html>
            """
            self.wfile.write(htmlContent.encode())
        elif self.path.startswith("/image"):
            if self.previewPath and os.path.exists(self.previewPath):
                try:
                    with open(self.previewPath, "rb") as f:
                        content = f.read()

                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", len(content))
                    self.send_header(
                        "Cache-Control", "no-cache, no-store, must-revalidate"
                    )
                    self.send_header("Pragma", "no-cache")
                    self.send_header("Expires", "0")
                    self.end_headers()
                    self.wfile.write(content)
                except Exception as e:
                    self.send_error(500, f"Error reading preview: {e}")
            else:
                self.send_error(404, "Preview not available yet")
        else:
            self.send_error(404, "Not found")

    def log_message(self, format, *args):
        pass


class Preview:
    def __init__(
        self,
        localHost: str = "127.0.0.1",
        port: int = 5000,
        previewPath: str = None,
        autoRefresh: bool = True,
        refreshInterval: int = 1000,
    ) -> None:
        self.localHost = localHost
        self.port = port
        self.previewPath = previewPath
        self.autoRefresh = autoRefresh
        self.refreshInterval = refreshInterval
        self.server = None
        self.serverThread = None

        PreviewHTTPHandler.previewPath = previewPath
        PreviewHTTPHandler.autoRefresh = autoRefresh
        PreviewHTTPHandler.refreshInterval = refreshInterval

    def start(self) -> None:
        try:
            self.server = HTTPServer((self.localHost, self.port), PreviewHTTPHandler)

            logAndPrint(
                f"Preview URL: http://{self.localHost}:{self.port}/",
                "green",
            )

            self.serverThread = threading.Thread(
                target=self.server.serve_forever, daemon=True
            )
            self.serverThread.start()

        except Exception as e:
            logging.error(f"Preview server error: {e}")

    def close(self) -> None:
        if self.server:
            try:
                self.server.shutdown()
                self.server.server_close()
            except Exception as e:
                logging.warning(f"Error stopping preview server: {e}")

        if self.serverThread and self.serverThread.is_alive():
            self.serverThread.join(timeout=2)

        if self.previewPath and os.path.exists(self.previewPath):
            try:
                os.remove(self.previewPath)
            except Exception:
                pass

        logAndPrint("Preview server stopped", "green")

    def add(self, frame: np.ndarray) -> None:
        pass
