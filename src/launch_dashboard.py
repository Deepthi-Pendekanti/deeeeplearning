"""
Launch the hackathon dashboard.

Browsers block fetch() of local files over file://, so we serve the results/
folder over a tiny local HTTP server and open the dashboard automatically.

Usage (from anywhere):
    python src/launch_dashboard.py
Then press Ctrl+C to stop.
"""
import http.server
import socketserver
import webbrowser
from functools import partial
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"
PORT = 8000


def main():
    if not (RESULTS / "dashboard.html").exists():
        print("dashboard.html not found; nothing to serve.")
        return
    if not (RESULTS / "demo_data.json").exists():
        print("demo_data.json missing — run: python src/phase12_export_demo.py")
        return
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(RESULTS))
    with socketserver.TCPServer(("127.0.0.1", PORT), handler) as httpd:
        url = f"http://127.0.0.1:{PORT}/dashboard.html"
        print(f"Serving {RESULTS} at {url}")
        print("Press Ctrl+C to stop.")
        try:
            webbrowser.open(url)
        except Exception:
            pass
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped.")


if __name__ == "__main__":
    main()
