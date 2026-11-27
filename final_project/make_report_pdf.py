"""report.md -> report.pdf (Moodle report and ISIC manuscript).

  python make_report_pdf.py

Markdown is rendered to HTML (package `markdown`), then printed to PDF by a headless
Edge or Chrome, which every Windows machine has. Re-run after editing report.md.
"""

import subprocess
from pathlib import Path

import markdown

import config

BROWSERS = [
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
]
CSS = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: "Segoe UI", Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; color: #191919; }
h1 { font-size: 19pt; margin: 0 0 6pt; }
h2 { font-size: 13.5pt; margin: 16pt 0 5pt; border-bottom: 1px solid #ddd; padding-bottom: 2pt; }
table { border-collapse: collapse; margin: 6pt 0 10pt; font-size: 9.5pt; page-break-inside: avoid; }
th, td { border: 1px solid #cfcfcf; padding: 3pt 6pt; vertical-align: top; }
th { background: #f3f3f3; text-align: left; }
code { font-family: Consolas, monospace; font-size: 9pt; background: #f5f5f5; padding: 0 2pt; }
img { max-width: 100%; display: block; margin: 6pt auto; page-break-inside: avoid; }
p, li { orphans: 3; widows: 3; }
"""


def main() -> None:
    src = config.HERE / "report.md"
    body = markdown.markdown(src.read_text(encoding="utf-8"), extensions=["tables"])
    html = (f'<!doctype html><html><head><meta charset="utf-8"><base href="{config.HERE.as_uri()}/">'
            f"<style>{CSS}</style></head><body>{body}</body></html>")
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    page = config.CACHE_DIR / "report.html"
    page.write_text(html, encoding="utf-8")
    out = config.HERE / "report.pdf"
    browser = next(b for b in BROWSERS if b.is_file())
    subprocess.run([str(browser), "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    "--print-to-pdf-no-header", f"--print-to-pdf={out}", page.as_uri()],
                   check=True, timeout=120, capture_output=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
