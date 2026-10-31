"""Shared Orbital presentation assets; no manager or data side effects."""

from pathlib import Path


ASSET_DIR = Path(__file__).with_name("assets")
APP_STYLE = "<style>" + (ASSET_DIR / "orbital.css").read_text(encoding="utf-8") + "</style>"
TEAM_PITCH_STYLE = "<style>" + (ASSET_DIR / "orbital_pitch.css").read_text(encoding="utf-8") + "</style>"


def app_brand_html(*, sidebar: bool = False) -> str:
    modifier = " orbital-brand--sidebar" if sidebar else ""
    return (
        f'<div class="orbital-brand{modifier}">'
        '<span class="orbital-brand__mark" aria-hidden="true">◈</span>'
        '<span>FPL <span class="orbital-brand__slash">/</span> '
        '<strong>ORBITAL</strong></span></div>'
    )
