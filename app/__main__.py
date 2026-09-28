"""Entry point: `python -m app` starts the server and prints how to open it on this computer and a phone."""

from __future__ import annotations

import logging
import socket
import sys

import httpx
import uvicorn
from openai import AsyncOpenAI

from .config import load_settings
from .server import create_app


def lan_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("10.255.255.255", 1))  # no packet is sent; this only picks the outgoing interface
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def print_qr(url: str) -> None:
    try:
        import qrcode

        qr = qrcode.QRCode(border=1)
        qr.add_data(url)
        qr.print_ascii(invert=True)
    except Exception:
        pass  # consoles without Unicode block characters just skip the QR code


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("app").setLevel(logging.INFO)
    settings = load_settings()
    if not settings.openai_api_key:
        sys.exit("OPENAI_API_KEY is not set. Copy .env.example to .env and put your key in it.")
    client = AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url or None,
        max_retries=1,
        timeout=httpx.Timeout(60.0, connect=5.0),
    )
    app = create_app(settings, client)

    print(f"\n  GPT Associate is running.\n\n  On this computer:  http://localhost:{settings.port}")
    if settings.lan_access:
        url = f"http://{lan_ip()}:{settings.port}/?code={settings.access_code}"
        print(f"  On your phone:     {url}\n  Access code:       {settings.access_code}\n")
        print_qr(url)
    print("\n  Press Ctrl+C to stop.\n", flush=True)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="warning")


if __name__ == "__main__":
    main()
