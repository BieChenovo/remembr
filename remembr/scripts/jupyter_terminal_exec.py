#!/usr/bin/env python3
"""Execute one shell command through an unauthenticated Jupyter terminal API.

This is intended for the private HKUST Web GPU network, where the WebIDE
container is reachable only from a management node.  It creates a terminal,
sends one command, and waits for an explicit shell marker before returning.
"""

from __future__ import annotations

import argparse
import asyncio
import http.cookiejar
import json
import shlex
import urllib.request
import uuid

import websockets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url", help="Jupyter base URL, including its service prefix")
    parser.add_argument("command", help="Shell command to execute")
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


def create_terminal(base_url: str) -> tuple[str, str]:
    cookie_jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        urllib.request.HTTPCookieProcessor(cookie_jar),
    )
    # Loading JupyterLab establishes the anonymous-user and XSRF cookies.
    with opener.open(f"{base_url.rstrip('/')}/lab", timeout=10):
        pass
    xsrf = next(
        (cookie.value for cookie in cookie_jar if cookie.name == "_xsrf"),
        None,
    )
    if xsrf is None:
        raise RuntimeError("JupyterLab did not issue an _xsrf cookie")

    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/terminals",
        data=b"{}",
        headers={
            "Content-Type": "application/json",
            "X-XSRFToken": xsrf,
        },
        method="POST",
    )
    with opener.open(request, timeout=10) as response:
        name = json.load(response)["name"]
    cookie_header = "; ".join(
        f"{cookie.name}={cookie.value}" for cookie in cookie_jar
    )
    return name, cookie_header


async def execute(
    base_url: str,
    terminal_name: str,
    command: str,
    timeout: float,
    cookie_header: str,
) -> None:
    marker_prefix = "__CODEX_TERMINAL_DONE_"
    marker_id = uuid.uuid4().hex
    marker = f"{marker_prefix}{marker_id}__"
    # Keep the complete marker out of the echoed command line; otherwise a
    # terminal with input echo enabled could look complete before execution.
    marked_command = (
        f"{command}; command_rc=$?; "
        f"printf '\\n%s%s__:%s\\n' {shlex.quote(marker_prefix)} "
        f"{shlex.quote(marker_id)} \"$command_rc\""
    )
    scheme = "wss" if base_url.startswith("https://") else "ws"
    address = base_url.split("://", 1)[1].rstrip("/")
    websocket_url = f"{scheme}://{address}/terminals/websocket/{terminal_name}"

    async with websockets.connect(
        websocket_url,
        extra_headers={"Cookie": cookie_header},
        open_timeout=10,
        close_timeout=2,
        ping_interval=None,
        max_size=2**22,
    ) as websocket:
        await websocket.send(json.dumps(["set_size", 40, 160, 1600, 800]))
        await websocket.send(json.dumps(["stdin", marked_command + "\n"]))

        async def wait_for_marker() -> None:
            recent_output = ""
            while True:
                raw = await websocket.recv()
                message = json.loads(raw)
                if message and message[0] == "stdout":
                    output = message[1]
                    print(output, end="", flush=True)
                    recent_output = (recent_output + output)[-4096:]
                    if marker in recent_output:
                        return

        await asyncio.wait_for(wait_for_marker(), timeout=timeout)


def main() -> int:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    terminal_name, cookie_header = create_terminal(base_url)
    print(f"Created Jupyter terminal {shlex.quote(terminal_name)}", flush=True)
    asyncio.run(
        execute(base_url, terminal_name, args.command, args.timeout, cookie_header)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
