#!/usr/bin/env python3
"""Execute one shell command through a Jupyter terminal API.

This is intended for the private HKUST Web GPU network, where the WebIDE
container is reachable only from a management node. It creates a terminal,
sends one command, waits for an explicit shell marker, then removes the
terminal. The implementation uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import base64
import http.cookiejar
import json
import os
import shlex
import socket
import ssl
import struct
import time
import urllib.parse
import urllib.request
import uuid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url", help="Jupyter base URL, including its service prefix")
    parser.add_argument("command", help="Shell command to execute")
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


class JupyterSession:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.cookies = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            urllib.request.HTTPCookieProcessor(self.cookies),
        )

    def request(self, path: str, method: str = "GET", body: bytes | None = None):
        request = urllib.request.Request(
            f"{self.base_url}/{path.lstrip('/')}", data=body, method=method
        )
        request.add_header("Content-Type", "application/json")
        xsrf = next(
            (cookie.value for cookie in self.cookies if cookie.name == "_xsrf"),
            None,
        )
        if xsrf is not None:
            request.add_header("X-XSRFToken", xsrf)
        return self.opener.open(request, timeout=10)

    def create_terminal(self) -> str:
        # Loading JupyterLab establishes the anonymous-user and XSRF cookies.
        self.request("lab").close()
        with self.request("api/terminals", "POST", b"{}") as response:
            return json.load(response)["name"]

    def delete_terminal(self, name: str) -> None:
        self.request(
            f"api/terminals/{urllib.parse.quote(name)}", "DELETE"
        ).close()

    def cookie_header(self) -> str:
        return "; ".join(
            f"{cookie.name}={cookie.value}" for cookie in self.cookies
        )


def recv_exact(connection: socket.socket, length: int) -> bytes:
    chunks = []
    while length:
        chunk = connection.recv(length)
        if not chunk:
            raise RuntimeError("WebSocket closed")
        chunks.append(chunk)
        length -= len(chunk)
    return b"".join(chunks)


def recv_frame(connection: socket.socket) -> tuple[int, bytes]:
    first, second = recv_exact(connection, 2)
    opcode = first & 0x0F
    length = second & 0x7F
    if length == 126:
        length = struct.unpack("!H", recv_exact(connection, 2))[0]
    elif length == 127:
        length = struct.unpack("!Q", recv_exact(connection, 8))[0]
    mask = recv_exact(connection, 4) if second & 0x80 else None
    payload = recv_exact(connection, length)
    if mask:
        payload = bytes(
            byte ^ mask[index % 4] for index, byte in enumerate(payload)
        )
    return opcode, payload


def send_text(connection: socket.socket, value: str) -> None:
    payload = value.encode()
    mask = os.urandom(4)
    header = bytearray([0x81])
    if len(payload) < 126:
        header.append(0x80 | len(payload))
    elif len(payload) < 65536:
        header.append(0x80 | 126)
        header.extend(struct.pack("!H", len(payload)))
    else:
        header.append(0x80 | 127)
        header.extend(struct.pack("!Q", len(payload)))
    masked = bytes(
        byte ^ mask[index % 4] for index, byte in enumerate(payload)
    )
    connection.sendall(bytes(header) + mask + masked)


def open_websocket(
    base_url: str, terminal_name: str, cookie_header: str
) -> socket.socket:
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    connection = socket.create_connection((parsed.hostname, port), timeout=10)
    if parsed.scheme == "https":
        connection = ssl.create_default_context().wrap_socket(
            connection, server_hostname=parsed.hostname
        )

    websocket_path = (
        parsed.path.rstrip("/")
        + "/terminals/websocket/"
        + urllib.parse.quote(terminal_name)
    )
    key = base64.b64encode(os.urandom(16)).decode()
    handshake = (
        f"GET {websocket_path} HTTP/1.1\r\n"
        f"Host: {parsed.netloc}\r\n"
        f"Origin: {parsed.scheme}://{parsed.netloc}\r\n"
        f"Cookie: {cookie_header}\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        "Sec-WebSocket-Version: 13\r\n\r\n"
    )
    connection.sendall(handshake.encode())
    response = b""
    while b"\r\n\r\n" not in response:
        response += connection.recv(4096)
    status_line = response.split(b"\r\n", 1)[0]
    if b" 101 " not in status_line:
        connection.close()
        raise RuntimeError(status_line.decode(errors="replace"))
    return connection


def execute(
    base_url: str,
    terminal_name: str,
    command: str,
    timeout: float,
    cookie_header: str,
) -> int:
    marker_prefix = "__CODEX_TERMINAL_DONE_"
    marker_id = uuid.uuid4().hex
    marker = f"{marker_prefix}{marker_id}__"
    marked_command = (
        f"{command}; command_rc=$?; "
        f"printf '\n%s%s__:%s\n' {shlex.quote(marker_prefix)} "
        f"{shlex.quote(marker_id)} \"$command_rc\""
    )
    connection = open_websocket(base_url, terminal_name, cookie_header)
    try:
        send_text(connection, json.dumps(["set_size", 40, 160, 1600, 800]))
        send_text(connection, json.dumps(["stdin", "stty -echo\r"]))
        connection.settimeout(0.5)
        while True:
            try:
                recv_frame(connection)
            except socket.timeout:
                break

        send_text(connection, json.dumps(["stdin", marked_command + "\r"]))
        connection.settimeout(min(timeout, 5.0))
        deadline = time.monotonic() + timeout
        recent_output = ""
        while time.monotonic() < deadline:
            try:
                opcode, payload = recv_frame(connection)
            except socket.timeout:
                continue
            if opcode == 8:
                break
            if opcode != 1:
                continue
            message = json.loads(payload.decode())
            if not message or message[0] != "stdout":
                continue
            output = message[1]
            print(output, end="", flush=True)
            recent_output = (recent_output + output)[-4096:]
            if marker in recent_output:
                suffix = recent_output.rsplit(marker + ":", 1)[-1]
                return int(suffix.splitlines()[0])
        raise TimeoutError(f"Command did not finish within {timeout:g} seconds")
    finally:
        connection.close()


def main() -> int:
    args = parse_args()
    session = JupyterSession(args.base_url)
    terminal_name = session.create_terminal()
    print(f"Created Jupyter terminal {shlex.quote(terminal_name)}", flush=True)
    try:
        return execute(
            session.base_url,
            terminal_name,
            args.command,
            args.timeout,
            session.cookie_header(),
        )
    finally:
        session.delete_terminal(terminal_name)


if __name__ == "__main__":
    raise SystemExit(main())
