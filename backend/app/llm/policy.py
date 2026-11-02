"""Local-only inference policy.

With LLM_LOCAL_ONLY=true (the default) every configured LLM endpoint — the
answer/agent model and the RAGAS judge — must resolve to a loopback, private-network or
link-local address (e.g. localhost, the docker-compose `ollama` service, a LAN
machine). Anything else is refused at client construction with a clear error,
so no query or retrieved passage can leave the local network by
misconfiguration.
"""
from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse


class RemoteEndpointError(ValueError):
    pass


def _resolve(host: str) -> list[str]:
    infos = socket.getaddrinfo(host, None)
    return sorted({info[4][0] for info in infos})


def is_local_address(ip: str) -> bool:
    addr = ipaddress.ip_address(ip.split("%", 1)[0])
    return addr.is_loopback or addr.is_private or addr.is_link_local


def ensure_local_url(url: str, what: str = "LLM endpoint", resolver=_resolve) -> None:
    host = urlparse(url).hostname
    if not host:
        raise RemoteEndpointError(f"{what}: cannot parse a host from {url!r}")
    try:
        ips = [host] if _is_ip(host) else resolver(host)
    except OSError as exc:
        raise RemoteEndpointError(
            f"{what}: cannot resolve {host!r} to verify it is local ({exc.__class__.__name__}); "
            "set LLM_LOCAL_ONLY=false only if you intend to use a non-local endpoint"
        ) from exc
    remote = [ip for ip in ips if not is_local_address(ip)]
    if remote or not ips:
        raise RemoteEndpointError(
            f"{what}: {host!r} resolves to non-local address(es) {remote or ips}; "
            "LLM_LOCAL_ONLY=true allows only loopback/private-network endpoints"
        )


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False
