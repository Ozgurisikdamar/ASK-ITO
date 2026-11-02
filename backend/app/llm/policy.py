"""Local-only inference policy.

With LLM_LOCAL_ONLY=true (the default) every configured LLM endpoint — the
answer/agent model and the RAGAS judge — must be **loopback**:

  * scheme http or https, no credentials in the URL, a valid port;
  * the host is `localhost`, an address in 127.0.0.0/8, `::1` or an
    IPv4-mapped loopback address, or a hostname that resolves *only* to such
    addresses. Private-network (RFC 1918 / ULA), link-local, wildcard,
    multicast, reserved and public addresses are refused, and so is a hostname
    whose addresses mix loopback with anything else.

Explicit exception for container networks: hostnames listed in
LLM_ALLOWED_HOSTS (exact, case-insensitive; empty by default) may resolve to
private-network addresses (RFC 1918 / IPv6 ULA) — never to public, link-local,
wildcard or multicast addresses. docker-compose lists only its `ollama`
service there.

The check runs when a client is built and again on every outgoing request (an
httpx request hook), and the HTTP client never follows redirects. Residual risk:
DNS is resolved by this check and again by the HTTP stack, so a hostname whose
DNS answer changes between the two (DNS rebinding) is not fully excluded; use
IP literals or /etc/hosts entries for the strictest setup.
"""
from __future__ import annotations

import ipaddress
import socket
from typing import Callable, Iterable
from urllib.parse import urlsplit

Resolver = Callable[[str], list[str]]


class RemoteEndpointError(ValueError):
    pass


def _resolve(host: str) -> list[str]:
    infos = socket.getaddrinfo(host, None)
    return sorted({info[4][0] for info in infos})


def _ip(value: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    addr = ipaddress.ip_address(value.split("%", 1)[0])
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        return addr.ipv4_mapped
    return addr


def is_loopback(value: str) -> bool:
    try:
        return _ip(value).is_loopback
    except ValueError:
        return False


def is_private_network(value: str) -> bool:
    """RFC 1918 / IPv6 ULA (container bridge networks). Excludes loopback,
    link-local, wildcard, multicast and reserved ranges."""
    try:
        addr = _ip(value)
    except ValueError:
        return False
    if addr.is_loopback or addr.is_link_local or addr.is_unspecified or addr.is_multicast \
            or addr.is_reserved:
        return False
    if isinstance(addr, ipaddress.IPv4Address):
        return any(addr in ipaddress.ip_network(n)
                   for n in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"))
    return addr in ipaddress.ip_network("fc00::/7")


def _is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host.split("%", 1)[0])
        return True
    except ValueError:
        return False


def ensure_local_url(url: str, what: str = "LLM endpoint",
                     allowed_hosts: Iterable[str] = (),
                     resolver: Resolver = _resolve) -> None:
    """Raise RemoteEndpointError unless `url` is a loopback endpoint (or an
    explicitly allowed container hostname resolving to private addresses)."""
    try:
        parts = urlsplit(url)
        port = parts.port  # raises ValueError on an invalid port
    except ValueError as exc:
        raise RemoteEndpointError(f"{what}: malformed URL {url!r} ({exc})") from exc
    if parts.scheme not in ("http", "https"):
        raise RemoteEndpointError(f"{what}: scheme must be http or https, got {parts.scheme!r}")
    if parts.username is not None or parts.password is not None:
        raise RemoteEndpointError(f"{what}: credentials in the URL are not allowed")
    host = (parts.hostname or "").lower()
    if not host:
        raise RemoteEndpointError(f"{what}: cannot parse a host from {url!r}")
    if port is not None and not 0 < port < 65536:
        raise RemoteEndpointError(f"{what}: invalid port {port}")

    allowed = {h.strip().lower() for h in allowed_hosts if h and h.strip()}
    if _is_ip_literal(host):
        ips = [host]
    else:
        try:
            ips = resolver(host)
        except OSError as exc:
            raise RemoteEndpointError(
                f"{what}: cannot resolve {host!r} to verify it is local "
                f"({exc.__class__.__name__})") from exc
    if not ips:
        raise RemoteEndpointError(f"{what}: {host!r} did not resolve to any address")

    if all(is_loopback(ip) for ip in ips):
        return
    if host in allowed and not _is_ip_literal(host) and all(is_private_network(ip) for ip in ips):
        return
    if host in allowed:
        raise RemoteEndpointError(
            f"{what}: allowed host {host!r} must resolve only to private-network "
            f"addresses, got {ips}")
    raise RemoteEndpointError(
        f"{what}: {host!r} resolves to {ips}; LLM_LOCAL_ONLY=true allows only loopback "
        "endpoints (add a container hostname to LLM_ALLOWED_HOSTS to trust a private "
        "network explicitly)")


def allowed_hosts_from_setting(value: str) -> list[str]:
    return [h.strip() for h in (value or "").split(",") if h.strip()]
