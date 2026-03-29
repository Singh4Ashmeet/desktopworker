from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urlparse

from config import SidConfig


@dataclass(slots=True)
class NetworkDecision:
    allowed: bool
    reason: str


class NetworkFirewall:
    def __init__(self, config: SidConfig) -> None:
        self.config = config

    def check_url(self, url: str, purpose: str = "runtime") -> NetworkDecision:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        if not host:
            return NetworkDecision(False, "Blocked: URL host missing.")
        if host in {item.lower() for item in self.config.permitted_runtime_hosts}:
            return NetworkDecision(True, "Allowed local runtime host.")
        if purpose == "explicit_web_fetch":
            return NetworkDecision(True, "Allowed explicit user-requested web fetch.")
        if purpose == "model_download" and self.config.model_download_consent:
            return NetworkDecision(True, "Allowed user-consented model download.")
        if host in {item.lower() for item in self.config.explicit_fetch_hosts}:
            return NetworkDecision(True, "Allowed explicit host.")
        return NetworkDecision(False, f"Blocked outbound network call to {host}.")

    def filter_urls(self, urls: Iterable[str], purpose: str = "runtime") -> list[str]:
        allowed: list[str] = []
        for url in urls:
            decision = self.check_url(url, purpose=purpose)
            if decision.allowed:
                allowed.append(url)
        return allowed


__all__ = ["NetworkFirewall", "NetworkDecision"]
