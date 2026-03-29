from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import httpx
from tqdm import tqdm  # type: ignore

from config import SidConfig
from network_guard import NetworkFirewall


@dataclass(slots=True)
class DownloadSpec:
    name: str
    url: str
    destination: Path
    sha256: str = ""


class ModelDownloadManager:
    def __init__(self, config: SidConfig) -> None:
        self.config = config
        self.firewall = NetworkFirewall(config)

    async def ensure_file(self, spec: DownloadSpec) -> Path:
        spec.destination.parent.mkdir(parents=True, exist_ok=True)
        if spec.destination.exists() and (
            not spec.sha256 or self._verify_sha256(spec.destination, spec.sha256)
        ):
            return spec.destination
        decision = self.firewall.check_url(spec.url, purpose="model_download")
        if not decision.allowed:
            raise RuntimeError(decision.reason)
        if not self.config.allow_model_downloads:
            raise RuntimeError("Model downloads are disabled in config.")
        await self._download(spec)
        if spec.sha256 and not self._verify_sha256(spec.destination, spec.sha256):
            raise RuntimeError(f"Checksum mismatch for {spec.name}")
        return spec.destination

    async def ensure_hf_snapshot(self, repo_id: str, local_dir: Path) -> Path:
        if local_dir.exists() and any(local_dir.iterdir()):
            return local_dir
        if (
            not self.config.allow_model_downloads
            or not self.config.model_download_consent
        ):
            raise RuntimeError(
                f"Missing local model {repo_id} and downloads are not permitted."
            )

        def _download_snapshot() -> str:
            try:
                from huggingface_hub import snapshot_download  # type: ignore
            except Exception as exc:
                raise RuntimeError(
                    "huggingface_hub is required for model downloads"
                ) from exc
            return snapshot_download(
                repo_id=repo_id, local_dir=str(local_dir), local_dir_use_symlinks=False
            )

        await asyncio.to_thread(_download_snapshot)
        return local_dir

    async def _download(self, spec: DownloadSpec) -> None:
        """Download file with progress bar."""
        logging.info("Downloading model asset: %s", spec.name)

        async with httpx.AsyncClient(timeout=None) as client:
            # First, get the total file size
            head_response = await client.head(spec.url)
            head_response.raise_for_status()
            total_size = int(head_response.headers.get("content-length", 0))

            # Download with progress
            progress_bar = tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=f"Downloading {spec.name}",
            )

            try:
                async with client.stream("GET", spec.url) as response:
                    response.raise_for_status()
                    with spec.destination.open("wb") as handle:
                        async for chunk in response.aiter_bytes():
                            if chunk:
                                handle.write(chunk)
                                progress_bar.update(len(chunk))
            finally:
                progress_bar.close()

    def _verify_sha256(self, path: Path, expected: str) -> bool:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return digest.lower() == expected.lower()


__all__ = ["DownloadSpec", "ModelDownloadManager"]
