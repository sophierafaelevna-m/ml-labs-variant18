"""Model registry and atomic version switching."""

from __future__ import annotations

import json
import os
import threading
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from src.lab6.model_loader import LoadedModel, load_model

ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = ROOT / "reports/LAB6/registry.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ModelRegistry:
    def __init__(self, path: Path = REGISTRY_PATH):
        self.path = Path(path)
        self.lock = threading.RLock()

        self.data = json.loads(self.path.read_text(encoding="utf-8"))

        self.models: dict[str, LoadedModel] = {}

        for version, metadata in self.data["versions"].items():
            model_path = (ROOT / metadata["path"]).resolve()

            # Model files must remain inside the project.
            if not model_path.is_relative_to(ROOT.resolve()):
                raise ValueError("Model path is outside the project")

            self.models[version] = load_model(
                path=model_path,
                version=version,
                expected_sha256=metadata["sha256"],
            )

        self.active_version = self.data["active_version"]

        if self.active_version not in self.models:
            raise ValueError("Active model is not registered")

        working = [
            version
            for version, metadata in self.data["versions"].items()
            if metadata["status"] == "working"
        ]

        if working != [self.active_version]:
            raise ValueError("Registry must contain exactly one working model")

        self.active_model = self.models[self.active_version]

    def _save(self, data: dict) -> None:
        """Write registry atomically to avoid partial JSON files."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".json.tmp")

        try:
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(
                    data,
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
                handle.flush()
                os.fsync(handle.fileno())

            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def get_active_model(self) -> LoadedModel:
        """Return one stable model reference for a request."""
        with self.lock:
            return self.active_model

    def get_registry(self) -> dict:
        with self.lock:
            return deepcopy(self.data)

    def promote(self, version: str, reason: str) -> dict:
        with self.lock:
            if version not in self.models:
                raise KeyError(f"Unknown model version: {version}")

            if version == self.active_version:
                raise ValueError("Model is already active")

            if self.data["versions"][version]["status"] != "canary":
                raise ValueError("Only a canary model can be promoted")

            previous = self.active_version
            updated = deepcopy(self.data)

            updated["versions"][previous]["status"] = "retired"
            updated["versions"][version]["status"] = "working"
            updated["active_version"] = version
            updated["rollback_target"] = previous

            updated["history"].append(
                {
                    "timestamp": utc_now(),
                    "action": "promote",
                    "version": version,
                    "previous_version": previous,
                    "reason": reason,
                }
            )

            # Persist first; publish the new pointer only on success.
            self._save(updated)

            self.data = updated
            self.active_version = version
            self.active_model = self.models[version]

            return {
                "action": "promote",
                "previous_version": previous,
                "active_version": version,
                "message": "Model promoted without service restart",
            }

    def rollback(self) -> dict:
        with self.lock:
            previous = self.data.get("rollback_target")

            if not previous or previous not in self.models:
                raise ValueError("No rollback target available")

            current = self.active_version
            updated = deepcopy(self.data)

            updated["versions"][current]["status"] = "canary"
            updated["versions"][previous]["status"] = "working"
            updated["active_version"] = previous
            updated["rollback_target"] = None

            updated["history"].append(
                {
                    "timestamp": utc_now(),
                    "action": "rollback",
                    "version": previous,
                    "previous_version": current,
                    "reason": "Restore previous working model",
                }
            )

            self._save(updated)

            self.data = updated
            self.active_version = previous
            self.active_model = self.models[previous]

            return {
                "action": "rollback",
                "previous_version": current,
                "active_version": previous,
                "message": "Previous model restored without restart",
            }
