"""Content-addressed recovery-side index over verified teacher-cache batches."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def sha256_bytes(encoded: bytes) -> str:
    return hashlib.sha256(encoded).hexdigest()


class VerifiedTeacherCache:
    def __init__(
        self,
        verification_paths: Iterable[Path],
        teacher_revision: str,
        teacher_provenance_sha256: str,
    ) -> None:
        self.teacher_revision = teacher_revision
        self.teacher_provenance_sha256 = teacher_provenance_sha256
        self.batches: list[dict[str, Any]] = []
        self.artifacts: list[dict[str, Any]] = []
        seen_ids = set()
        for verification_path in verification_paths:
            encoded = verification_path.read_bytes()
            document = json.loads(encoded)
            if document.get("format") != "ctox.teacher-cache-verification.v1":
                raise ValueError(f"unsupported teacher verification {verification_path}")
            if document.get("status") != "passed":
                raise ValueError(f"teacher verification did not pass: {verification_path}")
            if document.get("teacher_revision") != teacher_revision:
                raise ValueError(f"teacher revision differs in {verification_path}")
            if document.get("teacher_provenance_sha256") != teacher_provenance_sha256:
                raise ValueError(f"teacher provenance differs in {verification_path}")
            cache = Path(document["cache"])
            local_artifacts = document.get("artifacts", [])
            if len(local_artifacts) != int(document.get("samples", -1)):
                raise ValueError(f"artifact count differs in {verification_path}")
            if sum(int(item["bytes"]) for item in local_artifacts) != int(
                document.get("artifact_bytes", -1)
            ):
                raise ValueError(f"artifact bytes differ in {verification_path}")
            batch_record = {
                "verification": str(verification_path.resolve()),
                "verification_bytes": len(encoded),
                "verification_sha256": sha256_bytes(encoded),
                "cache": str(cache.resolve()),
                "samples": len(local_artifacts),
                "artifact_bytes": int(document["artifact_bytes"]),
                "artifact_root_sha256": document["artifact_root_sha256"],
            }
            self.batches.append(batch_record)
            for item in local_artifacts:
                sample_id = str(item["id"])
                filename = str(item["file"])
                if sample_id in seen_ids:
                    raise ValueError(f"duplicate cached teacher sample {sample_id}")
                if Path(filename).name != filename or filename != f"{sample_id}.safetensors":
                    raise ValueError(f"unsafe cached artifact filename {filename}")
                seen_ids.add(sample_id)
                self.artifacts.append(
                    {
                        "id": sample_id,
                        "path": str((cache / filename).resolve()),
                        "bytes": int(item["bytes"]),
                        "sha256": str(item["sha256"]),
                        "batch_verification_sha256": batch_record["verification_sha256"],
                    }
                )
        if not self.artifacts:
            raise ValueError("verified teacher cache is empty")

    def verified_artifact_path(self, index: int) -> Path:
        artifact = self.artifacts[index]
        path = Path(artifact["path"])
        encoded = path.read_bytes()
        if len(encoded) != artifact["bytes"]:
            raise ValueError(f"cached teacher artifact changed size: {path}")
        if sha256_bytes(encoded) != artifact["sha256"]:
            raise ValueError(f"cached teacher artifact changed content: {path}")
        return path

    def manifest(self) -> dict[str, Any]:
        digest = hashlib.sha256()
        for artifact in self.artifacts:
            digest.update(artifact["id"].encode("ascii"))
            digest.update(bytes.fromhex(artifact["sha256"]))
        return {
            "format": "ctox.teacher-cache-set.v1",
            "teacher_revision": self.teacher_revision,
            "teacher_provenance_sha256": self.teacher_provenance_sha256,
            "batches": self.batches,
            "samples": len(self.artifacts),
            "artifact_bytes": sum(artifact["bytes"] for artifact in self.artifacts),
            "artifact_root_sha256": digest.hexdigest(),
        }
