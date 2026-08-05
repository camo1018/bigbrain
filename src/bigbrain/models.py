"""Data model for a single stored memory."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field


def now_ts() -> int:
    return int(time.time())


@dataclass
class Memory:
    """One unit of knowledge.

    `topic` is the short semantic key that gets embedded; `content` holds the
    detailed knowledge. Scoring fields (`similarity`, `score`) are only populated
    on recall results and are never persisted.
    """

    id: int
    topic: str
    content: str
    tags: list[str] = field(default_factory=list)
    source: str = ""
    importance: float = 0.5
    created_at: int = field(default_factory=now_ts)
    updated_at: int = field(default_factory=now_ts)
    access_count: int = 0
    last_accessed: int = 0

    # Populated only on recall; excluded from persistence.
    similarity: float | None = None
    score: float | None = None

    _PERSISTED_FIELDS = (
        "id",
        "topic",
        "content",
        "tags",
        "source",
        "importance",
        "created_at",
        "updated_at",
        "access_count",
        "last_accessed",
    )

    def to_row(self, vector: list[float]) -> dict:
        row = {k: getattr(self, k) for k in self._PERSISTED_FIELDS}
        row["vector"] = vector
        return row

    @classmethod
    def from_entity(cls, entity: dict) -> "Memory":
        return cls(
            id=int(entity["id"]),
            topic=entity.get("topic", ""),
            content=entity.get("content", ""),
            tags=list(entity.get("tags") or []),
            source=entity.get("source", ""),
            importance=float(entity.get("importance", 0.5)),
            created_at=int(entity.get("created_at", 0)),
            updated_at=int(entity.get("updated_at", 0)),
            access_count=int(entity.get("access_count", 0)),
            last_accessed=int(entity.get("last_accessed", 0)),
        )

    def to_dict(self) -> dict:
        data = asdict(self)
        for key in ("similarity", "score"):
            if data.get(key) is None:
                data.pop(key, None)
        return data
