"""bigbrain — a hand-rolled semantic agent-memory store.

Vector-indexed by topic embedding (the fast key), with detailed content as the
retrieved knowledge. Backed by Milvus Lite and local embeddings.
"""

from .config import Config
from .models import Memory
from .store import MemoryStore

__all__ = ["Config", "Memory", "MemoryStore"]
__version__ = "0.1.0"
