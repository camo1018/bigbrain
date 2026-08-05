"""One-time model download helper.

Behind a TLS-intercepting network the interceptor's CA may not be in Python's
trust store, which breaks the initial HuggingFace model download. `insecure=True`
opts into an unverified client for that single fetch. Prefer pointing
`SSL_CERT_FILE`/`REQUESTS_CA_BUNDLE` at the CA instead; runtime embedding loads
the cached model from disk and performs no network I/O either way.
"""

from __future__ import annotations

from .config import Config
from .embeddings import Embedder


def _install_insecure_hf_backend() -> None:
    import httpx
    from huggingface_hub.utils._http import hf_request_event_hook, set_client_factory

    def factory() -> httpx.Client:
        return httpx.Client(
            event_hooks={"request": [hf_request_event_hook]},
            follow_redirects=True,
            timeout=None,
            verify=False,
        )

    set_client_factory(factory)


def download_model(config: Config | None = None, *, insecure: bool = True) -> str:
    """Download and cache the embedding model. Returns the resolved model name."""
    config = config or Config.from_env()
    if insecure:
        _install_insecure_hf_backend()
    embedder = Embedder(config.embed_model)
    # Triggers the actual download + a real embedding to validate the pipeline.
    vec = embedder.embed_one("bigbrain model warmup")
    if len(vec) != config.embed_dim:
        raise RuntimeError(
            f"Model produced dim {len(vec)}, expected {config.embed_dim}. "
            "Set BIGBRAIN_EMBED_DIM to match the model."
        )
    return config.embed_model
