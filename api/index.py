"""Vercel entrypoint that reuses the local GraphCite GCN FastAPI app."""

from main import app

__all__ = ["app"]
