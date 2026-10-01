"""Central configuration: paths, seeds, model choices.

Everything that could make a run non-reproducible lives here and is echoed into
every artifact manifest. `GLOBAL_SEED` is the single root of all randomness in
the project; modules derive sub-seeds from it rather than inventing their own.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]

GLOBAL_SEED = 20250911
"""Root seed. Never change: every committed artifact was produced under it."""


def seed_for(purpose: str) -> int:
    """Derive a stable sub-seed from GLOBAL_SEED and a purpose string.

    Using one seed everywhere silently correlates the golden-set sample with the
    train/test split. Deriving per purpose keeps them independent *and*
    reproducible, and the purpose string lands in the manifest.
    """
    h = hashlib.sha256(f"{GLOBAL_SEED}:{purpose}".encode()).digest()
    return int.from_bytes(h[:4], "big")


class Paths:
    raw = ROOT / "data" / "raw" / "twcs" / "twcs.csv"
    interim = ROOT / "data" / "interim"
    processed = ROOT / "data" / "processed"
    golden = ROOT / "data" / "golden"
    cache = ROOT / ".cache" / "llm"
    runs = ROOT / "runs"
    prompts = Path(__file__).resolve().parent / "prompts"


def load_prompt(name: str) -> str:
    """A versioned prompt from `src/support_agent/prompts/<name>.txt`.

    Header lines starting with `#` are dropped and the rest is stripped, so the
    header can document the prompt without entering the request. The text is part
    of every cache key: edit a prompt by adding a new version, never in place.
    """
    text = (Paths.prompts / f"{name}.txt").read_text(encoding="utf-8").replace("\r\n", "\n")
    return "\n".join(ln for ln in text.splitlines() if not ln.startswith("#")).strip()


class Settings(BaseSettings):
    """Runtime settings. Secrets come from `.env`, never from code."""

    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    brand: str = "AmericanAir"

    gemini_api_key: str = ""
    mistral_api_key: str = ""
    nvidia_api_key: str = ""
    cerebras_api_key: str = ""
    openrouter_api_key: str = ""

    # Hard ceiling per process. A runaway loop over a 2.8M-row dataset is the
    # realistic way this project burns a quota; the adapter refuses past this.
    llm_max_calls_per_run: int = 1200


settings = Settings()
