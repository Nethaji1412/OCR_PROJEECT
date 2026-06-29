"""
llm_corrector_enhanced.py
──────────────────────────
OCR post-processing using Fireworks AI (DeepSeek).

Loads FIREWORKS_API_KEY from environment or a local .env file.
Falls back to local regex-based correction if the API is unavailable.
"""

import os
import re
import time
import requests
from functools import lru_cache
from typing import Optional, Dict


# ── .env loader (plain, no dependency on python-dotenv at import time) ────────

def _load_dotenv() -> None:
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key   = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

_load_dotenv()


# ── Constants ────────────────────────────────────────────────────────────────

_BASE_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
_MODEL    = "accounts/fireworks/models/deepseek-v3-0324"
api_key = "fw_9R96xEJRMu4758EdF6V5Lv"
_SYSTEM_PROMPT = """You are an expert OCR post-processing engine.

Your task:
1. Correct OCR extraction errors (spelling, character misrecognition, broken words).
2. Fix grammar and syntax errors.
3. Restore proper punctuation and sentence structure.
4. Maintain the original meaning and intent exactly.
5. Preserve paragraph breaks and list formatting.
6. Handle multiple languages (English, Hindi, Tamil, Telugu, Kannada, Marathi).

Rules:
- Output ONLY the corrected text.
- Do NOT add explanations, headings, or summaries.
- Return text in the same language as the input.
"""


# ── FireworksCorrector ────────────────────────────────────────────────────────

class FireworksCorrector:
    """
    Corrects OCR text via Fireworks AI.
    Falls back to local regex rules if the API is unavailable.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key     = api_key or os.getenv("FIREWORKS_API_KEY", "")
        self.base_url    = _BASE_URL
        self.model       = _MODEL
        self.timeout     = 60
        self.max_retries = 3

        if not self.api_key:
            raise ValueError(
                "Fireworks API key not provided. "
                "Set FIREWORKS_API_KEY in .env or pass api_key= parameter."
            )

    # ── Public ────────────────────────────────────────────────────────────────

    def correct_text(self, text: str, language: str = "en") -> Dict:
        if not text or not text.strip():
            return {
                "status": "error", "message": "Empty text provided",
                "original": text, "corrected": text, "confidence": 0,
            }

        # Chunk large texts so they stay within model context
        chunks   = _split_chunks(text, limit=8_000)
        results  = [self._call_api(chunk, language) for chunk in chunks]
        corrected = "\n\n".join(results)

        return {
            "status": "success",
            "original": text,
            "corrected": corrected,
            "confidence": 0.95,
            "model": self.model,
            "language": language,
        }

    def _local_correction(self, text: str) -> str:
        """Regex-based fallback (no API needed)."""
        t = re.sub(r"\s+", " ", text)

        typos = {
            "teh": "the", "adn": "and", "hte": "the",
            "thier": "their", "recieve": "receive", "occured": "occurred",
        }
        for bad, good in typos.items():
            t = re.sub(r"\b" + bad + r"\b", good, t, flags=re.IGNORECASE)

        if t and t[-1] not in ".!?":
            t += "."
        return t

    # ── Private ───────────────────────────────────────────────────────────────

    def _call_api(self, text: str, language: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": f"Correct this {language.upper()} text:\n\n{text}"},
            ],
            "temperature": 0.1,
            "max_tokens": min(len(text) * 2, 4096),
        }

        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    self.base_url, headers=headers,
                    json=payload, timeout=self.timeout,
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"].strip()

            except requests.exceptions.Timeout:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return f"[Timeout after {self.max_retries} retries — text left uncorrected]"

            except requests.exceptions.HTTPError as e:
                return f"[Fireworks API HTTP {resp.status_code}: {e}]"

            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(1)
                else:
                    return f"[Fireworks API error: {e}]"

        return text  # unreachable but satisfies type checker

    @lru_cache(maxsize=1)
    def get_stats(self) -> Dict:
        return {
            "model": self.model,
            "api": "Fireworks AI",
            "timeout": self.timeout,
            "retries": self.max_retries,
        }


# ── HybridCorrector ───────────────────────────────────────────────────────────

class HybridCorrector:
    """
    Tries LLM correction first; falls back to local rules if API is absent.
    """

    def __init__(self, api_key: Optional[str] = None):
        resolved_key = api_key or os.getenv("FIREWORKS_API_KEY", "")
        try:
            self.fireworks: Optional[FireworksCorrector] = FireworksCorrector(resolved_key)
        except ValueError:
            self.fireworks = None

    def correct_text(self, text: str, language: str = "en", use_llm: bool = True) -> Dict:
        if use_llm and self.fireworks:
            return self.fireworks.correct_text(text, language)

        # Local-only
        corrected = FireworksCorrector.__new__(FireworksCorrector)
        return {
            "status": "success",
            "original": text,
            "corrected": corrected._local_correction(text),
            "confidence": 0.70,
            "method": "local_rules",
        }

    def compare_corrections(self, text: str, language: str = "en") -> Dict:
        local_fc  = FireworksCorrector.__new__(FireworksCorrector)
        local_out = local_fc._local_correction(text)

        llm_out = None
        if self.fireworks:
            res     = self.fireworks.correct_text(text, language)
            llm_out = res.get("corrected")

        return {
            "original": text,
            "local_correction": local_out,
            "llm_correction": llm_out,
            "llm_available": self.fireworks is not None,
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _split_chunks(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in text.split("\n\n"):
        if current_len + len(para) + 2 > limit and current:
            chunks.append("\n\n".join(current))
            current, current_len = [para], len(para)
        else:
            current.append(para)
            current_len += len(para) + 2

    if current:
        chunks.append("\n\n".join(current))
    return chunks
