"""
translation_engine.py
──────────────────────
Multilingual translation using argostranslate (offline, no ctranslate2).

WHY THIS REPLACES ctranslate2:
  ctranslate2 fails on Streamlit Cloud with:
    ImportError: libctranslate2.so.X: cannot enable executable stack
  This is a kernel-level seccomp restriction on Streamlit Cloud's sandbox —
  ctranslate2's shared library requires executable stack memory, which the
  cloud environment blocks. It cannot be fixed by reinstalling.

  argostranslate is the correct alternative:
    - Pure Python + PyTorch backend (no native .so with exec-stack)
    - Works fine on Streamlit Cloud
    - Downloads language packs on demand
    - Covers the same language pairs

SUPPORTED LANGUAGES (install packs from sidebar):
  Arabic, Chinese, French, German, Hindi, Italian, Japanese,
  Korean, Portuguese, Russian, Spanish, Tamil, Turkish, Urdu
"""

from __future__ import annotations
from typing import Optional


# ── Language metadata ─────────────────────────────────────────────────────────

LANGUAGE_MAP: dict[str, str] = {
    "ar": "Arabic",
    "zh": "Chinese",
    "fr": "French",
    "de": "German",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "ru": "Russian",
    "es": "Spanish",
    "ta": "Tamil",
    "tr": "Turkish",
    "ur": "Urdu",
    "en": "English",
}

# Codes used in the original app (PaddleOCR style → argos style)
PADDLE_TO_ARGOS: dict[str, str] = {
    "en": "en",
    "hi": "hi",
    "ta": "ta",
    "te": "te",   # Telugu — may not have an argos pack; falls back gracefully
    "ka": "ka",   # Kannada — same
    "mr": "mr",   # Marathi — same
}


# ── Public API ────────────────────────────────────────────────────────────────

class TranslationEngine:
    """
    Drop-in replacement for the old ctranslate2-based TranslationEngine.
    Uses argostranslate under the hood.
    """

    def get_supported_languages(self) -> dict[str, str]:
        return LANGUAGE_MAP

    def detect_language(self, text: str) -> str:
        """Heuristic language detection by Unicode block."""
        if not text:
            return "en"

        counts: dict[str, int] = {
            "hi": 0, "ta": 0, "te": 0, "ka": 0,
        }
        for ch in text:
            cp = ord(ch)
            if 0x0900 <= cp <= 0x097F:
                counts["hi"] += 1
            elif 0x0B80 <= cp <= 0x0BFF:
                counts["ta"] += 1
            elif 0x0C00 <= cp <= 0x0C7F:
                counts["te"] += 1
            elif 0x0C80 <= cp <= 0x0CFF:
                counts["ka"] += 1

        total = len(text)
        for lang, cnt in counts.items():
            if cnt > total * 0.25:
                return lang
        return "en"

    def translate(
        self,
        text: str,
        source_lang: str = "en",
        target_lang: str = "hi",
    ) -> str:
        """
        Translate *text* from *source_lang* to *target_lang*.

        Returns the original text if no pack is installed or if
        source == target.
        """
        if source_lang == target_lang or not text.strip():
            return text
        return translate_text(text, source_lang, target_lang)

    def translate_batch(
        self,
        texts: list[str],
        source_lang: str = "en",
        target_lang: str = "hi",
    ) -> list[str]:
        return [self.translate(t, source_lang, target_lang) for t in texts]


# ── Standalone helpers (used by app.py directly) ──────────────────────────────

def get_available_languages() -> list[dict]:
    """
    Return installed language packs that can be translated *to* from English.
    Each item: {"code": "es", "name": "Spanish"}
    """
    try:
        from argostranslate import translate
        installed = translate.get_installed_languages()
        installed_codes = {lang.code for lang in installed}

        result = []
        if "en" not in installed_codes:
            return result

        en_obj = _get_lang_obj("en", installed)
        if en_obj is None:
            return result

        for code, name in LANGUAGE_MAP.items():
            if code == "en" or code not in installed_codes:
                continue
            tgt = _get_lang_obj(code, installed)
            if tgt and en_obj.get_translation(tgt):
                result.append({"code": code, "name": name})

        return result
    except ImportError:
        return []
    except Exception:
        return []


def install_language_pair(from_code: str, to_code: str) -> tuple[bool, str]:
    """Download and install an argostranslate pack."""
    try:
        from argostranslate import package

        package.update_package_index()
        available = package.get_available_packages()

        target = next(
            (p for p in available if p.from_code == from_code and p.to_code == to_code),
            None,
        )

        if target is None:
            return False, (
                f"No Argos pack found for {from_code}→{to_code}. "
                "See https://www.argosopentech.com/argospm/index/"
            )

        package.install_from_path(target.download())
        return True, f"✅ Installed {from_code}→{to_code} language pack."

    except ImportError:
        return False, "argostranslate not installed. Run: pip install argostranslate"
    except Exception as e:
        return False, f"Installation failed: {e}"


def translate_text(text: str, from_code: str, to_code: str) -> str:
    """Translate paragraph-by-paragraph to keep memory low."""
    if not text.strip() or from_code == to_code:
        return text

    try:
        from argostranslate import translate

        installed = translate.get_installed_languages()
        src = _get_lang_obj(from_code, installed)
        tgt = _get_lang_obj(to_code, installed)

        if src is None:
            return f"[Source language '{from_code}' not installed.]"
        if tgt is None:
            return f"[Target language '{to_code}' not installed. Install from sidebar.]"

        tr_obj = src.get_translation(tgt)
        if tr_obj is None:
            return (
                f"[No translation pack for {from_code}→{to_code}. "
                "Install it from the sidebar.]"
            )

        paragraphs = text.split("\n\n")
        return "\n\n".join(
            tr_obj.translate(p) if p.strip() else p for p in paragraphs
        )

    except ImportError:
        return "[argostranslate not installed. Run: pip install argostranslate]"
    except Exception as e:
        return f"[Translation error: {e}]"


# ── Internal ──────────────────────────────────────────────────────────────────

def _get_lang_obj(code: str, installed_langs):
    return next((l for l in installed_langs if l.code == code), None)
