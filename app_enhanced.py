"""
app_enhanced.py
──────────────
Advanced OCR System with LLM Correction and Multilingual Translation.

Pipeline: Upload → Extract (PyMuPDF) → Correct (Fireworks AI) → Translate (Argos)
"""

import os
import json
from datetime import datetime

import streamlit as st

# ── Custom modules ────────────────────────────────────────────────────────────
from ocr_engine_enhanced   import EnhancedOCREngine, TextBlock
from llm_corrector_enhanced import FireworksCorrector, HybridCorrector
from translation_engine    import (
    TranslationEngine,
    get_available_languages,
    install_language_pair,
    translate_text,
)
from file_handler_enhanced import FileHandler

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Advanced OCR System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Language maps (Paddle-style codes) ───────────────────────────────────────
DISPLAY_LANGS = {
    "English": "en", "Hindi": "hi", "Tamil": "ta",
    "Telugu": "te", "Kannada": "ka", "Marathi": "mr",
}

# ── Session state defaults ────────────────────────────────────────────────────
for key, default in [
    ("ocr_result", None),
    ("corrected_text", None),
    ("translated_text", None),
    ("processing_log", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.title("⚙️ Configuration")

# API key
with st.sidebar.expander("🔑 API Keys", expanded=False):
    fireworks_key = st.text_input(
        "Fireworks API Key",
        type="password",
        value=os.getenv("FIREWORKS_API_KEY", ""),
        help="Get from https://fireworks.ai",
    )
    if fireworks_key:
        os.environ["FIREWORKS_API_KEY"] = fireworks_key

st.sidebar.subheader("🔤 OCR Languages")
ocr_languages = st.sidebar.multiselect(
    "Select OCR Languages",
    options=list(DISPLAY_LANGS.keys()),
    default=["English"],
    help="First language used as PaddleOCR primary (images only)",
)
ocr_langs = [DISPLAY_LANGS[l] for l in ocr_languages] if ocr_languages else ["en"]

st.sidebar.subheader("⚡ Processing Features")
enable_llm_correction = st.sidebar.checkbox("LLM Correction (Fireworks)", value=True)
enable_translation    = st.sidebar.checkbox("Translation (Argos Translate)", value=False)

translation_target_code = None
if enable_translation:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🌐 Translation Settings")

    # Install language packs
    with st.sidebar.expander("⬇️ Install Language Packs", expanded=False):
        all_target_langs = [
            "Arabic", "Chinese", "French", "German", "Hindi",
            "Italian", "Japanese", "Korean", "Portuguese",
            "Russian", "Spanish", "Tamil", "Turkish", "Urdu",
        ]
        install_lang = st.selectbox("Language to install", all_target_langs)
        # Map display name → argos 2-letter code (first 2 chars, lowercase)
        install_code_map = {
            "Arabic": "ar", "Chinese": "zh", "French": "fr", "German": "de",
            "Hindi": "hi", "Italian": "it", "Japanese": "ja", "Korean": "ko",
            "Portuguese": "pt", "Russian": "ru", "Spanish": "es",
            "Tamil": "ta", "Turkish": "tr", "Urdu": "ur",
        }
        if st.button("⬇️ Install"):
            code = install_code_map[install_lang]
            with st.spinner(f"Downloading {install_lang} pack…"):
                ok, msg = install_language_pair("en", code)
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

    # Pick target language
    available = get_available_languages()
    if available:
        lang_labels = [f"{l['name']} ({l['code']})" for l in available]
        selected    = st.sidebar.selectbox("Translate to", lang_labels)
        translation_target_code = selected.split("(")[-1].rstrip(")")
    else:
        st.sidebar.info("No language packs installed. Use the install panel above.")

    translation_source = st.sidebar.selectbox(
        "Source language",
        list(DISPLAY_LANGS.keys()),
        index=0,
    )
    translation_source_code = DISPLAY_LANGS[translation_source]

st.sidebar.subheader("🔧 Advanced Options")
ocr_confidence_threshold = st.sidebar.slider("OCR Confidence Threshold", 0.0, 1.0, 0.85, 0.05)
pdf_dpi = st.sidebar.slider("PDF DPI", 72, 300, 150, 25)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN TITLE
# ─────────────────────────────────────────────────────────────────────────────

st.title("📄 Advanced OCR System")
st.markdown(
    "**Pipeline:** Upload → Extract (PyMuPDF) → Correct (Fireworks AI) → Translate (Argos)"
)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(
    ["📤 Upload & Extract", "✏️ Correction", "🌐 Translation", "📊 Results"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Upload & Extract
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.subheader("Upload Document")

    file_handler = FileHandler()
    all_exts = [e.lstrip(".") for e in file_handler.get_supported_extensions()]

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=all_exts,
        help="PDF, DOCX, PPTX, XLSX, EPUB, TXT, images (JPG/PNG/…)",
    )

    if uploaded_file:
        c1, c2, c3 = st.columns(3)
        c1.metric("File", uploaded_file.name)
        c2.metric("Size", f"{uploaded_file.size / 1_048_576:.2f} MB")
        c3.metric("Type", uploaded_file.type or "—")

        save_result = file_handler.save_file(uploaded_file)

        if not save_result["success"]:
            st.error(save_result["message"])
        else:
            file_path = save_result["file_path"]
            st.success(f"✅ Saved: `{save_result['saved_filename']}`")

            if st.button("🚀 Extract Text"):
                with st.spinner("Extracting text…"):
                    try:
                        st.session_state.processing_log.append(
                            f"[{datetime.now():%H:%M:%S}] Initialising OCR engine…"
                        )
                        engine = EnhancedOCREngine(languages=ocr_langs)
                        result = engine.extract_text(file_path)
                        st.session_state.ocr_result = result
                        st.session_state.processing_log.append(
                            f"[{datetime.now():%H:%M:%S}] Extraction complete — "
                            f"{result['block_count']} blocks, source={result['source']}"
                        )
                        st.success("✅ Extraction complete!")

                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Confidence", f"{result['confidence']:.0%}")
                        m2.metric("Blocks",     result["block_count"])
                        m3.metric("Characters", len(result["raw_text"]))
                        m4.metric("Source",     result["source"].upper())

                    except Exception as e:
                        st.error(f"❌ {e}")
                        st.session_state.processing_log.append(
                            f"[{datetime.now():%H:%M:%S}] ERROR: {e}"
                        )

    if st.session_state.ocr_result:
        st.subheader("📝 Extracted Text")
        st.text_area(
            "Raw output", value=st.session_state.ocr_result["raw_text"],
            height=300, disabled=True, key="tab1_raw",
        )

        with st.expander("📋 Text Blocks (first 20)"):
            rows = []
            for i, b in enumerate(st.session_state.ocr_result.get("blocks", [])[:20]):
                rows.append({
                    "#": i + 1,
                    "Text": b.text[:40] + "…" if len(b.text) > 40 else b.text,
                    "Conf": f"{b.confidence:.0%}",
                    "Size": f"{b.font_size:.0f}px",
                    "Bold": "✓" if b.is_bold else "",
                    "Italic": "✓" if b.is_italic else "",
                    "Align": b.alignment,
                })
            if rows:
                st.table(rows)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — LLM Correction
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.subheader("✏️ Text Correction with LLM")

    if st.session_state.ocr_result is None:
        st.info("👈 Extract text first in the Upload tab.")
    else:
        raw = st.session_state.ocr_result["raw_text"]
        st.text_area("Extracted text", value=raw, height=220, disabled=True, key="tab2_raw")

        method = st.radio(
            "Method",
            ["LLM Correction (Fireworks AI)", "Local Correction (Regex fallback)"],
        )

        if st.button("🔧 Correct Text"):
            with st.spinner("Correcting…"):
                try:
                    if method.startswith("LLM"):
                        if not os.getenv("FIREWORKS_API_KEY"):
                            st.error("❌ Fireworks API key not set (sidebar → API Keys).")
                        else:
                            corrector = FireworksCorrector()
                            res = corrector.correct_text(raw)
                            st.session_state.corrected_text = res["corrected"]
                            st.success("✅ LLM correction done!")
                            st.session_state.processing_log.append(
                                f"[{datetime.now():%H:%M:%S}] LLM correction — status={res['status']}"
                            )
                    else:
                        fc = FireworksCorrector.__new__(FireworksCorrector)
                        st.session_state.corrected_text = fc._local_correction(raw)
                        st.success("✅ Local correction done.")
                        st.session_state.processing_log.append(
                            f"[{datetime.now():%H:%M:%S}] Local correction done."
                        )
                except Exception as e:
                    st.error(f"❌ {e}")

        if st.session_state.corrected_text:
            st.text_area(
                "Corrected text", value=st.session_state.corrected_text,
                height=220, disabled=True, key="tab2_corrected",
            )
            with st.expander("📊 Side-by-side comparison"):
                l, r = st.columns(2)
                l.markdown("**Original**")
                l.write(raw)
                r.markdown("**Corrected**")
                r.write(st.session_state.corrected_text)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Translation
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.subheader("🌐 Translation (Argos Translate)")

    if not enable_translation:
        st.info("👈 Enable Translation in the sidebar to use this tab.")
    elif st.session_state.ocr_result is None:
        st.info("👈 Extract text first.")
    else:
        source_text = (
            st.session_state.corrected_text
            or st.session_state.ocr_result["raw_text"]
        )
        st.text_area("Text to translate", value=source_text, height=200, disabled=True, key="tab3_src")

        if translation_target_code is None:
            st.warning("No language pack installed. Go to sidebar → Install Language Packs.")
        else:
            col_info, col_btn = st.columns([3, 1])
            with col_info:
                st.info(f"**From:** {translation_source_code.upper()}  →  **To:** {translation_target_code.upper()}")
            with col_btn:
                if st.button("🔄 Translate"):
                    with st.spinner("Translating…"):
                        try:
                            translated = translate_text(
                                source_text,
                                translation_source_code,
                                translation_target_code,
                            )
                            st.session_state.translated_text = translated
                            st.success("✅ Translation done!")
                            st.session_state.processing_log.append(
                                f"[{datetime.now():%H:%M:%S}] Translation "
                                f"{translation_source_code}→{translation_target_code} done."
                            )
                        except Exception as e:
                            st.error(f"❌ {e}")

        if st.session_state.translated_text:
            st.text_area(
                "Translated output", value=st.session_state.translated_text,
                height=200, disabled=True, key="tab3_out",
            )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Results & Export
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.subheader("📊 Processing Results")

    with st.expander("📋 Processing Log"):
        if st.session_state.processing_log:
            for entry in st.session_state.processing_log:
                st.text(entry)
        else:
            st.info("No log entries yet.")

    has_data = any([
        st.session_state.ocr_result,
        st.session_state.corrected_text,
        st.session_state.translated_text,
    ])

    if has_data:
        st.subheader("💾 Export")
        c1, c2, c3 = st.columns(3)

        # TXT download
        txt_parts = []
        if st.session_state.ocr_result:
            txt_parts.append("=== OCR OUTPUT ===\n" + st.session_state.ocr_result["raw_text"])
        if st.session_state.corrected_text:
            txt_parts.append("=== CORRECTED ===\n" + st.session_state.corrected_text)
        if st.session_state.translated_text:
            txt_parts.append("=== TRANSLATED ===\n" + st.session_state.translated_text)

        with c1:
            st.download_button(
                "⬇️ Download TXT",
                "\n\n".join(txt_parts),
                file_name=f"ocr_{datetime.now():%Y%m%d_%H%M%S}.txt",
                mime="text/plain",
            )

        # JSON download
        with c2:
            export = {
                "timestamp": datetime.now().isoformat(),
                "ocr_text": st.session_state.ocr_result["raw_text"] if st.session_state.ocr_result else None,
                "corrected": st.session_state.corrected_text,
                "translated": st.session_state.translated_text,
                "meta": {
                    k: v for k, v in (st.session_state.ocr_result or {}).items()
                    if k not in ("blocks", "pages")
                },
            }
            st.download_button(
                "⬇️ Download JSON",
                json.dumps(export, indent=2, ensure_ascii=False),
                file_name=f"ocr_{datetime.now():%Y%m%d_%H%M%S}.json",
                mime="application/json",
            )

        # Clear
        with c3:
            if st.button("🗑️ Clear All"):
                st.session_state.ocr_result     = None
                st.session_state.corrected_text  = None
                st.session_state.translated_text = None
                st.session_state.processing_log  = []
                st.rerun()
    else:
        st.info("👈 Process a document first.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
fc1, fc2, fc3 = st.columns(3)
fc1.metric("Session",  "🟢 Active")
fc2.metric("Log entries", len(st.session_state.processing_log))
fc3.metric("Time", datetime.now().strftime("%H:%M:%S"))
st.caption("Advanced OCR System | PyMuPDF · PaddleOCR · Fireworks AI · Argos Translate")
