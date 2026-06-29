"""
Enhanced OCR System with LLM Correction and Translation
Complete workflow: Upload → OCR → Correct → Translate
"""

import streamlit as st
from pathlib import Path
import os
from datetime import datetime

# Import custom modules
from ocr_engine_enhanced import EnhancedOCREngine, TextBlock
from llm_corrector_enhanced import FireworksCorrector, HybridCorrector
from translation_engine import TranslationEngine
from file_handler_enhanced import FileHandler

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Advanced OCR System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

st.sidebar.title("⚙️ Configuration")

# API Key configuration
with st.sidebar.expander("API Keys", expanded=False):
    fireworks_key = st.text_input(
        "Fireworks API Key",
        type="password",
        value=os.getenv("FIREWORKS_API_KEY", ""),
        help="Get from https://fireworks.ai"
    )
    if fireworks_key:
        os.environ["FIREWORKS_API_KEY"] = fireworks_key

# Processing options
st.sidebar.subheader("Processing Options")

# OCR Language selection
ocr_languages = st.sidebar.multiselect(
    "OCR Languages",
    options=['English', 'Hindi', 'Tamil', 'Telugu', 'Kannada', 'Marathi'],
    default=['English', 'Hindi'],
    help="Select languages for OCR processing"
)

language_map = {
    'English': 'en',
    'Hindi': 'hi',
    'Tamil': 'ta',
    'Telugu': 'te',
    'Kannada': 'ka',
    'Marathi': 'mr'
}

ocr_langs = [language_map[lang] for lang in ocr_languages]

# Processing features
enable_llm_correction = st.sidebar.checkbox("Enable LLM Correction", value=True)
enable_translation = st.sidebar.checkbox("Enable Translation", value=False)

if enable_translation:
    translation_source = st.sidebar.selectbox(
        "Translate from",
        ['English', 'Hindi', 'Tamil', 'Telugu', 'Kannada', 'Marathi'],
        index=0
    )
    
    translation_target = st.sidebar.selectbox(
        "Translate to",
        ['English', 'Hindi', 'Tamil', 'Telugu', 'Kannada', 'Marathi'],
        index=1
    )
else:
    translation_source = 'English'
    translation_target = 'Hindi'

# Processing parameters
st.sidebar.subheader("Advanced Options")

ocr_confidence_threshold = st.sidebar.slider(
    "OCR Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.85,
    step=0.05,
    help="Only extract text with confidence above this value"
)

pdf_dpi = st.sidebar.slider(
    "PDF DPI (Resolution)",
    min_value=72,
    max_value=300,
    value=150,
    step=25,
    help="Higher DPI = better quality but slower"
)

# ============================================================================
# INITIALIZE SESSION STATE
# ============================================================================

if 'ocr_result' not in st.session_state:
    st.session_state.ocr_result = None

if 'corrected_text' not in st.session_state:
    st.session_state.corrected_text = None

if 'translated_text' not in st.session_state:
    st.session_state.translated_text = None

if 'processing_log' not in st.session_state:
    st.session_state.processing_log = []

# ============================================================================
# MAIN APP
# ============================================================================

st.title("📄 Advanced OCR System")
st.markdown("""
Advanced OCR with:
- **Multi-format support**: Images, PDFs, DOCX
- **Font differentiation**: Preserves bold, italic, alignment
- **LLM Correction**: Uses Fireworks AI for error correction
- **Translation**: Multi-language support with ctranslate2
""")

# ============================================================================
# TAB LAYOUT
# ============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📤 Upload & Extract",
    "✏️ Correction",
    "🌐 Translation",
    "📊 Results"
])

# ============================================================================
# TAB 1: UPLOAD & EXTRACT
# ============================================================================

with tab1:
    st.subheader("Upload Document")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a file to process",
        type=["jpg", "jpeg", "png", "bmp", "tiff", "pdf", "docx", "doc"],
        help="Maximum file size: 100 MB"
    )
    
    if uploaded_file:
        # File information
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("File Name", uploaded_file.name)
        with col2:
            st.metric("File Size", f"{uploaded_file.size / 1024 / 1024:.2f} MB")
        with col3:
            st.metric("File Type", uploaded_file.type)
        
        # Initialize file handler
        file_handler = FileHandler()
        
        # Save file
        save_result = file_handler.save_file(uploaded_file)
        
        if save_result['success']:
            file_path = save_result['file_path']
            
            st.success(f"✅ File saved: {save_result['saved_filename']}")
            
            # Start OCR
            if st.button("🚀 Start OCR Processing", key="extract_button"):
                with st.spinner("Processing... This may take a minute"):
                    try:
                        # Initialize OCR engine
                        st.session_state.processing_log.append(
                            f"[{datetime.now().strftime('%H:%M:%S')}] Initializing OCR engine..."
                        )
                        
                        ocr_engine = EnhancedOCREngine(languages=ocr_langs)
                        
                        st.session_state.processing_log.append(
                            f"[{datetime.now().strftime('%H:%M:%S')}] Extracting text..."
                        )
                        
                        # Extract text
                        result = ocr_engine.extract_text(file_path)
                        
                        st.session_state.ocr_result = result
                        st.session_state.processing_log.append(
                            f"[{datetime.now().strftime('%H:%M:%S')}] OCR completed successfully"
                        )
                        
                        # Display results
                        st.success("✅ OCR Processing Complete!")
                        
                        # Show statistics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Confidence", f"{result['confidence']:.2%}")
                        with col2:
                            st.metric("Blocks/Lines", result['block_count'])
                        with col3:
                            char_count = len(result['raw_text'])
                            st.metric("Characters", char_count)
                        with col4:
                            st.metric("Source", result['source'].upper())
                    
                    except Exception as e:
                        st.error(f"❌ Error during OCR: {str(e)}")
                        st.session_state.processing_log.append(
                            f"[{datetime.now().strftime('%H:%M:%S')}] ERROR: {str(e)}"
                        )
        else:
            st.error(f"❌ {save_result['message']}")
    
    # Display extracted text
    if st.session_state.ocr_result:
        st.subheader("📝 Extracted Text")
        
        extracted_text = st.session_state.ocr_result['raw_text']
        
        st.text_area(
            "Raw OCR Output",
            value=extracted_text,
            height=300,
            disabled=True,
            key="ocr_output"
        )
        
        # Show formatting details if available
        if 'blocks' in st.session_state.ocr_result:
            with st.expander("📋 Text Blocks Detail"):
                blocks = st.session_state.ocr_result['blocks']
                
                block_info = []
                for i, block in enumerate(blocks[:20]):  # Show first 20 blocks
                    block_info.append({
                        '#': i + 1,
                        'Text': block.text[:30] + "..." if len(block.text) > 30 else block.text,
                        'Confidence': f"{block.confidence:.2%}",
                        'Font Size': f"{block.font_size:.0f}px",
                        'Bold': "✓" if block.is_bold else "",
                        'Italic': "✓" if block.is_italic else "",
                        'Align': block.alignment
                    })
                
                st.table(block_info)

# ============================================================================
# TAB 2: CORRECTION
# ============================================================================

with tab2:
    st.subheader("✏️ Text Correction with LLM")
    
    if st.session_state.ocr_result is None:
        st.info("👈 Please extract text first in the Upload tab")
    else:
        extracted_text = st.session_state.ocr_result['raw_text']
        
        # Show extracted text
        st.text_area(
            "Original Extracted Text",
            value=extracted_text,
            height=250,
            disabled=True,
            key="correction_input"
        )
        
        # Correction options
        col1, col2 = st.columns(2)
        
        with col1:
            correction_method = st.radio(
                "Correction Method",
                ["LLM Correction (Fireworks)", "Local Correction (Regex)"],
                help="LLM provides better results but requires API key"
            )
        
        with col2:
            if st.button("🔧 Correct Text", key="correct_button"):
                with st.spinner("Correcting text..."):
                    try:
                        st.session_state.processing_log.append(
                            f"[{datetime.now().strftime('%H:%M:%S')}] Starting text correction..."
                        )
                        
                        if correction_method == "LLM Correction (Fireworks)":
                            if not os.getenv("FIREWORKS_API_KEY"):
                                st.error("❌ Fireworks API key not set. Add it in sidebar.")
                            else:
                                corrector = FireworksCorrector(os.getenv("FIREWORKS_API_KEY"))
                                result = corrector.correct_text(extracted_text)
                                
                                st.session_state.corrected_text = result['corrected']
                                
                                st.success("✅ Text corrected with LLM!")
                                st.session_state.processing_log.append(
                                    f"[{datetime.now().strftime('%H:%M:%S')}] LLM correction completed"
                                )
                        else:
                            # Local correction
                            corrector = FireworksCorrector.__new__(FireworksCorrector)
                            corrected = corrector._local_correction(extracted_text)
                            
                            st.session_state.corrected_text = corrected
                            
                            st.success("✅ Text corrected with local rules!")
                            st.session_state.processing_log.append(
                                f"[{datetime.now().strftime('%H:%M:%S')}] Local correction completed"
                            )
                    
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        st.session_state.processing_log.append(
                            f"[{datetime.now().strftime('%H:%M:%S')}] ERROR: {str(e)}"
                        )
        
        # Show corrected text
        if st.session_state.corrected_text:
            st.subheader("Corrected Text")
            
            st.text_area(
                "Corrected Output",
                value=st.session_state.corrected_text,
                height=250,
                disabled=True,
                key="correction_output"
            )
            
            # Comparison
            with st.expander("📊 Comparison"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Original**")
                    st.write(extracted_text)
                
                with col2:
                    st.write("**Corrected**")
                    st.write(st.session_state.corrected_text)

# ============================================================================
# TAB 3: TRANSLATION
# ============================================================================

with tab3:
    st.subheader("🌐 Text Translation")
    
    if not enable_translation:
        st.info("👈 Enable translation in sidebar to use this feature")
    elif st.session_state.corrected_text is None and st.session_state.ocr_result is None:
        st.info("👈 Please extract and correct text first")
    else:
        # Source text selection
        text_to_translate = st.session_state.corrected_text or st.session_state.ocr_result['raw_text']
        
        st.text_area(
            "Text to Translate",
            value=text_to_translate,
            height=200,
            disabled=True,
            key="translation_input"
        )
        
        # Translation configuration
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info(f"📍 From: {translation_source}")
        with col2:
            st.info(f"📍 To: {translation_target}")
        with col3:
            if st.button("🔄 Translate", key="translate_button"):
                with st.spinner("Translating..."):
                    try:
                        st.session_state.processing_log.append(
                            f"[{datetime.now().strftime('%H:%M:%S')}] Starting translation..."
                        )
                        
                        translator = TranslationEngine()
                        
                        translated = translator.translate(
                            text_to_translate,
                            source_lang=language_map.get(translation_source, 'en'),
                            target_lang=language_map.get(translation_target, 'hi')
                        )
                        
                        st.session_state.translated_text = translated
                        
                        st.success(f"✅ Translated to {translation_target}!")
                        st.session_state.processing_log.append(
                            f"[{datetime.now().strftime('%H:%M:%S')}] Translation completed"
                        )
                    
                    except Exception as e:
                        st.error(f"❌ Translation error: {str(e)}")
        
        # Show translated text
        if st.session_state.translated_text:
            st.subheader(f"📝 Translation ({translation_target})")
            
            st.text_area(
                "Translated Output",
                value=st.session_state.translated_text,
                height=200,
                disabled=True,
                key="translation_output"
            )

# ============================================================================
# TAB 4: RESULTS & EXPORT
# ============================================================================

with tab4:
    st.subheader("📊 Processing Results")
    
    # Processing log
    with st.expander("📋 Processing Log"):
        if st.session_state.processing_log:
            for log_entry in st.session_state.processing_log:
                st.text(log_entry)
        else:
            st.info("No processing log available")
    
    # Export options
    if st.session_state.ocr_result or st.session_state.corrected_text or st.session_state.translated_text:
        st.subheader("💾 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Download as TXT"):
                # Combine all results
                content = "=== OCR EXTRACTION RESULTS ===\n\n"
                
                if st.session_state.ocr_result:
                    content += f"OCR Output:\n{st.session_state.ocr_result['raw_text']}\n\n"
                
                if st.session_state.corrected_text:
                    content += f"Corrected Text:\n{st.session_state.corrected_text}\n\n"
                
                if st.session_state.translated_text:
                    content += f"Translated Text:\n{st.session_state.translated_text}\n"
                
                st.download_button(
                    "⬇️ Download TXT",
                    content,
                    file_name=f"ocr_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    key="download_txt"
                )
        
        with col2:
            if st.button("📥 Download as JSON"):
                import json
                
                export_data = {
                    'timestamp': datetime.now().isoformat(),
                    'ocr': st.session_state.ocr_result,
                    'corrected': st.session_state.corrected_text,
                    'translated': st.session_state.translated_text
                }
                
                # Convert TextBlock objects to dicts for JSON serialization
                if export_data['ocr'] and 'blocks' in export_data['ocr']:
                    export_data['ocr']['blocks'] = [
                        {
                            'text': b.text,
                            'confidence': b.confidence,
                            'bbox': b.bbox,
                            'font_size': b.font_size,
                            'font_name': b.font_name,
                            'is_bold': b.is_bold,
                            'is_italic': b.is_italic,
                            'alignment': b.alignment
                        }
                        for b in export_data['ocr']['blocks']
                    ]
                
                st.download_button(
                    "⬇️ Download JSON",
                    json.dumps(export_data, indent=2),
                    file_name=f"ocr_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    key="download_json"
                )
        
        with col3:
            if st.button("🗑️ Clear All"):
                st.session_state.ocr_result = None
                st.session_state.corrected_text = None
                st.session_state.translated_text = None
                st.session_state.processing_log = []
                st.success("✅ All results cleared")
    
    else:
        st.info("👈 No results to export. Process a document first!")

# ============================================================================
# FOOTER
# ============================================================================

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Session Status", "🟢 Active")

with col2:
    st.metric("Processing Steps", len(st.session_state.processing_log))

with col3:
    st.metric("Last Updated", datetime.now().strftime("%H:%M:%S"))

st.caption("Advanced OCR System v1.0 | Built with Streamlit, PaddleOCR, Fireworks AI")
