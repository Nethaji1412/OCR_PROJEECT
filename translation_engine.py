"""
Translation Module using ctranslate2
Fast and efficient translation with language detection
"""

import ctranslate2
import sentencepiece
from typing import List, Dict, Optional
import os

class TranslationEngine:
    """
    Translation engine using ctranslate2
    Supports fast neural machine translation
    """
    
    def __init__(self):
        """Initialize translation engine"""
        print("Initializing Translation Engine...")
        
        # Model paths for different language pairs
        self.models = {
            'en2hi': 'nllb-200-distilled-600M',  # English to Hindi
            'en2ta': 'nllb-200-distilled-600M',  # English to Tamil
            'en2te': 'nllb-200-distilled-600M',  # English to Telugu
            'en2ka': 'nllb-200-distilled-600M',  # English to Kannada
            'en2mr': 'nllb-200-distilled-600M',  # English to Marathi
        }
        
        self.language_codes = {
            'en': 'eng_Latn',
            'hi': 'hin_Deva',
            'ta': 'tam_Tamil',
            'te': 'tel_Telu',
            'ka': 'kan_Knda',
            'mr': 'mar_Deva',
        }
        
        self.translators = {}
        self._load_models()
    
    def _load_models(self):
        """Load translation models"""
        try:
            # This is a simplified approach
            # In production, download actual NLLB models
            print("Note: Download NLLB-200 models from HuggingFace for production use")
            print("Command: pip install -q https://github.com/OpenNMT/CTranslate2/releases/download/v3.14.0/ctranslate2-3.14.0-cp310-cp310-linux_x86_64.whl")
        except Exception as e:
            print(f"Warning: Could not load translation models: {e}")
    
    def _download_model(self, model_name: str, target_dir: str = "models"):
        """Download model from HuggingFace"""
        try:
            from huggingface_hub import snapshot_download
            
            model_path = snapshot_download(
                repo_id=f"facebook/{model_name}",
                local_dir=os.path.join(target_dir, model_name)
            )
            return model_path
        except Exception as e:
            print(f"Error downloading model: {e}")
            return None
    
    def translate(self, text: str, source_lang: str = 'en', target_lang: str = 'hi') -> str:
        """
        Translate text from source to target language
        
        Args:
            text: Text to translate
            source_lang: Source language code (e.g., 'en')
            target_lang: Target language code (e.g., 'hi')
        
        Returns:
            Translated text
        """
        try:
            # For demonstration, using a simple approach
            # In production, use actual ctranslate2 models
            
            if source_lang == target_lang:
                return text
            
            # Get language codes for NLLB
            src_code = self.language_codes.get(source_lang, 'eng_Latn')
            tgt_code = self.language_codes.get(target_lang, 'hin_Deva')
            
            # This is where actual translation would happen
            # For now, return placeholder
            return self._translate_with_ctranslate2(text, src_code, tgt_code)
        
        except Exception as e:
            print(f"Translation error: {e}")
            return text  # Return original if translation fails
    
    def _translate_with_ctranslate2(self, text: str, source_code: str, target_code: str) -> str:
        """
        Perform actual translation using ctranslate2
        
        Requires:
        1. Download NLLB model
        2. Convert to ctranslate2 format
        3. Load translator
        """
        try:
            model_path = os.path.join("models", "nllb-200-distilled-600M")
            
            # Check if model exists
            if not os.path.exists(model_path):
                print(f"Model not found at {model_path}")
                print("Download from: https://huggingface.co/facebook/nllb-200-distilled-600M")
                return text
            
            # Initialize translator
            translator = ctranslate2.Translator(model_path, device="cpu")
            
            # Prepare text with language tags (NLLB format)
            text_with_tag = f"{target_code} {text}"
            
            # Translate
            result = translator.translate_batch([[text_with_tag]])
            
            if result and result[0]:
                translated = result[0][0]['tgt'][0].replace(target_code, "").strip()
                return translated
            
            return text
        
        except Exception as e:
            print(f"ctranslate2 error: {e}")
            return text
    
    def translate_batch(self, texts: List[str], source_lang: str = 'en', 
                       target_lang: str = 'hi') -> List[str]:
        """Translate multiple texts"""
        return [self.translate(text, source_lang, target_lang) for text in texts]
    
    def detect_language(self, text: str) -> str:
        """
        Detect language of text
        
        Returns:
            Language code (e.g., 'en', 'hi')
        """
        try:
            # Simple heuristic detection
            if not text:
                return 'en'
            
            # Count character patterns
            hindi_chars = sum(1 for c in text if ord(c) >= 0x0900 and ord(c) <= 0x097F)
            tamil_chars = sum(1 for c in text if ord(c) >= 0x0B80 and ord(c) <= 0x0BFF)
            telugu_chars = sum(1 for c in text if ord(c) >= 0x0C00 and ord(c) <= 0x0C7F)
            kannada_chars = sum(1 for c in text if ord(c) >= 0x0C80 and ord(c) <= 0x0CFF)
            marathi_chars = sum(1 for c in text if ord(c) >= 0x0900 and ord(c) <= 0x097F)
            
            total_chars = len(text)
            
            # Determine language based on character density
            if hindi_chars > total_chars * 0.3:
                return 'hi'
            elif tamil_chars > total_chars * 0.3:
                return 'ta'
            elif telugu_chars > total_chars * 0.3:
                return 'te'
            elif kannada_chars > total_chars * 0.3:
                return 'ka'
            else:
                return 'en'  # Default to English
        
        except Exception as e:
            print(f"Language detection error: {e}")
            return 'en'
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get all supported languages"""
        return {
            'en': 'English',
            'hi': 'Hindi (हिंदी)',
            'ta': 'Tamil (தமிழ்)',
            'te': 'Telugu (తెలుగు)',
            'ka': 'Kannada (ಕನ್ನಡ)',
            'mr': 'Marathi (मराठी)',
        }


# Simplified fallback translator using dictionary
class SimpleDictTranslator:
    """
    Fallback translator using simple dictionary approach
    For demo/POC purposes
    """
    
    def __init__(self):
        """Initialize dictionary translator"""
        self.common_words = {
            'en': {
                'the': {'hi': 'यह', 'ta': 'அ', 'te': 'ఈ'},
                'a': {'hi': 'एक', 'ta': 'ஒரு', 'te': 'ఒక'},
                'is': {'hi': 'है', 'ta': 'உள்ளது', 'te': 'ఉంది'},
                'and': {'hi': 'और', 'ta': 'மற்றும்', 'te': 'మరియు'},
            }
        }
    
    def translate(self, text: str, source_lang: str = 'en', target_lang: str = 'hi') -> str:
        """Simple word-by-word translation"""
        if source_lang == target_lang:
            return text
        
        words = text.lower().split()
        translated_words = []
        
        for word in words:
            clean_word = word.strip('.,!?;:')
            
            if clean_word in self.common_words.get(source_lang, {}):
                translated = self.common_words[source_lang][clean_word].get(target_lang, word)
                translated_words.append(translated)
            else:
                translated_words.append(word)
        
        return ' '.join(translated_words)


# Example usage
if __name__ == "__main__":
    # Initialize translator
    translator = TranslationEngine()
    
    # Example translation
    text = "Machine Learning is transforming the world"
    
    print("Original:", text)
    print("To Hindi:", translator.translate(text, 'en', 'hi'))
    print("To Tamil:", translator.translate(text, 'en', 'ta'))
    print("Detected Language:", translator.detect_language(text))
    print("Supported Languages:", translator.get_supported_languages())
