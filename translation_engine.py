"""
Simple Translation Engine - Streamlit Cloud Compatible
Works without ctranslate2 or heavy ML models
Uses dictionary-based and heuristic translation
"""

from typing import List, Dict, Optional
import re


class SimpleTranslationEngine:
    """
    Simple translation engine using dictionary-based approach
    Compatible with Streamlit Cloud (no heavy dependencies)
    """
    
    def __init__(self):
        """Initialize translation engine"""
        self.language_codes = {
            'en': 'English',
            'hi': 'Hindi (हिंदी)',
            'ta': 'Tamil (தமிழ்)',
            'te': 'Telugu (తెలుగు)',
            'ka': 'Kannada (ಕನ್ನಡ)',
            'mr': 'Marathi (मराठी)',
        }
        
        # Comprehensive word dictionary for basic translation
        self.translation_dict = {
            'en': {
                # Common words
                'the': {'hi': 'यह', 'ta': 'அ', 'te': 'ఈ', 'ka': 'ಇದು', 'mr': 'हे'},
                'a': {'hi': 'एक', 'ta': 'ஒரு', 'te': 'ఒక', 'ka': 'ಒಂದು', 'mr': 'एक'},
                'is': {'hi': 'है', 'ta': 'உள்ளது', 'te': 'ఉంది', 'ka': 'ಆಗಿದೆ', 'mr': 'आहे'},
                'and': {'hi': 'और', 'ta': 'மற்றும்', 'te': 'మరియు', 'ka': 'ಮತ್ತು', 'mr': 'आणि'},
                'of': {'hi': 'का', 'ta': 'இன்', 'te': 'యొక్క', 'ka': 'ನ', 'mr': 'चा'},
                'to': {'hi': 'को', 'ta': 'க்கு', 'te': 'కు', 'ka': 'ಗೆ', 'mr': 'ला'},
                'in': {'hi': 'में', 'ta': 'இல்', 'te': 'లో', 'ka': 'ನಲ್ಲಿ', 'mr': 'मध्ये'},
                'for': {'hi': 'के लिए', 'ta': 'க்கான', 'te': 'కోసం', 'ka': 'ಗಾಗಿ', 'mr': 'साठी'},
                'that': {'hi': 'कि', 'ta': 'அது', 'te': 'అది', 'ka': 'ಅದು', 'mr': 'ते'},
                'with': {'hi': 'के साथ', 'ta': 'உடன்', 'te': 'తో', 'ka': 'ಜೊತೆಗೆ', 'mr': 'सह'},
                
                # Tech/Business terms
                'machine': {'hi': 'मशीन', 'ta': 'இயந்திரம்', 'te': 'యంత్రం', 'ka': 'ಯಂತ್ರ', 'mr': 'यंत्र'},
                'learning': {'hi': 'सीखना', 'ta': 'கற்றல்', 'te': 'నేర్చుకోవడం', 'ka': 'ಕಲಿಕೆ', 'mr': 'शिक्षण'},
                'data': {'hi': 'डेटा', 'ta': 'தரவு', 'te': 'డేటా', 'ka': 'ಡೇಟಾ', 'mr': 'डेटा'},
                'system': {'hi': 'प्रणाली', 'ta': 'அமைப்பு', 'te': 'వ్యవస్థ', 'ka': 'ವ್ಯವಸ್ಥೆ', 'mr': 'प्रणाली'},
                'computer': {'hi': 'कंप्यूटर', 'ta': 'கணினி', 'te': 'కంప్యూటర్', 'ka': 'ಕಂಪ್ಯೂಟರ್', 'mr': 'संगणक'},
                'software': {'hi': 'सॉफ्टवेयर', 'ta': 'மென்பொருள்', 'te': 'సాఫ్ట్‌వేర్', 'ka': 'ಸಾಫ್ಟ್‌ವೇರ್', 'mr': 'सॉफ्टवेयर'},
                'technology': {'hi': 'प्रौद्योगिकी', 'ta': 'தொழில்நுட்பம்', 'te': 'సాంకేతికత', 'ka': 'ತಂತ್ರಜ್ಞಾನ', 'mr': 'तंत्रज्ञान'},
                'intelligence': {'hi': 'बुद्धिमत्ता', 'ta': 'அறிவுசாலி', 'te': 'గ్రహణశక్తి', 'ka': 'ಬುದ್ಧಿಮತ್ತೆ', 'mr': 'बुद्धिमत्ता'},
                'artificial': {'hi': 'कृत्रिम', 'ta': 'செயற்கை', 'te': 'కృత్రిమ', 'ka': 'ಕೃತ್ರಿಮ', 'mr': 'कृत्रिम'},
                'network': {'hi': 'नेटवर्क', 'ta': 'வலையமைப்பு', 'te': 'నెట్‌వర్క్', 'ka': 'ನೆಟ್‌ವರ್ಕ್', 'mr': 'नेटवर्क'},
                'algorithm': {'hi': 'एल्गोरिदम', 'ta': 'வழிமுறை', 'te': 'అల్గారిథమ్', 'ka': 'ಅಲ್ಗಾರಿದಮ್', 'mr': 'अल्गोरिदम'},
                
                # Common adjectives
                'good': {'hi': 'अच्छा', 'ta': 'நல்ல', 'te': 'మంచి', 'ka': 'ಒಳ್ಳೆಯ', 'mr': 'चांगला'},
                'bad': {'hi': 'बुरा', 'ta': 'கெட்ட', 'te': 'చెడు', 'ka': 'ಕೆಟ್ಟ', 'mr': 'वाईट'},
                'large': {'hi': 'बड़ा', 'ta': 'பெரிய', 'te': 'పెద్ద', 'ka': 'ದೊಡ್ಡ', 'mr': 'मोठा'},
                'small': {'hi': 'छोटा', 'ta': 'சிறிய', 'te': 'చిన్న', 'ka': 'ಚಿಕ್ಕ', 'mr': 'लहान'},
                'new': {'hi': 'नया', 'ta': 'புதிய', 'te': 'కొత్త', 'ka': 'ಹೊಸ', 'mr': 'नवीन'},
                'old': {'hi': 'पुराना', 'ta': 'பழைய', 'te': 'పాతది', 'ka': 'ಹಳೆಯ', 'mr': 'जुना'},
            }
        }
    
    def translate(self, text: str, source_lang: str = 'en', target_lang: str = 'hi') -> str:
        """
        Translate text using dictionary-based approach
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
        
        Returns:
            Translated text or original if not possible
        """
        if source_lang == target_lang:
            return text
        
        if source_lang not in self.translation_dict:
            return text  # Can only translate from English
        
        # Split text into words
        words = text.lower().split()
        translated_words = []
        untranslated_count = 0
        
        for word in words:
            # Remove punctuation
            clean_word = re.sub(r'[^\w\s]', '', word)
            
            # Try to translate
            if clean_word in self.translation_dict[source_lang]:
                translated = self.translation_dict[source_lang][clean_word].get(target_lang, word)
                translated_words.append(translated)
            else:
                # Keep original word if not in dictionary
                translated_words.append(word)
                untranslated_count += 1
        
        result = ' '.join(translated_words)
        
        # If too many untranslated words, return original
        if untranslated_count / len(words) > 0.7:
            return text
        
        return result
    
    def translate_batch(self, texts: List[str], source_lang: str = 'en', 
                       target_lang: str = 'hi') -> List[str]:
        """Translate multiple texts"""
        return [self.translate(text, source_lang, target_lang) for text in texts]
    
    def detect_language(self, text: str) -> str:
        """
        Detect language of text based on character patterns
        
        Returns:
            Language code (en, hi, ta, te, ka, mr)
        """
        if not text:
            return 'en'
        
        # Count script-specific characters
        hindi_chars = sum(1 for c in text if ord(c) >= 0x0900 and ord(c) <= 0x097F)
        tamil_chars = sum(1 for c in text if ord(c) >= 0x0B80 and ord(c) <= 0x0BFF)
        telugu_chars = sum(1 for c in text if ord(c) >= 0x0C00 and ord(c) <= 0x0C7F)
        kannada_chars = sum(1 for c in text if ord(c) >= 0x0C80 and ord(c) <= 0x0CFF)
        
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
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get all supported languages"""
        return self.language_codes
    
    def add_translations(self, source_lang: str, word: str, translations: Dict[str, str]) -> None:
        """Add custom word translations"""
        if source_lang not in self.translation_dict:
            self.translation_dict[source_lang] = {}
        
        self.translation_dict[source_lang][word.lower()] = translations
    
    def get_dictionary_stats(self) -> Dict:
        """Get statistics about the translation dictionary"""
        stats = {}
        for lang, words in self.translation_dict.items():
            stats[lang] = len(words)
        return stats


# Alias for compatibility
TranslationEngine = SimpleTranslationEngine


# Example usage
if __name__ == "__main__":
    translator = SimpleTranslationEngine()
    
    # Test translations
    test_texts = [
        "Machine Learning is powerful",
        "Artificial Intelligence is transforming the world",
        "The data system is important"
    ]
    
    print("🌐 Translation Demo (Dictionary-Based)\n")
    print("=" * 60)
    
    for text in test_texts:
        hindi = translator.translate(text, 'en', 'hi')
        tamil = translator.translate(text, 'en', 'ta')
        
        print(f"English: {text}")
        print(f"Hindi:   {hindi}")
        print(f"Tamil:   {tamil}")
        print()
    
    # Show supported languages
    print("=" * 60)
    print("\n📚 Supported Languages:")
    for code, name in translator.get_supported_languages().items():
        print(f"  {code}: {name}")
    
    # Show dictionary stats
    print("\n📖 Dictionary Statistics:")
    stats = translator.get_dictionary_stats()
    for lang, count in stats.items():
        print(f"  {lang}: {count} words")
