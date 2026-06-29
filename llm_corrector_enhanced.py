"""
Enhanced LLM Corrector - Streamlit Cloud Compatible
Works with or without Fireworks API
Includes advanced local correction with TextBlob
"""

import requests
from typing import Optional, Dict
import os
import time
import re
from functools import lru_cache

# Try to load TextBlob for better spell correction
try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False

def _load_dotenv_file() -> None:
    """Load environment variables from .env file if present"""
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if key and key not in os.environ:
                os.environ[key] = value


_load_dotenv_file()


class FireworksCorrector:
    """
    LLM-based corrector using Fireworks AI API
    Falls back to advanced local correction if API unavailable
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Fireworks corrector"""
        self.api_key = api_key or os.getenv("fw_9R96xEJRMu4758EdF6V5Lv")
        self.has_api = bool(self.api_key)
        
        self.base_url = "https://api.fireworks.ai/inference/v1/chat/completions"
        
        self.system_prompt = """You are an expert OCR post-processing and text correction engine.

Your task is to:
1. Correct OCR extraction errors (spelling, character misrecognition)
2. Fix grammar and syntax errors
3. Restore proper punctuation
4. Maintain original meaning and intent
5. Preserve text formatting and structure

Rules:
- Output ONLY the corrected text
- Do NOT add explanations or metadata
- Do NOT change the overall structure unless necessary
- Preserve paragraph breaks and line spacing
- Return text in the same language as input
"""
        
        self.model = "accounts/fireworks/models/deepseek-v4-pro"
        self.timeout = 30
        self.max_retries = 3
    
    def correct_text(self, text: str, language: str = "en") -> Dict[str, str]:
        """
        Correct OCR-extracted text
        
        Args:
            text: Raw OCR text to correct
            language: Language code (en, hi, ta, te, ka, mr)
        
        Returns:
            Dict with original and corrected text
        """
        if not text or len(text.strip()) == 0:
            return {
                'status': 'error',
                'message': 'Empty text provided',
                'original': text,
                'corrected': text,
                'confidence': 0,
                'method': 'none'
            }
        
        # Try API first if available
        if self.has_api:
            try:
                corrected = self._call_fireworks_api(text, language)
                return {
                    'status': 'success',
                    'original': text,
                    'corrected': corrected,
                    'confidence': 0.95,
                    'method': 'fireworks_api',
                    'model': self.model
                }
            except Exception as e:
                print(f"⚠️ Fireworks API error: {e}. Using advanced local correction...")
        
        # Fallback to advanced local correction
        corrected = self._advanced_local_correction(text, language)
        
        return {
            'status': 'success',
            'original': text,
            'corrected': corrected,
            'confidence': 0.85,
            'method': 'advanced_local_correction',
            'message': 'Using advanced local correction'
        }
    
    def _call_fireworks_api(self, text: str, language: str = "en") -> str:
        """Call Fireworks API with retry logic"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": f"Correct the following text:\n\n{text}"
                }
            ],
            "temperature": 0.2,
            "max_tokens": min(len(text) * 2, 4096),
            "top_p": 0.9
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                
                response.raise_for_status()
                data = response.json()
                
                if "choices" in data and len(data["choices"]) > 0:
                    return data["choices"][0]["message"]["content"].strip()
                else:
                    raise ValueError("No choices in API response")
            
            except requests.exceptions.Timeout:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                else:
                    raise
            
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    time.sleep(1)
                else:
                    raise
        
        raise Exception("Failed after maximum retries")
    
    def _advanced_local_correction(self, text: str, language: str = "en") -> str:
        """
        Advanced local correction using multiple techniques
        - TextBlob spelling correction
        - Regex-based OCR error fixing
        - Grammar checks
        - Punctuation restoration
        """
        corrected = text
        
        # 1. Fix common OCR errors (character misrecognition)
        corrected = self._fix_ocr_errors(corrected)
        
        # 2. Use TextBlob for spell correction if available
        if HAS_TEXTBLOB:
            corrected = self._textblob_correction(corrected)
        
        # 3. Fix spacing and formatting
        corrected = self._fix_spacing(corrected)
        
        # 4. Fix common typos
        corrected = self._fix_common_typos(corrected)
        
        # 5. Restore punctuation if missing
        corrected = self._restore_punctuation(corrected)
        
        return corrected
    
    def _fix_ocr_errors(self, text: str) -> str:
        """Fix common OCR character misrecognition errors"""
        fixes = {
            # Digit to letter confusions
            r'\b0\b': 'O',           # Zero to O
            r'\b1\b(?!\d)': 'I',     # 1 to I (not in numbers)
            r'\b1l\b': 'Il',         # 1l to Il
            r'\bO0\b': 'OO',         # O0 to OO
            r'\bl\b': 'I',           # l to I (single letter)
            
            # Common OCR pairs
            r'\brn\b': 'm',          # rn to m
            r'\bni\b': 'm',          # ni to m
            r'\bvv\b': 'w',          # vv to w
            r'\bcl\b': 'd',          # cl to d
            r'\bji\b': 'u',          # ji to u
            r'\b!\b(?=\w)': 'i',     # ! to i when followed by word
            
            # Remove extra characters
            r'\|': 'l',              # Pipe to l
            r'\^': 'A',              # Caret to A
        }
        
        for pattern, replacement in fixes.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def _textblob_correction(self, text: str) -> str:
        """Use TextBlob for spelling correction"""
        try:
            # Split into sentences to avoid over-correction
            sentences = re.split(r'(?<=[.!?])\s+', text)
            corrected_sentences = []
            
            for sentence in sentences:
                if len(sentence.split()) > 1:  # Only correct if multiple words
                    blob = TextBlob(sentence)
                    corrected = str(blob.correct())
                    corrected_sentences.append(corrected)
                else:
                    corrected_sentences.append(sentence)
            
            return ' '.join(corrected_sentences)
        except Exception as e:
            print(f"TextBlob correction error: {e}")
            return text
    
    def _fix_spacing(self, text: str) -> str:
        """Fix spacing issues"""
        # Multiple spaces to single
        text = re.sub(r'\s+', ' ', text)
        
        # Remove space before punctuation
        text = re.sub(r'\s+([.!?,;:])', r'\1', text)
        
        # Add space after punctuation if missing
        text = re.sub(r'([.!?,;:])([A-Za-z])', r'\1 \2', text)
        
        return text.strip()
    
    def _fix_common_typos(self, text: str) -> str:
        """Fix common typos and misspellings"""
        typo_map = {
            # Common OCR/typo corrections
            'teh': 'the',
            'adn': 'and',
            'hte': 'the',
            'thr': 'the',
            'thier': 'their',
            'recieve': 'receive',
            'occured': 'occurred',
            'seperate': 'separate',
            'bussiness': 'business',
            'untill': 'until',
            'wich': 'which',
            'thier': 'their',
            'writting': 'writing',
            'comming': 'coming',
            'begining': 'beginning',
            'refered': 'referred',
            'aquire': 'acquire',
            'realy': 'really',
            'truely': 'truly',
        }
        
        for typo, correction in typo_map.items():
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + typo + r'\b'
            text = re.sub(pattern, correction, text, flags=re.IGNORECASE)
        
        return text
    
    def _restore_punctuation(self, text: str) -> str:
        """Restore proper punctuation"""
        # Add period at end if missing and text doesn't end with punctuation
        if text and not text[-1] in '.!?,;:':
            # Check if it's not a single word
            if len(text.split()) > 1:
                text += '.'
        
        # Fix spacing around punctuation
        text = re.sub(r',(?! )', ', ', text)  # Add space after comma
        text = re.sub(r':(?! )', ': ', text)  # Add space after colon
        
        return text
    
    def correct_batch(self, texts: list, language: str = "en") -> list:
        """Correct multiple texts"""
        results = []
        for i, text in enumerate(texts):
            result = self.correct_text(text, language)
            results.append(result)
        return results
    
    @lru_cache(maxsize=100)
    def get_stats(self) -> Dict:
        """Get correction statistics"""
        return {
            'has_api': self.has_api,
            'api_model': self.model if self.has_api else None,
            'local_methods': [
                'OCR error fixing',
                'TextBlob correction' if HAS_TEXTBLOB else 'Regex correction',
                'Typo fixing',
                'Punctuation restoration'
            ],
            'timeout': self.timeout
        }


class HybridCorrector:
    """Combines LLM and local correction strategies"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize hybrid corrector"""
        self.corrector = FireworksCorrector(api_key)
    
    def correct_text(self, text: str, language: str = "en", use_llm: bool = True) -> Dict:
        """Correct text using best available method"""
        if use_llm and self.corrector.has_api:
            return self.corrector.correct_text(text, language)
        else:
            return self.corrector.correct_text(text, language)  # Uses local fallback
    
    def compare_methods(self, text: str) -> Dict:
        """Compare different correction methods"""
        result = self.corrector.correct_text(text)
        
        return {
            'original': text,
            'corrected': result['corrected'],
            'method_used': result['method'],
            'confidence': result['confidence']
        }


# Example usage
if __name__ == "__main__":
    corrector = FireworksCorrector()
    
    # Test cases
    test_texts = [
        "Maching Learnlng is powrful",
        "Teh quick brown fox jumps ovr the lzy dog",
        "Artifical Inteligence is transformng the wrld"
    ]
    
    print("🔧 Text Correction Demo\n")
    print("=" * 60)
    
    for text in test_texts:
        result = corrector.correct_text(text)
        print(f"Original:  {result['original']}")
        print(f"Corrected: {result['corrected']}")
        print(f"Method:    {result['method']}")
        print(f"Confidence: {result['confidence']:.0%}\n")
    
    # Show available methods
    print("=" * 60)
    stats = corrector.get_stats()
    print("\n✨ Available Correction Methods:")
    for method in stats['local_methods']:
        print(f"  ✓ {method}")
