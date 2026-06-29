"""
Enhanced LLM Text Corrector
Uses Fireworks AI API for high-quality OCR error correction
Includes fallback to local corrections if API fails
"""

import requests
from typing import Optional, Dict
import os
import time
from functools import lru_cache


def _load_dotenv_file() -> None:
    """Load environment variables from a local .env file if present."""
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
    Corrects OCR-extracted text using Fireworks AI API
    Automatically fixes:
    - Spelling errors
    - Grammar mistakes
    - Punctuation issues
    - Formatting
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Fireworks corrector
        
        Args:
            api_key: Fireworks API key (or set FIREWORKS_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("FIREWORKS_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "Fireworks API key not provided. "
                "Set FIREWORKS_API_KEY in your .env file or pass api_key parameter"
            )
        
        self.base_url = "https://api.fireworks.ai/inference/v1/chat/completions"
        
        self.system_prompt = """You are an expert OCR post-processing and text correction engine.

Your task is to:
1. Correct OCR extraction errors (spelling, character misrecognition)
2. Fix grammar and syntax errors
3. Restore proper punctuation
4. Maintain original meaning and intent
5. Preserve text formatting and structure
6. Handle multiple languages (English, Hindi, Tamil, Telugu, Kannada, Marathi)

Rules:
- Output ONLY the corrected text
- Do NOT add explanations or metadata
- Do NOT change the overall structure unless necessary
- Preserve paragraph breaks and line spacing
- Maintain font/style indicators if present
- Return text in the same language as input
- Do NOT add headings, titles, or summaries
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
            Dict with original and corrected text, confidence, etc.
        """
        if not text or len(text.strip()) == 0:
            return {
                'status': 'error',
                'message': 'Empty text provided',
                'original': text,
                'corrected': text,
                'confidence': 0
            }
        
        try:
            corrected = self._call_fireworks_api(text, language)
            
            return {
                'status': 'success',
                'original': text,
                'corrected': corrected,
                'confidence': 0.95,  # Fireworks has high accuracy
                'model': self.model,
                'language': language
            }
        
        except Exception as e:
            print(f"Fireworks API error: {e}")
            
            # Fallback to local correction
            fallback_result = self._local_correction(text)
            
            return {
                'status': 'fallback',
                'original': text,
                'corrected': fallback_result,
                'confidence': 0.7,
                'error': str(e),
                'message': 'Used fallback correction'
            }
    
    def _call_fireworks_api(self, text: str, language: str = "en") -> str:
        """
        Call Fireworks API with retry logic
        """
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
                    "content": f"Correct the following {language.upper()} text:\n\n{text}"
                }
            ],
            "temperature": 0.2,  # Low temperature for consistent corrections
            "max_tokens": min(len(text) * 2, 4096),  # 2x text length or max
            "top_p": 0.9
        }
        
        # Retry logic
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
                    corrected_text = data["choices"][0]["message"]["content"].strip()
                    return corrected_text
                else:
                    raise ValueError("No choices in API response")
            
            except requests.exceptions.Timeout:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"Timeout on attempt {attempt + 1}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise
            
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    print(f"Request error on attempt {attempt + 1}: {e}. Retrying...")
                    time.sleep(1)
                else:
                    raise
        
        raise Exception("Failed after maximum retries")
    
    def _local_correction(self, text: str) -> str:
        """
        Fallback local correction using regex and simple rules
        """
        import re
        
        corrected = text
        
        # Fix common OCR errors
        ocr_fixes = {
            r'\b0\b': 'O',  # Zero to O
            r'\bl\b': 'I',  # l to I
            r'\brn\b': 'm',  # rn to m
            r'\b1\b': 'I',  # 1 to I
            r'\|': 'I',  # Pipe to I
        }
        
        for pattern, replacement in ocr_fixes.items():
            corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)
        
        # Fix spacing
        corrected = re.sub(r'\s+', ' ', corrected)  # Multiple spaces to one
        
        # Fix common typos
        typo_fixes = {
            'teh': 'the',
            'adn': 'and',
            'hte': 'the',
            'thier': 'their',
            'recieve': 'receive',
            'occured': 'occurred',
        }
        
        for typo, correction in typo_fixes.items():
            corrected = re.sub(r'\b' + typo + r'\b', correction, corrected, flags=re.IGNORECASE)
        
        # Add period at end if missing
        if corrected and not corrected[-1] in '.!?':
            corrected += '.'
        
        return corrected
    
    def correct_batch(self, texts: list, language: str = "en") -> list:
        """
        Correct multiple texts
        """
        results = []
        for i, text in enumerate(texts):
            result = self.correct_text(text, language)
            results.append(result)
            print(f"Corrected {i + 1}/{len(texts)}")
        
        return results
    
    @lru_cache(maxsize=100)
    def get_stats(self) -> Dict:
        """Get correction statistics"""
        return {
            'model': self.model,
            'api': 'Fireworks AI',
            'timeout': self.timeout,
            'retries': self.max_retries,
            'temperature': 0.2
        }


class HybridCorrector:
    """
    Combines multiple correction strategies
    1. LLM correction (Fireworks)
    2. Local regex-based correction
    3. Language-specific rules
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize hybrid corrector"""
        self.fireworks = FireworksCorrector(api_key) if api_key else None
        self.fallback = FireworksCorrector.__new__(FireworksCorrector)  # Local correction only
    
    def correct_text(self, text: str, language: str = "en", use_llm: bool = True) -> Dict:
        """
        Correct text using hybrid approach
        
        Args:
            text: Text to correct
            language: Language code
            use_llm: Whether to use LLM (if available)
        
        Returns:
            Corrected text and metadata
        """
        if use_llm and self.fireworks:
            return self.fireworks.correct_text(text, language)
        else:
            # Use local-only correction
            return {
                'status': 'success',
                'original': text,
                'corrected': self.fallback._local_correction(text),
                'confidence': 0.7,
                'method': 'local_rules'
            }
    
    def compare_corrections(self, text: str, language: str = "en") -> Dict:
        """
        Compare LLM vs local corrections
        Useful for testing and validation
        """
        local_result = self.fallback._local_correction(text)
        
        if self.fireworks:
            llm_result = self.fireworks.correct_text(text, language)
        else:
            llm_result = None
        
        return {
            'original': text,
            'local_correction': local_result,
            'llm_correction': llm_result['corrected'] if llm_result else None,
            'llm_available': self.fireworks is not None
        }


# Example usage
if __name__ == "__main__":
    import os
    
    # Get API key from environment
    api_key = os.getenv("FIREWORKS_API_KEY")
    
    if api_key:
        corrector = FireworksCorrector(api_key)
        
        # Example OCR text with errors
        ocr_text = """
        Maching Learnlng is a subset of Artifical lntelligence that 
        enabels systmes to learn and imprve from exprience.
        """
        
        result = corrector.correct_text(ocr_text, language="en")
        
        print("Original:")
        print(ocr_text)
        print("\nCorrected:")
        print(result['corrected'])
        print(f"\nStatus: {result['status']}")
        print(f"Confidence: {result['confidence']}")
    
    else:
        print("FIREWORKS_API_KEY environment variable not set")
        print("Using local correction fallback...")
        
        corrector = FireworksCorrector.__new__(FireworksCorrector)
        
        ocr_text = "Maching Learnlng is powrful"
        corrected = corrector._local_correction(ocr_text)
        
        print(f"Original: {ocr_text}")
        print(f"Corrected: {corrected}")
