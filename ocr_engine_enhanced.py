"""
Enhanced OCR Engine with PDF/DOC Support
Handles images, PDFs, and maintains formatting
"""

import cv2
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image
import fitz  # PyMuPDF
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple
import tempfile

@dataclass
class TextBlock:
    """Represents a text block with formatting info"""
    text: str
    confidence: float
    bbox: List  # [x0, y0, x1, y1]
    font_size: float = 12
    font_name: str = "Unknown"
    is_bold: bool = False
    is_italic: bool = False
    alignment: str = "left"  # left, center, right, justify

class EnhancedOCREngine:
    """OCR Engine with support for multiple document formats"""
    
    def __init__(self, languages=['en', 'hi', 'ta', 'te', 'ka', 'mr']):
        """
        Initialize OCR engine
        Args:
            languages: List of supported languages
        """
        print("Initializing Enhanced OCR Engine...")
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang=languages,
            enable_mkldnn=True,
            show_log=False
        )
        self.languages = languages
        
    def preprocess_image(self, image_path: str) -> str:
        """
        Preprocess image for better OCR accuracy
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(binary, h=10)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Save preprocessed image
        temp_path = image_path + '_processed.jpg'
        cv2.imwrite(temp_path, enhanced)
        
        return temp_path
    
    def extract_text_from_image(self, image_path: str) -> Dict:
        """
        Extract text from image with formatting
        """
        # Preprocess
        processed_path = self.preprocess_image(image_path)
        
        try:
            # Run OCR
            results = self.ocr.ocr(processed_path, cls=True)
            
            text_blocks = []
            raw_text = ""
            
            if results:
                for line_result in results[0]:
                    bbox, (text, confidence) = line_result[0], line_result[1]
                    
                    # Calculate alignment based on position
                    x_positions = [point[0] for point in bbox]
                    center_x = np.mean(x_positions)
                    
                    # Determine alignment (simplified)
                    if abs(center_x - 0.5) < 0.1:
                        alignment = "center"
                    elif center_x < 0.3:
                        alignment = "left"
                    else:
                        alignment = "right"
                    
                    # Estimate font size from bbox height
                    y_positions = [point[1] for point in bbox]
                    height = max(y_positions) - min(y_positions)
                    font_size = max(8, int(height * 0.8))
                    
                    # Check for bold/italic patterns (heuristic)
                    is_bold = len(text) < 20 and text.isupper()
                    is_italic = any(char.lower() in "ijl" for char in text.lower())
                    
                    block = TextBlock(
                        text=text,
                        confidence=float(confidence),
                        bbox=bbox,
                        font_size=font_size,
                        font_name="Arial",  # Default
                        is_bold=is_bold,
                        is_italic=is_italic,
                        alignment=alignment
                    )
                    
                    text_blocks.append(block)
                    raw_text += text + "\n"
            
            # Calculate average confidence
            avg_confidence = np.mean([b.confidence for b in text_blocks]) if text_blocks else 0
            
            return {
                'status': 'success',
                'raw_text': raw_text.strip(),
                'blocks': text_blocks,
                'confidence': avg_confidence,
                'block_count': len(text_blocks),
                'source': 'image'
            }
        
        finally:
            # Cleanup
            if os.path.exists(processed_path):
                os.remove(processed_path)
    
    def extract_images_from_pdf(self, pdf_path: str, dpi: int = 150) -> List[str]:
        """
        Extract images from PDF
        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for image extraction
        Returns:
            List of image file paths
        """
        try:
            doc = fitz.open(pdf_path)
            image_paths = []
            
            zoom = dpi / 72  # Convert DPI to zoom factor
            mat = fitz.Matrix(zoom, zoom)
            
            temp_dir = tempfile.gettempdir()
            
            for page_num, page in enumerate(doc):
                # Render page to image
                pix = page.get_pixmap(matrix=mat)
                
                # Save image
                image_path = os.path.join(
                    temp_dir,
                    f"page_{page_num:03d}.png"
                )
                pix.save(image_path)
                image_paths.append(image_path)
                
                print(f"Extracted page {page_num + 1}/{len(doc)}")
            
            doc.close()
            return image_paths
        
        except Exception as e:
            raise ValueError(f"Error extracting PDF: {str(e)}")
    
    def extract_text_from_pdf(self, pdf_path: str) -> Dict:
        """
        Extract text from PDF with formatting
        """
        try:
            doc = fitz.open(pdf_path)
            all_blocks = []
            all_text = ""
            page_data = []
            
            for page_num, page in enumerate(doc):
                # Extract text with formatting
                blocks = page.get_text("dict")["blocks"]
                page_text = ""
                page_blocks = []
                
                for block in blocks:
                    if "lines" in block:  # Text block
                        for line in block["lines"]:
                            for span in line["spans"]:
                                text = span["text"].strip()
                                
                                if text:
                                    # Extract formatting info
                                    bbox = span["bbox"]
                                    font_size = span["size"]
                                    font_name = span["font"]
                                    
                                    # Check for bold/italic
                                    flags = span.get("flags", 0)
                                    is_bold = bool(flags & 0b0001)
                                    is_italic = bool(flags & 0b0010)
                                    
                                    # Determine alignment
                                    x_center = (bbox[0] + bbox[2]) / 2
                                    page_width = page.rect.width
                                    
                                    if abs(x_center - page_width / 2) < page_width * 0.1:
                                        alignment = "center"
                                    elif x_center < page_width * 0.3:
                                        alignment = "left"
                                    else:
                                        alignment = "right"
                                    
                                    text_block = TextBlock(
                                        text=text,
                                        confidence=0.95,  # PDF text is accurate
                                        bbox=list(bbox),
                                        font_size=font_size,
                                        font_name=font_name,
                                        is_bold=is_bold,
                                        is_italic=is_italic,
                                        alignment=alignment
                                    )
                                    
                                    page_blocks.append(text_block)
                                    page_text += text + " "
                                    all_blocks.append(text_block)
                
                page_data.append({
                    'page_number': page_num + 1,
                    'text': page_text.strip(),
                    'blocks': page_blocks
                })
                
                all_text += page_text + "\n"
            
            doc.close()
            
            avg_confidence = 0.95  # PDFs are accurate
            
            return {
                'status': 'success',
                'raw_text': all_text.strip(),
                'blocks': all_blocks,
                'pages': page_data,
                'confidence': avg_confidence,
                'page_count': len(page_data),
                'block_count': len(all_blocks),
                'source': 'pdf'
            }
        
        except Exception as e:
            raise ValueError(f"Error extracting PDF: {str(e)}")
    
    def extract_text_from_doc(self, doc_path: str) -> Dict:
        """
        Extract text from DOCX file
        """
        try:
            from docx import Document
            
            doc = Document(doc_path)
            all_text = ""
            all_blocks = []
            
            for para_num, para in enumerate(doc.paragraphs):
                text = para.text.strip()
                
                if text:
                    # Extract formatting
                    is_bold = False
                    is_italic = False
                    font_size = 12
                    
                    for run in para.runs:
                        if run.bold:
                            is_bold = True
                        if run.italic:
                            is_italic = True
                        if run.font.size:
                            font_size = run.font.size.pt
                    
                    # Determine alignment
                    alignment_map = {
                        None: "left",  # Default
                        0: "left",
                        1: "center",
                        2: "right",
                        3: "justify"
                    }
                    alignment = alignment_map.get(para.alignment, "left")
                    
                    block = TextBlock(
                        text=text,
                        confidence=0.98,  # Documents are very accurate
                        bbox=[0, para_num * 20, 100, (para_num + 1) * 20],
                        font_size=font_size,
                        font_name="Calibri",
                        is_bold=is_bold,
                        is_italic=is_italic,
                        alignment=alignment
                    )
                    
                    all_blocks.append(block)
                    all_text += text + "\n"
            
            return {
                'status': 'success',
                'raw_text': all_text.strip(),
                'blocks': all_blocks,
                'confidence': 0.98,
                'block_count': len(all_blocks),
                'source': 'docx'
            }
        
        except ImportError:
            raise ValueError("python-docx is required for DOCX support. Install with: pip install python-docx")
        except Exception as e:
            raise ValueError(f"Error extracting DOCX: {str(e)}")
    
    def extract_text(self, file_path: str) -> Dict:
        """
        Main extraction method - handles all file types
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            return self.extract_text_from_image(file_path)
        
        elif file_ext == '.pdf':
            return self.extract_text_from_pdf(file_path)
        
        elif file_ext in ['.docx', '.doc']:
            return self.extract_text_from_doc(file_path)
        
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def format_text_with_alignment(self, blocks: List[TextBlock]) -> str:
        """
        Format extracted text preserving alignment and font styles
        """
        formatted = []
        
        for block in blocks:
            # Add formatting indicators
            text = block.text
            
            if block.is_bold:
                text = f"**{text}**"
            
            if block.is_italic:
                text = f"_{text}_"
            
            # Add alignment
            if block.alignment == "center":
                text = f"\n{text:^80}\n"
            elif block.alignment == "right":
                text = f"\n{text:>80}\n"
            
            formatted.append(text)
        
        return "\n".join(formatted)


# Example usage
if __name__ == "__main__":
    engine = EnhancedOCREngine()
    
    # Test with image
    # result = engine.extract_text("test.jpg")
    
    # Test with PDF
    # result = engine.extract_text("test.pdf")
    
    print("OCR Engine ready!")
