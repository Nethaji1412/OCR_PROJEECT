"""
Enhanced File Handler
Handles file uploads, validation, and management
Supports images, PDFs, and DOCX files
"""

import os
import shutil
from pathlib import Path
from typing import Optional, Dict, Tuple
from datetime import datetime
import hashlib

class FileHandler:
    """
    Manages file uploads and processing
    """
    
    # Supported file types
    SUPPORTED_FORMATS = {
        'image': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'],
        'pdf': ['.pdf'],
        'document': ['.docx', '.doc']
    }
    
    # Size limits (in bytes)
    SIZE_LIMITS = {
        'image': 50 * 1024 * 1024,    # 50 MB
        'pdf': 100 * 1024 * 1024,     # 100 MB
        'document': 50 * 1024 * 1024   # 50 MB
    }
    
    def __init__(self, upload_dir: str = "uploads", temp_dir: str = "temp"):
        """
        Initialize file handler
        
        Args:
            upload_dir: Directory to store uploaded files
            temp_dir: Directory for temporary processing files
        """
        self.upload_dir = Path(upload_dir)
        self.temp_dir = Path(temp_dir)
        
        # Create directories if they don't exist
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Upload directory: {self.upload_dir.absolute()}")
        print(f"Temp directory: {self.temp_dir.absolute()}")
    
    def get_file_type(self, filename: str) -> Optional[str]:
        """Get file type category"""
        ext = Path(filename).suffix.lower()
        
        for file_type, extensions in self.SUPPORTED_FORMATS.items():
            if ext in extensions:
                return file_type
        
        return None
    
    def validate_file(self, file) -> Tuple[bool, str]:
        """
        Validate uploaded file
        
        Args:
            file: File object (from streamlit or flask)
        
        Returns:
            (is_valid, error_message)
        """
        if not file:
            return False, "No file provided"
        
        filename = file.name if hasattr(file, 'name') else str(file)
        
        # Check file extension
        file_type = self.get_file_type(filename)
        if not file_type:
            return False, f"Unsupported file type. Supported: {self._get_supported_extensions()}"
        
        # Check file size
        file_size = file.size if hasattr(file, 'size') else len(file.getvalue())
        size_limit = self.SIZE_LIMITS.get(file_type)
        
        if file_size > size_limit:
            size_mb = size_limit / (1024 * 1024)
            return False, f"File too large. Maximum size: {size_mb:.0f} MB"
        
        # Check for empty files
        if file_size == 0:
            return False, "File is empty"
        
        return True, "Valid"
    
    def _get_supported_extensions(self) -> str:
        """Get comma-separated list of supported extensions"""
        all_exts = []
        for extensions in self.SUPPORTED_FORMATS.values():
            all_exts.extend(extensions)
        return ", ".join(all_exts)
    
    def save_file(self, file, subfolder: str = "") -> Dict[str, any]:
        """
        Save uploaded file
        
        Args:
            file: File object (from streamlit or flask)
            subfolder: Optional subfolder within upload_dir
        
        Returns:
            Dict with file info (path, filename, size, file_type, etc.)
        """
        # Validate
        is_valid, message = self.validate_file(file)
        if not is_valid:
            return {
                'status': 'error',
                'message': message,
                'success': False
            }
        
        # Get original filename
        original_filename = file.name if hasattr(file, 'name') else str(file)
        
        # Create unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_without_ext = Path(original_filename).stem
        extension = Path(original_filename).suffix
        unique_filename = f"{name_without_ext}_{timestamp}{extension}"
        
        # Determine save directory
        if subfolder:
            save_dir = self.upload_dir / subfolder
        else:
            save_dir = self.upload_dir
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Full path
        file_path = save_dir / unique_filename
        
        try:
            # Save file
            if hasattr(file, 'read'):
                content = file.read()
            else:
                content = file.getvalue()
            
            with open(file_path, 'wb') as f:
                f.write(content)
            
            # Calculate file hash
            file_hash = self._calculate_file_hash(file_path)
            
            # Get file size
            file_size = os.path.getsize(file_path)
            
            # Get file type
            file_type = self.get_file_type(original_filename)
            
            return {
                'status': 'success',
                'success': True,
                'original_filename': original_filename,
                'saved_filename': unique_filename,
                'file_path': str(file_path),
                'relative_path': str(file_path.relative_to(self.upload_dir)),
                'file_size': file_size,
                'file_size_mb': round(file_size / (1024 * 1024), 2),
                'file_type': file_type,
                'file_hash': file_hash,
                'upload_time': datetime.now().isoformat()
            }
        
        except Exception as e:
            return {
                'status': 'error',
                'success': False,
                'message': f"Error saving file: {str(e)}"
            }
    
    def _calculate_file_hash(self, file_path: Path, algorithm: str = "md5") -> str:
        """Calculate file hash for integrity checking"""
        hash_obj = hashlib.new(algorithm)
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_obj.update(chunk)
        
        return hash_obj.hexdigest()
    
    def delete_file(self, file_path: str) -> Dict[str, any]:
        """
        Delete a file
        
        Args:
            file_path: Path to file to delete
        
        Returns:
            Status dict
        """
        try:
            path = Path(file_path)
            
            if not path.exists():
                return {
                    'status': 'error',
                    'message': 'File not found'
                }
            
            if path.is_file():
                path.unlink()
                return {
                    'status': 'success',
                    'message': f'File deleted: {path.name}'
                }
            else:
                return {
                    'status': 'error',
                    'message': 'Path is not a file'
                }
        
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error deleting file: {str(e)}'
            }
    
    def cleanup_temp_files(self, days: int = 1) -> Dict[str, any]:
        """
        Clean up temporary files older than specified days
        
        Args:
            days: Delete files older than this many days
        
        Returns:
            Cleanup statistics
        """
        from datetime import timedelta, timezone
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=days)
        deleted_count = 0
        freed_space = 0
        
        try:
            for file_path in self.temp_dir.glob('**/*'):
                if file_path.is_file():
                    mod_time = datetime.fromtimestamp(
                        file_path.stat().st_mtime,
                        tz=timezone.utc
                    )
                    
                    if mod_time < cutoff_time:
                        freed_space += file_path.stat().st_size
                        file_path.unlink()
                        deleted_count += 1
            
            return {
                'status': 'success',
                'deleted_files': deleted_count,
                'freed_space_mb': round(freed_space / (1024 * 1024), 2)
            }
        
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def get_file_info(self, file_path: str) -> Dict[str, any]:
        """Get detailed file information"""
        try:
            path = Path(file_path)
            
            if not path.exists():
                return {
                    'status': 'error',
                    'message': 'File not found'
                }
            
            stat = path.stat()
            
            return {
                'status': 'success',
                'filename': path.name,
                'path': str(path),
                'size_bytes': stat.st_size,
                'size_mb': round(stat.st_size / (1024 * 1024), 2),
                'file_type': self.get_file_type(path.name),
                'created_time': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                'modified_time': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'file_hash': self._calculate_file_hash(path)
            }
        
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def list_uploaded_files(self) -> Dict[str, any]:
        """List all uploaded files"""
        try:
            files = []
            
            for file_path in self.upload_dir.glob('**/*'):
                if file_path.is_file():
                    files.append({
                        'filename': file_path.name,
                        'path': str(file_path),
                        'size_mb': round(file_path.stat().st_size / (1024 * 1024), 2),
                        'file_type': self.get_file_type(file_path.name),
                        'modified_time': datetime.fromtimestamp(
                            file_path.stat().st_mtime
                        ).isoformat()
                    })
            
            return {
                'status': 'success',
                'file_count': len(files),
                'files': sorted(files, key=lambda x: x['modified_time'], reverse=True)
            }
        
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }


# Example usage
if __name__ == "__main__":
    handler = FileHandler()
    
    # Example: List supported formats
    print("Supported formats:")
    for file_type, extensions in FileHandler.SUPPORTED_FORMATS.items():
        print(f"  {file_type}: {', '.join(extensions)}")
    
    # Example: List uploaded files
    files_info = handler.list_uploaded_files()
    print(f"\nUploaded files: {files_info}")
