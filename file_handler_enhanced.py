"""
file_handler_enhanced.py
────────────────────────
Manages file uploads, validation, and metadata.
Aligned with the format list that PyMuPDF + office fallbacks support.
"""

import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple


class FileHandler:
    """Manages file uploads, validation, size checks, and metadata."""

    # ── Supported formats (keyed by processing category) ─────────────────────
    SUPPORTED_FORMATS: Dict[str, list] = {
        "image":    [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"],
        "pdf":      [".pdf"],
        "document": [".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls"],
        "ebook":    [".epub", ".mobi", ".fb2"],
        "other":    [".xps", ".svg", ".txt"],
    }

    # ── Per-category size caps ────────────────────────────────────────────────
    SIZE_LIMITS: Dict[str, int] = {
        "image":    50  * 1024 * 1024,   # 50 MB
        "pdf":      100 * 1024 * 1024,   # 100 MB
        "document": 50  * 1024 * 1024,   # 50 MB
        "ebook":    50  * 1024 * 1024,
        "other":    20  * 1024 * 1024,
    }

    def __init__(self, upload_dir: str = "uploads", temp_dir: str = "temp"):
        self.upload_dir = Path(upload_dir)
        self.temp_dir   = Path(temp_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def get_file_type(self, filename: str) -> Optional[str]:
        ext = Path(filename).suffix.lower()
        for category, exts in self.SUPPORTED_FORMATS.items():
            if ext in exts:
                return category
        return None

    def get_supported_extensions(self) -> list:
        exts = []
        for lst in self.SUPPORTED_FORMATS.values():
            exts.extend(lst)
        return sorted(set(exts))

    def validate_file(self, file) -> Tuple[bool, str]:
        if not file:
            return False, "No file provided."

        filename  = file.name if hasattr(file, "name") else str(file)
        file_type = self.get_file_type(filename)

        if not file_type:
            supported = ", ".join(self.get_supported_extensions())
            return False, f"Unsupported type. Supported: {supported}"

        size = file.size if hasattr(file, "size") else len(file.getvalue())

        if size == 0:
            return False, "File is empty."

        limit = self.SIZE_LIMITS.get(file_type, 50 * 1024 * 1024)
        if size > limit:
            return False, f"File too large. Max: {limit // (1024 * 1024)} MB"

        return True, "Valid"

    # ── Save ──────────────────────────────────────────────────────────────────

    def save_file(self, file, subfolder: str = "") -> Dict:
        ok, msg = self.validate_file(file)
        if not ok:
            return {"success": False, "status": "error", "message": msg}

        original_name = file.name if hasattr(file, "name") else "upload"
        ts            = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem          = Path(original_name).stem
        ext           = Path(original_name).suffix
        unique_name   = f"{stem}_{ts}{ext}"

        save_dir = self.upload_dir / subfolder if subfolder else self.upload_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        file_path = save_dir / unique_name

        try:
            content = file.read() if hasattr(file, "read") else file.getvalue()
            with open(file_path, "wb") as fh:
                fh.write(content)

            size = os.path.getsize(file_path)

            return {
                "success": True,
                "status": "success",
                "original_filename": original_name,
                "saved_filename": unique_name,
                "file_path": str(file_path),
                "file_size": size,
                "file_size_mb": round(size / (1024 * 1024), 2),
                "file_type": self.get_file_type(original_name),
                "file_hash": self._md5(file_path),
                "upload_time": datetime.now().isoformat(),
            }

        except Exception as e:
            return {"success": False, "status": "error", "message": f"Save error: {e}"}

    # ── Metadata ──────────────────────────────────────────────────────────────

    def get_file_info(self, file_path: str) -> Dict:
        path = Path(file_path)
        if not path.exists():
            return {"status": "error", "message": "File not found"}
        s = path.stat()
        return {
            "status": "success",
            "filename": path.name,
            "path": str(path),
            "size_mb": round(s.st_size / (1024 * 1024), 2),
            "file_type": self.get_file_type(path.name),
            "modified_time": datetime.fromtimestamp(s.st_mtime).isoformat(),
        }

    def list_uploaded_files(self) -> Dict:
        try:
            files = [
                {
                    "filename": p.name,
                    "path": str(p),
                    "size_mb": round(p.stat().st_size / (1024 * 1024), 2),
                    "file_type": self.get_file_type(p.name),
                    "modified_time": datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
                }
                for p in self.upload_dir.glob("**/*")
                if p.is_file()
            ]
            return {
                "status": "success",
                "file_count": len(files),
                "files": sorted(files, key=lambda x: x["modified_time"], reverse=True),
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def delete_file(self, file_path: str) -> Dict:
        try:
            p = Path(file_path)
            if not p.exists():
                return {"status": "error", "message": "File not found"}
            p.unlink()
            return {"status": "success", "message": f"Deleted: {p.name}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def cleanup_temp_files(self, days: int = 1) -> Dict:
        from datetime import timedelta, timezone
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        deleted, freed = 0, 0
        try:
            for p in self.temp_dir.glob("**/*"):
                if p.is_file():
                    mt = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
                    if mt < cutoff:
                        freed += p.stat().st_size
                        p.unlink()
                        deleted += 1
            return {
                "status": "success",
                "deleted_files": deleted,
                "freed_space_mb": round(freed / (1024 * 1024), 2),
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _md5(file_path: Path) -> str:
        h = hashlib.md5()
        with open(file_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(4096), b""):
                h.update(chunk)
        return h.hexdigest()
