from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from app.models.ecg_record import ECGRecord


class BaseECGStore(ABC):
    source_format: str = ""
    display_name: str = ""
    load_extensions: tuple[str, ...] = ()
    save_extensions: tuple[str, ...] = ()
    default_save_extension: str = ""

    @abstractmethod
    def load(self, file_path: str) -> ECGRecord:
        raise NotImplementedError

    @abstractmethod
    def save(self, record: ECGRecord, file_path: str) -> str:
        raise NotImplementedError

    @classmethod
    def supports_loading_extension(cls, extension: str) -> bool:
        return extension.lower() in cls.load_extensions

    @classmethod
    def supports_saving_extension(cls, extension: str) -> bool:
        return extension.lower() in cls.save_extensions

    @classmethod
    def save_file_filter(cls) -> str:
        patterns = " ".join(f"*{extension}" for extension in cls.save_extensions)
        return f"{cls.display_name} ({patterns})"

    @classmethod
    def preferred_save_extension(cls, original_file_path: str | None = None) -> str:
        if original_file_path is not None:
            original_extension = Path(original_file_path).suffix.lower()
            if cls.supports_saving_extension(original_extension):
                return original_extension
        return cls.default_save_extension

    @classmethod
    def ensure_save_extension(cls, file_path: str, *, preferred_extension: str | None = None) -> str:
        path = Path(file_path)
        current_extension = path.suffix.lower()
        if cls.supports_saving_extension(current_extension):
            return str(path)

        target_extension = preferred_extension or cls.default_save_extension
        if not target_extension.startswith("."):
            target_extension = f".{target_extension}"
        return str(path.with_suffix(target_extension))
