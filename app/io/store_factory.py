from __future__ import annotations

from pathlib import Path

from app.io.base_store import BaseECGStore
from app.io.csv_store import CSVECGStore
from app.io.dicom_store import DICOMECGStore
from app.io.edf_store import EDFECGStore
from app.io.wfdb_store import WFDBECGStore
from app.utils.file_utils import file_extension


class StoreFactory:
    _STORE_CLASSES = (CSVECGStore, EDFECGStore, WFDBECGStore, DICOMECGStore)

    @staticmethod
    def create_loader(file_path: str) -> BaseECGStore:
        extension = Path(file_path).suffix.lower()
        for store_class in StoreFactory._STORE_CLASSES:
            if store_class.supports_loading_extension(extension):
                return store_class()
        raise ValueError(f"Unsupported file format: {extension or 'unknown'}")

    @staticmethod
    def create_store_for_format(source_format: str) -> BaseECGStore:
        normalized = source_format.strip().lower()
        for store_class in StoreFactory._STORE_CLASSES:
            if store_class.source_format == normalized:
                return store_class()
        raise ValueError(f"Unsupported source format: {source_format or 'unknown'}")

    @staticmethod
    def save_filters() -> list[str]:
        return [store_class.save_file_filter() for store_class in StoreFactory._STORE_CLASSES]

    @staticmethod
    def default_save_filter(source_format: str) -> str:
        return StoreFactory.create_store_for_format(source_format).save_file_filter()

    @staticmethod
    def preferred_save_extension(source_format: str, original_file_path: str | None = None) -> str:
        return StoreFactory.create_store_for_format(source_format).preferred_save_extension(original_file_path)

    @staticmethod
    def resolve_save_target(
        file_path: str,
        selected_filter: str,
        *,
        fallback_format: str,
        original_file_path: str | None = None,
    ) -> tuple[BaseECGStore, str]:
        extension_store = StoreFactory._store_from_save_extension(file_extension(file_path))
        store = extension_store or StoreFactory._store_from_save_filter(selected_filter)
        if store is None:
            store = StoreFactory.create_store_for_format(fallback_format)

        preferred_extension = store.preferred_save_extension(original_file_path)
        normalized_path = store.ensure_save_extension(file_path, preferred_extension=preferred_extension)
        extension = file_extension(normalized_path)
        if not store.supports_saving_extension(extension):
            normalized_path = store.ensure_save_extension(normalized_path, preferred_extension=preferred_extension)
        return store, normalized_path

    @staticmethod
    def _store_from_save_filter(selected_filter: str) -> BaseECGStore | None:
        for store_class in StoreFactory._STORE_CLASSES:
            if store_class.save_file_filter() == selected_filter:
                return store_class()
        return None

    @staticmethod
    def _store_from_save_extension(extension: str) -> BaseECGStore | None:
        for store_class in StoreFactory._STORE_CLASSES:
            if store_class.supports_saving_extension(extension):
                return store_class()
        return None
