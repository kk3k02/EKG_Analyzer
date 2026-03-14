from __future__ import annotations

from abc import ABC, abstractmethod

from app.models.ecg_record import ECGRecord


class BaseECGLoader(ABC):
    @abstractmethod
    def load(self, file_path: str) -> ECGRecord:
        raise NotImplementedError
