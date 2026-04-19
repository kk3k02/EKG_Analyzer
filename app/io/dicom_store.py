from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from pydicom import dcmread, dcmwrite
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, TwelveLeadECGWaveformStorage, generate_uid

from app.io.base_store import BaseECGStore
from app.models.ecg_record import ECGRecord
from app.services.validation import build_time_axis
from app.utils.file_utils import normalize_path


ECG_KEYWORDS = ("ecg", "lead", "einthoven", "electrocardiogram")


class DicomWaveformError(ValueError):
    """Base error for DICOM waveform import problems."""


class UnsupportedDicomWaveformError(DicomWaveformError):
    """Raised when the file contains waveform data, but not supported ECG data."""


class InvalidDicomECGError(DicomWaveformError):
    """Raised when DICOM ECG waveform content is missing or malformed."""


@dataclass(slots=True)
class _WaveformCandidate:
    index: int
    item: Dataset
    score: tuple[int, int, int]


class DICOMECGStore(BaseECGStore):
    """Load DICOM ECG waveform data into the common ECGRecord model.

    The loader focuses on ECG waveform datasets and uses pydicom's native
    waveform decoding. If multiple waveform groups are present, it prefers the
    group that looks most like a primary rhythm strip: ECG-labelled channels,
    then a group labelled "RHYTHM", then larger channel/sample counts.
    """

    source_format = "dicom"
    display_name = "DICOM Waveform"
    load_extensions = (".dcm",)
    save_extensions = (".dcm",)
    default_save_extension = ".dcm"

    def load(self, file_path: str) -> ECGRecord:
        normalized_path = normalize_path(file_path)
        dataset = dcmread(normalized_path)

        waveform_sequence = getattr(dataset, "WaveformSequence", None)
        if waveform_sequence is None or len(waveform_sequence) == 0:
            raise InvalidDicomECGError("DICOM file does not contain a WaveformSequence.")

        waveform_item, sequence_index = self._select_ecg_waveform_item(dataset, waveform_sequence)
        sampling_rate = self._extract_sampling_rate(waveform_item)
        signal = self._decode_waveform(dataset, waveform_item, sequence_index)
        lead_names = self._extract_lead_names(waveform_item, signal.shape[1])
        units = self._extract_units(waveform_item)
        time_axis = build_time_axis(signal.shape[0], sampling_rate)
        metadata = self._build_metadata(dataset, waveform_item, sequence_index, signal.shape, units)

        return ECGRecord(
            source_format="dicom",
            file_path=normalized_path,
            sampling_rate=sampling_rate,
            lead_names=lead_names,
            signal=signal,
            time_axis=time_axis,
            units=units,
            metadata=metadata,
            annotations=[],
        )

    def save(self, record: ECGRecord, file_path: str) -> str:
        normalized_path = normalize_path(self.ensure_save_extension(file_path))
        output_path = Path(normalized_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = TwelveLeadECGWaveformStorage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        file_meta.ImplementationClassUID = generate_uid()

        dataset = FileDataset(str(output_path), {}, file_meta=file_meta, preamble=b"\0" * 128)
        dataset.is_little_endian = True
        dataset.is_implicit_VR = False
        dataset.SOPClassUID = file_meta.MediaStorageSOPClassUID
        dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        dataset.Modality = "ECG"
        dataset.PatientID = str(record.metadata.get("patient_id", "") or "")
        dataset.StudyID = str(record.metadata.get("study_id", "") or "")
        dataset.Manufacturer = str(record.metadata.get("manufacturer", "") or "EKG Viewer")
        study_date = str(record.metadata.get("study_date", "") or "")
        if study_date:
            dataset.StudyDate = study_date
        dataset.ContentDate = datetime.now().strftime("%Y%m%d")
        dataset.ContentTime = datetime.now().strftime("%H%M%S")
        dataset.SeriesInstanceUID = generate_uid()
        dataset.StudyInstanceUID = generate_uid()

        waveform_item = Dataset()
        waveform_item.MultiplexGroupLabel = str(record.metadata.get("waveform_group_label", "") or "RHYTHM")
        waveform_item.NumberOfWaveformChannels = int(record.n_leads)
        waveform_item.NumberOfWaveformSamples = int(record.n_samples)
        waveform_item.SamplingFrequency = float(record.sampling_rate)
        waveform_item.WaveformBitsAllocated = 16
        waveform_item.WaveformBitsStored = 16
        waveform_item.WaveformSampleInterpretation = "SS"
        waveform_item.ChannelDefinitionSequence = Sequence(
            [
                self._build_channel_definition(index, lead_name, record.units, np.asarray(record.signal[:, index], dtype=float))
                for index, lead_name in enumerate(record.lead_names)
            ]
        )
        waveform_item.WaveformData = self._encode_waveform_bytes(np.asarray(record.signal, dtype=float))
        dataset.WaveformSequence = Sequence([waveform_item])

        dcmwrite(str(output_path), dataset, write_like_original=False)
        return str(output_path)

    def _encode_waveform_bytes(self, signal: np.ndarray) -> bytes:
        encoded = np.zeros_like(signal, dtype=np.int16)
        for lead_index in range(signal.shape[1]):
            encoded[:, lead_index] = self._encode_channel(signal[:, lead_index])
        return encoded.astype("<i2", copy=False).tobytes()

    def _encode_channel(self, channel_signal: np.ndarray) -> np.ndarray:
        finite = np.asarray(channel_signal, dtype=float)
        peak = float(np.max(np.abs(finite))) if finite.size else 0.0
        if peak <= 0.0:
            return np.zeros(finite.shape[0], dtype=np.int16)
        sensitivity = peak / 32767.0
        raw = np.rint(finite / sensitivity)
        raw = np.clip(raw, -32768, 32767)
        return raw.astype(np.int16)

    def _build_channel_definition(self, index: int, lead_name: str, units: str, signal: np.ndarray) -> Dataset:
        channel = Dataset()
        channel.ChannelLabel = lead_name
        peak = float(np.max(np.abs(signal))) if signal.size else 0.0
        channel.ChannelSensitivity = peak / 32767.0 if peak > 0.0 else 1.0
        channel.ChannelBaseline = 0.0
        channel.ChannelSensitivityCorrectionFactor = 1.0

        units_item = Dataset()
        units_item.CodeMeaning = units
        units_item.CodeValue = units
        units_item.CodingSchemeDesignator = "99LOCAL"
        channel.ChannelSensitivityUnitsSequence = Sequence([units_item])

        source_item = Dataset()
        source_item.CodeMeaning = lead_name
        source_item.CodeValue = f"LEAD{index + 1}"
        source_item.CodingSchemeDesignator = "99LOCAL"
        channel.ChannelSourceSequence = Sequence([source_item])
        return channel

    def _select_ecg_waveform_item(self, dataset: Dataset, waveform_sequence: Sequence) -> tuple[Dataset, int]:
        candidates: list[_WaveformCandidate] = []
        for index, item in enumerate(waveform_sequence):
            if not self._has_waveform_payload(item):
                continue
            if not self._looks_like_ecg(dataset, item):
                continue
            candidates.append(
                _WaveformCandidate(
                    index=index,
                    item=item,
                    score=self._candidate_score(item),
                )
            )

        if not candidates:
            raise UnsupportedDicomWaveformError(
                "DICOM file contains waveform data, but no supported ECG waveform group was found."
            )

        best = max(candidates, key=lambda candidate: candidate.score)
        return best.item, best.index

    def _has_waveform_payload(self, item: Dataset) -> bool:
        return bool(getattr(item, "WaveformData", None)) and int(getattr(item, "NumberOfWaveformChannels", 0)) > 0

    def _looks_like_ecg(self, dataset: Dataset, item: Dataset) -> bool:
        modality = str(getattr(dataset, "Modality", "")).lower()
        sop_name = str(getattr(getattr(dataset, "SOPClassUID", None), "name", "")).lower()
        multiplex_label = str(getattr(item, "MultiplexGroupLabel", "")).lower()

        if any(keyword in modality for keyword in ECG_KEYWORDS):
            return True
        if any(keyword in sop_name for keyword in ECG_KEYWORDS):
            return True
        if any(keyword in multiplex_label for keyword in ECG_KEYWORDS):
            return True

        for channel in getattr(item, "ChannelDefinitionSequence", []):
            source_sequence = getattr(channel, "ChannelSourceSequence", [])
            for source in source_sequence:
                source_text = " ".join(
                    str(getattr(source, attr, ""))
                    for attr in ("CodeMeaning", "CodeValue", "CodingSchemeDesignator")
                ).lower()
                if any(keyword in source_text for keyword in ECG_KEYWORDS):
                    return True
            channel_label = str(getattr(channel, "ChannelLabel", "")).lower()
            if any(keyword in channel_label for keyword in ECG_KEYWORDS):
                return True

        return False

    def _candidate_score(self, item: Dataset) -> tuple[int, int, int]:
        label = str(getattr(item, "MultiplexGroupLabel", "")).strip().lower()
        rhythm_bonus = 1 if label == "rhythm" else 0
        channels = int(getattr(item, "NumberOfWaveformChannels", 0))
        samples = int(getattr(item, "NumberOfWaveformSamples", 0))
        return (rhythm_bonus, channels, samples)

    def _extract_sampling_rate(self, item: Dataset) -> float:
        sampling_rate = float(getattr(item, "SamplingFrequency", 0.0))
        if sampling_rate <= 0:
            raise InvalidDicomECGError("DICOM ECG waveform does not define a valid sampling frequency.")
        return sampling_rate

    def _decode_waveform(self, dataset: Dataset, item: Dataset, sequence_index: int) -> np.ndarray:
        try:
            signal = np.asarray(dataset.waveform_array(sequence_index), dtype=float)
        except Exception as exc:  # pragma: no cover - exercised via failure path in tests
            raise InvalidDicomECGError(f"Failed to decode DICOM waveform samples: {exc}") from exc

        if signal.ndim == 1:
            signal = signal[:, np.newaxis]
        if signal.ndim != 2:
            raise InvalidDicomECGError("Decoded DICOM waveform has an invalid number of dimensions.")

        expected_channels = int(getattr(item, "NumberOfWaveformChannels", 0))
        expected_samples = int(getattr(item, "NumberOfWaveformSamples", 0))
        if signal.shape != (expected_samples, expected_channels):
            raise InvalidDicomECGError(
                "Decoded DICOM waveform dimensions do not match the metadata in WaveformSequence."
            )
        if expected_samples <= 0:
            raise InvalidDicomECGError("DICOM ECG waveform contains no samples.")
        return signal

    def _extract_lead_names(self, item: Dataset, n_channels: int) -> list[str]:
        lead_names: list[str] = []
        for index, channel in enumerate(getattr(item, "ChannelDefinitionSequence", [])):
            lead_name = self._channel_name(channel)
            lead_names.append(lead_name or f"Lead {index + 1}")

        if len(lead_names) < n_channels:
            lead_names.extend(f"Lead {index + 1}" for index in range(len(lead_names), n_channels))
        return lead_names[:n_channels]

    def _channel_name(self, channel: Dataset) -> str | None:
        channel_label = str(getattr(channel, "ChannelLabel", "")).strip()
        if channel_label:
            return channel_label

        source_sequence = getattr(channel, "ChannelSourceSequence", [])
        for source in source_sequence:
            code_meaning = str(getattr(source, "CodeMeaning", "")).strip()
            if code_meaning:
                return code_meaning

        return None

    def _extract_units(self, item: Dataset) -> str:
        unit_names: list[str] = []
        for channel in getattr(item, "ChannelDefinitionSequence", []):
            units_sequence = getattr(channel, "ChannelSensitivityUnitsSequence", [])
            if not units_sequence:
                continue
            code = units_sequence[0]
            unit = str(getattr(code, "CodeMeaning", "")).strip() or str(getattr(code, "CodeValue", "")).strip()
            if unit:
                unit_names.append(unit)

        if not unit_names:
            return "a.u."
        unique_units = list(dict.fromkeys(unit_names))
        if len(unique_units) == 1:
            return unique_units[0]
        return "mixed"

    def _build_metadata(
        self,
        dataset: Dataset,
        item: Dataset,
        sequence_index: int,
        signal_shape: tuple[int, int],
        units: str,
    ) -> dict[str, Any]:
        return {
            "patient_id": str(getattr(dataset, "PatientID", "")) or None,
            "study_id": str(getattr(dataset, "StudyID", "")) or None,
            "modality": str(getattr(dataset, "Modality", "")) or None,
            "manufacturer": str(getattr(dataset, "Manufacturer", "")) or None,
            "study_date": str(getattr(dataset, "StudyDate", "")) or None,
            "waveform_source": str(getattr(getattr(dataset, "SOPClassUID", None), "name", "")) or "DICOM Waveform",
            "waveform_sequence_index": sequence_index,
            "waveform_group_label": str(getattr(item, "MultiplexGroupLabel", "")) or None,
            "waveform_bits_allocated": int(getattr(item, "WaveformBitsAllocated", 0)),
            "waveform_sample_interpretation": str(getattr(item, "WaveformSampleInterpretation", "")) or None,
            "channel_count": int(signal_shape[1]),
            "sample_count": int(signal_shape[0]),
            "units_source": units,
            "sampling_rate_editable": False,
            "sampling_rate_note": "Sampling rate was read from DICOM waveform metadata and is treated as authoritative.",
        }
