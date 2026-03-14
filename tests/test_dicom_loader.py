from __future__ import annotations

import pytest
from pydicom import Dataset, examples
from pydicom.sequence import Sequence

from app.io.dicom_loader import DICOMECGLoader, InvalidDicomECGError, UnsupportedDicomWaveformError


def test_dicom_loader_parses_real_ecg_waveform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.io.dicom_loader.dcmread", lambda path: examples.waveform)

    record = DICOMECGLoader().load("sample.dcm")

    assert record.source_format == "dicom"
    assert record.n_samples == 10000
    assert record.n_leads == 12
    assert record.sampling_rate == 1000.0
    assert "Lead I" in record.lead_names[0]
    assert record.metadata["waveform_sequence_index"] == 0
    assert record.metadata["modality"] == "ECG"
    assert record.metadata["sampling_rate_editable"] is False


def test_dicom_loader_rejects_dataset_without_waveform_sequence(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = Dataset()
    dataset.Modality = "ECG"
    monkeypatch.setattr("app.io.dicom_loader.dcmread", lambda path: dataset)

    with pytest.raises(InvalidDicomECGError, match="WaveformSequence"):
        DICOMECGLoader().load("missing_waveform.dcm")


def test_dicom_loader_rejects_unsupported_non_ecg_waveform(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = Dataset()
    dataset.Modality = "RESP"
    item = Dataset()
    item.WaveformData = b"\x00\x00"
    item.NumberOfWaveformChannels = 1
    item.NumberOfWaveformSamples = 1
    item.SamplingFrequency = 250.0
    channel = Dataset()
    source = Dataset()
    source.CodeMeaning = "Respiration"
    channel.ChannelSourceSequence = Sequence([source])
    item.ChannelDefinitionSequence = Sequence([channel])
    dataset.WaveformSequence = Sequence([item])
    monkeypatch.setattr("app.io.dicom_loader.dcmread", lambda path: dataset)

    with pytest.raises(UnsupportedDicomWaveformError, match="supported ECG waveform"):
        DICOMECGLoader().load("resp_waveform.dcm")
