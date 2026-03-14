# EKG Viewer - Etap 1

Desktopowa aplikacja w Pythonie do importu i interaktywnej wizualizacji sygnałów EKG.

## Struktura katalogów

```text
app/
  main.py
  models/
  io/
  services/
  gui/
  utils/
tests/
requirements.txt
README.md
```

## Obsługiwane formaty

- WFDB (`.hea` + `.dat`)
- EDF / EDF+ (`.edf`)
- CSV / TXT (`.csv`, `.txt`)
- DICOM waveform (`.dcm`) jako przygotowany punkt rozszerzeń z kontrolowanym stubem

## Uruchomienie

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m app.main
```

## Testy

```powershell
pytest
```

## Najważniejsze decyzje architektoniczne

- `ECGRecord` jest wspólnym modelem wewnętrznym dla wszystkich loaderów.
- Warstwa `io/` odpowiada wyłącznie za odczyt i normalizację danych do `ECGRecord`.
- Warstwa `services/` trzyma walidację, preprocessing podglądowy i statystyki zaznaczenia.
- GUI jest rozbite na panel sterowania, panel metadanych i widget wykresu oparty o `pyqtgraph`.
- Ładowanie plików działa w tle przez `QThreadPool`, żeby nie blokować GUI.

## Przygotowanie pod kolejne etapy

- `ECGRecord.annotations` jest miejscem pod adnotacje beatów i zdarzeń.
- `services/preprocessing.py` jest punktem wejścia pod bardziej zaawansowany preprocessing w Etapie 2.
- `services/selection_stats.py` i architektura `gui/plot_widget.py` nadają się do rozszerzenia o R-peaki i znaczniki diagnostyczne.
- `io/dicom_loader.py` jest jawnie wydzielonym stubem dla późniejszego loadera waveform DICOM.
- W przyszłości można dodać `services/features.py` i `ml/` bez mieszania logiki ML z warstwą GUI.

## Interakcje w GUI

- Pan i zoom myszą po wykresie `pyqtgraph`
- Wspólny pionowy kursor czasu
- Zaznaczanie obszaru z `Shift + lewy klik`
- Widok stacked multi-lead i single lead focus
- Przełączanie odprowadzeń
- Overview paska całego sygnału
- Surowy i filtrowany sygnał do podglądu

## Ograniczenia obecnej iteracji

- DICOM waveform nie jest jeszcze w pełni zaimplementowany.
- Zmiana sampling rate po wczytaniu dotyczy głównie CSV/TXT bez jawnej osi czasu.
- Statystyki zaznaczenia są liczone dla aktywnego lub najbliższego odprowadzenia, bez zaawansowanych adnotacji klinicznych.
