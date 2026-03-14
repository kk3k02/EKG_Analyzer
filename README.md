# EKG Viewer - Etap 1

Desktopowa aplikacja w Pythonie do importu i interaktywnej wizualizacji sygnalow EKG.

## Struktura katalogow

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

## Obslugiwane formaty

- WFDB (`.hea` + `.dat`)
- EDF / EDF+ (`.edf`)
- CSV / TXT (`.csv`, `.txt`)
- DICOM waveform (`.dcm`) jako przygotowany punkt rozszerzen z kontrolowanym stubem

## Uruchomienie

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m app.main
```

## Testy

```powershell
pytest tests -p no:cacheprovider
```

## Najwazniejsze decyzje architektoniczne

- `ECGRecord` jest wspolnym modelem wewnetrznym dla wszystkich loaderow.
- Warstwa `io/` odpowiada wylacznie za odczyt i normalizacje danych do `ECGRecord`.
- Warstwa `services/` trzyma walidacje, preprocessing podgladowy i statystyki zaznaczenia.
- GUI jest rozbite na panel sterowania, panel metadanych i widget wykresu oparty o `pyqtgraph`.
- Ladowanie plikow dziala w tle przez `QThreadPool`, zeby nie blokowac GUI.

## Przygotowanie pod kolejne etapy

- `ECGRecord.annotations` jest miejscem pod adnotacje beatow i zdarzen.
- `services/preprocessing.py` jest punktem wejscia pod bardziej zaawansowany preprocessing w Etapie 2.
- `services/selection_stats.py` i architektura `gui/plot_widget.py` nadaja sie do rozszerzenia o R-peaki i znaczniki diagnostyczne.
- `io/dicom_loader.py` jest jawnie wydzielonym stubem dla pozniejszego loadera waveform DICOM.
- W przyszlosci mozna dodac `services/features.py` i `ml/` bez mieszania logiki ML z warstwa GUI.

## Interakcje w GUI

- Pan i zoom mysza po wykresie `pyqtgraph`
- Wspolny pionowy kursor czasu
- Zaznaczanie obszaru z `Shift + lewy klik`
- Widok stacked multi-lead i single lead focus
- Przelaczanie odprowadzen
- Overview paska calego sygnalu
- Surowy i filtrowany sygnal do podgladu

## Ograniczenia obecnej iteracji

1. DICOM waveform
   Obsluga DICOM waveform nie jest jeszcze pelna. `app/io/dicom_loader.py` pozostaje swiadomym stubem architektonicznym i przy probie wczytania `.dcm` aplikacja komunikuje, ze ta sciezka jest jeszcze niepelna.
2. Sampling rate
   Reczna zmiana `sampling_rate` ma sens glownie dla CSV/TXT bez jawnej osi czasu. Dla WFDB i EDF wartosc z pliku jest traktowana jako zrodlo prawdy. Dla CSV/TXT z jawna kolumna czasu sampling rate jest wyliczany z osi czasu, a reczna zmiana pozostaje celowo ograniczona.
3. Statystyki zaznaczenia
   Pokazywane wartosci `min`, `max`, `mean`, `std` sa technicznymi statystykami jednego odprowadzenia: aktywnego w trybie single albo najblizszego interakcji w trybie stacked. To nie sa statystyki kliniczne i aplikacja nie realizuje jeszcze adnotacji klinicznych, detekcji zalamkow, analizy PQ/QRS/QT/ST ani analizy wieloodprowadzeniowej.
