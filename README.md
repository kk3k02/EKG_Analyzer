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
- DICOM waveform (`.dcm`) dla wspieranych plikow ECG waveform

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

## DICOM waveform

- Pliki `.dcm` mozna otwierac tak samo jak inne wspierane formaty z poziomu przycisku `Wczytaj plik`.
- Loader korzysta z `pydicom.dcmread(...)` i dekoduje `WaveformSequence` przez natywne `waveform_array()`.
- Obslugiwane sa przypadki, w ktorych plik zawiera waveform ECG oraz sensowne metadane kanalu.
- Gdy plik ma wiele grup waveform, loader preferuje te najbardziej zblizone do glownego rytmu ECG: grupy z kanalami ECG, potem `RHYTHM`, potem z wieksza liczba kanalow i probek.

## Ograniczenia obecnej iteracji

1. DICOM waveform
   Implementacja obsluguje DICOM ECG waveform, ale nie celuje jeszcze we wszystkie mozliwe biosygnaly DICOM. Jezeli plik `.dcm` nie zawiera `WaveformSequence`, ma uszkodzone probki albo reprezentuje niewspierany waveform nie-EKG, aplikacja zwroci kontrolowany blad.
2. Sampling rate
   Reczna zmiana `sampling_rate` ma sens glownie dla CSV/TXT bez jawnej osi czasu. Dla WFDB i EDF wartosc z pliku jest traktowana jako zrodlo prawdy. Dla CSV/TXT z jawna kolumna czasu sampling rate jest wyliczany z osi czasu, a reczna zmiana pozostaje celowo ograniczona.
   To samo dotyczy DICOM waveform: sampling rate jest odczytywany z metadanych waveform i nie jest traktowany jako pole do recznej korekty.
3. Statystyki zaznaczenia
   Pokazywane wartosci `min`, `max`, `mean`, `std` sa technicznymi statystykami jednego odprowadzenia: aktywnego w trybie single albo najblizszego interakcji w trybie stacked. To nie sa statystyki kliniczne i aplikacja nie realizuje jeszcze adnotacji klinicznych, detekcji zalamkow, analizy PQ/QRS/QT/ST ani analizy wieloodprowadzeniowej.
