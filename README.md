# EKG Viewer

Desktopowa aplikacja w Pythonie do wczytywania, normalizacji, filtrowania i interaktywnej wizualizacji sygnalow EKG z kilku formatow plikow. Projekt jest zbudowany jako lokalny viewer techniczny: sluzy do otwarcia zapisu, obejrzenia przebiegow, sprawdzenia podstawowych parametrow sygnalu i wykonania prostych operacji podgladowych. Nie jest to system diagnostyczny ani modul automatycznej interpretacji medycznej.

## Cel programu

Program sluzy do:

- wczytania sygnalu EKG z roznych formatow,
- sprowadzenia danych do jednego wspolnego modelu wewnetrznego,
- wyswietlenia przebiegu w formie wieloodprowadzeniowej albo jednego aktywnego odprowadzenia,
- podgladowego filtrowania sygnalu bez zmiany pliku zrodlowego,
- przegladania sygnalu w oknach czasowych,
- sprawdzania podstawowych informacji o pliku, liczbie probek, liczbie odprowadzen i czestotliwosci probkowania,
- pokazania prostego przegladu czestotliwosciowego aktualnie widocznego fragmentu sygnalu.

W praktyce sygnal EKG jest wczytywany po to, aby operator mogl:

- otworzyc zapis z badan lub z bazy referencyjnej,
- obejrzec ksztalt przebiegu i relacje miedzy odprowadzeniami,
- sprawdzic, czy plik ma poprawna os czasu i sensowna czestotliwosc probkowania,
- porownac wersje surowa i przetworzona,
- ocenic obecnosc dryfu, szumu wysokocestotliwosciowego lub zaklocen sieciowych 50/60 Hz,
- odpowiedziec na pytania interesantow o to, co aplikacja potrafi odczytac i jak prezentuje dane.

## Co program robi po uruchomieniu

Punkt startowy znajduje sie w `app/main.py`.

Po uruchomieniu:

1. tworzona jest aplikacja Qt (`PySide6`),
2. otwierane jest glowne okno `MainWindow`,
3. budowany jest interfejs z lewym panelem sterowania i glownym obszarem wykresu,
4. aplikacja czeka na wybor pliku przez uzytkownika.

Okno glowne jest zdefiniowane w `app/gui/main_window.py` i spina wszystkie warstwy programu:

- GUI,
- ladowanie plikow w tle,
- aktualizacje metadanych,
- preprocessing,
- sterowanie widokiem i odtwarzaniem,
- przekazanie danych do widgetu wykresu.

## Pelny przeplyw danych

Po kliknieciu `Wczytaj plik` aplikacja wykonuje nastepujacy scenariusz:

1. `MainWindow` otwiera okno wyboru pliku.
2. Wybrane zadanie ladowania trafia do `QThreadPool` jako `LoadFileTask`.
3. `LoaderFactory` sprawdza rozszerzenie i wybiera odpowiedni loader.
4. Loader czyta plik i zamienia go na wspolny obiekt `ECGRecord`.
5. `ECGRecord` przechodzi walidacje:
   - sygnal jest zamieniany na `numpy.ndarray`,
   - sygnal 1D jest promowany do ukladu 2D,
   - `NaN` i `Inf` sa zastapione wartosciami skonczonymi,
   - sprawdzana jest zgodnosc liczby probek, osi czasu i liczby odprowadzen.
6. Jesli to CSV/TXT bez jawnej osi czasu, uzytkownik dostaje okno potwierdzenia `sampling rate`.
7. Rekord jest zapisywany jako `current_record`.
8. Uruchamiany jest preprocessing podgladowy zgodnie z aktualna konfiguracja filtrow.
9. Widget wykresu dostaje sygnal surowy i przetworzony.
10. Interfejs aktualizuje:
    - panel metadanych,
    - liste odprowadzen,
    - status pliku,
    - elementy odtwarzania,
    - dolny przeglad czestotliwosci aktualnego okna.

Wazne: aplikacja nie nadpisuje pliku zrodlowego. Wszystkie operacje odbywaja sie w pamieci.

## Wspolny model danych: `ECGRecord`

Centralnym modelem projektu jest `app/models/ecg_record.py`.

Kazdy loader, niezaleznie od formatu, zwraca obiekt `ECGRecord` z polami:

- `source_format` - typ zrodla, np. `wfdb`, `edf`, `csv`, `dicom`,
- `file_path` - sciezka do pliku,
- `sampling_rate` - czestotliwosc probkowania w Hz,
- `lead_names` - lista nazw odprowadzen,
- `signal` - macierz `numpy` o ksztalcie `(n_samples, n_leads)`,
- `time_axis` - os czasu w sekundach,
- `units` - jednostki amplitudy,
- `metadata` - slownik dodatkowych metadanych z loadera,
- `annotations` - lista adnotacji, obecnie glownie dla WFDB.

To jest najwazniejsza decyzja architektoniczna w projekcie: reszta programu nie musi wiedziec, czy dane przyszly z CSV, EDF, WFDB czy DICOM. Dalej wszystkie moduly pracuja juz na tym samym typie rekordu.

## Obslugiwane formaty i dokladne zasady importu

### 1. WFDB (`.hea`, `.dat`, opcjonalnie `.atr`)

Implementacja: `app/io/wfdb_loader.py`  
Biblioteka: `wfdb`

Jak to dziala:

- jesli uzytkownik wskaze `.dat`, loader sprawdza, czy istnieje pasujacy `.hea`,
- jesli wskaze `.hea`, loader sprawdza, czy istnieje pasujacy `.dat`,
- rekord jest czytany przez `wfdb.rdrecord(...)`,
- sygnal pobierany jest z `p_signal`, a jesli go brak, z `d_signal`,
- czestotliwosc probkowania pochodzi z `record.fs`,
- os czasu jest budowana automatycznie na podstawie liczby probek i `fs`,
- jesli obok istnieje plik `.atr`, loader czyta go przez `wfdb.rdann(...)` i zapisuje adnotacje do `record.annotations`.

Jakie dane dodatkowe trafiaja do `metadata`:

- nazwa rekordu,
- data bazowa i czas bazowy, jesli sa obecne,
- komentarze z naglowka,
- dlugosc sygnalu (`sig_len`).

Do czego ten tryb jest szczegolnie przydatny:

- praca z klasycznymi zbiorami badawczymi,
- wykorzystanie gotowych adnotacji z plikow `.atr`,
- analiza zapisow z baz typu MIT-BIH.

### 2. EDF (`.edf`)

Implementacja: `app/io/edf_loader.py`  
Biblioteka: `mne`

Jak to dziala:

- plik jest wczytywany przez `mne.io.read_raw_edf(..., preload=True)`,
- sygnal pobierany jest przez `raw.get_data().T`,
- czestotliwosc probkowania pochodzi z `raw.info["sfreq"]`,
- nazwy kanalow pochodza z `raw.ch_names`,
- os czasu jest budowana automatycznie,
- jednostki sa ustawiane na `uV`.

Do `metadata` trafiaja m.in.:

- `meas_date`,
- `subject_info`,
- liczba kanalow,
- parametry `highpass` i `lowpass` zapisane w pliku.

Do czego sluzy:

- otwieranie biosygnalow zapisanych w EDF,
- szybkie obejrzenie kanalow i parametrow zapisu,
- wykorzystanie `mne` jako stabilnego loadera dla formatu pomiarowego.

### 3. CSV / TXT (`.csv`, `.txt`)

Implementacja: `app/io/csv_loader.py`  
Biblioteki: `pandas`, `numpy`, standardowe `csv`

Jak to dziala:

- loader probuje wykryc separator `,` albo `;`,
- probuje okreslic, czy plik ma naglowek,
- sprawdza, czy pierwsza kolumna wyglada jak os czasu,
- wczytuje dane do `pandas.DataFrame`,
- usuwa puste wiersze i kolumny,
- zamienia wartosci na liczbowe,
- wykonuje interpolacje brakow i uzupelnienie `bfill/ffill`,
- jesli nadal pozostaja nieczytelne wartosci, zwraca kontrolowany blad.

Mozliwe dwa glowne scenariusze:

1. Plik ma jawna kolumne czasu  
   Wtedy:
   - pierwsza kolumna jest traktowana jako `time_axis`,
   - pozostale kolumny sa traktowane jako odprowadzenia,
   - `sampling_rate` jest wyliczany z mediany roznic czasu,
   - reczna zmiana `sampling rate` jest zablokowana.

2. Plik nie ma kolumny czasu  
   Wtedy:
   - wszystkie kolumny sa traktowane jako odprowadzenia,
   - tworzona jest sztuczna os czasu,
   - domyslne `sampling_rate` wynosi `250 Hz`,
   - po zaladowaniu uzytkownik moze potwierdzic lub zmienic te wartosc,
   - reczna zmiana `sampling rate` w panelu jest dozwolona.

Nazwy odprowadzen:

- jesli jest naglowek, nazwy sa brane z nazw kolumn,
- bez naglowka tworzone sa `Lead 1`, `Lead 2`, itd.

Do czego sluzy:

- import wlasnych tabelarycznych danych,
- szybkie otwieranie zapisow wyeksportowanych z innych narzedzi,
- praca z danymi eksperymentalnymi i technicznymi.

### 4. DICOM waveform (`.dcm`)

Implementacja: `app/io/dicom_loader.py`  
Biblioteka: `pydicom`

Jak to dziala:

- plik jest czytany przez `pydicom.dcmread(...)`,
- loader sprawdza, czy w pliku istnieje `WaveformSequence`,
- jesli w pliku jest kilka grup waveform, wybierana jest ta, ktora najbardziej przypomina glowny sygnal EKG,
- dekodowanie probek odbywa sie przez natywne `dataset.waveform_array(sequence_index)`,
- `sampling_rate` jest odczytywany z `SamplingFrequency`,
- nazwy kanalow sa probowane z `ChannelLabel` albo `ChannelSourceSequence`,
- jednostki sa probowane z `ChannelSensitivityUnitsSequence`.

Jak wybierana jest najlepsza grupa waveform:

- preferowane sa grupy, ktore wygladaja na EKG po `Modality`, `SOPClassUID`, `MultiplexGroupLabel` lub opisach kanalow,
- dodatkowy priorytet dostaje grupa oznaczona jako `RHYTHM`,
- dalej preferowane sa grupy z wieksza liczba kanalow i probek.

Do `metadata` trafiaja m.in.:

- `patient_id`,
- `study_id`,
- `modality`,
- `manufacturer`,
- `study_date`,
- indeks wybranej grupy waveform,
- liczba kanalow i probek,
- informacja, ze `sampling_rate` z DICOM jest traktowany jako autorytatywny.

Do czego sluzy:

- otwieranie medycznych plikow DICOM zawierajacych waveform EKG,
- zachowanie podstawowych metadanych badania,
- praca z klinicznymi lub referencyjnymi zapisami w standardzie DICOM.

## Biblioteki i do czego sa uzywane

Plik zaleznosci: `requirements.txt`

### `PySide6`

Warstwa GUI:

- tworzenie okna aplikacji,
- przyciski, listy, suwaki, panele,
- `QThreadPool` i `QRunnable` do ladowania plikow w tle,
- `QTimer` do odtwarzania okna sygnalu.

### `pyqtgraph`

Warstwa wykresow:

- glowny wykres EKG,
- pionowy kursor czasu,
- rysowanie wielu odprowadzen,
- dolny wykres przegladu czestotliwosci,
- szybka obsluga duzych zbiorow probek.

### `numpy`

Podstawowa reprezentacja sygnalu:

- macierze probek,
- os czasu,
- operacje numeryczne,
- wyliczenia statystyk i widma FFT.

### `scipy`

Przetwarzanie sygnalu:

- filtry Butterwortha,
- filtracja zerofazowa `filtfilt`,
- filtr `iirnotch`,
- metoda Welcha do analizy czestotliwosciowej.

Jesli `SciPy` nie jest dostepne, logika ma czesciowy fallback:

- filtrowanie jest pomijane z ostrzezeniem,
- analiza czestotliwosci moze przejsc na prostsza FFT z `numpy`.

### `pandas`

Import danych tabelarycznych:

- czytanie CSV/TXT,
- usuwanie pustych danych,
- konwersja do liczb,
- interpolacja brakow.

### `wfdb`

Obsluga rekordow WFDB:

- czytanie naglowkow i danych sygnalu,
- czytanie adnotacji `.atr`.

### `mne`

Obsluga EDF:

- stabilny odczyt plikow EDF/EDF+,
- dostep do metadanych pomiarowych,
- pobranie danych kanalow.

### `pydicom`

Obsluga DICOM waveform:

- odczyt datasetu,
- dekodowanie `WaveformSequence`,
- pobieranie metadanych klinicznych i technicznych.

### `pytest`

Testy jednostkowe:

- walidacja modelu,
- loader factory,
- import CSV,
- import DICOM,
- preprocessing,
- statystyki i analiza czestotliwosci.

## Operacje dostepne w programie

Na obecnym etapie aplikacja pozwala na:

- wczytanie pliku `WFDB`, `EDF`, `CSV/TXT` lub `DICOM`,
- wyswietlenie sygnalu jako:
  - widok wieloodprowadzeniowy,
  - widok jednego aktywnego odprowadzenia,
- wlaczanie i wylaczanie widocznosci odprowadzen,
- wybor aktywnego odprowadzenia,
- przejscie na gotowe okna czasu `2 s`, `5 s`, `10 s`, `30 s`, `Caly`,
- wlaczenie lub wylaczenie siatki,
- wlaczenie podgladu sygnalu przetworzonego,
- wlaczenie podgladu surowego sygnalu jako nakladki,
- sterowanie odtwarzaniem okna sygnalu:
  - start,
  - pauza,
  - stop,
  - zmiana predkosci,
  - petla,
  - przesuwanie pozycji suwakiem,
- przeglad aktualnego widma czestotliwosciowego,
- podejrzenie podstawowych informacji o pliku.

## Filtrowanie i preprocessing

Implementacja: `app/services/preprocessing.py`

To nie jest pipeline diagnostyczny. To zestaw filtrow podgladowych uruchamianych na aktualnym rekordzie w pamieci.

Dostepne operacje:

- usuniecie skladowej stalej (`dc_removal`),
- filtr gornoprzepustowy,
- filtr dolnoprzepustowy,
- filtr pasmowoprzepustowy,
- filtr notch dla `50 Hz` lub `60 Hz`.

Wazne zasady:

- filtry dzialaja na wszystkich odprowadzeniach jednoczesnie,
- jesli wlaczony jest bandpass, ma priorytet nad osobnymi filtrami highpass i lowpass,
- filtrowanie uzywa `scipy.signal.butter(...)` i `scipy.signal.filtfilt(...)`,
- notch jest budowany przez `scipy.signal.iirnotch(...)`,
- przy niepoprawnych parametrach filtr jest pomijany z ostrzezeniem zamiast wywolywania awarii,
- przy zbyt krotkim sygnale filtracja moze zostac pominieta.

Typowe zastosowanie:

- usuniecie dryfu linii bazowej,
- stlumienie szumu o wysokich czestotliwosciach,
- redukcja zaklocen sieciowych,
- szybki podglad, jak filtr zmienia przebieg.

## Analiza czestotliwosciowa

Implementacja: `app/services/frequency_overview.py`

Dolny panel wykresu pokazuje analize czestotliwosci aktualnie widocznego fragmentu sygnalu.

Jak to dziala:

- brany jest tylko widoczny fragment aktualnego odprowadzenia,
- sygnal jest centrowany przez odjecie sredniej,
- jesli dostepne jest `scipy.signal.welch`, liczona jest gestosc mocy widmowej,
- w przeciwnym razie liczona jest amplituda FFT z `numpy`,
- wynik jest ograniczany domyslnie do `0-40 Hz`,
- uzytkownik moze przelaczyc skale logarytmiczna.

To jest przeglad techniczny, nie analiza kliniczna.

## Metadane i informacje pokazywane w interfejsie

Panel metadanych pokazuje:

- nazwe pliku,
- format,
- czestotliwosc probkowania,
- liczbe probek,
- liczbe odprowadzen,
- czas trwania,
- jednostki.

Pasek statusu pokazuje dodatkowo:

- informacje o aktualnym pliku,
- stan odtwarzania,
- pozycje kursora,
- techniczne statystyki zaznaczenia.

## Adnotacje

Struktura `ECGRecord.annotations` istnieje jako punkt rozszerzenia.

Na obecnym etapie:

- faktycznie zasilane adnotacjami sa rekordy WFDB z plikiem `.atr`,
- widget wykresu potrafi przygotowac elementy tekstowe dla adnotacji,
- nie ma jeszcze rozbudowanego systemu klinicznych znacznikow, detekcji R-peakow ani opisu zalamkow P/QRS/T.

## Ograniczenia obecnej wersji

To jest bardzo wazna czesc przy odpowiadaniu interesantom.

Program obecnie:

- nie diagnozuje pacjenta,
- nie wykonuje automatycznej klasyfikacji arytmii,
- nie mierzy automatycznie odcinkow PQ, QRS, QT, ST,
- nie liczy HRV,
- nie eksportuje wynikow do raportu,
- nie zapisuje przetworzonego sygnalu do nowego pliku,
- nie implementuje jeszcze pelnej klinicznej pracy na adnotacjach.

Dodatkowo w kodzie sa elementy przygotowane, ale jeszcze nieukonczone:

- przyciski `Poczatek` i `Koniec` sa podlaczone w GUI, ale metody `go_to_start()` i `go_to_end()` w `plot_widget.py` sa jeszcze puste,
- obsluga klikniecia myszy pod zaznaczanie jest przygotowana, ale `_on_mouse_clicked()` jest obecnie puste,
- backend statystyk zaznaczenia istnieje, ale pelna interakcja zaznaczania w aktualnym kodzie nie jest jeszcze domknieta.

To oznacza, ze projekt ma solidny fundament do przegladania sygnalu, ale czesc funkcji interakcyjnych jest jeszcze w trakcie rozbudowy.

## Struktura projektu

```text
app/
  main.py                 # punkt startowy aplikacji
  models/
    ecg_record.py         # wspolny model rekordu EKG
  io/
    base_loader.py        # interfejs loadera
    loader_factory.py     # wybor loadera po rozszerzeniu
    wfdb_loader.py        # import WFDB
    edf_loader.py         # import EDF
    csv_loader.py         # import CSV/TXT
    dicom_loader.py       # import DICOM waveform
  services/
    validation.py         # walidacja sygnalu i osi czasu
    preprocessing.py      # filtry i podglad przetwarzania
    selection_stats.py    # statystyki zaznaczenia
    frequency_overview.py # analiza czestotliwosci aktualnego okna
  gui/
    main_window.py        # glowne okno aplikacji
    controls_panel.py     # panel sterowania
    metadata_panel.py     # panel metadanych
    plot_widget.py        # glowny widget wykresu
    dialogs.py            # dialog potwierdzenia sampling rate
  utils/
    file_utils.py         # rozszerzenia i normalizacja sciezek
    time_utils.py         # formatowanie czasu
tests/
requirements.txt
README.md
```

## Uruchomienie lokalne

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

## Jak najkrocej opisac program interesantom

Najprostszy poprawny opis jest taki:

"EKG Viewer to desktopowa aplikacja do otwierania i technicznej analizy zapisow EKG z plikow WFDB, EDF, CSV/TXT i DICOM. Program normalizuje dane do wspolnego modelu, pokazuje przebiegi wielu odprowadzen, pozwala wlaczyc podstawowe filtry podgladowe i pokazuje przeglad widma czestotliwosciowego. To narzedzie do wizualizacji i kontroli sygnalu, a nie do automatycznej diagnozy."

## Najwazniejsze odpowiedzi na typowe pytania

### Do czego wczytywany jest sygnal EKG?

Do obejrzenia przebiegu, sprawdzenia technicznej jakosci sygnalu, porownania odprowadzen, podgladowego filtrowania i podstawowej analizy czestotliwosciowej.

### Czy program interpretuje badanie?

Nie. Pokazuje sygnal i dane techniczne, ale nie stawia diagnozy.

### Czy mozna otwierac pliki kliniczne?

Tak, jesli sa zapisane w obslugiwanym formacie, szczegolnie WFDB, EDF albo DICOM waveform ECG.

### Czy mozna zmieniac czestotliwosc probkowania?

Tak, ale praktycznie tylko dla CSV/TXT bez jawnej osi czasu. Dla WFDB, EDF i DICOM wartosc z pliku jest traktowana jako poprawna i nie jest przeznaczona do recznej korekty.

### Czy program filtruje sygnal?

Tak, podgladowo. Dostepne sa: usuniecie skladowej stalej, high-pass, low-pass, band-pass i notch 50/60 Hz.

### Czy program obsluguje adnotacje?

Czesciowo. WFDB moze wczytac adnotacje `.atr`, ale pelna kliniczna obsluga znacznikow nie jest jeszcze gotowa.
