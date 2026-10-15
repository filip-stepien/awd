# Analiza statystyczna League of Legends w Pythonie

Projekt do części **I. Dane numeryczne**: czyszczenie, statystyki opisowe,
korelacje Pearsona, dwie proste regresje liniowe i komplet właściwych dla danych wykresów.

Najlepiej zacząć od **`notebooks/01_analiza_numeryczna.ipynb`**. Notebook ma wykonane
komórki, tabele, wykresy i komentarze po polsku. Zwięzły raport z wynikami:
**`reports/analiza.md`**; wersja do przeglądarki: **`reports/analiza.html`**.

## Uruchomienie (PowerShell / Windows)

Python 3.11 lub nowszy (sprawdzono na 3.13).

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/download_data.py
.\.venv\Scripts\python.exe -m src.analysis
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\jupyter-lab.exe notebooks/01_analiza_numeryczna.ipynb
```

Gotowe `.venv` jest już zainstalowane w tym katalogu. Przy odtwarzaniu dokładnie tego
środowiska użyj `requirements-lock.txt` zamiast `requirements.txt` (lock dla Windows/Python 3.13).
Na Linux/macOS odpowiednikiem jest `.venv/bin/python` i `.venv/bin/jupyter-lab`.
Notebook można też otworzyć w VS Code i wybrać interpreter `.venv`.

Ponowne utworzenie i wykonanie notebooka oraz eksport HTML:

```powershell
.\.venv\Scripts\python.exe scripts/build_notebook.py
```

## Dane i ich interpretacja

Źródło: [League of Legends Diamond Ranked Games (10 min), bobbyscience / Yi Lan Ma, Kaggle](https://www.kaggle.com/datasets/bobbyscience/league-of-legends-diamond-ranked-games-10-min).
Pobrano bezpośrednio z publicznego API Kaggle. Adres i SHA-256 pliku zapisano
w `data/raw/source.json`. Nie korzystamy z kopii innych autorów.

- `data/raw/high_diamond_ranked_10min.csv`: niezmieniony plik źródłowy, 9879 × 40.
- `data/processed/high_diamond_ranked_10min_clean.csv`: wynik czyszczenia.
- Jeden wiersz to mecz. Cechy opisują stan po pierwszych 10 minutach.
- `blueWins` to **końcowy** wynik: 1 = blue wygrało, 0 = red wygrało.
- `gameId` to identyfikator, a nie cecha ilościowa do interpretacji.
- `reports/tables/data_dictionary.csv`: opis wszystkich 40 kolumn.

**Brak szeregu czasowego**: jeden pomiar na mecz nie pozwala analizować zmian
w czasie. Kolejność wierszy i numery meczów nie zastępują osi czasu. Ten element
wymaga dodatkowych danych, jeśli prowadząca uzna go za obowiązkowy dla każdego zbioru.

## Pliki projektu

```text
data/raw/                 oryginalny CSV i informacja o pochodzeniu
data/processed/           oczyszczony CSV
notebooks/                analiza krok po kroku z wykonanymi wynikami
src/analysis.py           funkcje czyszczenia, obliczeń i wizualizacji oraz CLI
scripts/download_data.py  powtarzalne pobieranie danych
scripts/build_notebook.py budowa i wykonanie notebooka, eksport HTML
tests/test_cleaning.py    testy naprawy błędów i zachowania poprawnych danych
reports/analiza.md        raport wyników i pokrycie punktów I.1-I.5
reports/analiza.html      samodzielny raport do przeglądarki z obrazami
reports/quality_report.json audyt czyszczenia
reports/tables/           statystyki, korelacje, regresje, klasy, obserwacje odstające
reports/figures/          7 plansz PNG (kołowy, słupkowy, histogramy, boxplot,
                         violinplot, rozrzut z regresją, korelacje i reszty)
```

Statystyki obejmują min, max, średnią, odchylenie standardowe próbki, medianę,
Q1, Q3, P10 i P90. Czyszczenie imputuje cechy, ale nie etykietę wyniku ani ID;
kolumny powiązane algebraicznie przelicza. Wartości odstające są oznaczane,
bez automatycznego uznawania ich za błędy. Testy używają osobnej uszkodzonej kopii.

Regresje opisują zależności w próbie. R² i RMSE nie są oceną na zbiorze testowym,
a korelacje nie wskazują przyczynowości. Nie wykonujemy tutaj zadania klasyfikacji
ani prognozowania szeregów czasowych.

Przed dalszym udostępnianiem danych sprawdź warunki ich wykorzystania na stronie
Kaggle; projekt nie nadaje zbiorowi własnej licencji.
