"""Zbuduj wykonany notebook i samodzielny raport HTML."""

from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
for variable, subdir in (
    ("JUPYTER_RUNTIME_DIR", "jupyter-runtime"),
    ("JUPYTER_CONFIG_DIR", "jupyter-config"),
    ("IPYTHONDIR", "ipython"),
):
    folder = ROOT / "tmp" / subdir
    folder.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(variable, str(folder))

import nbformat as nbf
from nbclient import NotebookClient
from nbconvert import HTMLExporter


def main():
    cells = []

    def md(text):
        cells.append(nbf.v4.new_markdown_cell(text.strip()))

    def code(text):
        cells.append(nbf.v4.new_code_cell(text.strip()))

    md("""
# Analiza statystyczna meczów League of Legends
### Część I projektu: dane numeryczne

**Pytanie badawcze:** jak statystyki po 10 minutach wiążą się ze sobą i z końcowym wynikiem?

Źródło: [Yi Lan Ma / bobbyscience, Kaggle](https://www.kaggle.com/datasets/bobbyscience/league-of-legends-diamond-ranked-games-10-min).
Jeden wiersz oznacza jeden mecz. `blueWins` jest kategorią końcowego wyniku (0 = wygrana red, 1 = wygrana blue).
Pozostałe statystyki opisują pierwsze 10 minut. Nie są to kolejne chwile jednego meczu.

Notebook można wykonać od początku (`Restart Kernel and Run All Cells`). Funkcje w `src/analysis.py`
są wspólne dla notebooka i skryptu. W HTML kod jest ukryty; notebook zawiera pełne komórki.
""")
    code("""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
from IPython.display import display, Markdown, Image

ROOT = Path.cwd()
if ROOT.name == 'notebooks':
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.analysis import (clean_data, descriptive_statistics, pearson_analysis,
                          linear_regressions, generate_figures, outlier_summary,
                          data_dictionary, write_report)
pd.set_option('display.max_columns', 12)
pd.set_option('display.float_format', lambda x: f'{x:,.4f}')
raw = pd.read_csv(ROOT / 'data/raw/high_diamond_ranked_10min.csv')
display(Markdown(f'**Wczytano {raw.shape[0]:,} meczów i {raw.shape[1]} kolumn.**'))
display(raw.head())
""")
    md("""
## I.1. Kontrola jakości i czyszczenie

Sprawdzamy braki, typy, skończoność wartości, dziedziny, duplikaty i spójność kolumn pochodnych.
Liczniki są nieujemnymi liczbami całkowitymi, poziom mieści się w [1, 18], a kategorie binarne w {0, 1}.
Różnice złota i doświadczenia **mogą być ujemne**.

Braki cech: mediana (zaokrąglona dla liczników) lub dominanta (kategorie).
ID i wyniku nie uzupełniamy. Kolumny pochodne przeliczamy z podstawowych statystyk.
Przy sprzeczności pierwszeństwo mają `Kills`, `TotalGold`, `TotalExperience`,
`TotalMinionsKilled`, `Dragons`, `Heralds` i `blueFirstBlood`.
Szczegóły audytu zapisujemy w JSON. Oryginalnego CSV nie zmieniamy.
""")
    code("""
data, audit = clean_data(raw)
display(pd.DataFrame({
    'miara': ['Wiersze przed', 'Wiersze po', 'Brakujące komórki przed',
              'Błędne/brakujące komórki przed', 'Duplikaty', 'Konfliktowe rekordy ID',
              'Błędny ID/wynik', 'Zmienione komórki', 'Brakujące komórki po'],
    'wartość': [audit['rows_before'], audit['rows_after'], audit['missing_cells_before'],
                sum(audit['invalid_cells_by_column'].values()), audit['dropped_exact_duplicates'],
                audit['dropped_conflicting_id_rows'], audit['dropped_invalid_id_or_target'],
                audit['changed_cells_in_retained_rows'], audit['missing_cells_after']]
}))
""")
    md("""
### Wartości odstające

Reguła 1,5 IQR oznacza nietypowe wartości, a nie automatycznie błędy.
Zachowujemy je w analizie. Wardy mają długi prawy ogon; nie przyjmujemy arbitralnego
progu poprawności. Przy IQR = 0 pomijamy regułę, ponieważ dla rzadkich liczników byłaby myląca.

Plik `tests/test_cleaning.py` sprawdza naprawy na **osobnej sztucznie uszkodzonej kopii**,
w tym tekst, NaN, nieskończoność, ujemne i ułamkowe liczniki oraz błędne kategorie.
""")
    code("""
outliers = outlier_summary(data)
display(outliers.sort_values('liczba_poza_IQR', ascending=False).head(8))
display(data[['blueWardsPlaced', 'redWardsPlaced']].describe().T)
""")
    md("""
## I.2. Statystyki opisowe

Obliczamy min, max, średnią, odchylenie standardowe próbki (`ddof=1`), medianę,
Q1, Q3, P10 i P90. Kwantyle: interpolacja liniowa. `gameId` pomijamy,
ponieważ to identyfikator. Dla zmiennych 0/1 średnia oznacza udział kategorii 1.
Smoki i Heroldy są licznikami, choć w tym pliku mają jedynie wartości 0/1.
""")
    code("""
descriptive = descriptive_statistics(data)
display(descriptive)
""")
    md("""
## I.3. Korelacja liniowa Pearsona

Wybrane pary dotyczą ekonomii, walki i wyniku. Dla zmiennej liczbowej i kategorii 0/1
Pearson jest równoważny korelacji punktowo-dwuseryjnej. To opis współwystępowania,
nie dowód przyczynowości. Dwustronne wartości p mają charakter eksploracyjny,
bez korekty za wielokrotne porównania. Numeryczne p = 0 oznacza niedomiar numeryczny.
""")
    code("""
correlations = pearson_analysis(data)
display(correlations)
gold_result = correlations.query("x == 'blueGoldDiff' and y == 'blueWins'").iloc[0]
display(Markdown(f"Związek przewagi złota z końcową wygraną blue: **r = {gold_result.r:.4f}**."))
""")
    md("""
## I.4. Prosta regresja liniowa

Główna para: **x = różnica złota**, **y = różnica doświadczenia** (blue minus red).
Druga para: **x = zabójstwa blue**, **y = łączne złoto blue**.
Każdy model ma jedną zmienną objaśniającą i wyraz wolny.

Omijamy tożsamości algebraiczne, np. złoto i złoto/minutę. Nie traktujemy zwykłej
regresji liniowej wyniku 0/1 jako modelu prawdopodobieństwa wygranej.
""")
    code("""
regressions = linear_regressions(data)
display(regressions)
for row in regressions.itertuples():
    display(Markdown(f'**{row.y} = {row.intercept:.4f} + {row.slope:.6f} × {row.x}**, '
                     f'R² = **{row.R2:.4f}**, RMSE w próbie = **{row.RMSE_in_sample:.2f}**.'))
main = regressions.iloc[0]
display(Markdown(f'W dopasowanym modelu dodatkowe 1000 jednostek przewagi złota '
                 f'wiąże się przeciętnie z **{1000*main.slope:.1f} XP** większej przewagi. '
                 f'Model wyjaśnia **{100*main.R2:.2f}%** zmienności różnicy doświadczenia w tej próbie.'))
""")
    md("""
## I.5. Wizualizacje

Wykresy obejmują wszystkie poprawne rekordy, również wartości odstające.
Najpierw porównujemy klasy wyniku i rozkłady cech, potem zależności i reszty regresji.
""")
    code("""
figures = generate_figures(data, regressions, ROOT / 'reports/figures')
counts = data.blueWins.value_counts().reindex([0, 1], fill_value=0)
display(Markdown(f'Blue wygrało **{counts[1]:,} ({100*counts[1]/len(data):.2f}%)**, '
                 f'red **{counts[0]:,} ({100*counts[0]/len(data):.2f}%)**. '
                 'Klasy są niemal zrównoważone.'))
display(Image(filename=str(figures[0])))
""")
    md("### Histogramy: środek, rozrzut i nietypowe obserwacje")
    code("display(Image(filename=str(figures[1])))")
    md("""
### Pudełka i wioliny: rozkłady w klasach wyniku

Porównujemy różnicę złota i doświadczenia w meczach wygranych przez blue i red.
Wioliny przedstawiają wygładzone oszacowanie gęstości. Pudełka pokazują medianę,
kwartyle oraz obserwacje poza wąsami (1,5 IQR).
""")
    code("""
display(Image(filename=str(figures[2])))
display(Image(filename=str(figures[3])))
""")
    md("### Rozrzut i proste regresji")
    code("display(Image(filename=str(figures[4])))")
    md("""
### Mapa korelacji

Mapa zawiera wybrane zmienne; pełna macierz jest w CSV. Niektóre pozostałe kolumny
są redundantne, np. `redGoldDiff = -blueGoldDiff`, `redDeaths = blueKills`.
Takie korelacje wynikają z definicji i nie są odkryciem badawczym.
""")
    code("display(Image(filename=str(figures[5])))")
    md("""
### Diagnostyka reszt

Na wykresie reszt nie widać wyraźnej krzywizny ani mocnego efektu lejka w środku danych.
Q-Q jest bliski prostej w centralnej części, z odchyleniami na krańcach, zwłaszcza w lewym ogonie.
To ocena wizualna, nie formalne potwierdzenie założeń. R² i RMSE dotyczą danych dopasowania;
nie mierzą jakości prognozy dla nowych meczów.
""")
    code("display(Image(filename=str(figures[6])))")
    md("""
## Wnioski i ograniczenia

1. Cechy liczbowe i binarny wynik spełniają zakres części I („Dane numeryczne”).
2. Audyt oryginalnego pliku nie wymagał zmiany wartości. Funkcja napraw jest sprawdzona testami.
3. Klasy wyniku są niemal równe. Przewagi złota i XP współwystępują z końcową wygraną.
4. Regresja różnicy doświadczenia względem różnicy złota opisuje silną zależność liniową,
   jednak pozostawia niewyjaśnioną zmienność i nie dowodzi przyczynowości.
5. **Brak szeregu czasowego**: po jednym pomiarze w 10. minucie na mecz. Indeks wiersza
   i kolejność `gameId` nie są osią czasu. Jeśli wykres czasowy jest bezwzględnie wymagany,
   trzeba użyć dodatkowych danych z pomiarami w wielu chwilach.
6. Nie uogólniamy wyników na wszystkie rangi i aktualną wersję gry. Nieznany dobór próby
   oraz możliwe powtarzanie się graczy ograniczają interpretację wnioskowania statystycznego.

### Mapa wymagań

| Punkt | Wynik |
| --- | --- |
| I.1 | Funkcja czyszczenia, audyt, testy uszkodzonych danych |
| I.2 | Pełna tabela opisowa 39 kolumn poza ID |
| I.3 | Współczynniki Pearsona i mapa korelacji |
| I.4 | Dwie proste regresje, wzory, R² i diagnostyka |
| I.5 | Kołowy, słupkowy, histogramy, pudełka, wioliny, rozrzut z regresją |
| Szeregi czasowe | Ograniczenie źródła danych, opisane powyżej |
""")
    md("## Eksport wyników\n\nPoniższa komórka aktualizuje tabele CSV, audyt, raport Markdown i oczyszczony plik.")
    code("""
output = ROOT / 'reports'
tables = output / 'tables'
tables.mkdir(parents=True, exist_ok=True)
descriptive.to_csv(tables / 'descriptive_statistics.csv')
correlations.to_csv(tables / 'pearson.csv', index=False)
regressions.to_csv(tables / 'regressions.csv', index=False)
outliers.to_csv(tables / 'outliers.csv', index=False)
data_dictionary().to_csv(tables / 'data_dictionary.csv', index=False)
counts.rename_axis('blueWins').reset_index(name='n').to_csv(tables / 'class_counts.csv', index=False)
data.drop(columns='gameId').corr().to_csv(tables / 'correlation_matrix.csv')
(output / 'quality_report.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
(ROOT / 'data/processed').mkdir(parents=True, exist_ok=True)
data.to_csv(ROOT / 'data/processed/high_diamond_ranked_10min_clean.csv', index=False)
write_report(data, audit, descriptive, correlations, regressions, output)
print('Zapisano raporty i tabele w reports/, dane w data/processed/.')
""")

    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3 (.venv)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
    })
    destination = ROOT / "notebooks/01_analiza_numeryczna.ipynb"
    destination.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, destination)
    print("Wykonywanie notebooka...", flush=True)
    NotebookClient(notebook, timeout=180, startup_timeout=60, kernel_name="python3",
                   resources={"metadata": {"path": str(ROOT)}}).execute()
    nbf.write(notebook, destination)
    exporter = HTMLExporter(template_name="lab")
    exporter.exclude_input = True
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True
    body, _ = exporter.from_notebook_node(notebook)
    (ROOT / "reports/analiza.html").write_text(body, encoding="utf-8")
    print(f"OK: {destination}")
    print(f"OK: {ROOT / 'reports/analiza.html'}")


if __name__ == "__main__":
    main()
