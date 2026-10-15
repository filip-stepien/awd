"""Powtarzalna analiza punktu I: python -m src.analysis."""

from pathlib import Path
import argparse
import json
import os

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp" / "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

SUFFIXES = {
    "WardsPlaced": "Liczba postawionych wardów",
    "WardsDestroyed": "Liczba zniszczonych wardów przeciwnika",
    "FirstBlood": "Pierwsze zabójstwo: 1 = ta drużyna, 0 = przeciwnik",
    "Kills": "Liczba zabójstw przeciwników",
    "Deaths": "Liczba śmierci graczy drużyny",
    "Assists": "Liczba asyst",
    "EliteMonsters": "Liczba smoków i Heroldów łącznie",
    "Dragons": "Liczba zabitych smoków",
    "Heralds": "Liczba zabitych Heroldów",
    "TowersDestroyed": "Liczba zniszczonych wież przeciwnika",
    "TotalGold": "Łączne złoto drużyny",
    "AvgLevel": "Średni poziom graczy drużyny",
    "TotalExperience": "Łączne doświadczenie drużyny",
    "TotalMinionsKilled": "Liczba zabitych minionów z linii",
    "TotalJungleMinionsKilled": "Liczba zabitych potworów w dżungli",
    "GoldDiff": "Złoto drużyny minus złoto przeciwnika",
    "ExperienceDiff": "Doświadczenie drużyny minus doświadczenie przeciwnika",
    "CSPerMin": "Miniony z linii na minutę: TotalMinionsKilled / 10",
    "GoldPerMin": "Złoto na minutę: TotalGold / 10",
}
COLUMNS = ["gameId", "blueWins"] + [t + s for t in ("blue", "red") for s in SUFFIXES]
BINARY = {"blueWins", "blueFirstBlood", "redFirstBlood"}
CONTINUOUS = {t + s for t in ("blue", "red") for s in ("AvgLevel", "CSPerMin", "GoldPerMin")}
DERIVED = {t + s for t in ("blue", "red") for s in (
    "Deaths", "EliteMonsters", "GoldDiff", "ExperienceDiff", "CSPerMin", "GoldPerMin"
)} | {"redFirstBlood"}
PAIRS = [
    ("blueGoldDiff", "blueExperienceDiff"),
    ("blueKills", "blueTotalGold"),
    ("blueTotalMinionsKilled", "blueTotalGold"),
    ("blueGoldDiff", "blueWins"),
    ("blueExperienceDiff", "blueWins"),
    ("blueKills", "blueWins"),
    ("blueDeaths", "blueWins"),
    ("blueFirstBlood", "blueWins"),
]
REGRESSION_PAIRS = PAIRS[:2]
COLORS = ["#cf5262", "#2878b5"]
LABELS = ["Wygrana red", "Wygrana blue"]


def invalid_mask(series, column):
    """Sprawdź dziedzinę; różnice złota/XP mogą być ujemne."""
    bad = series.isna() | ~np.isfinite(series)
    if column in BINARY:
        return bad | ~series.isin([0, 1])
    if column not in CONTINUOUS:
        bad |= series.notna() & (series != np.floor(series))
    if column == "gameId" or column.endswith(("TotalGold", "GoldPerMin")):
        bad |= series <= 0
    elif column.endswith("AvgLevel"):
        bad |= (series < 1) | (series > 18)
    elif not column.endswith(("GoldDiff", "ExperienceDiff")):
        bad |= series < 0
    return bad


def derived_values(df):
    result = {"redFirstBlood": 1 - df["blueFirstBlood"]}
    for team, other in (("blue", "red"), ("red", "blue")):
        result.update({
            team + "Deaths": df[other + "Kills"],
            team + "EliteMonsters": df[team + "Dragons"] + df[team + "Heralds"],
            team + "GoldDiff": df[team + "TotalGold"] - df[other + "TotalGold"],
            team + "ExperienceDiff": df[team + "TotalExperience"] - df[other + "TotalExperience"],
            team + "CSPerMin": df[team + "TotalMinionsKilled"] / 10,
            team + "GoldPerMin": df[team + "TotalGold"] / 10,
        })
    return result


def clean_data(raw):
    """Zwróć oczyszczoną kopię i audyt. Oryginał pozostaje niezmieniony.

    Nie zgadujemy wyniku/ID. Konfliktowe rekordy tego samego meczu usuwamy.
    Cechy uzupełniamy medianą (liczniki zaokrągloną) lub dominantą (0/1).
    Cechy pochodne przeliczamy, by zachować spójność. Odstające wartości
    bez naruszenia dziedziny pozostają w zbiorze.
    """
    missing = set(COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"Brak wymaganych kolumn: {sorted(missing)}")
    df = raw[COLUMNS].apply(pd.to_numeric, errors="coerce").copy()
    audit = {
        "rows_before": len(raw), "columns": len(COLUMNS),
        "missing_cells_before": int(raw[COLUMNS].isna().sum().sum()),
        "invalid_cells_by_column": {}, "imputed_by_column": {},
        "derived_repaired_by_column": {},
    }
    for col in COLUMNS:
        mask = invalid_mask(df[col], col)
        audit["invalid_cells_by_column"][col] = int(mask.sum())
        df.loc[mask, col] = np.nan

    unusable = df[["gameId", "blueWins"]].isna().any(axis=1)
    audit["dropped_invalid_id_or_target"] = int(unusable.sum())
    df = df.loc[~unusable].copy()
    duplicates = df.duplicated()
    audit["dropped_exact_duplicates"] = int(duplicates.sum())
    df = df.loc[~duplicates].copy()
    conflicts = df.gameId.duplicated(keep=False)
    audit["dropped_conflicting_id_rows"] = int(conflicts.sum())
    df = df.loc[~conflicts].copy()
    if df.empty:
        raise ValueError("Brak poprawnych rekordów do analizy.")

    # Poprawny odpowiednik pozwala odtworzyć brak bez imputacji statystycznej.
    for col, counterpart in (("blueKills", "redDeaths"), ("redKills", "blueDeaths")):
        df[col] = df[col].fillna(df[counterpart])
    df["blueFirstBlood"] = df["blueFirstBlood"].fillna(1 - df["redFirstBlood"])

    for col in COLUMNS:
        if col in DERIVED or col in {"gameId", "blueWins"}:
            continue
        count = int(df[col].isna().sum())
        audit["imputed_by_column"][col] = count
        if count:
            observed = df[col].dropna()
            if observed.empty:
                raise ValueError(f"Brak poprawnych wartości w {col}; imputacja niemożliwa.")
            fill = observed.mode().iloc[0] if col in BINARY else observed.median()
            if col not in CONTINUOUS:
                fill = np.rint(fill)
            df[col] = df[col].fillna(fill)

    for col, expected in derived_values(df).items():
        audit["derived_repaired_by_column"][col] = int((~np.isclose(df[col], expected)).sum())
        df[col] = expected
    for col in COLUMNS:
        if invalid_mask(df[col], col).any():
            raise ValueError(f"Walidacja po czyszczeniu nie powiodła się: {col}")
        if col not in CONTINUOUS:
            df[col] = df[col].astype("int64")

    original = raw.loc[df.index, COLUMNS].apply(pd.to_numeric, errors="coerce")
    same = np.isclose(df.to_numpy(), original.to_numpy(), equal_nan=True)
    audit["changed_cells_in_retained_rows"] = int((~same).sum())
    audit["rows_after"] = len(df)
    audit["missing_cells_after"] = int(df.isna().sum().sum())
    return df.reset_index(drop=True), audit


def descriptive_statistics(df):
    """Odchylenie standardowe próbki (ddof=1), kwantyle z interpolacją liniową."""
    features = df.drop(columns="gameId")
    result = features.describe(percentiles=[.1, .25, .5, .75, .9]).T
    return result.rename(columns={
        "count": "n", "mean": "srednia", "std": "odchylenie_std",
        "10%": "P10", "25%": "Q1", "50%": "mediana", "75%": "Q3", "90%": "P90",
    })[["n", "min", "max", "srednia", "odchylenie_std", "mediana", "Q1", "Q3", "P10", "P90"]]


def pearson_analysis(df):
    records = []
    for x, y in PAIRS:
        r = stats.pearsonr(df[x], df[y])
        records.append({"x": x, "y": y, "n": len(df), "r": r.statistic, "p_value": r.pvalue})
    return pd.DataFrame(records)


def linear_regressions(df):
    records = []
    for x, y in REGRESSION_PAIRS:
        fit = stats.linregress(df[x], df[y])
        residuals = df[y] - (fit.intercept + fit.slope * df[x])
        records.append({
            "x": x, "y": y, "n": len(df), "intercept": fit.intercept,
            "slope": fit.slope, "r": fit.rvalue, "R2": fit.rvalue ** 2,
            "RMSE_in_sample": float(np.sqrt(np.mean(residuals ** 2))),
        })
    return pd.DataFrame(records)


def outlier_summary(df):
    """IQR służy do oznaczenia obserwacji, nie jest testem poprawności rekordu."""
    cols = [c for c in df if c not in BINARY and c != "gameId"]
    records = []
    for col in cols:
        q1, q3 = df[col].quantile([.25, .75])
        iqr = q3 - q1
        if iqr == 0:  # degeneracja IQR dla rzadkich liczników
            records.append({"zmienna": col, "dolna_granica": np.nan,
                            "gorna_granica": np.nan, "liczba_poza_IQR": np.nan})
        else:
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            count = int(((df[col] < lower) | (df[col] > upper)).sum())
            records.append({"zmienna": col, "dolna_granica": lower,
                            "gorna_granica": upper, "liczba_poza_IQR": count})
    return pd.DataFrame(records)


def data_dictionary():
    rows = [
        {"kolumna": "gameId", "rola": "identyfikator", "opis": "ID meczu; wyłączony ze statystyk i korelacji"},
        {"kolumna": "blueWins", "rola": "kategoria binarna", "opis": "Końcowy wynik: 1 = wygrana blue, 0 = wygrana red"},
    ]
    for team in ("blue", "red"):
        for suffix, description in SUFFIXES.items():
            col = team + suffix
            role = "kategoria binarna" if col in BINARY else "liczbowa"
            rows.append({"kolumna": col, "rola": role, "opis": f"{team}: {description}"})
    return pd.DataFrame(rows)


def generate_figures(df, regressions, folder):
    folder.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font="DejaVu Sans", context="notebook")
    paths = []

    def save(fig, name):
        path = folder / f"{name}.png"
        fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        paths.append(path)

    counts = df.blueWins.value_counts().reindex([0, 1], fill_value=0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    bars = axes[0].bar(LABELS, counts, color=COLORS, width=.6)
    axes[0].bar_label(bars, padding=4)
    axes[0].set(ylabel="Liczba meczów", title="Liczebność klas wyniku", ylim=(0, counts.max()*1.17))
    axes[1].pie(counts, labels=LABELS, colors=COLORS, autopct="%.2f%%", startangle=90,
                textprops={"fontsize": 11})
    axes[1].set_title(f"Udział klas (n = {len(df):,})")
    save(fig, "01_klasy")

    features = [("blueGoldDiff", "Różnica złota: blue - red"),
                ("blueExperienceDiff", "Różnica doświadczenia: blue - red"),
                ("blueKills", "Zabójstwa blue"),
                ("blueWardsPlaced", "Postawione wardy blue")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for ax, (col, label) in zip(axes.flat, features):
        sns.histplot(data=df, x=col, bins=35 if col != "blueKills" else range(24),
                     discrete=col == "blueKills", color="#2878b5", ax=ax)
        ax.axvline(df[col].median(), color="#ae334e", ls="--", label="Mediana")
        ax.set(xlabel=label, ylabel="Liczba meczów", title=label)
        ax.legend()
    save(fig, "02_histogramy")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for ax, (col, label) in zip(axes, features[:2]):
        sns.boxplot(data=df, x="blueWins", y=col, hue="blueWins", palette=COLORS,
                    legend=False, fliersize=2, ax=ax)
        ax.set_xticks([0, 1], LABELS)
        ax.set(xlabel="Końcowy wynik meczu", ylabel=label, title=label)
    save(fig, "03_pudelkowe")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for ax, (col, label) in zip(axes, features[:2]):
        sns.violinplot(data=df, x="blueWins", y=col, hue="blueWins", palette=COLORS,
                       legend=False, inner="quart", cut=0, ax=ax)
        ax.set_xticks([0, 1], LABELS)
        ax.set(xlabel="Końcowy wynik meczu", ylabel=label, title=label)
    save(fig, "04_wiolinowe")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), layout="constrained")
    axis_labels = [("Różnica złota: blue - red", "Różnica doświadczenia: blue - red"),
                   ("Zabójstwa blue", "Łączne złoto blue")]
    for ax, (_, reg), (xl, yl) in zip(axes, regressions.iterrows(), axis_labels):
        ax.scatter(df[reg.x], df[reg.y], s=7, alpha=.16, color="#2878b5", rasterized=True)
        line_x = np.linspace(df[reg.x].min(), df[reg.x].max(), 100)
        ax.plot(line_x, reg.intercept + reg.slope*line_x, color="#ae334e", lw=2,
                label=f"y = {reg.intercept:.2f} + {reg.slope:.4f}x\nR² = {reg.R2:.3f}")
        ax.set(xlabel=xl, ylabel=yl, title="Regresja liniowa OLS")
        ax.legend(loc="upper left")
    save(fig, "05_regresje")

    cols = ["blueGoldDiff", "blueExperienceDiff", "blueKills", "blueDeaths", "blueAssists",
            "blueTotalMinionsKilled", "blueWardsPlaced", "blueFirstBlood", "blueWins"]
    fig, ax = plt.subplots(figsize=(11, 9), layout="constrained")
    sns.heatmap(df[cols].corr(method="pearson"), annot=True, fmt=".2f", cmap="vlag",
                vmin=-1, vmax=1, center=0, square=True, ax=ax, annot_kws={"size": 9})
    ax.set_title("Korelacje Pearsona - wybrane zmienne")
    ax.tick_params(axis="x", labelrotation=45)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    save(fig, "06_korelacje")

    reg = regressions.iloc[0]
    predicted = reg.intercept + reg.slope * df[reg.x]
    residuals = df[reg.y] - predicted
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    axes[0].scatter(predicted, residuals, s=7, alpha=.18, color="#2878b5")
    axes[0].axhline(0, color="#ae334e", ls="--")
    axes[0].set(xlabel="Dopasowana różnica doświadczenia", ylabel="Reszta modelu",
                title="Reszty względem wartości dopasowanych")
    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set(title="Wykres Q-Q reszt", xlabel="Kwantyle rozkładu normalnego", ylabel="Kwantyle reszt")
    save(fig, "07_reszty")
    return paths


def markdown_table(frame, digits=4):
    def cell(value):
        if isinstance(value, (float, np.floating)):
            if abs(value) < .0001 and value != 0:
                return f"{value:.2e}"
            return f"{value:.{digits}f}"
        return str(value)
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |",
             "| " + " | ".join(["---"]*len(frame.columns)) + " |"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join(lines)


def write_report(df, audit, descriptive, correlations, regressions, folder):
    r = regressions.iloc[0]
    r2 = regressions.iloc[1]
    selected = ["blueTotalGold", "blueKills", "blueGoldDiff", "blueExperienceDiff", "blueWardsPlaced"]
    counts = df.blueWins.value_counts().reindex([0, 1], fill_value=0)
    report = f"""# Analiza statystyczna meczów League of Legends

## Cel i zakres

Realizacja części I („Dane numeryczne”) wymagań projektu. Pytania: jak wyglądają
rozkłady statystyk po 10 minutach, jak różnica złota wiąże się z różnicą doświadczenia
i jak te przewagi wiążą się z końcowym wynikiem meczu?

Źródło: [Yi Lan Ma / bobbyscience, Kaggle](https://www.kaggle.com/datasets/bobbyscience/league-of-legends-diamond-ranked-games-10-min).
Analizowany plik ma **{len(df):,} rekordów i {len(df.columns)} kolumn**.
Jednostką obserwacji jest jeden mecz; `blue` i `red` oznaczają drużyny.
Cechy są agregatami z pierwszych 10 minut, a `blueWins` to końcowy wynik (0 = red, 1 = blue).
`gameId` jest identyfikatorem i nie bierze udziału w statystykach ani korelacjach.

## I.1. Weryfikacja i czyszczenie

- Brakujące komórki przed czyszczeniem: **{audit['missing_cells_before']}**.
- Wartości poza zakodowaną dziedziną, w tym braki: **{sum(audit['invalid_cells_by_column'].values())}**.
- Usunięte dokładne duplikaty: **{audit['dropped_exact_duplicates']}**.
- Usunięte rekordy z błędnym ID/wynikiem: **{audit['dropped_invalid_id_or_target']}**.
- Usunięte rekordy z konfliktowymi ID: **{audit['dropped_conflicting_id_rows']}**.
- Zmienione komórki w zachowanych rekordach: **{audit['changed_cells_in_retained_rows']}**.
- Braki po czyszczeniu: **{audit['missing_cells_after']}**.

Funkcja `clean_data` wykrywa tekst w polach liczbowych, NaN, nieskończoność,
ujemne liczniki, ułamkowe liczniki, niepoprawne wartości binarne i poziomy poza [1, 18].
Ujemne różnice złota/XP są poprawne. Braki cech uzupełnia medianą, zaokrągloną
dla liczników, albo dominantą dla kategorii binarnych. Nie imputuje ID ani wyniku.
Kolumny pochodne przelicza z danych podstawowych; poprawne aliasy zabójstw/śmierci
wykorzystuje przed imputacją. Przy sprzeczności pierwszeństwo mają `Kills`, `TotalGold`,
`TotalExperience`, `TotalMinionsKilled`, `Dragons`, `Heralds` i `blueFirstBlood`.
Brak jakiejkolwiek poprawnej wartości cechy powoduje jawny błąd.

Obserwacje poza 1,5 IQR są oznaczane w `tables/outliers.csv`, lecz pozostają w danych.
Szczególnie wardy wymagają ostrożności: maksimum blue wynosi **{df.blueWardsPlaced.max()}**,
red **{df.redWardsPlaced.max()}**, a mediana blue **{df.blueWardsPlaced.median():.0f}**.
Duża wartość nie stanowi sama w sobie dowodu błędu. Dla IQR = 0 reguła jest pomijana.
Test na osobnej, sztucznie uszkodzonej kopii danych sprawdza rzeczywistą naprawę;
nie dodajemy sztucznych błędów do wyników analizy.

## I.2. Statystyki opisowe

Odchylenie standardowe liczymy dla próbki (`ddof=1`), a kwantyle z interpolacją liniową.
Pełna tabela obejmuje 39 kolumn poza ID: `tables/descriptive_statistics.csv`.
Średnia zmiennej binarnej jest udziałem kategorii 1; kwartyle nie opisują dla niej
rozkładu ciągłego. Smoki i Heroldy są licznikami, choć w tym pliku przyjmują tylko 0/1.

{markdown_table(descriptive.loc[selected].reset_index(names='zmienna'), 2)}

Blue wygrało **{counts[1]:,}** meczów (**{100*counts[1]/len(df):.2f}%**),
red **{counts[0]:,}** (**{100*counts[0]/len(df):.2f}%**). Klasy są niemal zrównoważone.

![Liczebności i udziały klas](figures/01_klasy.png)
![Histogramy](figures/02_histogramy.png)

## I.3. Korelacja Pearsona

{markdown_table(correlations)}

Korelacja różnicy złota z różnicą doświadczenia wynosi **r = {r.r:.4f}**:
większa przewaga złota wiąże się z większą przewagą XP. Korelacje z `blueWins`
opisują związek z kategorią wyniku; dla pary zmiennej liczbowej i 0/1 Pearson
jest korelacją punktowo-dwuseryjną. Nie oznacza to związku przyczynowego.
Wartości p są dwustronne, eksploracyjne, bez korekty za wielokrotne porównania.
Zapis p = 0 w CSV oznacza niedomiar numeryczny, a nie dosłowne zerowe prawdopodobieństwo.

![Wybrane korelacje](figures/06_korelacje.png)

## I.4. Prosta regresja liniowa

Główny model OLS (x = `blueGoldDiff`, y = `blueExperienceDiff`):

**ŷ = {r.intercept:.4f} + {r.slope:.6f} · x**, **R² = {r.R2:.4f}**.

Dodatkowe 1000 jednostek przewagi złota wiąże się przeciętnie z
**{r.slope*1000:.1f}** jednostkami większej przewagi doświadczenia w dopasowanym modelu.
Model wyjaśnia **{r.R2*100:.2f}%** zmienności różnicy XP w analizowanej próbie.
RMSE na danych dopasowania wynosi **{r.RMSE_in_sample:.2f} XP**.

Drugi model (x = `blueKills`, y = `blueTotalGold`):
**ŷ = {r2.intercept:.4f} + {r2.slope:.6f} · x**, **R² = {r2.R2:.4f}**.

To analiza zależności w próbie, bez oceny jakości predykcji na nowych meczach.
Nie regresujemy wyniku 0/1 zwykłą prostą jako modelu prawdopodobieństwa.
Nie wybieramy par będących tożsamościami algebraicznymi, np. TotalGold i GoldPerMin.

![Wykresy rozrzutu i proste regresji](figures/05_regresje.png)
![Diagnostyka reszt głównej regresji](figures/07_reszty.png)

Na wykresie reszt nie widać wyraźnej krzywizny ani mocnego efektu lejka w centralnej
części danych. Wykres Q-Q jest bliski prostej w środku, z odchyleniami na krańcach,
zwłaszcza w lewym ogonie. To ocena wizualna, nie formalne potwierdzenie założeń.
Nie wyprowadzamy z modelu wniosków przyczynowych.

## I.5. Wizualizacje i ograniczenia

![Wykresy pudełkowe](figures/03_pudelkowe.png)
![Wykresy wiolinowe](figures/04_wiolinowe.png)

Wykresy porównują przewagi drużyny blue w meczach później wygranych i przegranych.
Wioliny są wygładzonym oszacowaniem gęstości, a nie histogramem.

**Szereg czasowy: nie dotyczy tego zbioru.** Dla każdego meczu istnieje jeden
punkt pomiaru (10. minuta), bez kolejnych minut i daty rozpoczęcia. Indeks wiersza
ani sortowanie po `gameId` nie tworzą szeregu czasowego. Jeśli prowadząca wymaga
tego wykresu bez wyjątku, potrzebny będzie dodatkowy zbiór z pomiarami wielu chwil.

Wyniki dotyczą tej próby meczów o wysokiej randze. Nie są dowodem reprezentatywności
dla wszystkich graczy ani aktualnej wersji gry. Sposób doboru próby i możliwa obecność
tych samych graczy w różnych meczach ograniczają interpretację testów istotności.

## Pokrycie wymagań

| Wymaganie | Realizacja |
| --- | --- |
| I.1. Czyszczenie i braki | `clean_data`, audyt JSON, testy naprawy |
| I.2. Min/max, średnia, SD, mediana, Q1/Q3, P10/P90 | CSV wszystkich cech poza ID |
| I.3. Pearson | Tabela wybranych par i mapa korelacji |
| I.4. Regresja i wzór | Dwa modele OLS, współczynniki, R², rozrzut |
| I.5. Kołowy i słupkowy | Liczebności klas wyniku |
| I.5. Histogram, pudełkowy, wiolinowy | Rozkłady i podział według wyniku |
| I.5. Szereg czasowy | Brak odpowiednich danych; ograniczenie opisane powyżej |
"""
    (folder / "analiza.md").write_text(report, encoding="utf-8")


def run_analysis(data_path=None, output=None):
    data_path = Path(data_path or ROOT / "data/raw/high_diamond_ranked_10min.csv")
    output = Path(output or ROOT / "reports")
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(data_path)
    df, audit = clean_data(raw)
    descriptive = descriptive_statistics(df)
    correlations = pearson_analysis(df)
    regressions = linear_regressions(df)
    outliers = outlier_summary(df)
    for name, frame, index in [
        ("descriptive_statistics", descriptive, True), ("pearson", correlations, False),
        ("regressions", regressions, False), ("outliers", outliers, False),
        ("data_dictionary", data_dictionary(), False),
        ("class_counts", df.blueWins.value_counts().sort_index().rename_axis("blueWins").reset_index(name="n"), False),
        ("correlation_matrix", df.drop(columns="gameId").corr(), True),
    ]:
        frame.to_csv(tables / f"{name}.csv", index=index)
    (output / "quality_report.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    processed = ROOT / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed / "high_diamond_ranked_10min_clean.csv", index=False)
    figures = generate_figures(df, regressions, output / "figures")
    write_report(df, audit, descriptive, correlations, regressions, output)
    return {"data": df, "audit": audit, "descriptive": descriptive, "correlations": correlations,
            "regressions": regressions, "outliers": outliers, "figures": figures, "output": output}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = run_analysis(args.data, args.output)
    print(f"OK: {len(results['data'])} meczow, raport: {results['output'] / 'analiza.md'}")
    print(results["regressions"].to_string(index=False))


if __name__ == "__main__":
    main()
