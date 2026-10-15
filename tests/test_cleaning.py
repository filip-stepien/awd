"""Sprawdzamy naprawę błędów, spójność i ochronę etykiet wyniku."""

from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from src.analysis import clean_data, invalid_mask

ROOT = Path(__file__).resolve().parents[1]


class CleaningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = pd.read_csv(ROOT / "data/raw/high_diamond_ranked_10min.csv")

    def test_original_is_preserved_and_valid(self):
        original = self.raw.copy(deep=True)
        cleaned, audit = clean_data(self.raw)
        pd.testing.assert_frame_equal(self.raw, original)
        pd.testing.assert_frame_equal(cleaned, original)
        self.assertEqual(audit["changed_cells_in_retained_rows"], 0)
        self.assertFalse(cleaned.isna().any().any())

    def test_corrupt_copy_is_repaired_with_consistent_derivatives(self):
        damaged = self.raw.head(30).astype(object).copy()
        damaged.loc[0, "blueWardsPlaced"] = "tekst"
        damaged.loc[1, "blueKills"] = -1
        damaged.loc[1, "redDeaths"] = np.nan  # obie kopie uszkodzone
        damaged.loc[2, "blueTotalGold"] = np.inf
        damaged.loc[3, "blueAvgLevel"] = 19
        damaged.loc[4, "blueAssists"] = 1.5
        damaged.loc[5, ["blueFirstBlood", "redFirstBlood"]] = [2, -1]
        damaged.loc[6, "blueGoldDiff"] = 999999
        cleaned, audit = clean_data(damaged)
        self.assertFalse(cleaned.isna().any().any())
        for col in cleaned:
            self.assertFalse(invalid_mask(cleaned[col], col).any(), col)
        np.testing.assert_array_equal(cleaned.blueKills, cleaned.redDeaths)
        np.testing.assert_array_equal(cleaned.blueGoldDiff, cleaned.blueTotalGold - cleaned.redTotalGold)
        np.testing.assert_allclose(cleaned.blueGoldPerMin, cleaned.blueTotalGold / 10)
        np.testing.assert_array_equal(cleaned.blueFirstBlood + cleaned.redFirstBlood, np.ones(len(cleaned)))
        self.assertEqual(cleaned.loc[0, "blueWardsPlaced"], np.rint(self.raw.iloc[1:30].blueWardsPlaced.median()))
        self.assertGreater(audit["changed_cells_in_retained_rows"], 7)
        pd.testing.assert_series_equal(cleaned.blueWins, self.raw.head(30).blueWins)

    def test_missing_kills_are_recovered_from_opponent_deaths(self):
        damaged = self.raw.head(10).astype(object).copy()
        damaged.loc[0, "blueKills"] = np.nan
        cleaned, audit = clean_data(damaged)
        self.assertEqual(cleaned.loc[0, "blueKills"], self.raw.loc[0, "redDeaths"])
        self.assertEqual(audit["imputed_by_column"]["blueKills"], 0)

    def test_bad_labels_are_dropped_not_imputed(self):
        damaged = self.raw.head(10).astype(object).copy()
        damaged.loc[0, "blueWins"] = np.nan
        damaged.loc[1, "blueWins"] = 2
        damaged.loc[2, "gameId"] = -1
        cleaned, audit = clean_data(damaged)
        self.assertEqual(len(cleaned), 7)
        self.assertEqual(audit["dropped_invalid_id_or_target"], 3)

    def test_duplicates_and_conflicting_games(self):
        damaged = pd.concat([self.raw.head(10), self.raw.iloc[[0]], self.raw.iloc[[1]]], ignore_index=True)
        damaged.loc[11, "blueWins"] = 1 - damaged.loc[11, "blueWins"]
        cleaned, audit = clean_data(damaged)
        self.assertEqual(audit["dropped_exact_duplicates"], 1)
        self.assertEqual(audit["dropped_conflicting_id_rows"], 2)
        self.assertEqual(len(cleaned), 9)
        self.assertTrue(cleaned.gameId.is_unique)

    def test_no_valid_values_and_missing_schema_fail_explicitly(self):
        damaged = self.raw.head(10).astype(object).copy()
        damaged["blueWardsPlaced"] = np.nan
        with self.assertRaisesRegex(ValueError, "blueWardsPlaced"):
            clean_data(damaged)
        with self.assertRaisesRegex(ValueError, "Brak wymaganych kolumn"):
            clean_data(self.raw.drop(columns="blueKills"))


if __name__ == "__main__":
    unittest.main()
