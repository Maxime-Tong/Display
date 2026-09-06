import csv
import tempfile
import unittest
from pathlib import Path

from plot_training import plot_history


class PlotTrainingTest(unittest.TestCase):
    def test_writes_both_plots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fields = ["step", "total", "power", "metam", "weber", "ssim", "power_term", "metam_term", "weber_term", "ssim_term", "power_saving"]
            with (root / "history.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow({name: 1 for name in fields})
            plot_history(root / "history.csv", root / "plots")
            self.assertTrue((root / "plots/losses.png").is_file())
            self.assertTrue((root / "plots/power_saving.png").is_file())


if __name__ == "__main__":
    unittest.main()
