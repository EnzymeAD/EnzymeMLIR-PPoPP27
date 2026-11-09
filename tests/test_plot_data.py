import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("jaxmd_plot", ROOT / "plotting" / "jaxmd.py")
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)


class PlotDataTests(unittest.TestCase):
    def test_source_paths_and_missing_new_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref = root / "results" / "results_JAXMD.json"
            ref.parent.mkdir()
            ref.write_text("[]")
            selected, output = plot.resolve_paths(root=root)
            self.assertEqual(selected, ref)
            self.assertEqual(output, root / "results/plots/jaxmd/reference")
            with self.assertRaisesRegex(FileNotFoundError, "Run the benchmark first"):
                plot.resolve_paths("new", root=root)
            new = root / "results/jaxmd/results_JAXMD.json"
            new.parent.mkdir()
            new.write_text("[]")
            selected, output = plot.resolve_paths("new", root=root)
            self.assertEqual(selected, new)
            self.assertEqual(output, root / "results/plots/jaxmd/new")
            selected, output = plot.resolve_paths("new", ref, root / "custom output", root=root)
            self.assertEqual(selected, ref)
            self.assertEqual(output, root / "custom output")
        with self.assertRaisesRegex(ValueError, "Invalid source"):
            plot.resolve_paths("typo")

    def test_reference_normalization_and_rendering(self):
        df = plot.load_results(ROOT / "results/results_JAXMD.json")
        baseline = df[(df.pipeline == "Jax") & (df.key == "BothRev")].iloc[0].time_avg_s
        self.assertAlmostEqual(baseline, 0.004279242583333334)
        figures = plot.make_figures(df)
        self.assertEqual(set(figures), {"runtime", "speedup", "memory"})
        with tempfile.TemporaryDirectory() as tmp:
            plot.save_figures(figures, Path(tmp))
            self.assertEqual(len(list(Path(tmp).glob("*.pdf"))), 3)
            self.assertEqual(len(list(Path(tmp).glob("*.png"))), 3)
            self.assertTrue(all(p.stat().st_size > 1000 for p in Path(tmp).iterdir()))
        for figure in figures.values():
            plot.plt.close(figure)


if __name__ == "__main__":
    unittest.main()
