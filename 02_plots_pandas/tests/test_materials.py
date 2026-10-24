"""Validate the displayed code/result pairs and the lesson datasets."""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import tempfile
import unittest

import nbformat
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


class LessonMaterialsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.examples = json.loads((ROOT / "slides/examples.json").read_text())
        cls.notebook = nbformat.read(ROOT / "seminar/pandas_plots.ipynb", as_version=4)
        cls.raw = pd.read_csv(ROOT / "data/rides.csv")
        cls.clean = cls.raw.drop_duplicates()

    def test_rides_contract(self) -> None:
        self.assertEqual(self.raw.shape, (61, 4))
        self.assertEqual(self.raw.duplicated().sum(), 1)
        self.assertEqual(self.raw["distance_km"].isna().sum(), 2)
        self.assertEqual(self.raw["minutes"].isna().sum(), 3)
        self.assertEqual(len(self.clean), 60)
        self.assertTrue(self.clean["trip_id"].is_unique)
        self.assertEqual(len(self.clean.dropna(subset=["distance_km", "minutes"])), 55)
        self.assertTrue(self.clean["distance_km"].dropna().gt(0).all())
        self.assertTrue(self.clean["minutes"].dropna().gt(0).all())

    def test_weather_contract(self) -> None:
        weather = pd.read_csv(ROOT / "data/weather.csv")
        self.assertEqual(weather.shape, (14, 4))
        self.assertEqual(weather["day"].tolist(), list(range(1, 15)))
        self.assertFalse(weather.isna().any().any())
        self.assertTrue(weather["wind_ms"].ge(0).all())

    def test_reproducible_data(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "lesson_data", ROOT / "generate_data.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)
            module.main(destination)
            for filename in ("weather.csv", "rides.csv"):
                self.assertEqual(
                    (destination / filename).read_bytes(),
                    (ROOT / "data" / filename).read_bytes(),
                )

    def test_every_code_frame_has_its_own_result_on_the_right(self) -> None:
        tex = (ROOT / "slides/lesson02.tex").read_text()
        frames = re.findall(r"\\begin\{frame\}\[fragile\].*?\\end\{frame\}", tex, re.S)
        self.assertEqual(len(frames), len(self.examples))
        for frame, example in zip(frames, self.examples, strict=True):
            with self.subTest(example=example["id"]):
                code_column, result_column = frame.split(
                    r"\begin{column}{0.49\textwidth}"
                )[1:]
                self.assertIn(example["code"], code_column)
                self.assertIn("\\includegraphics", result_column)
                self.assertIn(example["pdf"], result_column)
                self.assertEqual(frame.count(r"\begin{lstlisting}"), 1)
                self.assertLessEqual(max(map(len, example["code"].splitlines())), 42)
                for extension in ("pdf", "png"):
                    image = ROOT / "slides" / example[extension]
                    self.assertTrue(image.exists())
                    self.assertGreater(image.stat().st_size, 1000)

    def test_notebook_uses_the_same_code_as_the_slides(self) -> None:
        for example in self.examples:
            cell = next(
                c for c in self.notebook.cells if c.id == f"example-{example['id']}"
            )
            self.assertEqual(cell.source, example["code"])
            self.assertEqual(
                hashlib.sha256(cell.source.encode()).hexdigest(), example["code_sha256"]
            )
            self.assertIsNotNone(cell.execution_count)
            if example["kind"] == "plot":
                self.assertTrue(
                    any(
                        "image/png" in output.get("data", {}) for output in cell.outputs
                    )
                )

    def test_introductory_order_and_no_compulsory_axes(self) -> None:
        ids = [example["id"] for example in self.examples]
        self.assertLess(ids.index("first-line"), ids.index("create-table"))
        self.assertLess(ids.index("sns-hue"), ids.index("pd-bar"))
        self.assertIn("plt.plot", self.examples[0]["code"])
        for example in self.examples[:-1]:
            self.assertNotIn("plt.subplots", example["code"])

    def test_small_examples_match_the_explanations(self) -> None:
        notebook_sources = {example["id"]: example["code"] for example in self.examples}
        namespace = {}
        for identifier in ("create-table", "series", "revenue", "groupby"):
            exec(compile(notebook_sources[identifier], identifier, "exec"), namespace)
        self.assertEqual(namespace["prices"].mean(), 62.5)
        self.assertEqual(namespace["table"]["sold"].sum(), 50)
        self.assertEqual(
            namespace["revenue_by_category"].to_dict(), {"Бумага": 1400, "Письмо": 900}
        )
        counts, _ = np.histogram(
            [1, 3, 2, 5, 4, 2, 6, 3, 1, 4, 7, 5, 2, 3], bins=[0, 2, 4, 6, 8]
        )
        self.assertEqual(counts.tolist(), [2, 6, 4, 2])

    def test_notebooks_are_valid_and_homework_has_no_solutions(self) -> None:
        for path in ROOT.glob("*/*.ipynb"):
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            for cell in notebook.cells:
                if cell.cell_type == "code":
                    ast.parse(cell.source)
                    self.assertFalse(
                        any(o.output_type == "error" for o in cell.outputs)
                    )
                    if path.parent.name == "hw":
                        self.assertIsNone(cell.execution_count)
                        self.assertEqual(cell.outputs, [])
        homework = nbformat.read(ROOT / "hw/homework.ipynb", as_version=4)
        headings = [
            line
            for cell in homework.cells
            if cell.cell_type == "markdown"
            for line in cell.source.splitlines()
        ]
        self.assertEqual(
            sum(bool(re.match(r"## \d+\.", line)) for line in headings), 10
        )

    def test_no_old_academic_dataset(self) -> None:
        self.assertFalse((ROOT / "data/students.csv").exists())
        self.assertFalse((ROOT / "data/groups.csv").exists())
        for example in self.examples:
            self.assertNotIn("score", example["code"])
            self.assertNotIn("student", example["code"])

    def test_html_reader_has_a_code_result_pair_for_every_example(self) -> None:
        reader = (ROOT / "seminar/pandas_plots.html").read_text()
        self.assertEqual(reader.count('class="pair"'), len(self.examples))
        self.assertEqual(
            reader.count('src="data:image/png;base64,'), len(self.examples)
        )


if __name__ == "__main__":
    unittest.main()
