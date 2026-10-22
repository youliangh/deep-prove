"""Acceptance-metric and artifact validation tests; no model downloads required."""

import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import qwen35_validation as qwen


class ValidationTests(unittest.TestCase):
    def test_semantic_checks_reject_repetition_and_wrong_answers(self):
        self.assertTrue(qwen.content_check(" Paris.\n", {"exact": ["Paris"]}))
        self.assertFalse(qwen.content_check("Not Paris", {"exact": ["Paris"]}))
        self.assertFalse(qwen.content_check("It's a very difficult thing to do. " * 4, {}))
        self.assertFalse(qwen.content_check("太阳发光，地球上的人能看到太阳。" * 4, {}))
        self.assertTrue(qwen.content_check("太阳发出的光使我们在白天看见它。", {
            "keyword_groups": [["太阳"], ["光"]], "one_sentence": True}))

    def test_streamed_metrics_include_last_row_but_not_last_nll_target(self):
        a = np.array([[1, 4, 0], [2, 0, 1], [0, 1, 5]], dtype=np.float32)
        b = a.copy()
        b[-1] = [6, 0, 1]
        ids = [0, 1, 2]
        totals = {}
        for row in range(3):
            metrics = qwen.row_metrics(a[row:row + 1], b[row:row + 1], ids[row + 1:row + 2])
            for key, value in metrics.items():
                totals[key] = max(totals.get(key, 0), value) if key == "max_abs" else totals.get(key, 0) + value
        summary = qwen.summarize(totals)
        self.assertEqual(summary["tokens"], 2)
        self.assertEqual(summary["logit_rows"], 3)
        self.assertAlmostEqual(summary["top1"], 2 / 3)
        self.assertEqual(summary["ppl_ratio"], 1.0)
        self.assertGreater(summary["relative_l2"], 0)
        dense = qwen.row_metrics(a, b, ids[1:])
        for key in totals:
            self.assertAlmostEqual(totals[key], dense[key])

    def test_nonfinite_logits_are_rejected(self):
        a = np.array([[1.0, float("nan")]], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            qwen.row_metrics(a, a, [])

    def test_file_shape_digest_and_path_are_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.ones((2, 3), dtype="<f4").tofile(root / "ok.f32")
            record = {"file": "ok.f32", "shape": [2, 3], "sha256": qwen.sha256(root / "ok.f32")}
            self.assertEqual(qwen.checked_array(root, record).shape, (2, 3))
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                qwen.checked_array(root, {**record, "sha256": "0" * 64})
            with self.assertRaisesRegex(ValueError, "file size"):
                qwen.checked_array(root, {**record, "shape": [3, 3]})
            with self.assertRaisesRegex(ValueError, "inside"):
                qwen.checked_array(root, {**record, "file": "../outside.f32"})

    def test_comparison_rejects_identity_token_or_sample_substitution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref, cand = root / "ref", root / "cand"
            ref.mkdir()
            cand.mkdir()
            values = np.array([[1, 4, 0], [2, 0, 1]], dtype="<f4")
            values.tofile(ref / "logits.f32")
            values.tofile(cand / "logits.f32")
            record = {"sample": 0, "mode": "float", "file": "logits.f32", "tokens": [0, 1], "shape": [2, 3]}
            manifest = {"records": [record], "hashes": {"model": "identity"}}
            qwen.write_json(ref / "manifest.json", manifest)
            qwen.write_json(cand / "manifest.json", manifest)
            args = argparse.Namespace(reference=ref, candidate=cand, mode="float", rows_per_chunk=1,
                                      expected_quantization_sha256=None, require_source_identity=True,
                                      output=root / "metrics.json")
            with contextlib.redirect_stdout(io.StringIO()):
                qwen.compare(args)
            self.assertTrue(qwen.read_json(args.output)["passed"])
            bad_cases = [
                ({**manifest, "hashes": {"model": "different"}}, "identity hashes"),
                ({**manifest, "records": [{**record, "tokens": [0, 2]}]}, "token histories"),
                ({**manifest, "records": [record, record]}, "Duplicate"),
            ]
            for changed, error in bad_cases:
                qwen.write_json(cand / "manifest.json", changed)
                with self.assertRaisesRegex(ValueError, error):
                    qwen.compare(args)

    def test_cached_generation_comparison_requires_exact_computed_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref, cand = root / "ref", root / "cand"
            ref.mkdir()
            cand.mkdir()
            values = np.array([[1, 4, 0], [2, 0, 1]], dtype="<f4")
            values.tofile(ref / "logits.f32")
            values.tofile(cand / "logits.f32")
            record = {"sample": 0, "mode": "float", "file": "logits.f32", "tokens": [0, 1], "shape": [2, 3]}
            qwen.write_json(ref / "manifest.json", {"records": [record], "hashes": {"model": "same"}})
            generated = {**record, "tokens": [0], "generated_tokens": [1, 2]}
            qwen.write_json(cand / "manifest.json", {"records": [generated], "hashes": {"model": "same"}})
            args = argparse.Namespace(reference=ref, candidate=cand, mode="float", rows_per_chunk=1,
                                      expected_quantization_sha256=None, require_source_identity=True,
                                      candidate_generated_history=True, output=root / "metrics.json")
            with contextlib.redirect_stdout(io.StringIO()):
                qwen.compare(args)
            self.assertTrue(qwen.read_json(args.output)["passed"])
            for ids, error in [([2, 1], "token histories"), ([248046, 2], "after EOS")]:
                changed = {**generated, "generated_tokens": ids}
                qwen.write_json(cand / "manifest.json", {"records": [changed], "hashes": {"model": "same"}})
                with self.assertRaisesRegex(ValueError, error):
                    qwen.compare(args)

    def test_context_and_vocabulary_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "samples.json"
            for ids in [[], [0] * 257, [-1], [248320], [True], [1.5]]:
                path.write_text(json.dumps([ids]))
                with self.assertRaises(ValueError):
                    qwen.checked_samples(path)
            path.write_text(json.dumps([[0] * 256]))
            self.assertEqual(len(qwen.checked_samples(path)[0]), 256)


if __name__ == "__main__":
    unittest.main()
