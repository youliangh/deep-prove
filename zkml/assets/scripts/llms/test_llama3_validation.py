"""Small acceptance-check regressions; no model weights or inference required."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

import llama3_validation as validation


class CompareTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="llama3-compare-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.reference = self.root / "reference"
        self.candidate = self.root / "candidate"
        self.reference.mkdir()
        self.candidate.mkdir()
        self.values = np.array([[1, 2, 3], [1, 2, 3]], dtype="<f4")
        self.hashes = {name: str(index) * 64 for index, name in enumerate(
            ["model.safetensors", "config.json", "tokenizer.json"])}

    def fixture(self, candidate=None, mode="float"):
        self.values.tofile(self.reference / "logits.f32")
        (self.values if candidate is None else candidate).astype("<f4").tofile(
            self.candidate / "logits.f32")
        record = {"sample": 0, "shape": list(self.values.shape), "tokens": [0, 1],
                  "file": "logits.f32"}
        self.reference_manifest = {"hashes": dict(self.hashes),
                                   "records": [{**record, "mode": "hf"}]}
        self.candidate_manifest = {"hashes": dict(self.hashes),
                                   "records": [{**record, "mode": mode}]}
        if mode == "int":
            validation.write_json(self.candidate / "quantization.json", {"schema": 1})
            self.candidate_manifest["quantization_sha256"] = validation.sha256(
                self.candidate / "quantization.json")
        self.args = SimpleNamespace(reference=self.reference, candidate=self.candidate,
                                    mode=mode, output=self.root / "result.json",
                                    expected_quantization_sha256=None)

    def compare(self):
        validation.write_json(self.reference / "manifest.json", self.reference_manifest)
        validation.write_json(self.candidate / "manifest.json", self.candidate_manifest)
        with contextlib.redirect_stdout(io.StringIO()):
            validation.compare(self.args)
        return json.loads(self.args.output.read_text())

    def test_final_logit_row_is_checked_but_has_no_nll_target(self):
        changed = self.values.copy()
        changed[-1] = [1000, -1000, 0]
        self.fixture(changed)
        with self.assertRaises(SystemExit):
            self.compare()
        result = json.loads(self.args.output.read_text())
        self.assertFalse(result["passed"])
        self.assertEqual(result["tokens"], 1)
        self.assertEqual(result["logit_rows"], 2)
        self.assertEqual(result["top1"], 0.5)
        self.assertGreater(result["relative_l2"], 1)
        self.assertEqual(result["reference_ppl"], result["candidate_ppl"])

    def test_each_source_identity_hash_must_match(self):
        for name in self.hashes:
            with self.subTest(name=name):
                self.fixture()
                self.candidate_manifest["hashes"][name] = "f" * 64
                with self.assertRaisesRegex(ValueError, "Source identity hash mismatch"):
                    self.compare()

    def test_legacy_manifest_remains_usable_with_unchecked_identity_reported(self):
        self.fixture()
        del self.candidate_manifest["hashes"]
        result = self.compare()
        self.assertTrue(result["passed"])
        self.assertFalse(result["source_identity_checked"])
        self.assertEqual(result["logit_rows"], 2)

    def test_integer_quantization_file_must_match_manifest_and_external_pin(self):
        self.fixture(mode="int")
        expected = self.candidate_manifest["quantization_sha256"]
        self.args.expected_quantization_sha256 = expected
        result = self.compare()
        self.assertTrue(result["passed"])
        self.assertTrue(result["source_identity_checked"])
        self.assertEqual(result["quantization_sha256"], expected)
        self.args.expected_quantization_sha256 = "0" * 64
        with self.assertRaisesRegex(ValueError, "externally pinned"):
            self.compare()
        self.args.expected_quantization_sha256 = None
        validation.write_json(self.candidate / "quantization.json", {"schema": 2})
        with self.assertRaisesRegex(ValueError, "does not match its manifest"):
            self.compare()

    def test_last_row_must_be_finite_and_token_histories_must_match(self):
        changed = self.values.copy()
        changed[-1, 0] = np.nan
        self.fixture(changed)
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            self.compare()
        self.fixture()
        self.candidate_manifest["records"][0]["tokens"] = [0, 2]
        with self.assertRaisesRegex(ValueError, "identical token histories"):
            self.compare()


if __name__ == "__main__":
    unittest.main()
