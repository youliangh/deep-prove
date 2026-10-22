"""Reject text/output substitution when accepting native semantic generations."""
import argparse
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import qwen35_check_artifacts as artifacts
import qwen35_validation as qwen


class ArtifactTests(unittest.TestCase):
    def test_trace_checks_digest_shape_and_removes_padding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.arange(16, dtype='<f4').tofile(root / 'tensor.f32')
            record = {'file': 'tensor.f32', 'shape': [4, 4], 'unpadded_shape': [3, 3],
                      'sha256': qwen.sha256(root / 'tensor.f32')}
            np.testing.assert_equal(artifacts.tensor(root, record), [[0, 1, 2], [4, 5, 6], [8, 9, 10]])
            with self.assertRaisesRegex(ValueError, 'digest'):
                artifacts.tensor(root, {**record, 'sha256': '0' * 64})
            with self.assertRaisesRegex(ValueError, 'escapes'):
                artifacts.tensor(root, {**record, 'file': '../tensor.f32'})

    def test_semantic_acceptance_binds_tokens_text_model_and_eos(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / 'native'
            candidate.mkdir()
            prompts = [{'name': 'capital', 'prompt': 'capital?', 'prompt_tokens': [10, 11],
                        'max_new_tokens': 4, 'rubric': {'exact': ['Paris']}}]
            qwen.write_json(root / 'prompts.json', prompts)
            qwen.write_json(root / 'reference.json', [{'name': 'capital', 'generated_tokens': [42, 248046]}])
            qwen.write_json(root / 'reference.json.metadata.json', {'hashes': {'model': 'same'}})
            record = {'mode': 'float', 'sample': 0, 'tokens': [10, 11],
                      'generated_tokens': [42, 248046], 'text': 'Paris',
                      'file': 'logits.f32', 'shape': [3, 248320]}
            logits = np.zeros((3, 248320), dtype='<f4')
            logits[1, 42] = 1
            logits[2, 248046] = 1
            logits.tofile(candidate / 'logits.f32')
            manifest = {'records': [record], 'hashes': {'model': 'same'}}
            args = argparse.Namespace(prompts=root / 'prompts.json', reference=root / 'reference.json',
                                      candidate=candidate, output=root / 'report.json', mode='float', model='mock')
            class Tokenizer:
                def decode(self, ids, **kwargs):
                    return 'Paris'
            with patch.object(qwen, 'tokenizer', return_value=Tokenizer()), contextlib.redirect_stdout(io.StringIO()):
                qwen.write_json(candidate / 'manifest.json', manifest)
                artifacts.generation(args)
                self.assertTrue(qwen.read_json(args.output)['passed'])
                for changed, message in [
                    ({**record, 'text': 'Wrong answer'}, 'text differs'),
                    ({**record, 'tokens': [10, 12]}, 'histories'),
                    ({**record, 'generated_tokens': [42, 248044]}, 'logit argmax'),
                    ({**record, 'generated_tokens': [42, 248046, 42]}, 'after EOS'),
                ]:
                    qwen.write_json(candidate / 'manifest.json', {**manifest, 'records': [changed]})
                    with self.assertRaisesRegex(ValueError, message):
                        artifacts.generation(args)
                qwen.write_json(candidate / 'manifest.json', {**manifest, 'hashes': {'model': 'other'}})
                with self.assertRaisesRegex(ValueError, 'identity'):
                    artifacts.generation(args)
                # A self-consistent replacement quantization is still rejected
                # when the held-out run must use the frozen development graph.
                qwen.write_json(candidate / 'quantization.json', {'policy': 'frozen'})
                digest = qwen.sha256(candidate / 'quantization.json')
                int_record = {**record, 'mode': 'int'}
                int_manifest = {**manifest, 'records': [int_record], 'quantization_sha256': digest}
                qwen.write_json(candidate / 'manifest.json', int_manifest)
                args.mode = 'int'
                args.expected_quantization_sha256 = digest
                artifacts.generation(args)
                self.assertEqual(qwen.read_json(args.output)['quantization_sha256'], digest)
                qwen.write_json(candidate / 'quantization.json', {'policy': 'replacement'})
                replacement = qwen.sha256(candidate / 'quantization.json')
                qwen.write_json(candidate / 'manifest.json', {**int_manifest, 'quantization_sha256': replacement})
                with self.assertRaisesRegex(ValueError, 'frozen quantization'):
                    artifacts.generation(args)
                args.mode = 'float'
                args.expected_quantization_sha256 = None
                logits[2, 248046] = 0
                logits[2, 43] = 1
                logits.tofile(candidate / 'logits.f32')
                qwen.write_json(candidate / 'manifest.json', {**manifest, 'records': [{**record, 'generated_tokens': [42, 43]}]})
                with self.assertRaises(SystemExit):
                    artifacts.generation(args)
                self.assertFalse(qwen.read_json(args.output)['passed'])


if __name__ == '__main__':
    unittest.main()
