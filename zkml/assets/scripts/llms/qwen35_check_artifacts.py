#!/usr/bin/env python3
"""Check native decoder block traces and generated text against frozen HF fixtures."""
import argparse
from pathlib import Path
import numpy as np
import qwen35_validation as qwen


def tensor(root, record, allow_negative_inf=False):
    root = Path(root).resolve()
    path = (root / record['file']).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Tensor path escapes its artifact directory')
    shape = tuple(record['shape'])
    if not shape or any(type(x) is not int or x <= 0 for x in shape):
        raise ValueError('Invalid tensor shape')
    if path.stat().st_size != int(np.prod(shape)) * 4:
        raise ValueError('Tensor byte count does not match shape')
    if record.get('sha256') and qwen.sha256(path) != record['sha256']:
        raise ValueError('Tensor digest mismatch')
    value = np.fromfile(path, dtype='<f4').reshape(shape)
    valid = np.isfinite(value) | (np.isneginf(value) if allow_negative_inf else False)
    if not valid.all():
        raise ValueError('Non-finite trace tensor')
    # Native traces can include power-of-two padding, which is not model output.
    return value[tuple(slice(0, x) for x in record.get('unpadded_shape', shape))]


def layers(args):
    reference, candidate = Path(args.reference), Path(args.candidate)
    ref = qwen.read_json(reference / 'tensor-manifest.json')
    cand = qwen.read_json(candidate / 'manifest.json')
    source = qwen.read_json(candidate.parent / 'manifest.json')
    if not ref.get('hashes') or any(source.get('hashes', {}).get(k) != v for k, v in ref['hashes'].items()):
        raise ValueError('Layer trace model/tokenizer identity mismatch')
    if not any(record['tokens'] == ref['tokens'] for record in source['records']):
        raise ValueError('Layer trace token histories differ')
    ref_tensors = {r['name']: r for r in ref['tensors']}
    native = {(r['node'], r['port']): r for r in cand['records']}
    blocks = cand['model_metadata']['transformers']
    if len(blocks) != 24:
        raise ValueError('Expected all 24 Qwen3.5 decoder blocks')
    # A block output is exactly the next block input; the final block feeds
    # final RMSNorm. These are input traces, not raw pre-requant output traces.
    final_norm = max(r['node'] for r in cand['records'] if r['layer'] == 'QNRM')
    records = []
    for block in range(len(blocks)):
        name = f'model.layers.{block}.output'
        node = blocks[block + 1]['norm_id'] if block + 1 < len(blocks) else final_norm
        a = tensor(reference, ref_tensors[name]).squeeze(axis=0).astype(np.float64)
        b = tensor(candidate, native[node, 0]).astype(np.float64)
        if a.shape != b.shape:
            raise ValueError(f'Block {block} shape differs: {a.shape} / {b.shape}')
        relative = float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-30))
        cosine = float(np.dot(a.ravel(), b.ravel()) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))
        records.append({'block': block, 'native_input_node': node, 'shape': list(a.shape),
                        'relative_l2': relative, 'cosine': cosine,
                        'max_abs': float(np.abs(a - b).max()),
                        'passed': relative <= 0.001 and cosine >= 0.99999})
    report = {'records': records, 'passed': all(r['passed'] for r in records),
              'reference_manifest_sha256': qwen.sha256(reference / 'tensor-manifest.json'),
              'candidate_manifest_sha256': qwen.sha256(candidate / 'manifest.json')}
    qwen.write_json(args.output, report)
    print(f"24 blocks: passed={report['passed']}, maximum relative L2={max(r['relative_l2'] for r in records):.8g}")
    report['diagnostic_only'] = getattr(args, 'diagnostic', False)
    qwen.write_json(args.output, report)
    if not report['passed'] and not report['diagnostic_only']:
        raise SystemExit(1)


def generation(args):
    prompts = qwen.read_json(args.prompts)
    ref = qwen.read_json(args.reference)
    manifest = qwen.read_json(Path(args.candidate) / 'manifest.json')
    candidates = [r for r in manifest['records'] if r['mode'] == args.mode]
    source = qwen.read_json(str(args.reference) + '.metadata.json')
    if not source.get('hashes') or any(manifest.get('hashes', {}).get(k) != v for k, v in source['hashes'].items()):
        raise ValueError('Candidate model/tokenizer identity differs from HF reference')
    if len(candidates) != len(prompts) or len(ref) != len(prompts):
        raise ValueError('Missing or extra semantic cases')
    candidates = {r['sample']: r for r in candidates}
    if set(candidates) != set(range(len(prompts))):
        raise ValueError('Duplicate or invalid sample IDs')
    tok = qwen.tokenizer(args.model)
    records = []
    for i, (case, expected) in enumerate(zip(prompts, ref)):
        candidate = candidates[i]
        if candidate['tokens'] != case['prompt_tokens'] or expected['name'] != case['name']:
            raise ValueError('Prompt histories or case order differ')
        ids = candidate['generated_tokens']
        if (not ids or len(ids) > case['max_new_tokens'] or
                len(ids) + len(candidate['tokens']) > qwen.CONTEXT_LIMIT or
                any(type(x) is not int or not 0 <= x < 248320 for x in ids)):
            raise ValueError('Invalid generation budget or tokens')
        if any(x in qwen.EOS_IDS for x in ids[:-1]):
            raise ValueError('Generation continued after EOS')
        logits = qwen.checked_array(args.candidate, candidate)
        if logits.shape != (len(candidate['tokens']) + len(ids) - 1, 248320):
            raise ValueError('Generation logit shape differs from the computed history')
        for row in range(0, len(logits), 8):
            if not np.isfinite(logits[row:row + 8]).all():
                raise ValueError('Non-finite generation logits')
        predictions = logits[len(candidate['tokens']) - 1:].argmax(axis=1).tolist()
        if predictions != ids:
            raise ValueError('Generated tokens do not match native logit argmax')
        text = tok.decode(ids, skip_special_tokens=True)
        if text != candidate['text']:
            raise ValueError('Native text differs from exact tokenizer decode')
        content = qwen.content_check(text, case['rubric'])
        stopped = ids[-1] in qwen.EOS_IDS
        records.append({'name': case['name'], 'prompt': case['prompt'], 'text': text,
                        'generated_tokens': ids, 'stopped_on_eos': stopped, 'content_check': content,
                        'matches_hf_tokens': ids == expected['generated_tokens'],
                        'passed': content and stopped, 'manual_semantic_review_required': True})
    report = {'mode': args.mode, 'records': records, 'passed': all(r['passed'] for r in records),
              'source_identity_checked': True, 'prompts_sha256': qwen.sha256(args.prompts),
              'candidate_manifest_sha256': qwen.sha256(Path(args.candidate) / 'manifest.json')}
    if args.mode == 'int':
        digest = qwen.sha256(Path(args.candidate) / 'quantization.json')
        if digest != manifest.get('quantization_sha256'):
            raise ValueError('Quantization manifest digest mismatch')
        expected = getattr(args, 'expected_quantization_sha256', None)
        if expected and digest != expected:
            raise ValueError('Candidate differs from the frozen quantization digest')
        report['quantization_sha256'] = digest
    elif getattr(args, 'expected_quantization_sha256', None):
        raise ValueError('Frozen quantization hash requires --mode int')
    qwen.write_json(args.output, report)
    for record in records:
        print(f"{record['name']}: passed={record['passed']} {record['text']!r}")
    if not report['passed']:
        raise SystemExit(1)



def inputs(args):
    """Localize integer error growth; this report is not a held-out acceptance test."""
    reference, candidate = Path(args.reference), Path(args.candidate)
    ref, cand = qwen.read_json(reference / 'manifest.json'), qwen.read_json(candidate / 'manifest.json')
    reference_source = qwen.read_json(reference.parent / 'manifest.json')
    candidate_source = qwen.read_json(candidate.parent / 'manifest.json')
    if not reference_source.get('hashes') or reference_source['hashes'] != candidate_source.get('hashes'):
        raise ValueError('Diagnostic source identities differ')
    if [r['tokens'] for r in reference_source['records']] != [r['tokens'] for r in candidate_source['records']]:
        raise ValueError('Diagnostic token histories differ')
    quantization_path = candidate.parent / 'quantization.json'
    quantization_digest = qwen.sha256(quantization_path)
    if candidate_source.get('quantization_sha256') != quantization_digest:
        raise ValueError('Diagnostic quantization digest mismatch')
    quantization = qwen.read_json(quantization_path)['compiled']
    scalings = quantization['calibration']['input_scales']
    refs = {(r['node'], r['port']): r for r in ref['records']}
    records = []
    for record in cand['records']:
        key = record['node'], record['port']
        if key not in refs or refs[key]['layer'] != record['layer']:
            raise ValueError(f'Original graph node identities changed at {key}')
        a = tensor(reference, refs[key], allow_negative_inf=record['layer'] == 'SFTM').astype(np.float64)
        b = tensor(candidate, record).astype(np.float64)
        if a.shape != b.shape:
            raise ValueError(f'Diagnostic shape mismatch at {key}')
        logical_shape = list(a.shape)
        finite = np.isfinite(a)
        masked = int((~finite).sum())
        a, b = a[finite], b[finite]
        norm_a, norm_b = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        result = {'node': key[0], 'port': key[1], 'layer': record['layer'], 'shape': logical_shape,
                  'causally_masked_elements_excluded': masked,
                  'relative_l2': float(np.linalg.norm(a - b) / max(norm_a, 1e-30)),
                  'cosine': float(np.dot(a.ravel(), b.ravel()) / max(norm_a * norm_b, 1e-30)),
                  'max_abs_error': float(np.abs(a - b).max()),
                  'reference_min': float(a.min()), 'reference_max': float(a.max()),
                  'candidate_min': float(b.min()), 'candidate_max': float(b.max())}
        if record['layer'] != 'EMBD':
            scaling = scalings[str(key[0])][str(key[1])]
            low, high = [bound * scaling['scale'] for bound in scaling['quantized_domain']]
            tolerance = abs(scaling['scale']) * 0.01
            result.update({'calibrated_domain': [low, high],
                           'reference_outside_calibrated_domain': int(((a < low - tolerance) | (a > high + tolerance)).sum()),
                           'candidate_at_domain_endpoint': int(((abs(b - low) < tolerance) | (abs(b - high) < tolerance)).sum()),
                           'elements': int(a.size)})
        records.append(result)
    report = {'diagnostic_only': True, 'records': records, 'quantization_sha256': quantization_digest,
              'reference_manifest_sha256': qwen.sha256(reference / 'manifest.json'),
              'candidate_manifest_sha256': qwen.sha256(candidate / 'manifest.json')}
    qwen.write_json(args.output, report)
    for record in sorted(records, key=lambda r: (r['node'], r['port'])):
        if record['relative_l2'] > 0.01:
            print(f"N{record['node']}:{record['port']} {record['layer']} "
                  f"L2={record['relative_l2']:.5g} cos={record['cosine']:.6g} "
                  f"outside={record.get('reference_outside_calibrated_domain', 0)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=qwen.MODEL_DEFAULT)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ['layers', 'generation', 'inputs']:
        command = sub.add_parser(name)
        command.add_argument('--reference', required=True)
        command.add_argument('--candidate', required=True)
        command.add_argument('--output', required=True)
        if name == 'layers':
            command.add_argument('--diagnostic', action='store_true')
        if name == 'generation':
            command.add_argument('--prompts', required=True)
            command.add_argument('--mode', required=True, choices=['float', 'int'])
            command.add_argument('--expected-quantization-sha256')
    args = parser.parse_args()
    {'layers': layers, 'generation': generation, 'inputs': inputs}[args.command](args)


if __name__ == '__main__':
    main()
