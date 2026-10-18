"""Build the tiny COCO subset used by configs/ostd/ostd_dior_smoke.py.

Picks the N DIOR train images with the most annotations and copies them into
data/smoke/ with a fresh COCO JSON. Run once after staging DIOR:

    python tools/make_smoke_subset.py \
        --source data/DIOR_coco --num-images 2 --output data/smoke

Only depends on the standard library.
"""
import argparse
import json
import shutil
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='data/DIOR_coco')
    parser.add_argument('--num-images', type=int, default=2)
    parser.add_argument('--output', default='data/smoke')
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    doc = json.loads(
        (source / 'annotations/instances_train.json').read_text(
            encoding='utf-8'))

    counts = Counter(a['image_id'] for a in doc['annotations'])
    best = sorted(counts, key=lambda i: counts[i], reverse=True)
    chosen = best[:args.num_images]
    keep = {im['id']: im for im in doc['images'] if im['id'] in chosen}
    anns = [a for a in doc['annotations'] if a['image_id'] in chosen]

    (output / 'annotations').mkdir(parents=True, exist_ok=True)
    (output / 'images').mkdir(parents=True, exist_ok=True)
    for im in keep.values():
        shutil.copy(
            source / 'images/train' / im['file_name'].split('/')[-1],
            output / 'images' / im['file_name'].split('/')[-1])

    subset = {
        'images': list(keep.values()),
        'annotations': anns,
        'categories': doc['categories'],
    }
    (output / 'annotations/instances_train.json').write_text(
        json.dumps(subset), encoding='utf-8')
    (output / 'annotations/instances_val.json').write_text(
        json.dumps(subset), encoding='utf-8')
    print(f'{len(keep)} images, {len(anns)} annotations -> {output}')


if __name__ == '__main__':
    main()
