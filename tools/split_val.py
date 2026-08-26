"""Stratified train/val split for the aircraft dataset.

Official pipeline: 80% train, 20% validation, no independent test split.
Reads YOLO labels from data/labels/train, draws images into
data/images/{train,val} + data/labels/{train,val} using stratified-by-
class sampling so every class appears in val. Skips images whose label
file is empty or missing.

Important: this script MOVES images/labels out of the original `train/`
pool into `val/`. `train/` keeps the remaining 80%. Run it once, then
re-run tools/convert_yolo_to_coco.py for both splits.

Usage:
    python tools/split_val.py --root data --ratios 0.8 0.2 --seed 0 --overwrite
"""

import argparse
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path


CLASS_NAMES = [
    'HM', 'LQS', 'QHS', 'MS',
    'A1_SU-35', 'A2_C-130', 'A3_C-17', 'A4_C-5', 'A5_F-16', 'A6_TU-160',
    'A7_E-3', 'A8_B-52', 'A9_P-3C', 'A10_B-1B', 'A11_E-8', 'A12_TU-22',
    'A13_F-15', 'A14_KC-135', 'A15_F-22', 'A16_FA-18', 'A17_TU-95',
    'A18_KC-10', 'A19_SU-34', 'A20_SU-24', 'FSC',
]
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}
# 极稀有类保护阈值: len(by_class[c]) < RARE_THRESHOLD 的类全部保留给 train,
# val 不分。这是防御 n=2 / n=3 边界情况: 如果按比例拿 1 张进 val,
# train 也只能剩 1-2 张, 出现 val ≈ train 甚至 val > train 的情况。
RARE_THRESHOLD = 5
# 每个类在 train split 里至少保留下限 (boxes). 如果 split 之后某类的
# train box 数 < MIN_TRAIN_PER_CLASS, 把 val 里多余的挪回 train.
# 这是关键防御: 类间重叠时 (如 HM 经常和 MS 同图出现) val_set 拿走一张
# 含 HM 的图, train 里 HM box 数可能降到 0。MIN_TRAIN_PER_CLASS 保证
# train 里每个类至少有几张 box。
MIN_TRAIN_PER_CLASS = 1


def collect_train_files(root: Path):
    """Return (image_files, label_files, stems) lists whose basenames match."""
    img_dir = root / 'images' / 'train'
    lbl_dir = root / 'labels' / 'train'
    images = sorted(
        p for p in img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )
    labels = {p.stem: p for p in lbl_dir.iterdir() if p.suffix == '.txt'}
    image_stems = {p.stem: p for p in images}
    common = sorted(set(image_stems) & set(labels))
    return (
        [image_stems[s] for s in common],
        [labels[s] for s in common],
        common,
    )


def image_classes(label_path: Path):
    """Return the set of class ids present in a YOLO label file."""
    classes = set()
    if not label_path.exists():
        return classes
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if not parts:
                continue
            try:
                classes.add(int(parts[0]))
            except ValueError:
                continue
    return classes


def _stratified_n_way(stems, labels, ratios, seed):
    """Generic stratified n-way split with shared pool + post-check.

    The crucial fix vs. naive per-class slice: once a stem is taken for
    any non-train split it is excluded from subsequent classes'
    candidate sets, so the same image never lands in two non-train splits.

    Algorithm:
      1. 对 n < RARE_THRESHOLD 的极稀有类, 全部保留给 train (其他 split 不分)。
      2. 对 n >= RARE_THRESHOLD 的类, 按 ratio 分配到每个非 train split,
         并保证 train 至少 1 张 (即 n - k >= 1)。
      3. Process rare classes first so they don't get starved.
      4. Everything else stays in train.
      5. **Post-check (关键防御)**: 计算 train split 里每个类的 box count,
         如果某类 < MIN_TRAIN_PER_CLASS, 从非 train 集合里把对应 stem
         挪回 train. 这能解决类间重叠导致的
         "train 集合里某类 box 数 < 非 train 集合里该类 box 数" 问题。
    """
    rng = random.Random(seed)
    if len(ratios) < 2:
        raise ValueError('ratios must contain at least (train, val)')
    if any(r < 0 for r in ratios):
        raise ValueError('ratios must be non-negative')
    total = sum(ratios)
    if total <= 0:
        raise ValueError('ratios sum must be positive')
    normalized = [r / total for r in ratios]

    by_class = defaultdict(list)
    for stem, lbl in zip(stems, labels):
        for c in image_classes(lbl):
            by_class[c].append(stem)

    classes_in_order = sorted(by_class.keys(), key=lambda c: len(by_class[c]))

    # n_split == len(ratios) - 1 (train + n-1 non-train splits).
    n_split = len(ratios) - 1
    split_sets = [set() for _ in range(n_split)]
    split_names = [f'split{i}' for i in range(n_split)]

    def take_for(into_set, ratio, *other_sets):
        for c in classes_in_order:
            taken_anywhere = set().union(*other_sets, into_set)
            candidates = set(by_class[c]) - taken_anywhere
            n = len(candidates)
            if n < RARE_THRESHOLD:
                continue
            members = sorted(candidates)
            k = int(round(n * ratio))
            if n - k < 1:
                k = max(0, n - 1)
            if k <= 0:
                continue
            rng.shuffle(members)
            into_set.update(members[:k])

    # Process splits in declaration order; other_sets contains the splits
    # that come later so we don't double-book a stem.
    for i in range(n_split):
        others = split_sets[i + 1:]
        take_for(split_sets[i], normalized[i + 1], *others)

    label_index = {lbl.stem: lbl for lbl in labels}

    def per_class_box_count(stem_set):
        counts = defaultdict(int)
        for stem in stem_set:
            lbl = label_index[stem]
            for c in image_classes(lbl):
                counts[c] += 1
        return counts

    train_set = set(stems)
    for s in split_sets:
        train_set -= s

    def _post_check():
        train_box_counts = per_class_box_count(train_set)
        moved = 0
        for i in range(n_split):
            split_box_counts = per_class_box_count(split_sets[i])
            for c in classes_in_order:
                tr_n = train_box_counts.get(c, 0)
                sp_n = split_box_counts.get(c, 0)
                if sp_n > 0 and sp_n >= tr_n:
                    for stem in list(split_sets[i]):
                        if c in image_classes(label_index[stem]):
                            split_sets[i].discard(stem)
                            train_set.add(stem)
                            moved += 1
                            break
                    else:
                        continue
                    break
        return moved

    while True:
        moved = _post_check()
        if moved == 0:
            break

    train = sorted(train_set)
    splits = [sorted(s) for s in split_sets]
    return train, splits, split_names


def stratified_train_val(stems, labels, ratios, seed):
    """Stratified split returning (train, val) lists.

    ``ratios`` is ``(r_train, r_val)`` summing to ~1.0.
    """
    train, splits, _ = _stratified_n_way(stems, labels, ratios, seed)
    if len(splits) != 1:
        raise ValueError(
            f'stratified_train_val expects exactly 1 non-train split, '
            f'got {len(splits)}')
    return train, splits[0]


def stratified_three_way(stems, labels, ratios, seed):
    """Backwards-compatible three-way splitter used by old run.sh paths."""
    train, splits, _ = _stratified_n_way(stems, labels, ratios, seed)
    if len(splits) != 2:
        raise ValueError(
            f'stratified_three_way expects 2 non-train splits, got '
            f'{len(splits)}')
    return train, splits[0], splits[1]


def move_into(root: Path, split_name, stems, image_index, train_lbl_dir):
    """Copy images+labels into the target split dir.

    Images are copied from the train pool; labels are copied from train
    labels. We copy rather than move so a re-run with --overwrite doesn't
    lose data — the source train/ pool always stays intact until you
    delete it yourself.
    """
    out_img = root / 'images' / split_name
    out_lbl = root / 'labels' / split_name
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        src = image_index[stem]
        dst_img = out_img / src.name
        src_lbl = train_lbl_dir / f'{stem}.txt'
        dst_lbl = out_lbl / f'{stem}.txt'
        if not dst_img.exists():
            shutil.copy2(src, dst_img)
        if src_lbl.exists() and not dst_lbl.exists():
            shutil.copy2(src_lbl, dst_lbl)


def wipe_split(root, split_name):
    for sub in ('images', 'labels'):
        d = root / sub / split_name
        if d.exists():
            for p in d.iterdir():
                if p.is_file():
                    p.unlink()


def count_boxes(root, split_name, stems):
    counts = defaultdict(int)
    for stem in stems:
        lbl = root / 'labels' / split_name / f'{stem}.txt'
        for c in image_classes(lbl):
            counts[c] += 1
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='data', type=str)
    parser.add_argument('--ratios', nargs='+', type=float,
                        default=[0.8, 0.2],
                        help='train val ratios; e.g. --ratios 0.8 0.2')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--overwrite', action='store_true',
                        help='Wipe existing val/ before writing.')
    args = parser.parse_args(argv)

    if len(args.ratios) != 2:
        print('ERROR: official pipeline requires exactly 2 ratios '
              '(train val), got', len(args.ratios), file=sys.stderr)
        return 1

    root = Path(args.root).resolve()
    images, labels, stems = collect_train_files(root)
    if not stems:
        print(f'No train images with matching labels under {root}',
              file=sys.stderr)
        return 1

    # Drop empty-label images.
    keep_idx = [i for i, lbl in enumerate(labels) if image_classes(lbl)]
    images = [images[i] for i in keep_idx]
    labels = [labels[i] for i in keep_idx]
    stems = [stems[i] for i in keep_idx]
    print(f'Pool: {len(stems)} images with at least one label.')

    r_train, r_val = args.ratios
    total = r_train + r_val
    r_train, r_val = r_train / total, r_val / total
    print(f'Ratios -> train={r_train:.2f} val={r_val:.2f}')

    train_stems, val_stems = stratified_train_val(
        stems, labels, (r_train, r_val), args.seed)
    print(f'Split -> train={len(train_stems)}  val={len(val_stems)}')

    image_index = {p.stem: p for p in images}
    train_lbl_dir = root / 'labels' / 'train'

    if args.overwrite:
        wipe_split(root, 'val')
        # Guard against an unrelated test/ directory lingering from old runs;
        # we don't want to copy val into it but we should not silently keep
        # a stale 20% either.
        wipe_split(root, 'test')

    move_into(root, 'val', val_stems, image_index, train_lbl_dir)
    # Do not create a test/ split — the official pipeline validates on the
    # 20% val pool only.

    # Remove val stems from the train pool so train/ truly holds only the
    # 80% remainder. This is what makes the split honest.
    removed = 0
    for stem in val_stems:
        img = image_index[stem]
        if img.exists():
            img.unlink()
            removed += 1
        lbl = train_lbl_dir / f'{stem}.txt'
        if lbl.exists():
            lbl.unlink()
    print(f'Removed {removed} images+labels from train/ pool (moved to val).')

    splits_info = (
        ('train', train_stems),
        ('val', val_stems),
    )
    coverage = {}
    for name, slist in splits_info:
        coverage[name] = count_boxes(root, name, slist)

    for name, _ in splits_info:
        print(f'{name} class coverage (boxes):')
        for i, n in enumerate(CLASS_NAMES):
            if coverage[name][i]:
                print(f'  {i:2d} {n:<10s} {coverage[name][i]}')

    # === 防御性检测: val 中任意类的 box 数 >= train 的对应类 ===
    warnings_emitted = 0
    for c, cls_name in enumerate(CLASS_NAMES):
        tr_n = coverage['train'].get(c, 0)
        sp_n = coverage['val'].get(c, 0)
        if tr_n > 0 and sp_n >= tr_n:
            print(
                f'WARNING: val {cls_name}({c}) count={sp_n} '
                f'>= train count={tr_n}')
            warnings_emitted += 1
    if warnings_emitted == 0:
        print('No class imbalance warning (all train > val).')

    return 0


if __name__ == '__main__':
    sys.exit(main())
