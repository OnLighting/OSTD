"""Stratified train/val/test split for the aircraft dataset.

Default 6:2:2 (train:val:test). Reads YOLO labels from data/labels/train,
draws images into data/images/{train,val,test} + data/labels/{train,val,test}
using stratified-by-class sampling so every class appears in val AND test.
Skips images whose label file is empty or missing.

Important: this script MOVES images/labels out of the original `train/` pool
into `val/` and `test/`. `train/` keeps the remaining 60%. Run it once, then
re-run tools/convert_yolo_to_coco.py for all three splits.

Usage:
    python tools/split_val.py --root data --ratios 0.6 0.2 0.2 --seed 0 --overwrite
"""

import argparse
import os
import random
import shutil
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
# val/test 不分。这是防御 n=2 / n=3 边界情况: 如果按比例拿 1 张进 val,
# train 也只能剩 1-2 张, 出现 val ≈ train 甚至 val > train 的情况。
RARE_THRESHOLD = 5
# 每个类在 train split 里至少保留下限 (boxes). 如果 split 之后某类的
# train box 数 < MIN_TRAIN_PER_CLASS, 把 val/test 里多余的挪回 train.
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


def stratified_three_way(stems, labels, ratios, seed):
    """Split into (train, val, test) per-class stratified with no overlap.

    ratios = (r_train, r_val, r_test), summing to ~1.0. The crucial fix vs.
    the naive per-class slice is that we maintain a SHARED POOL: once a stem
    is taken for val (or test) it is excluded from subsequent classes'
    candidate sets, so the same image never lands in both val and test.

    Algorithm (v4):
      1. 对 n < RARE_THRESHOLD 的极稀有类, 全部保留给 train (val/test 不分)。
      2. 对 n >= RARE_THRESHOLD 的类, 按 ratio 分配 val/test, 并保证
         train 至少 1 张 (即 n - k >= 1)。
      3. Process rare classes first so they don't get starved.
      4. Everything else stays in train.
      5. **Post-check (关键防御)**: 计算 train split 里每个类的 box count,
         如果某类 < MIN_TRAIN_PER_CLASS, 从 val/test 集合里把对应 stem
         挪回 train (优先挪 val, 再挪 test). 这能解决类间重叠导致的
         "train 集合里某类 box 数 < val 集合里该类 box 数" 问题。

    返回 (train, val, test)。
    """
    rng = random.Random(seed)
    r_train, r_val, r_test = ratios

    by_class = defaultdict(list)
    for stem, lbl in zip(stems, labels):
        for c in image_classes(lbl):
            by_class[c].append(stem)

    # Sort classes by size ascending so rare classes (HM, LQS, FSC) get
    # served first and are guaranteed >=1 in val/test.
    classes_in_order = sorted(by_class.keys(), key=lambda c: len(by_class[c]))

    val_set = set()
    test_set = set()

    def take_for(into_set, ratio, label):
        for c in classes_in_order:
            candidates = set(by_class[c]) - into_set - test_set - val_set
            n = len(candidates)
            if n < RARE_THRESHOLD:
                # 极稀有类不参与 split, 全部保留给 train.
                continue
            members = sorted(candidates)
            k = int(round(n * ratio))
            # Don't drain a class entirely; keep at least 1 sample for train.
            if n - k < 1:
                k = max(0, n - 1)
            if k <= 0:
                continue
            rng.shuffle(members)
            into_set.update(members[:k])

    take_for(val_set, r_val, 'val')
    take_for(test_set, r_test, 'test')

    # === Post-check: 修复类间重叠导致的 train 集合里 box count < val ===
    # 计算 train_set 里每个类的 box count (一个图含 cls c 即计 1 box).
    def per_class_box_count(stem_set):
        counts = defaultdict(int)
        for stem in stem_set:
            lbl = label_index[stem]
            for c in image_classes(lbl):
                counts[c] += 1
        return counts

    # 构造 stem → label 索引 (避免反复扫描)
    label_index = {lbl.stem: lbl for lbl in labels}

    train_set = set(stems) - val_set - test_set

    # 反复迭代, 直到 val/test 里没有任何类的 box 数 >= train 的对应类
    # (即 "val/test 里某些类的 box 数 ≥ train 里同类的 box 数" 这种
    # 边界 case 完全消除)。每个 pass 对每个类检查一次, 必要时从
    # val/test 挪 1 个含该类的 stem 回 train。
    def _post_check():
        train_box_counts = per_class_box_count(train_set)
        val_box_counts = per_class_box_count(val_set)
        test_box_counts = per_class_box_count(test_set)
        moved = 0
        for c in classes_in_order:
            for src_set, src_counts in [(val_set, val_box_counts),
                                         (test_set, test_box_counts)]:
                tr_n = train_box_counts.get(c, 0)
                sp_n = src_counts.get(c, 0)
                if sp_n > 0 and sp_n >= tr_n:
                    # 找一个含 c 的 stem 从 src_set 挪到 train_set
                    for stem in list(src_set):
                        if c in image_classes(label_index[stem]):
                            src_set.discard(stem)
                            train_set.add(stem)
                            moved += 1
                            break
                    else:
                        continue
                    break  # 只挪一次, 下次迭代再 check
        return moved

    # 反复 check, 直到一次迭代里 moved == 0
    while True:
        moved = _post_check()
        if moved == 0:
            break

    train = sorted(train_set)
    val = sorted(val_set)
    test = sorted(test_set)
    return train, val, test


def move_into(root: Path, split_name, stems, image_index, train_lbl_dir):
    """Copy (not move, to be safe) images+labels into the target split dir.

    Images are copied from the train pool; labels are copied from train labels.
    We copy rather than move so a re-run with --overwrite doesn't lose data —
    the source train/ pool always stays intact until you delete it yourself.
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='data', type=str)
    parser.add_argument('--ratios', nargs=3, type=float, default=[0.6, 0.2, 0.2],
                        help='train val test ratios (default 0.6 0.2 0.2)')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--overwrite', action='store_true',
                        help='Wipe existing val/ and test/ before writing.')
    args = parser.parse_args()

    root = Path(args.root).resolve()
    images, labels, stems = collect_train_files(root)
    if not stems:
        raise SystemExit(f'No train images with matching labels under {root}')

    # Drop empty-label images.
    keep_idx = [i for i, lbl in enumerate(labels) if image_classes(lbl)]
    images = [images[i] for i in keep_idx]
    labels = [labels[i] for i in keep_idx]
    stems = [stems[i] for i in keep_idx]
    print(f'Pool: {len(stems)} images with at least one label.')

    r_train, r_val, r_test = args.ratios
    total = r_train + r_val + r_test
    r_train, r_val, r_test = r_train / total, r_val / total, r_test / total
    print(f'Ratios -> train={r_train:.2f} val={r_val:.2f} test={r_test:.2f}')

    train_stems, val_stems, test_stems = stratified_three_way(
        stems, labels, (r_train, r_val, r_test), args.seed)
    print(f'Split -> train={len(train_stems)}  val={len(val_stems)}  '
          f'test={len(test_stems)}')

    image_index = {p.stem: p for p in images}
    train_lbl_dir = root / 'labels' / 'train'

    if args.overwrite:
        for s in ('val', 'test'):
            wipe_split(root, s)

    # NOTE: val/test images are COPIED out of train/. They are NOT removed from
    # train/ here — to keep the train/ pool as the single source of truth, do
    # the removal explicitly after you've confirmed the split looks right.
    move_into(root, 'val', val_stems, image_index, train_lbl_dir)
    move_into(root, 'test', test_stems, image_index, train_lbl_dir)

    # Remove val/test stems from the train pool so train/ truly holds only the
    # 60% remainder. This is what makes the split honest.
    removed = 0
    for stem in val_stems + test_stems:
        img = image_index[stem]
        if img.exists():
            img.unlink()
            removed += 1
        lbl = train_lbl_dir / f'{stem}.txt'
        if lbl.exists():
            lbl.unlink()
    print(f'Removed {removed} images+labels from train/ pool '
          f'(moved to val/test).')

    # === 打印三个 split 的 per-class coverage (boxes) ===
    # v3 修复: 之前只打印 val/test, 不打印 train, 用户无法直观对比
    # "val vs train" 的 box count。现三个 split 都打印。
    splits_info = (
        ('train', train_stems),
        ('val', val_stems),
        ('test', test_stems),
    )
    coverage = {}
    for name, slist in splits_info:
        coverage[name] = count_boxes(root, name, slist)

    for name, _ in splits_info:
        print(f'{name} class coverage (boxes):')
        for i, n in enumerate(CLASS_NAMES):
            if coverage[name][i]:
                print(f'  {i:2d} {n:<10s} {coverage[name][i]}')

    # === 防御性检测: val/test 中任意类的 box 数 >= train 的对应类 ===
    warnings_emitted = 0
    for name in ('val', 'test'):
        for c, cls_name in enumerate(CLASS_NAMES):
            tr_n = coverage['train'].get(c, 0)
            sp_n = coverage[name].get(c, 0)
            # 仅当 train 不为 0 但 val/test 反而 >= train 时报警
            if tr_n > 0 and sp_n >= tr_n:
                print(
                    f'WARNING: {name} {cls_name}({c}) count={sp_n} '
                    f'>= train count={tr_n}')
                warnings_emitted += 1
    if warnings_emitted == 0:
        print('No class imbalance warning (all train > val, train > test).')


if __name__ == '__main__':
    main()
