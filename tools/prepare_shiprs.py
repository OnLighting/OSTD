"""Convert ShipRSImageNet COCO annotations to competition COCO labels.

The competition evaluates horizontal boxes.  This converter therefore keeps a
native COCO horizontal box when one is available, and only derives an
axis-aligned envelope from rotated or polygon geometry when the native box is
absent.  Images are referenced in place; no ShipRS image is copied.

Carried over from model_v4 commit 3f11e6b (feat: add deterministic source-balanced
dataset wrapper) with two ShipRSImageNet-specific patches for the public 5-element
``bbox = (cx, cy, w, h, angle)`` representation shipped in the public COCO dumps:
  * ``_bbox_from_obb`` now accepts a 5-element list/tuple placed in the standard
    COCO ``bbox`` slot (ShipRS public export convention; verified against
    ``add_data/COCO_Format/ShipRSImageNet_*.json``: ``bbox`` = ``[cx, cy, w, h,
    angle]`` and ``segmentation`` = 4-corner polygon; the polygon center equals
    ``(cx, cy)``).
  * ``to_horizontal_bbox`` matches that 5-element case before treating
    ``bbox`` as a 4-element native HBB.
"""

from __future__ import print_function

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

from tools.dataset_utils.shiprs_mapping import CLASS_NAMES, map_shiprs_category


AUDIT_FIELDS = (
    'source_coco', 'source_image_id', 'source_image_file_name',
    'source_annotation_id', 'source_category_id', 'source_category_name',
    'action', 'mapping_reason', 'geometry_source', 'clamped', 'excluded',
    'output_annotation_id', 'message')


def _is_finite_number(value):
    """Return whether ``value`` can be represented as a finite float."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _float_list(values, expected_length, label):
    if not isinstance(values, (list, tuple)) or len(values) != expected_length:
        raise ValueError('{} must contain {} numeric values'.format(
            label, expected_length))
    if not all(_is_finite_number(value) for value in values):
        raise ValueError('{} contains a non-finite coordinate'.format(label))
    return [float(value) for value in values]


def _bbox_from_polygon(segmentation):
    """Return an xywh envelope from COCO polygon segmentation."""
    if isinstance(segmentation, tuple):
        segmentation = list(segmentation)
    if not isinstance(segmentation, list):
        raise ValueError('segmentation must be a polygon list')
    if segmentation and not isinstance(segmentation[0], (list, tuple)):
        polygons = [segmentation]
    else:
        polygons = segmentation
    coordinates = []
    for polygon in polygons:
        if not isinstance(polygon, (list, tuple)) or len(polygon) < 6:
            raise ValueError('polygon needs at least three points')
        if len(polygon) % 2:
            raise ValueError('polygon has an odd number of coordinates')
        if not all(_is_finite_number(value) for value in polygon):
            raise ValueError('polygon contains a non-finite coordinate')
        coordinates.extend(float(value) for value in polygon)
    if not coordinates:
        raise ValueError('segmentation has no polygon coordinates')
    xs = coordinates[0::2]
    ys = coordinates[1::2]
    return [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]


def _obb_points(obb):
    """Return corner points for common ShipRS rotated-box representations."""
    if isinstance(obb, dict):
        if 'points' in obb:
            obb = obb['points']
        else:
            keys = ('cx', 'cy', 'w', 'h', 'angle')
            if not all(key in obb for key in keys):
                raise ValueError('rotated box dictionary lacks center fields')
            obb = [obb[key] for key in keys]
    if not isinstance(obb, (list, tuple)):
        raise ValueError('rotated box must be a list or dictionary')
    if len(obb) == 8:
        values = _float_list(obb, 8, 'rotated box points')
        return list(zip(values[0::2], values[1::2]))
    cx, cy, width, height, angle = _float_list(obb, 5, 'rotated box')
    if width <= 0 or height <= 0:
        raise ValueError('rotated box width and height must be positive')
    # ShipRS/HRSC-style rotated boxes use radians.  Values outside a full
    # radian turn are treated as degrees to support equivalent COCO exports.
    if abs(angle) > 2.0 * math.pi:
        angle = math.radians(angle)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    half_width = width / 2.0
    half_height = height / 2.0
    points = []
    for x_offset, y_offset in ((-half_width, -half_height),
                               (half_width, -half_height),
                               (half_width, half_height),
                               (-half_width, half_height)):
        points.append((cx + x_offset * cosine - y_offset * sine,
                       cy + x_offset * sine + y_offset * cosine))
    return points


def _bbox_from_obb(annotation):
    """Return an xywh envelope of the rotated box.

    Accepts two ShipRS OBB encodings:
      * a dedicated field in ``robndbox / rbox / obb / rotated_bbox`` -- typically
        an 8-tuple of points, a 5-tuple ``(cx, cy, w, h, angle)``, or a dict
        carrying the same fields;
      * a 5-element ``bbox`` ``(cx, cy, w, h, angle)`` -- the ShipRSImageNet
        public convention that places the OBB inside the standard COCO bbox
        slot.  Verified against add_data/COCO_Format/ShipRSImageNet_*.json.
    """
    # Path 1: historical ShipRS rotated-box fields
    for key in ('robndbox', 'rbox', 'obb', 'rotated_bbox'):
        if annotation.get(key) is None:
            continue
        points = _obb_points(annotation[key])
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
    # Path 2: 5-element bbox = (cx, cy, w, h, angle) (ShipRSImageNet public export)
    bbox = annotation.get('bbox')
    if isinstance(bbox, (list, tuple)) and len(bbox) == 5:
        points = _obb_points(bbox)
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
    raise ValueError('annotation has no supported rotated-box geometry')


def to_horizontal_bbox(annotation):
    """Return a horizontal COCO ``[x, y, width, height]`` and its source.

    Resolution order:
      1. ``bbox`` is 4 elements -> native HBB.
      2. ``bbox`` is 5 elements ``(cx, cy, w, h, angle)`` -> OBB envelope.
      3. ``robndbox / rbox / obb / rotated_bbox`` -> OBB envelope.
      4. ``segmentation`` polygon -> AABB envelope.
    """
    bbox_raw = annotation.get('bbox')
    if isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) == 4:
        bbox = _float_list(bbox_raw, 4, 'bbox')
        if bbox[2] <= 0 or bbox[3] <= 0:
            raise ValueError('bbox width and height must be positive')
        return bbox, 'hbb'
    has_obb = (
        (isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) == 5)
        or any(annotation.get(key) is not None for key in
               ('robndbox', 'rbox', 'obb', 'rotated_bbox'))
    )
    if has_obb:
        bbox = _bbox_from_obb(annotation)
        geometry_source = 'obb-envelope'
    else:
        segmentation = annotation.get('segmentation')
        if segmentation in (None, []):
            raise ValueError('annotation has neither HBB, OBB, nor polygon')
        bbox = _bbox_from_polygon(segmentation)
        geometry_source = 'polygon-envelope'
    if bbox[2] <= 0 or bbox[3] <= 0:
        raise ValueError('{} has zero-area envelope'.format(geometry_source))
    return bbox, geometry_source


def _resolve_beneath_root(shiprs_root, file_name, coco_path):
    """Resolve a source image while prohibiting paths outside ShipRS root.

    Candidate resolution order:
      1. ``file_name`` absolute path directly.
      2. ``<shiprs_root>/<file_name>`` (when annotations/images share the
         same parent directory).
      3. ``<shiprs_root>/images/<file_name>`` (the standard ShipRSImageNet
         layout after the model_v4 dataset reorganization
         ``external_data/ShipRSImageNet/{annotations,images}/``).
      4. ``<coco_path.parent>/<file_name>`` (legacy COCO-sibling image).

    The returned ``relative_name`` strips a single leading ``images/`` segment
    so downstream consumers can join it with their own ``img_prefix`` without
    duplicating the directory name.
    """
    if not isinstance(file_name, str) or not file_name.strip():
        raise ValueError('image file_name must be a non-empty string')
    root = shiprs_root.resolve()
    raw_path = Path(file_name)
    candidates = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.extend((
            root / raw_path,
            root / 'images' / raw_path,
            coco_path.parent / raw_path,
        ))
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            relative = resolved.relative_to(root)
        except (OSError, ValueError):
            continue
        if resolved.is_file():
            relative_name = relative.as_posix()
            # Note: ``img_prefix`` in the ShipRS fine-tune config is
            # ``external_data/ShipRSImageNet/`` (no trailing ``images/``), so
            # ``relative_name`` MUST keep the leading ``images/`` segment.
            # ``img_prefix + relative_name`` then reconstructs the original
            # path without duplication.
            return resolved, relative_name
    raise ValueError('image is missing or outside ShipRS root: {}'.format(
        file_name))


def _clip_bbox(bbox, width, height):
    if not _is_finite_number(width) or not _is_finite_number(height):
        raise ValueError('image width and height must be finite')
    width = float(width)
    height = float(height)
    if width <= 0 or height <= 0:
        raise ValueError('image width and height must be positive')
    x, y, box_width, box_height = bbox
    x2 = x + box_width
    y2 = y + box_height
    clipped_x = min(max(x, 0.0), width)
    clipped_y = min(max(y, 0.0), height)
    clipped_x2 = min(max(x2, 0.0), width)
    clipped_y2 = min(max(y2, 0.0), height)
    if clipped_x2 <= clipped_x or clipped_y2 <= clipped_y:
        raise ValueError('bbox has no positive area inside image bounds')
    clipped = [clipped_x, clipped_y, clipped_x2 - clipped_x,
               clipped_y2 - clipped_y]
    return clipped, clipped != list(bbox)


def _load_coco(path):
    with path.open('r', encoding='utf-8') as handle:
        dataset = json.load(handle)
    if not isinstance(dataset, dict):
        raise ValueError('COCO file must contain an object: {}'.format(path))
    for key in ('images', 'annotations', 'categories'):
        if not isinstance(dataset.get(key), list):
            raise ValueError('COCO file lacks list field {!r}: {}'.format(
                key, path))
    return dataset


def discover_coco_annotations(root):
    """Discover annotated COCO manifests below a ShipRS root deterministically.

    Annotation-free files are deliberately excluded: ShipRS test subsets do
    not enter this external training manifest.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError('ShipRS root is not a directory: {}'.format(root))
    discovered = []
    for candidate in sorted(root.rglob('*.json'), key=lambda path: str(path)):
        try:
            dataset = _load_coco(candidate)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            continue
        if dataset['annotations']:
            discovered.append(candidate.resolve())
    return discovered


def _load_exclusions(exclude_list, shiprs_root):
    excluded = set()
    if exclude_list is None:
        return excluded
    list_path = Path(exclude_list)
    with list_path.open('r', encoding='utf-8') as handle:
        for raw_line in handle:
            value = raw_line.strip()
            if not value or value.startswith('#'):
                continue
            value = value.split(',', 1)[0].strip()
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = Path(shiprs_root) / candidate
            try:
                excluded.add(candidate.resolve())
            except OSError:
                raise ValueError('invalid exclusion path: {}'.format(value))
    return excluded


def _audit_row(coco_path, image, annotation, category, decision=None,
               geometry_source='', clamped=False, excluded=False,
               output_annotation_id='', message=''):
    return {
        'source_coco': str(coco_path),
        'source_image_id': image.get('id', ''),
        'source_image_file_name': image.get('file_name', ''),
        'source_annotation_id': annotation.get('id', ''),
        'source_category_id': annotation.get('category_id', ''),
        'source_category_name': '' if category is None else category.get(
            'name', ''),
        'action': '' if decision is None else decision.action,
        'mapping_reason': '' if decision is None else decision.reason,
        'geometry_source': geometry_source,
        'clamped': int(bool(clamped)),
        'excluded': int(bool(excluded)),
        'output_annotation_id': output_annotation_id,
        'message': message,
    }


def _validated_sources(coco_paths, shiprs_root, excluded_paths):
    """Load sources and construct stable, unique image records."""
    source_images = []
    source_annotations = []
    seen_image_paths = set()
    seen_annotation_records = set()
    declared_targets = set()
    for coco_path in sorted((Path(path).resolve() for path in coco_paths),
                            key=lambda path: str(path)):
        dataset = _load_coco(coco_path)
        categories = {}
        category_names = set()
        for category in dataset['categories']:
            if not isinstance(category, dict) or 'id' not in category or \
                    not isinstance(category.get('name'), str):
                raise ValueError('invalid category record in {}'.format(
                    coco_path))
            if category['id'] in categories:
                raise ValueError('duplicate category id {} in {}'.format(
                    category['id'], coco_path))
            if category['name'] in category_names:
                raise ValueError('duplicate category name {!r} in {}'.format(
                    category['name'], coco_path))
            categories[category['id']] = category
            category_names.add(category['name'])
        all_images = {}
        images = {}
        for image in dataset['images']:
            if not isinstance(image, dict) or 'id' not in image:
                raise ValueError('invalid image record in {}'.format(coco_path))
            if image['id'] in all_images:
                raise ValueError('duplicate image id {} in {}'.format(
                    image['id'], coco_path))
            resolved, relative_name = _resolve_beneath_root(
                shiprs_root, image.get('file_name'), coco_path)
            all_images[image['id']] = resolved
            if resolved in seen_image_paths:
                raise ValueError('duplicate resolved image path: {}'.format(
                    resolved))
            seen_image_paths.add(resolved)
            image_record = {
                'source_coco': coco_path,
                'source': image,
                'resolved_path': resolved,
                'relative_name': relative_name,
                'excluded': resolved in excluded_paths,
            }
            images[image['id']] = image_record
            source_images.append(image_record)
        for annotation in dataset['annotations']:
            if not isinstance(annotation, dict):
                raise ValueError('invalid annotation record in {}'.format(
                    coco_path))
            image_id = annotation.get('image_id')
            category_id = annotation.get('category_id')
            if image_id not in all_images:
                raise ValueError('annotation references unknown image id {} in {}'.format(
                    image_id, coco_path))
            if category_id not in categories:
                raise ValueError('annotation references unknown category id {} in {}'.format(
                    category_id, coco_path))
            if all_images[image_id] in excluded_paths:
                continue
            if images[image_id]['excluded']:
                continue
            duplicate_key = (all_images[image_id], annotation.get('id'))
            if duplicate_key in seen_annotation_records:
                raise ValueError('duplicate image/source annotation record: {}'.format(
                    duplicate_key))
            seen_annotation_records.add(duplicate_key)
            source_annotations.append((coco_path, images[image_id], annotation,
                                       categories[category_id]))
            if annotation.get('category_id') in (0, 1, 2, 3):
                declared_targets.add(annotation['category_id'])
    source_images.sort(key=lambda record: str(record['resolved_path']))
    source_annotations.sort(key=lambda record: (
        str(record[1]['resolved_path']), str(record[0]),
        str(record[2]['id'])))
    return source_images, source_annotations


def convert_shiprs(coco_paths, shiprs_root, enable_ms=False,
                   exclude_list=None, target_levels=(3,)):
    """Convert selected ShipRS COCO files into a 25-class COCO dataset.

    ``target_levels`` filters which level JSON manifests enter the conversion
    (default ``(3,)``: the 50-class fine-grained annotations).  ``exclude_list``
    is an optional leakage exclusion list.
    """
    root = Path(shiprs_root).resolve()
    excluded_paths = _load_exclusions(exclude_list, root)
    selected_paths = []
    for raw_path in coco_paths:
        coco_path = Path(raw_path).resolve()
        if not coco_path.is_file():
            continue
        # Match .../ShipRSImageNet_*_level_<N>.json for any digit N
        name = coco_path.name
        if 'level_' not in name:
            continue
        try:
            level = int(name.rsplit('level_', 1)[1].split('.', 1)[0])
        except (IndexError, ValueError):
            continue
        if level in target_levels:
            selected_paths.append(coco_path)
    if not selected_paths:
        raise ValueError('no ShipRS COCO manifests match target_levels={} under {}'.format(
            target_levels, root))
    source_images, source_annotations = _validated_sources(
        selected_paths, root, excluded_paths)
    output_images = []
    image_ids = {}
    for record in source_images:
        if record['excluded']:
            continue
        image = record['source']
        if not _is_finite_number(image.get('width')) or not _is_finite_number(
                image.get('height')) or float(image['width']) <= 0 or \
                float(image['height']) <= 0:
            raise ValueError('image dimensions must be positive: {}'.format(
                record['relative_name']))
        image_ids[record['resolved_path']] = len(output_images) + 1
        output_images.append({
            'id': len(output_images) + 1,
            'file_name': record['relative_name'],
            'width': image['width'],
            'height': image['height'],
        })
    output_annotations = []
    audit_rows = []
    mapped_counts = Counter()
    for coco_path, image_record, annotation, category in source_annotations:
        decision = map_shiprs_category(category['name'], enable_ms=enable_ms)
        if image_record['excluded']:
            audit_rows.append(_audit_row(
                coco_path, image_record['source'], annotation, category,
                decision=decision, excluded=True,
                message='excluded_by_leakage_list'))
            continue
        if decision.action == 'drop':
            audit_rows.append(_audit_row(
                coco_path, image_record['source'], annotation, category,
                decision=decision, message='dropped_non_ship_object'))
            continue
        try:
            bbox, geometry_source = to_horizontal_bbox(annotation)
            bbox, clamped = _clip_bbox(
                bbox, image_record['source']['width'],
                image_record['source']['height'])
        except ValueError as error:
            raise ValueError('invalid annotation {} in {}: {}'.format(
                annotation.get('id'), coco_path, error))
        output_id = len(output_annotations) + 1
        if decision.action == 'map':
            category_id = decision.target_id
            iscrowd = 0
            mapped_counts[category_id] += 1
        else:
            # mmdet's CocoDataset keeps iscrowd boxes as bboxes_ignore only
            # after category filtering; class zero is a harmless sentinel.
            category_id = 0
            iscrowd = 1
        output_annotations.append({
            'id': output_id,
            'image_id': image_ids[image_record['resolved_path']],
            'category_id': category_id,
            'bbox': bbox,
            'area': bbox[2] * bbox[3],
            'iscrowd': iscrowd,
            'segmentation': [],
            'source_category_id': annotation['category_id'],
            'source_category_name': category['name'],
            'mapping_reason': decision.reason,
            'geometry_source': geometry_source,
        })
        audit_rows.append(_audit_row(
            coco_path, image_record['source'], annotation, category,
            decision=decision, geometry_source=geometry_source, clamped=clamped,
            output_annotation_id=output_id,
            message='clamped_to_image_bounds' if clamped else 'converted'))
    missing = sorted(target_id for target_id in (0, 1, 2)
                     if mapped_counts[target_id] == 0)
    if missing:
        names = ', '.join(CLASS_NAMES[target_id] for target_id in missing)
        raise ValueError('mapped ShipRS categories have zero retained instances: {}'.format(
            names))
    output = {
        'images': output_images,
        'annotations': output_annotations,
        'categories': [
            {'id': category_id, 'name': name}
            for category_id, name in enumerate(CLASS_NAMES)
        ],
    }
    return output, audit_rows


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write('\n')


def _write_audit(path, audit_rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS,
                                extrasaction='ignore')
        writer.writeheader()
        writer.writerows(audit_rows)


def _summary(output, audit_rows, coco_paths):
    actions = Counter(row['action'] for row in audit_rows)
    geometries = Counter(row['geometry_source'] for row in audit_rows
                         if row['geometry_source'])
    mapped = Counter(annotation['category_id'] for annotation in
                     output['annotations'] if not annotation['iscrowd'])
    return {
        'source_coco_files': [str(Path(path).resolve()) for path in coco_paths],
        'images': len(output['images']),
        'annotations': len(output['annotations']),
        'mapped_instances_by_class': {
            CLASS_NAMES[index]: mapped.get(index, 0) for index in range(25)
        },
        'action_counts': dict(sorted(actions.items())),
        'geometry_source_counts': dict(sorted(geometries.items())),
        'clamped_annotations': sum(row['clamped'] for row in audit_rows),
        'excluded_annotations': sum(row['excluded'] for row in audit_rows),
        'invalid_output_boxes': 0,
        'missing_images': 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shiprs-root', required=True,
                        help='root containing ShipRSImageNet COCO JSON files')
    parser.add_argument('--out-json',
                        default='data/external/shiprs_mapped_train.json')
    parser.add_argument('--audit-csv',
                        default='data/external/shiprs_mapping_audit.csv')
    parser.add_argument('--summary-json',
                        default='data/external/shiprs_summary.json')
    parser.add_argument('--exclude-list', default=None,
                        help='line-oriented leakage exclusion list')
    parser.add_argument('--enable-ms', action='store_true',
                        help='enable the explicitly approved Container Ship -> MS mapping')
    parser.add_argument('--target-levels', default='3',
                        help='comma-separated ShipRS annotation levels to include')
    args = parser.parse_args()
    root = Path(args.shiprs_root).resolve()
    coco_paths = discover_coco_annotations(root)
    if not coco_paths:
        parser.error('no annotated COCO manifests found under {}'.format(root))
    target_levels = tuple(int(s) for s in args.target_levels.split(',') if s.strip())
    output, audit_rows = convert_shiprs(
        coco_paths, root, enable_ms=args.enable_ms,
        exclude_list=args.exclude_list, target_levels=target_levels)
    _write_json(args.out_json, output)
    _write_audit(args.audit_csv, audit_rows)
    _write_json(args.summary_json, _summary(output, audit_rows, coco_paths))
    print('Wrote {}'.format(args.out_json))
    print('Wrote {}'.format(args.audit_csv))
    print('Wrote {}'.format(args.summary_json))


if __name__ == '__main__':
    main()