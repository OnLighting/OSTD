from .coco import CocoDataset
from .builder import DATASETS

from mmdet.core.evaluation import (CANDIDATE_SCORE_FLOOR,
                                   evaluate_mmdet_results,
                                   filter_mmdet_results,
                                   normalize_score_thresholds)


@DATASETS.register_module()
class AircraftDataset(CocoDataset):
    """25-class optical satellite aircraft/vehicle detection.

    Annotation layout follows data/annotations/instances_{train,val}.json built
    by tools/convert_yolo_to_coco.py. Class ids match data/dataset.yaml's
    0..24 ordering.

    When ``evaluate`` is called with ``metric`` containing ``'official'``,
    the dataset writes per-class, superclass, official, and merged metrics
    using the shared ``mmdet.core.evaluation.official_metrics`` module at one
    fixed training threshold (default ``0.30``). Threshold calibration never
    runs inside training evaluation.
    Stable scalar keys (``official_recall``, ``official_fdr``,
    ``ship_recall``, ``ship_fdr``, ...) are emitted so EvalHook consumers
    can read them without parsing nested dicts.
    """

    CLASSES = (
        'HM', 'LQS', 'QHS', 'MS',
        'A1_SU-35', 'A2_C-130', 'A3_C-17', 'A4_C-5', 'A5_F-16', 'A6_TU-160',
        'A7_E-3', 'A8_B-52', 'A9_P-3C', 'A10_B-1B', 'A11_E-8', 'A12_TU-22',
        'A13_F-15', 'A14_KC-135', 'A15_F-22', 'A16_FA-18', 'A17_TU-95',
        'A18_KC-10', 'A19_SU-34', 'A20_SU-24', 'FSC',
    )
    OFFICIAL_SCORE_THRESHOLD = 0.30

    def __init__(self, *args, official_score_threshold=0.30, **kwargs):
        thresholds = normalize_score_thresholds(official_score_threshold)
        if len(set(thresholds)) != 1:
            raise ValueError(
                'training official_score_threshold must be one scalar')
        self.official_score_threshold = thresholds[0]
        super().__init__(*args, **kwargs)

    def evaluate(self, results, metric=['bbox', 'official'], **kwargs):
        """Run standard COCO and official metrics when requested.

        Standard metrics (e.g. ``'bbox'``) are computed by ``CocoDataset``.
        ``'official'`` is computed by the shared metric module and writes
        scalar keys directly into the returned OrderedDict so that hooks
        can read them without parsing nested structures.
        """
        if isinstance(metric, str):
            metrics_requested = [metric]
        else:
            metrics_requested = list(metric)
        standard_metrics = [m for m in metrics_requested if m != 'official']
        run_official = 'official' in metrics_requested

        eval_results = super().evaluate(
            results, metric=standard_metrics if standard_metrics else 'bbox',
            **kwargs)

        if not run_official:
            return eval_results

        # Build per-image GT annotation lists expected by the metric module.
        # ``get_ann_info`` returns parsed dicts with ``bboxes`` (xyxy) and
        # ``labels`` (contiguous 0..N-1); the metric module consumes raw
        # ``bbox`` (xywh) + ``category_id`` (dataset-native ids), so we
        # convert via ``label2cat`` here.
        label2cat = {label: cat for cat, label in self.cat2label.items()}
        gt_infos = []
        for i in range(len(self)):
            ann_info = self.get_ann_info(i)
            anns = []
            for box, label in zip(ann_info.get('bboxes', []),
                                  ann_info.get('labels', [])):
                x1, y1, x2, y2 = box
                anns.append({
                    'bbox': [float(x1), float(y1),
                             float(x2 - x1), float(y2 - y1)],
                    'category_id': int(label2cat[int(label)]),
                })
            gt_infos.append(anns)
        threshold = float(getattr(
            self, 'official_score_threshold', self.OFFICIAL_SCORE_THRESHOLD))
        # Filter before matching so low-score candidates do not consume CPU
        # during every training validation epoch. Final 25-class calibration
        # remains a separate, one-time post-training step in run.sh.
        filtered_results = [
            filter_mmdet_results(image_results, threshold)
            for image_results in results
        ]
        metrics = evaluate_mmdet_results(
            filtered_results, gt_infos, CANDIDATE_SCORE_FLOOR)
        thresholds_by_super = {
            'ship': threshold,
            'aircraft': threshold,
            'vehicle': threshold,
        }

        official = metrics['official']
        # When any official superclass lacks GT, recall/fdr are NaN. Emit
        # them as-is (Python NaN) so the checkpoint hook can detect
        # ``math.isnan`` and refuse to save this run as best.
        eval_results['official_recall'] = official['recall']
        eval_results['official_fdr'] = official['fdr']
        eval_results['official_available'] = bool(official['available'])
        eval_results['official_unavailable_superclasses'] = list(
            official['unavailable_superclasses'])
        eval_results['training_score_thresholds'] = dict(
            thresholds_by_super)
        for super_name, threshold in thresholds_by_super.items():
            eval_results[f'official_threshold_{super_name}'] = (
                float(threshold)
                if threshold is not None else float('nan'))
        for super_name, vals in metrics['by_super'].items():
            eval_results[f'{super_name}_recall'] = (
                float(vals['recall']) if vals['recall'] is not None else float('nan'))
            eval_results[f'{super_name}_fdr'] = (
                float(vals['fdr']) if vals['fdr'] is not None else float('nan'))
        eval_results['merged_recall'] = float(metrics['merged']['recall'])
        eval_results['merged_fdr'] = float(metrics['merged']['fdr'])
        # Per-class breakdown (compact, used by hooks that want to log it).
        eval_results['official_per_class'] = metrics['per_class']
        return eval_results
