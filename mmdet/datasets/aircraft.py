from .coco import CocoDataset
from .builder import DATASETS

from mmdet.core.evaluation import evaluate_mmdet_results


@DATASETS.register_module()
class AircraftDataset(CocoDataset):
    """25-class optical satellite aircraft/vehicle detection.

    Annotation layout follows data/annotations/instances_{train,val}.json built
    by tools/convert_yolo_to_coco.py. Class ids match data/dataset.yaml's
    0..24 ordering.

    When ``evaluate`` is called with ``metric`` containing ``'official'``,
    the dataset writes per-class, superclass, official, and merged metrics
    using the shared ``mmdet.core.evaluation.official_metrics`` module.
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
        gt_infos = [self.get_ann_info(i) for i in range(len(self))]
        flat_results = list(results)
        metrics = evaluate_mmdet_results(flat_results, gt_infos)

        eval_results['official_recall'] = float(metrics['official']['recall'])
        eval_results['official_fdr'] = float(metrics['official']['fdr'])
        for super_name, vals in metrics['by_super'].items():
            eval_results[f'{super_name}_recall'] = (
                float(vals['recall']) if vals['recall'] is not None else 0.0)
            eval_results[f'{super_name}_fdr'] = (
                float(vals['fdr']) if vals['fdr'] is not None else 0.0)
        eval_results['merged_recall'] = float(metrics['merged']['recall'])
        eval_results['merged_fdr'] = float(metrics['merged']['fdr'])
        # Per-class breakdown (compact, used by hooks that want to log it).
        eval_results['official_per_class'] = metrics['per_class']
        return eval_results
