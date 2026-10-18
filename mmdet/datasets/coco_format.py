"""COCO-format detection datasets used by the OSTD experiments.

All target datasets use COCO-format JSON annotations with horizontal bounding
boxes, so the only difference from stock ``CocoDataset`` is the class list
and category-id mapping.
"""
from .builder import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class VisDroneDataset(CocoDataset):
    CLASSES = ('pedestrian', 'people', 'bicycle', 'car', 'van',
               'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor')


@DATASETS.register_module()
class HRRSDDataset(CocoDataset):
    """HRRSD as horizontal boxes in COCO format (13 classes)."""

    CLASSES = ('ship', 'bridge', 'ground track field', 'storage tank',
               'basketball court', 'tennis court', 'airplane',
               'baseball diamond', 'harbor', 'vehicle', 'crossroad',
               'T junction', 'parking lot')


@DATASETS.register_module()
class DIORDataset(CocoDataset):
    CLASSES = ('airplane', 'airport', 'baseballfield', 'basketballcourt',
               'bridge', 'chimney', 'expressway-service-area',
               'expressway-toll-station', 'dam', 'golffield',
               'groundtrackfield', 'harbor', 'overpass', 'ship',
               'stadium', 'storagetank', 'tenniscourt', 'trainstation',
               'vehicle', 'windmill')


@DATASETS.register_module()
class DOTAHBBDataset(CocoDataset):
    """DOTA-v1.5 converted to horizontal boxes (HBB) and COCO JSON.

    Categories follow the canonical DOTA class list; the JSON converter is
    expected to remap source ids to a contiguous 1..16 range so that this
    tuple matches the JSON's ``categories`` order.
    """
    CLASSES = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
               'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank', 'soccer-ball-field',
               'roundabout', 'harbor', 'swimming-pool', 'helicopter',
               'container-crane')


@DATASETS.register_module()
class SODAADataset(CocoDataset):
    """SODA-A patches represented as horizontal boxes in COCO format."""

    CLASSES = ('airplane', 'helicopter', 'small-vehicle', 'large-vehicle',
               'ship', 'container', 'storage-tank', 'swimming-pool',
               'windmill')
