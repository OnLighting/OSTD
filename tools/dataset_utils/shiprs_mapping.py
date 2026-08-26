"""Conservative, auditable mapping from ShipRS labels to competition classes."""

from dataclasses import dataclass
import re
from typing import Optional, Tuple

try:
    from typing import Literal
except ImportError:  # Python 3.6--3.7
    from typing_extensions import Literal


CLASS_NAMES: Tuple[str, ...] = (
    'HM', 'LQS', 'QHS', 'MS',
    'A1_SU-35', 'A2_C-130', 'A3_C-17', 'A4_C-5', 'A5_F-16',
    'A6_TU-160', 'A7_E-3', 'A8_B-52', 'A9_P-3C', 'A10_B-1B',
    'A11_E-8', 'A12_TU-22', 'A13_F-15', 'A14_KC-135',
    'A15_F-22', 'A16_FA-18', 'A17_TU-95', 'A18_KC-10',
    'A19_SU-34', 'A20_SU-24', 'FSC')


@dataclass(frozen=True)
class MappingDecision:
    """Disposition of one ShipRS source category.

    ``ignore`` preserves a real but unmapped ship as an ignore region in the
    downstream COCO conversion. ``drop`` is reserved for non-ship objects.
    """

    action: Literal['map', 'ignore', 'drop']
    target_id: Optional[int]
    reason: str


def normalize_shiprs_name(name: str) -> str:
    """Normalize case, whitespace, underscores, and hyphens in source names."""
    return re.sub(r'\s+', ' ', re.sub(r'[_-]+', ' ', name.strip())).upper()


def _normalized_names(names):
    return frozenset(normalize_shiprs_name(name) for name in names)


_TARGET_NAMES = {
    0: _normalized_names((
        'Other Aircraft Carrier', 'Enterprise', 'Nimitz', 'Midway')),
    1: _normalized_names((
        'Other Landing', 'YuTing LL', 'YuDeng LL', 'YuDao LL', 'YuZhao LL',
        'Austin LL', 'Osumi LL', 'Wasp LL', 'LSD 41 LL', 'LHA LL',
        'Amphibious Transport Dock', 'Amphibious Assault Ship')),
    2: _normalized_names((
        'Ticonderoga', 'Other Destroyer', 'Atago DD', 'Arleigh Burke DD',
        'Hatsuyuki DD', 'Hyuga DD', 'Asagiri DD', 'Other Frigate',
        'Perry FF')),
}

# This is the only commercial class approved for MS mapping by the task brief.
# Additional commercial labels must be added explicitly after category-review.
_MS_NAMES = _normalized_names(('Container Ship',))

_IGNORE_NAMES = _normalized_names((
    'Other Ship', 'Other Warship', 'Submarine', 'Patrol', 'Commander',
    'Auxiliary Ship', 'Medical Ship', 'Test Ship', 'Training Ship'))

_DOCK_NAME = normalize_shiprs_name('Dock')


def map_shiprs_category(name: str, enable_ms: bool = False) -> MappingDecision:
    """Return the explicit, conservative decision for a ShipRS category name.

    No keyword or substring inference is used: categories absent from the
    reviewed tables remain ignored so genuine ships are never treated as
    background by downstream conversion.
    """
    normalized_name = normalize_shiprs_name(name)
    for target_id, source_names in _TARGET_NAMES.items():
        if normalized_name in source_names:
            return MappingDecision('map', target_id, 'approved_shiprs_mapping')
    if normalized_name in _MS_NAMES:
        if enable_ms:
            return MappingDecision('map', 3, 'approved_ms_mapping')
        return MappingDecision('ignore', None, 'ms_mapping_disabled')
    if normalized_name == _DOCK_NAME:
        return MappingDecision('drop', None, 'non_ship_dock')
    if normalized_name in _IGNORE_NAMES:
        return MappingDecision('ignore', None, 'explicitly_unmapped_ship')
    return MappingDecision('ignore', None, 'unreviewed_shiprs_category')