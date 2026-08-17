# External datasets for HM/LQS/QHS/FSC augmentation

Lookup date: 2026-08-17

## Primary candidates

1. ShipRSImageNet V1.1
   - 3,435 optical remote-sensing images and 17,573 ship instances; HBB, OBB and polygon annotations; hierarchical labels with 50 types.
   - Relevant types include aircraft carriers, destroyers, frigates, and multiple landing/amphibious ships.
   - Academic use only; Google Earth terms also apply.
   - Source: https://github.com/zzndream/ShipRSImageNet

2. HRSC2016
   - 1,061 high-resolution optical remote-sensing images; hierarchical ship/category/type annotations.
   - Relevant labels include aircraft carriers, destroyers/frigates and amphibious/landing ships.
   - Dataset mirror/description: https://www.kaggle.com/datasets/guofeng/hrsc2016
   - Hierarchy discussion: https://doi.org/10.1109/jstars.2025.3570872

3. FGSC-23
   - 4,080 cropped optical remote-sensing samples, 23 categories, classification labels only.
   - Directly relevant categories: air carrier, destroyer, frigate, landing craft, amphibious transport dock, Tarawa-class amphibious assault ship, amphibious assault ship.
   - Research use only.
   - Source: https://github.com/xiong577/ship-datasets

4. FGSCR-42
   - 9,320 cropped remote-sensing ship images in 42 categories, roughly 200 samples per category; classification labels only.
   - Source: https://github.com/DYH666/FGSCR-42
   - Paper: https://www.mdpi.com/2072-4292/13/4/747

5. UOW-Vessel
   - 3,500 optical satellite images, 35,598 polygon-annotated instances, 10 vessel categories.
   - Includes aircraft carrier, landing, destroyer and frigate categories; useful as scene-level detection/segmentation data.
   - Paper/source: https://openaccess.thecvf.com/content/WACV2024/html/Bui_UOW-Vessel_A_Benchmark_Dataset_of_High-Resolution_Optical_Satellite_Images_for_WACV_2024_paper.html

## Broad-domain auxiliary datasets

6. FAIR1M
   - High-resolution optical remote-sensing detection data (0.3--0.8 m RGB).
   - Relevant coarse labels: Warship, other ship types, Truck Tractor, Trailer, Cargo Truck and other vehicles.
   - Paper: https://arxiv.org/abs/2103.05569

7. DOTA v1/v2
   - Oriented aerial detection data; relevant coarse categories are ship, large vehicle, small vehicle and harbor.
   - Official project: https://captain-whu.github.io/DOTA/

8. xView
   - WorldView-3 overhead imagery, over one million objects in 60 categories.
   - Useful maritime and heavy-vehicle/background categories, but not direct HM/LQS/QHS/FSC fine-grained supervision.
   - Official project: https://xviewdataset.org/
   - Paper: https://arxiv.org/abs/1802.07856

9. MVRSD
   - 3,000 Google Earth remote-sensing images, 32,626 military-vehicle instances across more than 40 scenarios, stated 0.3 m resolution.
   - Category ontology and license need manual verification before adoption; useful only after an annotation audit.
   - Source: https://github.com/baidongls/MVRSD

## Mapping proposed for this competition

- HM: aircraft carrier classes/types.
- LQS: landing craft, landing ship/dock, amphibious transport dock, amphibious assault ship, LHA/LHD/LSD types.
- QHS: destroyer and frigate; include cruiser only if the organizer's own label examples establish that convention.
- FSC: direct missile-launcher/TEL samples only when the overhead appearance and annotation are verified. Generic large/military trucks are auxiliary pretraining or hard-negative data, not automatic FSC positives.

## Reproducibility and licensing notes

- Keep original download URL, version, archive checksum, license/terms snapshot, conversion script, label-mapping table and exclusion list.
- Do not redistribute Google Earth-derived imagery unless the applicable terms permit it.
- Detect and remove duplicates across HRSC2016, FGSD, ShipRSImageNet and derived/rehosted datasets; ShipRSImageNet explicitly incorporates HRSC2016 and FGSD content.
- Never map every generic ship or large vehicle to a fine-grained competition class.
