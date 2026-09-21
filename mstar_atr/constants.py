"""
Constants and metadata for the MSTAR (Moving and Stationary Target Acquisition and Recognition) benchmark.
"""

from typing import Dict, List, Tuple

# Standard 10-Class MSTAR Target Classes
CLASSES: List[str] = [
    "2S1",       # 0: Self-propelled Howitzer
    "BMP2",      # 1: Infantry Fighting Vehicle
    "BRDM2",     # 2: Armored Reconnaissance Vehicle
    "BTR60",     # 3: Armored Personnel Carrier (8x8)
    "BTR70",     # 4: Armored Personnel Carrier (8x8)
    "D7",        # 5: Caterpillar Bulldozer
    "T62",       # 6: Main Battle Tank
    "T72",       # 7: Main Battle Tank
    "ZIL131",    # 8: 6x6 Military Cargo Truck
    "ZSU23_4",   # 9: Self-propelled Anti-aircraft Gun System ("Shilka")
]

# Mapping between directory variations and standard class names
CLASS_ALIASES: Dict[str, str] = {
    "2S1": "2S1",
    "2s1": "2S1",
    "BMP2": "BMP2",
    "BMP_2": "BMP2",
    "bmp2": "BMP2",
    "BRDM2": "BRDM2",
    "BRDM_2": "BRDM2",
    "brdm2": "BRDM2",
    "BTR60": "BTR60",
    "BTR_60": "BTR60",
    "btr60": "BTR60",
    "BTR70": "BTR70",
    "BTR_70": "BTR70",
    "btr70": "BTR70",
    "D7": "D7",
    "d7": "D7",
    "T62": "T62",
    "t62": "T62",
    "T72": "T72",
    "t72": "T72",
    "ZIL131": "ZIL131",
    "ZIL_131": "ZIL131",
    "zil131": "ZIL131",
    "ZSU23_4": "ZSU23_4",
    "ZSU_23_4": "ZSU23_4",
    "ZSU234": "ZSU23_4",
    "zsu23_4": "ZSU23_4",
}

CLASS_TO_IDX: Dict[str, int] = {cls_name: i for i, cls_name in enumerate(CLASSES)}
IDX_TO_CLASS: Dict[int, str] = {i: cls_name for i, cls_name in enumerate(CLASSES)}

# Detailed Target Metadata and Tactical Descriptions
TARGET_METADATA: Dict[str, Dict[str, str]] = {
    "2S1": {
        "full_name": "2S1 Gvozdika",
        "category": "Self-propelled Artillery",
        "origin": "Soviet Union",
        "radar_signature": "Prominent central turret scatterer, elongated gun barrel cavity, distinctive boxy chassis track reflections.",
        "length_width": "7.26 m x 2.85 m",
        "weight": "15.7 tonnes",
    },
    "BMP2": {
        "full_name": "BMP-2",
        "category": "Infantry Fighting Vehicle (IFV)",
        "origin": "Soviet Union",
        "radar_signature": "Sloped frontal armor reflection, low profile silhouette, distinct 30mm 2A42 autocannon return.",
        "length_width": "6.72 m x 3.15 m",
        "weight": "14.3 tonnes",
    },
    "BRDM2": {
        "full_name": "BRDM-2",
        "category": "Armored Reconnaissance Vehicle",
        "origin": "Soviet Union",
        "radar_signature": "Compact 4x4 wheeled hull with rounded boat-like nose, belly belly wheel scatterers, small conical turret.",
        "length_width": "5.75 m x 2.35 m",
        "weight": "7.0 tonnes",
    },
    "BTR60": {
        "full_name": "BTR-60PB",
        "category": "Armored Personnel Carrier (APC)",
        "origin": "Soviet Union",
        "radar_signature": "Eight-wheeled chassis with characteristic multi-point tire ground-bounce scatterers and faceted side armor.",
        "length_width": "7.56 m x 2.83 m",
        "weight": "10.3 tonnes",
    },
    "BTR70": {
        "full_name": "BTR-70",
        "category": "Armored Personnel Carrier (APC)",
        "origin": "Soviet Union",
        "radar_signature": "Upgraded eight-wheeled hull, twin engine bay corner reflections, lower roofline than BTR-60.",
        "length_width": "7.54 m x 2.80 m",
        "weight": "11.5 tonnes",
    },
    "D7": {
        "full_name": "Caterpillar D7G",
        "category": "Heavy Engineering / Bulldozer",
        "origin": "United States",
        "radar_signature": "Extremely high radar cross section (RCS) from vertical dozer blade dihedral reflector, roll-cage/ROPS lattice.",
        "length_width": "5.28 m x 2.60 m",
        "weight": "20.4 tonnes",
    },
    "T62": {
        "full_name": "T-62",
        "category": "Main Battle Tank (MBT)",
        "origin": "Soviet Union",
        "radar_signature": "Cast dome turret strong specular reflection, smooth 115mm gun barrel, five large road wheels per side.",
        "length_width": "9.34 m x 3.30 m",
        "weight": "37.0 tonnes",
    },
    "T72": {
        "full_name": "T-72M1",
        "category": "Main Battle Tank (MBT)",
        "origin": "Soviet Union",
        "radar_signature": "Very low profile hemispherical turret, 125mm 2A46 smoothbore gun, V-shaped frontal splash plate dihedral return.",
        "length_width": "9.53 m x 3.59 m",
        "weight": "41.5 tonnes",
    },
    "ZIL131": {
        "full_name": "ZIL-131",
        "category": "6x6 Military Cargo Truck",
        "origin": "Soviet Union",
        "radar_signature": "Distinctive soft-skin truck profile, high cargo bed cavity return, front cab dihedral, non-armored wheel wells.",
        "length_width": "7.04 m x 2.50 m",
        "weight": "6.7 tonnes",
    },
    "ZSU23_4": {
        "full_name": "ZSU-23-4 Shilka",
        "category": "Self-propelled Anti-aircraft Gun",
        "origin": "Soviet Union",
        "radar_signature": "High complexity RCS signature from quad liquid-cooled 23mm guns and prominent 'Gun Dish' radar dish cylinder.",
        "length_width": "6.54 m x 3.13 m",
        "weight": "19.0 tonnes",
    },
}

# Standard Operating Conditions (SOC) angles
SOC_TRAIN_DEPRESSION: float = 17.0  # degrees
SOC_TEST_DEPRESSION: float = 15.0   # degrees

# Default Image Dimensions
DEFAULT_INPUT_SIZE: Tuple[int, int] = (128, 128)
DEFAULT_CROP_SIZE: Tuple[int, int] = (88, 88)
