"""
Snowflake Cortex AI & Data Cloud integration utilities.
Provides simulators and connectors for:
- Snowflake Cortex Analyst (Semantic Model text-to-SQL)
- Snowflake Cortex Search (Hybrid Vector + Keyword RAG)
- Snowflake Cortex Complete (LLM Tactical Debriefs)
- Snowflake Radar Telemetry Data Generator
"""

from datetime import datetime, timedelta
import random
from typing import Any, Dict, List, Tuple
import pandas as pd

from mstar_atr.constants import CLASSES, TARGET_METADATA


def get_sample_radar_telemetry_df(n_samples: int = 120) -> pd.DataFrame:
    """Generates synthetic Snowflake-resident radar detection telemetry."""
    random.seed(42)
    records = []
    base_time = datetime.now() - timedelta(hours=48)

    categories = {
        "2S1": ("Artillery", 15.7, 7.26, 122.0),
        "BMP2": ("Infantry Fighting Vehicle", 14.3, 6.72, 30.0),
        "BRDM2": ("Armored Reconnaissance", 7.0, 5.75, 14.5),
        "BTR60": ("Armored Personnel Carrier", 10.3, 7.56, 14.5),
        "BTR70": ("Armored Personnel Carrier", 11.5, 7.54, 14.5),
        "D7": ("Engineering Bulldozer", 15.0, 4.30, 0.0),
        "T62": ("Main Battle Tank", 37.0, 9.34, 115.0),
        "T72": ("Main Battle Tank", 41.5, 9.53, 125.0),
        "ZIL131": ("Cargo / Logistics Truck", 6.7, 7.04, 0.0),
        "ZSU23_4": ("Anti-Aircraft Weapon", 19.0, 6.54, 23.0),
        "ZSU234": ("Anti-Aircraft Weapon", 19.0, 6.54, 23.0),
    }

    sectors = ["Sector-Alpha (North)", "Sector-Bravo (Ridge)", "Sector-Charlie (Plain)", "Sector-Delta (Depot)"]

    for i in range(n_samples):
        true_cls = random.choice(CLASSES)
        cat, weight, length, caliber = categories[true_cls]
        
        # Simulating classification accuracy ~98%
        if random.random() < 0.98:
            pred_cls = true_cls
            confidence = round(random.uniform(0.91, 0.998), 4)
        else:
            pred_cls = random.choice([c for c in CLASSES if c != true_cls])
            confidence = round(random.uniform(0.55, 0.88), 4)

        depr_angle = random.choice([15.0, 17.0, 30.0, 45.0])
        aspect_angle = round(random.uniform(0.0, 359.9), 1)
        rcs_db = round(random.uniform(12.0, 42.0), 2)
        event_time = base_time + timedelta(minutes=i * 24 + random.randint(0, 15))

        records.append({
            "DETECTION_ID": f"SAR-DET-{1000 + i}",
            "TIMESTAMP": event_time.strftime("%Y-%m-%d %H:%M:%S"),
            "SENSOR_ID": f"SAR-SAT-ORBIT-{random.randint(1, 4)}",
            "PREDICTED_CLASS": pred_cls,
            "TRUE_CLASS": true_cls,
            "TARGET_CATEGORY": cat,
            "CONFIDENCE": confidence,
            "IS_HIGH_CONFIDENCE": confidence >= 0.90,
            "DEPRESSION_ANGLE_DEG": depr_angle,
            "ASPECT_ANGLE_DEG": aspect_angle,
            "RCS_DB": rcs_db,
            "HULL_LENGTH_M": length,
            "COMBAT_WEIGHT_TON": weight,
            "GUN_CALIBER_MM": caliber,
            "SECTOR": random.choice(sectors),
            "LATITUDE": round(34.5000 + random.uniform(-0.15, 0.15), 5),
            "LONGITUDE": round(43.6000 + random.uniform(-0.15, 0.15), 5),
        })

    return pd.DataFrame(records)


def simulate_cortex_analyst_query(user_question: str, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Simulates Snowflake Cortex Analyst: Translates natural language into Snowflake SQL
    using the Semantic Model and executes it on the telemetry DataFrame.
    """
    q_lower = user_question.lower()

    if "main battle tank" in q_lower or "tanks" in q_lower or "t72" in q_lower or "t62" in q_lower:
        sql = """
SELECT 
    DETECTION_ID,
    PREDICTED_CLASS,
    TARGET_CATEGORY,
    CONFIDENCE,
    RCS_DB,
    SECTOR,
    TIMESTAMP
FROM MSTAR_RADAR_DB.ANALYTICS.SAR_TARGET_DETECTIONS
WHERE TARGET_CATEGORY = 'Main Battle Tank'
ORDER BY CONFIDENCE DESC
LIMIT 10;
        """.strip()
        filtered = df[df["TARGET_CATEGORY"] == "Main Battle Tank"].sort_values(by="CONFIDENCE", ascending=False).head(10)
        explanation = "Filtered `SAR_TARGET_DETECTIONS` where `TARGET_CATEGORY = 'Main Battle Tank'` and sorted by `CONFIDENCE` descending."

    elif "high confidence" in q_lower or "greater than 90" in q_lower or "> 90" in q_lower or ">90" in q_lower or "confidence >" in q_lower:
        sql = """
SELECT 
    PREDICTED_CLASS,
    COUNT(*) AS TOTAL_DETECTIONS,
    ROUND(AVG(CONFIDENCE) * 100, 2) AS AVG_CONFIDENCE_PCT,
    ROUND(AVG(RCS_DB), 2) AS AVG_RCS_DB
FROM MSTAR_RADAR_DB.ANALYTICS.SAR_TARGET_DETECTIONS
WHERE CONFIDENCE >= 0.90
GROUP BY PREDICTED_CLASS
ORDER BY TOTAL_DETECTIONS DESC;
        """.strip()
        grouped = df[df["CONFIDENCE"] >= 0.90].groupby("PREDICTED_CLASS").agg(
            TOTAL_DETECTIONS=("DETECTION_ID", "count"),
            AVG_CONFIDENCE_PCT=("CONFIDENCE", lambda x: round(x.mean() * 100, 2)),
            AVG_RCS_DB=("RCS_DB", lambda x: round(x.mean(), 2)),
        ).reset_index().sort_values(by="TOTAL_DETECTIONS", ascending=False)
        filtered = grouped
        explanation = "Aggregated high-confidence detections (CONFIDENCE >= 0.90) by target class with count and average RCS."

    elif "sector" in q_lower:
        sql = """
SELECT 
    SECTOR,
    COUNT(*) AS CONTACT_COUNT,
    ROUND(AVG(CONFIDENCE) * 100, 2) AS AVG_CONFIDENCE_PCT,
    COUNT(DISTINCT PREDICTED_CLASS) AS UNIQUE_TARGET_TYPES
FROM MSTAR_RADAR_DB.ANALYTICS.SAR_TARGET_DETECTIONS
GROUP BY SECTOR
ORDER BY CONTACT_COUNT DESC;
        """.strip()
        grouped = df.groupby("SECTOR").agg(
            CONTACT_COUNT=("DETECTION_ID", "count"),
            AVG_CONFIDENCE_PCT=("CONFIDENCE", lambda x: round(x.mean() * 100, 2)),
            UNIQUE_TARGET_TYPES=("PREDICTED_CLASS", "nunique"),
        ).reset_index().sort_values(by="CONTACT_COUNT", ascending=False)
        filtered = grouped
        explanation = "Grouped all radar detection events by operational defense `SECTOR`."

    elif "depression" in q_lower or "angle" in q_lower:
        sql = """
SELECT 
    DEPRESSION_ANGLE_DEG,
    COUNT(*) AS CHIP_COUNT,
    ROUND(AVG(CONFIDENCE) * 100, 2) AS ACCURACY_PCT,
    ROUND(AVG(RCS_DB), 2) AS MEAN_RCS_DB
FROM MSTAR_RADAR_DB.ANALYTICS.SAR_TARGET_DETECTIONS
GROUP BY DEPRESSION_ANGLE_DEG
ORDER BY DEPRESSION_ANGLE_DEG ASC;
        """.strip()
        grouped = df.groupby("DEPRESSION_ANGLE_DEG").agg(
            CHIP_COUNT=("DETECTION_ID", "count"),
            ACCURACY_PCT=("CONFIDENCE", lambda x: round(x.mean() * 100, 2)),
            MEAN_RCS_DB=("RCS_DB", lambda x: round(x.mean(), 2)),
        ).reset_index().sort_values(by="DEPRESSION_ANGLE_DEG", ascending=True)
        filtered = grouped
        explanation = "Aggregated target classification performance grouped by radar sensor `DEPRESSION_ANGLE_DEG` (SOC 15°/17° vs EOC 30°/45°)."

    else:
        # Default top detections
        sql = """
SELECT 
    DETECTION_ID,
    TIMESTAMP,
    PREDICTED_CLASS,
    TARGET_CATEGORY,
    ROUND(CONFIDENCE * 100, 1) AS CONFIDENCE_PCT,
    RCS_DB,
    SECTOR
FROM MSTAR_RADAR_DB.ANALYTICS.SAR_TARGET_DETECTIONS
ORDER BY TIMESTAMP DESC
LIMIT 12;
        """.strip()
        filtered = df.sort_values(by="TIMESTAMP", ascending=False).head(12)
        explanation = "Retrieved most recent synthetic aperture radar detection telemetry ordered chronologically."

    return {
        "sql": sql,
        "explanation": explanation,
        "results": filtered,
    }


def simulate_cortex_search(search_query: str) -> List[Dict[str, Any]]:
    """
    Simulates Snowflake Cortex Search (hybrid vector embeddings + BM25 keyword search)
    over SAR target intelligence corpus and NATO technical doctrine.
    """
    corpus = [
        {
            "doc_id": "TAC-DOC-001",
            "title": "T-72 Main Battle Tank Radar Signature & Scattering Centers",
            "category": "Armor Signature Analysis",
            "content": "The T-72 presents distinctive specular backscatter peaks from the cylindrical turret ring and frontal glacis plate at 60° slope. The auxiliary external fuel drums mounted on the rear hull produce distinctive double-bounce dihedral reflections when illuminated at 15° to 17° depression angles.",
            "keywords": ["t-72", "t72", "tank", "dihedral", "turret", "glacis", "armor", "fuel drums"],
            "relevance_score": 0.96,
        },
        {
            "doc_id": "TAC-DOC-002",
            "title": "BMP-2 Infantry Fighting Vehicle Dihedral Characterization",
            "category": "Light Armor Doctrine",
            "content": "BMP-2 exhibits strong dihedral reflections along the side skirts meeting the road wheels. The 30mm 2A42 autocannon barrel produces a localized flash when broadside to the radar line-of-sight. Troop compartment roof hatches provide multi-bounce cavity returns.",
            "keywords": ["bmp-2", "bmp2", "ifv", "autocannon", "skirts", "road wheels"],
            "relevance_score": 0.94,
        },
        {
            "doc_id": "TAC-DOC-003",
            "title": "ZSU-23-4 Shilka Quad Autocannon Air Defense Scatterer Dynamics",
            "category": "Air Defense",
            "content": "The RPK-2 'Tobol' radar dish antenna and the quad liquid-cooled 23mm 2A7 autocannons produce extreme scattering center concentration on the forward turret face. The dish curvature causes distinctive multi-path backscatter modulation.",
            "keywords": ["zsu-23-4", "zsu234", "shilka", "air defense", "radar dish", "quad"],
            "relevance_score": 0.92,
        },
        {
            "doc_id": "TAC-DOC-004",
            "title": "BTR-70 vs BTR-60 Wheeled Armored Carrier Aspect Discrimination",
            "category": "Reconnaissance Doctrine",
            "content": "Both 8x8 wheeled vehicles share similar hull lengths (~7.5m), but BTR-70 features twin ZMZ-4905 gasoline engines with distinct rear louver geometry that modulates X-band SAR returns. The side escape doors on BTR-70 create localized shadow gaps between the second and third axles.",
            "keywords": ["btr-70", "btr70", "btr-60", "btr60", "wheeled", "carrier", "engine"],
            "relevance_score": 0.89,
        },
        {
            "doc_id": "TAC-DOC-005",
            "title": "2S1 Gvozdika 122mm Self-Propelled Howitzer Shadow Silhouette",
            "category": "Artillery Threat Assessment",
            "content": "The 2S1 is based on the extended MT-LBu chassis. The long overhang of the 122mm 2A31 howitzer barrel casts a recognizable sharp radar shadow cavity behind the turret when imaged with low grazing angles (15°).",
            "keywords": ["2s1", "gvozdika", "howitzer", "artillery", "barrel", "shadow"],
            "relevance_score": 0.88,
        },
        {
            "doc_id": "TAC-DOC-006",
            "title": "D7 Heavy Engineering Bulldozer Corner Reflector Effect",
            "category": "Civilian / Dual-Use Engineering",
            "content": "The heavy steel blade of the Caterpillar D7 forms an almost ideal 90° dihedral reflector with the ground surface, producing extreme amplitude saturations in SAR chips. This is an essential signature for false-positive filtering in ATR systems.",
            "keywords": ["d7", "bulldozer", "blade", "dihedral", "corner reflector"],
            "relevance_score": 0.86,
        },
    ]

    q_words = set(search_query.lower().replace("-", "").split())
    scored = []
    for doc in corpus:
        doc_words = set(doc["keywords"] + doc["title"].lower().split())
        match_count = len(q_words.intersection(doc_words))
        score = 0.50 + 0.12 * match_count
        if score > 0.50 or not search_query.strip():
            doc_copy = doc.copy()
            doc_copy["score"] = round(min(score, 0.99), 3)
            scored.append(doc_copy)

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:4] if scored else corpus[:3]


def generate_cortex_tactical_debrief(pred_class: str, confidence: float, model_arch: str) -> str:
    """
    Simulates Snowflake Cortex Complete (LLM inference e.g. mistral-large / llama3-70b)
    to synthesize an operational tactical radar debrief from classification & Grad-CAM outputs.
    """
    meta = TARGET_METADATA.get(pred_class, {})
    full_name = meta.get("full_name", pred_class)
    category = meta.get("category", "Military Vehicle")
    radar_sig = meta.get("radar_signature", "High metallic backscatter")
    dims = meta.get("length_width", "Unknown")
    weight = meta.get("weight", "Unknown")

    debrief = f"""### 🛡️ SNOWFLAKE CORTEX AI TACTICAL INTELLIGENCE DEBRIEF
**Model:** `SNOWFLAKE.CORTEX.COMPLETE('mistral-large-2407')`  
**Classification System:** `{model_arch}` on Snowflake SPCS GPU Runtime  
**Target Designation:** **{full_name} ({pred_class})**  
**Confidence Score:** **{confidence * 100:.2f}%**  

#### 1. Radar Signature Assessment
- **Physical Scatterers:** The neural attention map (Grad-CAM) confirms dominant microwave reflections conforming to {pred_class} geometry ({dims}, {weight}).
- **Scattering Mechanism:** {radar_sig}.
- **Shadow Profile:** Low-grazing angle shadow cavity aligns with expected turret and hull profile under standard depression conditions.

#### 2. Tactical Threat Level
- **Classification:** `{category.upper()}`
- **Operational Status:** HIGH PRIORITY COMBATANT
- **Threat Vector:** Primary threat to mechanized infantry and armor formations. Recommended engagement priority: **TIER-1**.

#### 3. Data Governance & Lineage
- **Storage:** Telemetry and raw I/Q samples recorded to `MSTAR_RADAR_DB.RAW.SENSOR_ICEBERG_TABLE` (Apache Iceberg format).
- **Audit:** Model prediction logged in Snowflake Model Registry (`MSTAR_ACONVNET_V2`, run_id: `spcs-job-88194`).
"""
    return debrief
