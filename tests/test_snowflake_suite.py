"""
Tests Snowflake Cortex integration: Cortex Analyst, Cortex Search, Cortex Debrief,
telemetry generation, and Semantic Model YAML validation.
"""

import os
import pandas as pd
import yaml

from mstar_atr.snowflake.cortex import (
    get_sample_radar_telemetry_df,
    simulate_cortex_analyst_query,
    simulate_cortex_search,
    generate_cortex_tactical_debrief,
)


def test_snowflake_suite():
    print("=" * 60)
    print("Testing Snowflake Data Cloud & Cortex AI Integration")
    print("=" * 60)

    # 1. Test Telemetry DataFrame
    print("\n[1/5] Generating Snowflake Radar Telemetry Data...")
    df = get_sample_radar_telemetry_df(n_samples=50)
    assert len(df) == 50
    assert "DETECTION_ID" in df.columns
    assert "TARGET_CATEGORY" in df.columns
    assert "RCS_DB" in df.columns
    assert "SECTOR" in df.columns
    print(f"  --> Generated {len(df)} radar detection rows with columns: {list(df.columns[:6])}...")

    # 2. Test Cortex Analyst Text-to-SQL
    print("\n[2/5] Testing Snowflake Cortex Analyst (Text-to-SQL)...")
    questions = [
        "Which main battle tanks were detected with confidence > 90%?",
        "What is the count of detected vehicles and average confidence by sector?",
        "Show target classification accuracy across depression angles",
        "Show high confidence detections with RCS > 20 dB",
    ]
    for q in questions:
        res = simulate_cortex_analyst_query(q, df)
        assert "sql" in res and "SELECT" in res["sql"]
        assert "explanation" in res
        assert "results" in res and isinstance(res["results"], pd.DataFrame)
        print(f"  --> Query: '{q}'\n      SQL generated: {res['sql'].splitlines()[0]}... ({len(res['results'])} rows)")

    # 3. Test Cortex Search (RAG)
    print("\n[3/5] Testing Snowflake Cortex Search (Hybrid Vector + Keyword)...")
    queries = ["T-72 turret dihedral", "BMP-2 autocannon", "ZSU-23-4 Shilka", "Bulldozer blade"]
    for q in queries:
        docs = simulate_cortex_search(q)
        assert len(docs) > 0
        assert "doc_id" in docs[0] and "score" in docs[0]
        print(f"  --> Search query: '{q}' matched Top Doc: {docs[0]['doc_id']} ({docs[0]['title']}) with score {docs[0]['score']}")

    # 4. Test Cortex Complete LLM Tactical Debrief
    print("\n[4/5] Testing Snowflake Cortex Complete LLM Tactical Debrief...")
    debrief = generate_cortex_tactical_debrief("T72", 0.992, "A-ConvNet (SAR SOTA)")
    assert "SNOWFLAKE CORTEX AI TACTICAL INTELLIGENCE DEBRIEF" in debrief
    assert "T72" in debrief
    print("  --> Cortex debrief generated successfully.")

    # 5. Test Semantic Model YAML
    print("\n[5/5] Validating Cortex Analyst Semantic Model YAML Specification...")
    yaml_file = os.path.join(os.path.dirname(__file__), "..", "mstar_atr", "snowflake", "mstar_semantic_model.yaml")
    assert os.path.isfile(yaml_file), f"Semantic model file not found at {yaml_file}"
    with open(yaml_file, "r", encoding="utf-8") as f:
        parsed_yaml = yaml.safe_load(f)
    assert parsed_yaml["name"] == "mstar_sar_atr_intelligence"
    assert "tables" in parsed_yaml
    table = parsed_yaml["tables"][0]
    assert table["name"] == "SAR_TARGET_DETECTIONS"
    assert len(table["dimensions"]) >= 6
    assert len(table["measures"]) >= 5
    assert len(parsed_yaml["verified_queries"]) >= 2
    print(f"  --> Semantic Model '{parsed_yaml['name']}' verified with {len(table['dimensions'])} dimensions & {len(table['measures'])} measures.")

    print("\n" + "=" * 60)
    print("ALL SNOWFLAKE INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    test_snowflake_suite()
