"""
Snowflake Data Cloud & Cortex AI Integration for MSTAR SAR ATR.
Supports Cortex Analyst (Semantic Models), Cortex Search (RAG), Cortex LLMs,
Model Registry, Feature Store, and Streamlit in Snowflake (SiS) with SPCS Container Runtime.
"""

from .cortex import (
    simulate_cortex_analyst_query,
    simulate_cortex_search,
    generate_cortex_tactical_debrief,
    get_sample_radar_telemetry_df,
)

__all__ = [
    "simulate_cortex_analyst_query",
    "simulate_cortex_search",
    "generate_cortex_tactical_debrief",
    "get_sample_radar_telemetry_df",
]
