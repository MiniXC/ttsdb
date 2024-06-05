import streamlit as st

st.set_page_config(
    page_title="TTSDB (Text-to-Speech Distribution Benchmark)",
    page_icon=":microphone:",
)

st.write("# TTSDB (Text-to-Speech Distribution Benchmark)")

st.markdown(
    """
    Advanced TTS systems trained on extensive datasets have become publicly accessible, generating diverse voices. Unlike other fields, TTS has lacked comprehensive benchmarks for comparison. Traditional evaluations based on human-judged "naturalness" reached parity with real speech and often yielded inconsistent results. The MOS metric became outdated, as many state-of-the-art TTS systems stopped reporting it. Without a definitive metric, advancements were uncertain. Crowdsourced A/B preference tests were used but proved difficult to maintain and inconsistent due to varying user participation.

    The TTSDB (Text-to-Speech Distribution Benchmark), a multifaceted evaluation approach for TTS inspired by metrics in computer vision and NLP. Speech has several crucial subdomains, such as speaker realization, duration modeling, intelligibility, and prosody. TTSDB evaluates the distance between probability distributions of these subdomains, providing a robust benchmark for TTS systems.
"""
)
