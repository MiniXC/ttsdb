import streamlit as st
import time
import numpy as np

st.set_page_config(page_title="The Benchmarks", page_icon=":chart_with_upwards_trend:")

st.markdown(
    """
# The Benchmarks

This page contains information about the benchmarks used in the TTSDB (Text-to-Speech Distribution Benchmark).

## General
All the benchmarks in TTSDB are fundamentally distances between distributions.
At the moment, we always compare any given TTS dataset's distribution(s) to the distribution(s) of a reference dataset (LibriTTS).
The scores are calculated using the Wasserstein distance (for 1D distributions) and the Frechet distance (for nD distributions).
They are then rescaled to a range of 0 (identical to noise) to 100 (identical to the reference dataset).

## Categories

### Overall
These benchmarks are meant to capture the overall quality of the TTS system.
"""
)

with st.expander("Hubert"):
    st.markdown(
        """
#### Hubert
The Hubert benchmark uses the hidden states of a [pretrained Hubert model](https://huggingface.co/facebook/hubert-base-ls960).
"""
    )

with st.expander("MFCC"):
    st.markdown(
        """
#### MFCC
The Mel-Frequency Cepstral Coefficients (MFCC) benchmark uses the MFCCs of the audio files directly.
"""
    )

st.markdown(
    """
### Intelligibility
These benchmarks are meant to capture the intelligibility of the TTS system.
They compute the WER (Word Error Rate) for each audio file.
Then, the distances between the WER distributions of the TTS dataset and the reference dataset are calculated.
"""
)

with st.expander("Wav2Vec2 WER"):
    st.markdown(
        """
#### Wav2Vec2 WER
The Wav2Vec2 WER benchmark uses the WER of a pretrained Wav2Vec2 model.
"""
    )

with st.expander("Whisper"):
    st.markdown(
        """
#### Whisper
The Whisper benchmark uses the phone counts of a pretrained Whisper model.
"""
    )

st.markdown(
    """
### Phonetics
These benchmarks are meant to capture the phonetic quality of the TTS system.
"""
)

with st.expander("Allosaurus"):
    st.markdown(
        """
#### Allosaurus
This benchmark uses the phone counts of a pretrained Allosaurus model.
"""
    )

st.markdown(
    """
    ### Speakers
    These benchmarks are meant to capture the speaker quality of the TTS system.
    """
)

with st.expander("WeSpeaker"):
    st.markdown(
        """
    #### WeSpeaker
    This benchmark uses the speaker embeddings of a pretrained WeSpeaker model.
    """
    )

st.markdown(
    """
    ### Environment
    These benchmarks are meant to capture the environment quality of the TTS system.
    """
)

with st.expander("VoiceFixer"):
    st.markdown(
        """
    #### VoiceFixer
    This benchmark uses the mel spectrogram differences between the original and the restored audio.
    """
    )

with st.expander("WADA SNR"):
    st.markdown(
        """
    #### WADA SNR
    This benchmark uses the Signal-to-Noise Ratio (SNR) of the audio files.
    """
    )

st.markdown(
    """
    ### Prosody
    These benchmarks are meant to capture the prosody quality of the TTS system.
    """
)

with st.expander("MPM"):
    st.markdown(
        """
    #### MPM
    This benchmark uses representation of a Masked Prosody Model (MPM).
    """
    )

with st.expander("Pitch"):
    st.markdown(
        """
    #### Pitch
    This benchmark uses the pitch of the audio files.
    """
    )

st.sidebar.header("Benchmarks")
