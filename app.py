
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import hilbert

# Helper functions from your scripts

def freq_offset_compensation_block_ssb(tx_pilot_signal, demod_pilot_signal, Fs):
    """
    Frequency Offset Estimation and Correction
    """
    Tx_fft = np.fft.fftshift(np.fft.fft(tx_pilot_signal))
    N_fft = len(tx_pilot_signal)
    f_fft = np.linspace(-Fs/2, Fs/2 - Fs/N_fft, N_fft)
    max_idx = np.argmax(np.abs(Tx_fft))
    Tx_max_freq = f_fft[max_idx]

    pilot_fft = np.fft.fftshift(np.fft.fft(demod_pilot_signal))
    N_fft = len(demod_pilot_signal)
    f_fft = np.linspace(-Fs/2, Fs/2 - Fs/N_fft, N_fft)
    max_idx = np.argmax(np.abs(pilot_fft))
    Rx_max_freq = f_fft[max_idx]
    Freq_offset = Tx_max_freq - Rx_max_freq

    L = len(demod_pilot_signal)
    FO_estimate_Hz = Freq_offset
    Ts = 1 / Fs
    t = np.arange(0, L) * Ts

    FO_compensated = demod_pilot_signal * np.exp(1j * 2 * np.pi * FO_estimate_Hz * t)
    return FO_compensated, Freq_offset

def plot_fft(data_i, data_q, Fs):
    """
    Plots the FFT of I and Q signals.
    """
    N_q = len(data_q)
    N_i = len(data_i)

    yf_q = fft(data_q)
    yf_i = fft(data_i)

    xf_q = fftfreq(N_q, 1 / Fs)
    xf_i = fftfreq(N_i, 1 / Fs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    N_q_half = N_q // 2
    xf_q_half = xf_q[1:N_q_half]
    yf_q_half = 2.0/N_q * np.abs(yf_q[1:N_q_half])
    ax1.plot(xf_q_half, yf_q_half)
    ax1.set_title('Magnitude Spectrum of Q')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Magnitude')
    ax1.grid(True)
    ax1.set_xlim(0, Fs/2)

    N_i_half = N_i // 2
    xf_i_half = xf_i[1:N_i_half]
    yf_i_half = 2.0/N_i * np.abs(yf_i[1:N_i_half])
    ax2.plot(xf_i_half, yf_i_half)
    ax2.set_title('Magnitude Spectrum of I')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Magnitude')
    ax2.grid(True)
    ax2.set_xlim(0, Fs/2)

    plt.tight_layout()
    st.pyplot(fig)

def plot_combined_fft(data_i, data_q, Fs):
    """
    Plots the FFT of the combined I and Q signals.
    """
    min_len = min(len(data_i), len(data_q))
    data_i = data_i[:min_len]
    data_q = data_q[:min_len]
    complex_signal = data_i + 1j * data_q

    N = len(complex_signal)
    yf = fft(complex_signal)
    xf = fftfreq(N, 1 / Fs)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(xf, np.abs(yf))
    ax.set_title('Magnitude Spectrum of Combined I + jQ Signal')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude')
    ax.grid(True)
    ax.set_xlim(-Fs/2, Fs/2)
    st.pyplot(fig)

def plot_scatter(data_i, data_q):
    """
    Plots a scatter plot of the I and Q data.
    """
    demod_pilot_current = data_i + 1j * data_q
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(np.real(demod_pilot_current), np.imag(demod_pilot_current), alpha=0.7, s=5)
    ax.set_xlabel('Real Part')
    ax.set_ylabel('Imaginary Part')
    ax.set_title('Demodulated Pilot Current')
    ax.grid(True)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_aspect('equal', adjustable='box')
    st.pyplot(fig)

def plot_freq_offset_compensation(data_i, data_q, Fs, pilot_freq):
    """
    Performs frequency offset compensation and plots the results.
    """
    demod_pilot_current = data_i + 1j * data_q

    Duration = len(demod_pilot_current) / Fs
    Resolution_Time = 1 / Fs
    t = np.arange(0, Duration, Resolution_Time)

    if len(t) > len(demod_pilot_current):
        t = t[:len(demod_pilot_current)]
    elif len(t) < len(demod_pilot_current):
        t = np.arange(0, len(demod_pilot_current)) * Resolution_Time

    I_signal = np.cos(2 * np.pi * pilot_freq * t)
    Q_signal = np.imag(hilbert(I_signal))
    phase_shift = np.pi / 2
    tx_pilot_signal = I_signal + np.exp(1j * phase_shift) * Q_signal

    if len(tx_pilot_signal) != len(demod_pilot_current):
        min_len_signals = min(len(tx_pilot_signal), len(demod_pilot_current))
        tx_pilot_signal = tx_pilot_signal[:min_len_signals]
        demod_pilot_current = demod_pilot_current[:min_len_signals]

    Freq_Offset_comp_pilot_current, Freq_offset_estimated = freq_offset_compensation_block_ssb(tx_pilot_signal, demod_pilot_current, Fs)

    st.write(f"Estimated Frequency Offset: {Freq_offset_estimated:.2f} Hz")

    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.scatter(np.real(demod_pilot_current), np.imag(demod_pilot_current), alpha=0.7, s=5)
    ax1.set_xlabel('Real Part')
    ax1.set_ylabel('Imaginary Part')
    ax1.set_title('Before Frequency Offset Compensation')
    ax1.grid(True)
    ax1.axhline(0, color='gray', linewidth=0.5)
    ax1.axvline(0, color='gray', linewidth=0.5)
    ax1.axis('equal')
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.scatter(np.real(Freq_Offset_comp_pilot_current), np.imag(Freq_Offset_comp_pilot_current), alpha=0.7, s=5)
    ax2.set_xlabel('Real Part')
    ax2.set_ylabel('Imaginary Part')
    ax2.set_title('After Frequency Offset Compensation')
    ax2.grid(True)
    ax2.axhline(0, color='gray', linewidth=0.5)
    ax2.axvline(0, color='gray', linewidth=0.5)
    ax2.axis('equal')
    st.pyplot(fig2)


st.title("Signal Processing Application")

st.sidebar.header("Configuration")
uploaded_file_i = st.sidebar.file_uploader("Upload I data (CSV)", type="csv")
uploaded_file_q = st.sidebar.file_uploader("Upload Q data (CSV)", type="csv")

Fs = st.sidebar.number_input("Sampling Frequency (Fs)", value=2.5e9, format="%.1e")
pilot_freq = st.sidebar.number_input("Pilot Frequency (for Freq Offset)", value=500e6, format="%.1e")

analysis_type = st.sidebar.selectbox(
    "Choose Analysis",
    ["FFT", "Combine FFT", "Scatter Plot", "Frequency Offset Compensation"]
)

if uploaded_file_i is not None and uploaded_file_q is not None:
    try:
        data_i = pd.read_csv(uploaded_file_i, header=None, skiprows=4).iloc[:, 4].to_numpy()
        data_q = pd.read_csv(uploaded_file_q, header=None, skiprows=4).iloc[:, 4].to_numpy()

        st.header(analysis_type)

        if analysis_type == "FFT":
            plot_fft(data_i, data_q, Fs)
        elif analysis_type == "Combine FFT":
            plot_combined_fft(data_i, data_q, Fs)
        elif analysis_type == "Scatter Plot":
            plot_scatter(data_i, data_q)
        elif analysis_type == "Frequency Offset Compensation":
            plot_freq_offset_compensation(data_i, data_q, Fs, pilot_freq)

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload both I and Q data files to begin.")
