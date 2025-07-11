import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift
from scipy.signal import hilbert

# Helper functions

def freq_offset_compensation_block_ssb(tx_pilot_signal, demod_pilot_signal, Fs):
    """
    Frequency Offset Estimation and Correction based on test_cfo_pn_pilot.py
    """
    # Find the frequency of the peak in the transmitted pilot's FFT
    Tx_fft = fftshift(fft(tx_pilot_signal))
    N_fft_tx = len(tx_pilot_signal)
    f_fft_tx = np.linspace(-Fs/2, Fs/2 - Fs/N_fft_tx, N_fft_tx)
    max_idx_tx = np.argmax(np.abs(Tx_fft))
    Tx_max_freq = f_fft_tx[max_idx_tx]

    # Find the frequency of the peak in the demodulated pilot's FFT
    pilot_fft = fftshift(fft(demod_pilot_signal))
    N_fft_demod = len(demod_pilot_signal)
    f_fft_demod = np.linspace(-Fs/2, Fs/2 - Fs/N_fft_demod, N_fft_demod)
    max_idx_demod = np.argmax(np.abs(pilot_fft))
    Rx_max_freq = f_fft_demod[max_idx_demod]

    # Calculate the frequency offset
    Freq_offset =   Tx_max_freq - Rx_max_freq

    # Apply the correction
    L = len(demod_pilot_signal)
    t = np.arange(L) / Fs

    FO_compensated = demod_pilot_signal * np.exp(1j * 2 * np.pi * Freq_offset * t)
    
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

    data_i_dc_removed = data_i - 0*np.mean(data_i)
    data_q_dc_removed = data_q - 0*np.mean(data_q)
    complex_signal = data_i_dc_removed + 1j * data_q_dc_removed

    N = len(complex_signal)
    yf_shifted = fftshift(fft(complex_signal))
    xf_shifted = fftshift(fftfreq(N, 1 / Fs))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(xf_shifted, np.abs(yf_shifted))
    ax.set_title('Magnitude Spectrum of Combined I + jQ Signal (DC Removed)')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude')
    ax.grid(True)
    ax.set_xlim(-Fs/2, Fs/2)
    st.pyplot(fig)

def plot_scatter(data_i, data_q):
    """
    Plots a scatter plot of the I and Q data.
    """
    data_i_centered = data_i - 0*np.mean(data_i)
    data_q_centered = data_q - 0*np.mean(data_q)
    demod_pilot_current = data_i_centered + 1j * data_q_centered
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(np.real(demod_pilot_current), np.imag(demod_pilot_current), alpha=0.5)
    ax.set_xlabel('Real Part (I)')
    ax.set_ylabel('Imaginary Part (Q)')
    ax.set_title('Centered Scatter Plot of I and Q')
    ax.grid(True)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal', adjustable='box')
    st.pyplot(fig)

def plot_freq_offset_compensation(data_i, data_q, Fs, pilot_freq):
    """
    Performs frequency offset compensation and plots the results.
    """
    # Step 1: Remove DC offset from experimental data
    data_i_centered = data_i - 0*np.mean(data_i)
    data_q_centered = data_q - 0*np.mean(data_q)
    demod_pilot_current = data_i_centered + 1j * data_q_centered

    # Step 2: Generate an ideal reference pilot signal
    num_samples = len(demod_pilot_current)
    num_samples_per_pulse = Fs / pilot_freq
    resolution_time = 1 / Fs
    t = np.arange(0, resolution_time*num_samples, resolution_time).reshape(-1, 1)

    # --- Baseband I/Q Signal Generation ---
    i_signal = np.cos(2 * np.pi * pilot_freq * t)
    q_signal = np.imag(hilbert(i_signal, axis=0))
    phase_shift = np.pi / 2
    tx_pilot_signal = (i_signal + np.exp(1j * phase_shift) * q_signal).flatten()

    # Step 3: Perform Frequency Offset Compensation
    compensated_signal, offset_estimated = freq_offset_compensation_block_ssb(tx_pilot_signal, demod_pilot_current, Fs)

    st.write(f"Estimated Frequency Offset: {offset_estimated:.2f} Hz")

    # Step 4: Plot the corrected signal
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(np.real(compensated_signal), np.imag(compensated_signal), alpha=0.5, s=5)
    ax.set_xlabel('Real Part (I)')
    ax.set_ylabel('Imaginary Part (Q)')
    ax.set_title('After Frequency Offset Compensation')
    ax.grid(True)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal', adjustable='box')
    st.pyplot(fig)


# --- Streamlit UI ---
st.title("Signal Processing Application")

st.sidebar.header("Configuration")
uploaded_file_i = st.sidebar.file_uploader("Upload I data (CSV)", type="csv")
uploaded_file_q = st.sidebar.file_uploader("Upload Q data (CSV)", type="csv")

Fs = st.sidebar.number_input("Sampling Frequency (Fs)", value=1e9, step=1e6, format="%.2f")
pilot_freq = st.sidebar.number_input("Pilot Frequency (Hz)", value=200e6, step=1e6, format="%.2f")

analysis_type = st.sidebar.selectbox(
    "Choose Analysis",
    ["FFT", "Combine FFT", "Scatter Plot", "Frequency Offset Compensation"]
)

process_button = st.sidebar.button("Process")

# --- Main Logic ---
if process_button:
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


