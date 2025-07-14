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

def perform_phase_correction(freq_compensated_signal, tx_pilot_signal, Fs, pilot_freq, qc_symbol_rate, sample_position=3):
    """
    Performs phase correction based on the MATLAB script.
    Estimates the average phase error from the first symbol and applies it to the whole signal.
    """
    num_samples_per_pulse = Fs / pilot_freq
    if num_samples_per_pulse < 1:
        st.error("Fs/pilot_freq must be >= 1. Cannot have less than 1 sample per pulse.")
        return None, None

    no_pulses_per_symbol = pilot_freq / qc_symbol_rate
    if no_pulses_per_symbol < 1:
        st.error("pilot_freq / qc_symbol_rate must be >= 1. Cannot have less than one pulse per symbol.")
        return None, None

    segment_length = int(no_pulses_per_symbol * num_samples_per_pulse)
    if len(freq_compensated_signal) < segment_length:
        st.error(f"Signal is too short for phase estimation. Needs at least {segment_length} samples.")
        return None, None

    # Use the first symbol to estimate the phase error
    rx_pilot_segment = freq_compensated_signal[:segment_length]
    tx_pilot_segment = tx_pilot_signal[:segment_length]

    # Downsample based on the optimal sample position (from MATLAB)
    # MATLAB is 1-indexed, Python is 0-indexed
    down_sample_rx = rx_pilot_segment[sample_position - 1::int(num_samples_per_pulse)]
    down_sample_tx = tx_pilot_segment[sample_position - 1::int(num_samples_per_pulse)]

    if len(down_sample_rx) == 0 or len(down_sample_tx) == 0:
        st.error("Downsampling resulted in an empty array. Check your parameters (Fs, pilot_freq).")
        return None, None

    # Calculate the average phase error
    ref_phase = np.angle(down_sample_tx)
    measured_phase = np.angle(down_sample_rx)
    
    # The phase error is the difference between the reference and the measured phase
    phase_error_vector = ref_phase - measured_phase
    
    # Unwrap to handle phase jumps greater than pi and calculate the mean
    avg_phase_error = np.mean(np.unwrap(phase_error_vector))

    # Apply the correction (negative of the estimated error) to the entire signal
    phase_corrected_signal = freq_compensated_signal * np.exp(-1j * avg_phase_error)

    return phase_corrected_signal, avg_phase_error

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
    ax.set_aspect('equal', adjustable='box')
    st.pyplot(fig)

def plot_frequency_correction(data_i, data_q, Fs, pilot_freq):
    """
    Performs frequency offset compensation and plots the result.
    """
    st.subheader("Frequency Correction")
    # Step 1: Center data and create complex signal
    data_i_centered = data_i - np.mean(data_i)
    data_q_centered = data_q - np.mean(data_q)
    demod_pilot_current = data_i_centered + 1j * data_q_centered

    # Step 2: Generate an ideal reference pilot signal
    num_samples = len(demod_pilot_current)
    t = np.arange(num_samples) / Fs
    tx_pilot_signal = np.exp(1j * 2 * np.pi * pilot_freq * t)

    # Step 3: Perform Frequency Offset Compensation
    freq_compensated_signal, offset_estimated = freq_offset_compensation_block_ssb(tx_pilot_signal, demod_pilot_current, Fs)
    st.write(f"**Estimated Frequency Offset:** {offset_estimated:.2f} Hz")

    # Step 4: Plot the results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Plot 1: Before Correction
    ax1.scatter(np.real(demod_pilot_current), np.imag(demod_pilot_current), alpha=0.5, s=5)
    ax1.set_title('1. Before Correction (DC Centered)')
    ax1.set_xlabel('I'); ax1.set_ylabel('Q'); ax1.grid(True); ax1.set_aspect('equal', adjustable='box')

    # Plot 2: After Frequency Correction
    ax2.scatter(np.real(freq_compensated_signal), np.imag(freq_compensated_signal), alpha=0.5, s=5)
    ax2.set_title('2. After Frequency Correction')
    ax2.set_xlabel('I'); ax2.set_ylabel('Q'); ax2.grid(True); ax2.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    st.pyplot(fig)

def plot_phase_correction(data_i, data_q, Fs, pilot_freq, qc_symbol_rate):
    """
    Performs phase correction on the pilot signal based on the MATLAB script's logic.
    It processes the signal in chunks (Qc Symbols) and applies a specific
    phase correction to each chunk.
    """
    st.subheader("Pilot Signal Phase Correction")
    st.info("""
    This process corrects the phase noise on the **pilot signal only**.
    It works by:
    1.  First applying frequency correction to the entire signal.
    2.  Downsampling to isolate one sample per pilot pulse.
    3.  Grouping these pilot samples into chunks (Qc Symbols).
    4.  Calculating the average phase error for each chunk by comparing it to an ideal reference.
    5.  Applying a specific phase correction to each chunk of pilot samples.
    """)
    st.warning("""
    **Note on Phase Noise Calculation**: You referenced `Phase_noise_cal.py`. That script is for *simulating* phase noise.
    This application works with experimental data, so it *estimates* the real phase noise by comparing the received pilot's phase
    to an ideal reference. This is the correct approach for processing real-world signals.
    """)

    # --- Step 1: Initial Processing (Freq. Correction is a prerequisite) ---
    data_i_centered = data_i - np.mean(data_i)
    data_q_centered = data_q - np.mean(data_q)
    demod_pilot_current = data_i_centered + 1j * data_q_centered

    num_samples = len(demod_pilot_current)
    t = np.arange(num_samples) / Fs
    tx_pilot_signal = np.exp(1j * 2 * np.pi * pilot_freq * t)

    freq_compensated_signal, offset = freq_offset_compensation_block_ssb(tx_pilot_signal, demod_pilot_current, Fs)
    st.write(f"**Note:** A frequency offset of {offset:.2f} Hz was corrected first, as phase estimation requires a frequency-locked signal.")

    # --- Step 2: Setup parameters to isolate pilot samples ---
    num_samples_per_pulse = int(Fs / pilot_freq)
    no_pulses_per_symbol = int(pilot_freq / qc_symbol_rate)
    sample_position = 3  # Using the fixed sample position from the script (MATLAB is 1-indexed, so 3 -> Python index 2)

    st.write(f"- Samples per pilot pulse: `{num_samples_per_pulse}`")
    st.write(f"- Pilot pulses per Qc Symbol: `{no_pulses_per_symbol}`")
    st.write(f"- Using sample position: `{sample_position}` (Python index `{sample_position-1}`)")

    # --- Step 3: Downsample the signals to get only the pilot points---
    rx_pilot_points = freq_compensated_signal[sample_position - 1::num_samples_per_pulse]
    tx_pilot_points = tx_pilot_signal[sample_position - 1::num_samples_per_pulse]

    # --- Step 4: Perform chunk-based phase correction on the pilot points ---
    num_symbols = len(rx_pilot_points) // no_pulses_per_symbol
    corrected_pilot_points = np.zeros(num_symbols * no_pulses_per_symbol, dtype=np.complex128)
    estimated_phase_errors_rad = [] # To store the error for each chunk

    if num_symbols == 0:
        st.error("Signal is too short to form even one complete Qc symbol with the given parameters.")
        return

    for i in range(num_symbols):
        start = i * no_pulses_per_symbol
        end = start + no_pulses_per_symbol

        rx_chunk = rx_pilot_points[start:end]
        tx_chunk = tx_pilot_points[start:end]

        # Calculate the average phase error (the estimated phase noise) for this specific chunk
        estimated_phase_noise = np.mean(np.unwrap(np.angle(tx_chunk) - np.angle(rx_chunk)))
        estimated_phase_errors_rad.append(estimated_phase_noise)

        # Apply the correction to the chunk by rotating it by the estimated phase noise
        corrected_chunk = rx_chunk * np.exp(1j * estimated_phase_noise)
        corrected_pilot_points[start:end] = corrected_chunk

    st.success(f"Phase correction applied to {num_symbols} Qc symbol(s) of pilot data.")

    # --- Step 5: Display the overall calculated phase error for verification ---
    st.subheader("Verification: Overall Phase Estimation")
    overall_avg_phase_error = np.mean(estimated_phase_errors_rad)
    st.metric(label="Overall Average Phase Error", value=f"{overall_avg_phase_error:.4f} radians")

    # --- Step 6: Plot the phase error drift over time ---
    fig_drift, ax_drift = plt.subplots(figsize=(12, 6))
    ax_drift.plot(np.rad2deg(estimated_phase_errors_rad), marker='.', linestyle='-')
    ax_drift.set_title('Estimated Phase Error per Qc Symbol (Phase Drift)')
    ax_drift.set_xlabel('Qc Symbol Chunk Number')
    ax_drift.set_ylabel('Phase Error (degrees)')
    ax_drift.grid(True)
    st.pyplot(fig_drift)

    # --- Step 7: Plot the final downsampled constellations ---
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot Ideal TX Pilot (Blue Stars)
    ax.scatter(np.real(tx_pilot_points[:len(corrected_pilot_points)]), np.imag(tx_pilot_points[:len(corrected_pilot_points)]), 
               c='blue', marker='*', s=200, label='Ideal TX Pilot Points', zorder=3)
    
    # Plot Received RX Pilot (Red Circles)
    ax.scatter(np.real(rx_pilot_points[:len(corrected_pilot_points)]), np.imag(rx_pilot_points[:len(corrected_pilot_points)]), 
               c='red', marker='o', alpha=0.6, label='Received RX Pilot Points (Freq. Corrected)', zorder=2)
    
    # Plot Phase Corrected Signal (Green Plus)
    ax.scatter(np.real(corrected_pilot_points), np.imag(corrected_pilot_points), 
               c='green', marker='+', s=150, alpha=0.9, label='Phase Corrected Pilot Points', zorder=4)
    
    ax.set_title('Pilot Point Constellation after Phase Correction')
    ax.set_xlabel('I'); ax.set_ylabel('Q')
    ax.grid(True); ax.set_aspect('equal', adjustable='box')
    ax.legend()
    st.pyplot(fig)


# --- Streamlit UI ---
st.title("Optical Signal Correction")

st.sidebar.header("Configuration")
uploaded_file_i = st.sidebar.file_uploader("Upload Q data (CSV)", type="csv")
uploaded_file_q = st.sidebar.file_uploader("Upload I data (CSV)", type="csv")

st.sidebar.subheader("Signal Parameters")
Fs = st.sidebar.number_input("Sampling Frequency (Fs)", value=2.5e9, step=1e6, format="%.2e")
pilot_freq = st.sidebar.number_input("Pilot Frequency (Hz)", value=500e6, step=1e6, format="%.2e")
qc_symbol_rate = st.sidebar.number_input("Qc Symbol Rate (Hz)", value=100e6, step=1e6, format="%.2e")


analysis_type = st.sidebar.selectbox(
    "Choose Analysis",
    ["FFT", "Combine FFT", "Scatter Plot", "Frequency Correction", "Phase Correction"]
)

process_button = st.sidebar.button("Process")

# --- Main Logic ---
if process_button:
    if uploaded_file_i is not None and uploaded_file_q is not None:
        try:
            # Reading data now assumes a specific format from Tektronix scopes
            data_i = pd.read_csv(uploaded_file_i, header=None, skiprows=4).iloc[:, 4].to_numpy(dtype=float)
            data_q = pd.read_csv(uploaded_file_q, header=None, skiprows=4).iloc[:, 4].to_numpy(dtype=float)

            st.header(analysis_type)

            if analysis_type == "FFT":
                plot_fft(data_i, data_q, Fs)
            elif analysis_type == "Combine FFT":
                plot_combined_fft(data_i, data_q, Fs)
            elif analysis_type == "Scatter Plot":
                plot_scatter(data_i, data_q)
            elif analysis_type == "Frequency Correction":
                plot_frequency_correction(data_i, data_q, Fs, pilot_freq)
            elif analysis_type == "Phase Correction":
                plot_phase_correction(data_i, data_q, Fs, pilot_freq, qc_symbol_rate)

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.error("Please ensure the CSV files are in the correct format (e.g., from a Tektronix oscilloscope, with data in the 5th column and 4 header rows).")
    else:
        st.warning("Please upload both I and Q data files to begin.")
