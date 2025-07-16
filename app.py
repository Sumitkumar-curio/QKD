import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift
from scipy.signal import hilbert

# Helper functions

def freq_offset_compensation_block_ssb(tx_pilot_signal, demod_pilot_signal, Fs):
    """
    Frequency Offset Estimation and Correction.
    Estimates offset by comparing peak frequencies in FFT of TX reference and RX signal.
    Applies correction to the entire RX signal.
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
    Freq_offset = Tx_max_freq - Rx_max_freq

    # Apply the correction to the entire signal
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

    complex_signal = data_i + 1j * data_q

    N = len(complex_signal)
    yf_shifted = fftshift(fft(complex_signal))
    xf_shifted = fftshift(fftfreq(N, 1 / Fs))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(xf_shifted, np.abs(yf_shifted))
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
    ax.scatter(np.real(demod_pilot_current), np.imag(demod_pilot_current), alpha=0.5)
    ax.set_xlabel('Real Part (I)')
    ax.set_ylabel('Imaginary Part (Q)')
    ax.set_title('Scatter Plot of I and Q')
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')
    st.pyplot(fig)

def plot_frequency_correction(data_i, data_q, Fs, pilot_freq):
    """
    Performs frequency offset compensation on the pilot and plots the result.
    In CVQKD, this correction is applied to the entire signal (pilot + quantum).
    """
    st.subheader("Frequency Correction")
    # Create complex signal (no DC removal, as quantum signal may be centered)
    demod_pilot_current = data_i + 1j * data_q

    # Generate an ideal reference pilot signal
    num_samples = len(demod_pilot_current)
    t = np.arange(num_samples) / Fs
    I_signal = np.cos(2 * np.pi * pilot_freq * t)
    Q_signal = np.imag(hilbert(I_signal))
    tx_pilot_signal = I_signal + np.exp(1j * np.pi / 2) * Q_signal

    # Perform Frequency Offset Compensation
    freq_compensated_signal, offset_estimated = freq_offset_compensation_block_ssb(tx_pilot_signal, demod_pilot_current, Fs)
    st.write(f"**Estimated Frequency Offset:** {offset_estimated:.2f} Hz")

    # Plot the results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Plot 1: Before Correction
    ax1.scatter(np.real(demod_pilot_current), np.imag(demod_pilot_current), alpha=0.5, s=5)
    ax1.set_title('1. Before Correction')
    ax1.set_xlabel('I'); ax1.set_ylabel('Q'); ax1.grid(True); ax1.set_aspect('equal', adjustable='box')

    # Plot 2: After Frequency Correction
    ax2.scatter(np.real(freq_compensated_signal), np.imag(freq_compensated_signal), alpha=0.5, s=5)
    ax2.set_title('2. After Frequency Correction')
    ax2.set_xlabel('I'); ax2.set_ylabel('Q'); ax2.grid(True); ax2.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    st.pyplot(fig)

def plot_phase_correction(data_i, data_q, Fs, pilot_freq, qc_symbol_rate, sample_position):
    """
    Performs phase correction on the pilot signal, aligned with MATLAB logic.
    Uses point-by-point phase error estimation without buggy sign flip.
    Correction factor = ref_phase - measured_phase, added to measured to align to ref.
    """
    st.subheader("Pilot Signal Phase Correction")
    st.info("""
    This process corrects the phase noise on the **pilot signal only** for visualization.
    It works by:
    1. First applying frequency correction to the entire signal.
    2. Downsampling to isolate one sample per pilot pulse.
    3. Grouping these pilot samples into chunks (Qc Symbols).
    4. Calculating the phase error for each pilot point in the chunk by comparing to an ideal reference.
    5. Applying point-by-point phase correction to each pilot point.
    For CVQKD, use 'Full Signal Correction' to apply to quantum signal.
    Note: No sign flip on error; always add (ref - measured) to align phases.
    """)

    # Step 1: Initial Processing
    demod_pilot_current = data_i + 1j * data_q

    num_samples = len(demod_pilot_current)
    t = np.arange(num_samples) / Fs
    I_signal = np.cos(2 * np.pi * pilot_freq * t)
    Q_signal = np.imag(hilbert(I_signal))
    tx_pilot_signal = I_signal + np.exp(1j * np.pi / 2) * Q_signal

    freq_compensated_signal, offset = freq_offset_compensation_block_ssb(tx_pilot_signal, demod_pilot_current, Fs)
    st.write(f"**Note:** A frequency offset of {offset:.2f} Hz was corrected first, as phase estimation requires a frequency-locked signal.")

    # Step 2: Setup parameters
    num_samples_per_pulse = int(Fs / pilot_freq)
    if num_samples_per_pulse < 1:
        st.error("Fs/pilot_freq must be >= 1. Cannot have less than 1 sample per pulse.")
        return

    no_pulses_per_symbol = int(pilot_freq / qc_symbol_rate)
    if no_pulses_per_symbol < 1:
        st.error("pilot_freq / qc_symbol_rate must be >= 1. Cannot have less than one pulse per symbol.")
        return

    st.write(f"- Samples per pilot pulse: {num_samples_per_pulse}")
    st.write(f"- Pilot pulses per Qc Symbol: {no_pulses_per_symbol}")
    st.write(f"- Using sample position: {sample_position} (Python index {sample_position-1})")

    # Step 3: Downsample
    rx_pilot_points = freq_compensated_signal[sample_position - 1::num_samples_per_pulse]
    tx_pilot_points = tx_pilot_signal[sample_position - 1::num_samples_per_pulse]

    if len(rx_pilot_points) == 0:
        st.error("Downsampling resulted in an empty array. Check signal length and parameters.")
        return

    # Step 4: Point-by-point phase correction per chunk
    num_symbols = len(rx_pilot_points) // no_pulses_per_symbol
    if num_symbols == 0:
        st.error("Signal is too short to form even one complete Qc symbol with the given parameters.")
        return

    corrected_pilot_points = np.zeros(num_symbols * no_pulses_per_symbol, dtype=np.complex128)
    
    for i in range(num_symbols):
        start = i * no_pulses_per_symbol
        end = start + no_pulses_per_symbol

        rx_chunk = rx_pilot_points[start:end]
        tx_chunk = tx_pilot_points[start:end]

        # Point-by-point phase error
        ref_phase = np.angle(tx_chunk)
        measured_phase = np.angle(rx_chunk)
        est_phase_noise = ref_phase - measured_phase  # No unwrap needed if differences small; otherwise unwrap if jumps
        est_phase_noise = np.unwrap(est_phase_noise) if np.any(np.abs(est_phase_noise) > np.pi) else est_phase_noise

        # Correction: add est to measured to get ref
        corrected_theta = measured_phase + est_phase_noise
        corrected_chunk = np.abs(rx_chunk) * np.exp(1j * corrected_theta)
        corrected_pilot_points[start:end] = corrected_chunk

    st.success(f"Phase correction applied to {num_symbols} Qc symbol(s) of pilot data.")

    # Compute and display metrics using circular statistics
    def circular_mean(phases):
        return np.angle(np.mean(np.exp(1j * phases)))

    def circular_std(phases):
        mean = circular_mean(phases)
        return np.sqrt(-2 * np.log(np.mean(np.abs(np.exp(1j * (phases - mean))))))

    rx_phases = np.angle(rx_pilot_points[:len(corrected_pilot_points)])
    corr_phases = np.angle(corrected_pilot_points)
    tx_phases = np.angle(tx_pilot_points[:len(corrected_pilot_points)])

    mean_rx_phase = circular_mean(rx_phases)
    mean_corr_phase = circular_mean(corr_phases)
    mean_tx_phase = circular_mean(tx_phases)

    std_rx_phase = circular_std(rx_phases)
    std_corr_phase = circular_std(corr_phases)

    st.write(f"**Mean Phase (rad):** TX: {mean_tx_phase:.4f}, RX: {mean_rx_phase:.4f}, Corrected: {mean_corr_phase:.4f}")
    st.write(f"**Phase Std Dev (rad):** RX: {std_rx_phase:.4f}, Corrected: {std_corr_phase:.4f}")

    # Step 5: Plot normalized to unit circle
    tx_norm = np.exp(1j * np.angle(tx_pilot_points[:len(corrected_pilot_points)]))
    rx_norm = np.exp(1j * np.angle(rx_pilot_points[:len(corrected_pilot_points)]))
    corr_norm = np.exp(1j * np.angle(corrected_pilot_points))

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.scatter(np.real(tx_norm), np.imag(tx_norm), 
               c='blue', marker='*', s=200, label='Ideal TX Pilot Points (Norm)', zorder=3)
    
    ax.scatter(np.real(rx_norm), np.imag(rx_norm), 
               c='red', marker='o', alpha=0.6, label='Received RX Pilot Points (Norm)', zorder=2)
    
    ax.scatter(np.real(corr_norm), np.imag(corr_norm), 
               c='green', marker='+', s=150, alpha=0.9, label='Phase Corrected Pilot Points (Norm)', zorder=4)
    
    ax.set_title('Pilot Point Constellation after Phase Correction (Normalized to Unit Circle)')
    ax.set_xlabel('I'); ax.set_ylabel('Q')
    ax.grid(True); ax.set_aspect('equal', adjustable='box')
    ax.legend()
    st.pyplot(fig)

def plot_full_correction(data_i, data_q, Fs, pilot_freq, qc_symbol_rate, sample_position):
    """
    Performs full frequency and phase correction on the entire signal (pilot + quantum).
    Phase is estimated point-by-point from pilot and applied (averaged per block) to full signal blocks.
    """
    st.subheader("Full Signal Correction for CVQKD")
    st.info("""
    In CVQKD, the pilot tone is used to estimate frequency offset and phase noise.
    Corrections are applied to the entire received signal:
    1. Frequency offset correction to the whole signal.
    2. Point-by-point phase estimation from downsampled pilot per Qc symbol.
    3. Apply average phase correction per symbol block to the corresponding full signal chunk.
    This aligns the quadrature frame for quantum key extraction.
    """)

    # Step 1: Create complex signal
    demod_current = data_i + 1j * data_q

    num_samples = len(demod_current)
    t = np.arange(num_samples) / Fs
    I_signal = np.cos(2 * np.pi * pilot_freq * t)
    Q_signal = np.imag(hilbert(I_signal))
    tx_pilot_signal = I_signal + np.exp(1j * np.pi / 2) * Q_signal

    # Step 2: Frequency correction
    freq_compensated_signal, offset = freq_offset_compensation_block_ssb(tx_pilot_signal, demod_current, Fs)
    st.write(f"**Estimated Frequency Offset:** {offset:.2f} Hz")

    # Step 3: Setup parameters
    num_samples_per_pulse = int(Fs / pilot_freq)
    if num_samples_per_pulse < 1:
        st.error("Fs/pilot_freq must be >= 1.")
        return

    no_pulses_per_symbol = int(pilot_freq / qc_symbol_rate)
    if no_pulses_per_symbol < 1:
        st.error("pilot_freq / qc_symbol_rate must be >= 1.")
        return

    # Step 4: Downsample for phase estimation
    rx_pilot_points = freq_compensated_signal[sample_position - 1::num_samples_per_pulse]
    tx_pilot_points = tx_pilot_signal[sample_position - 1::num_samples_per_pulse]

    if len(rx_pilot_points) == 0:
        st.error("Downsampling empty. Check parameters.")
        return

    # Step 5: Estimate phase per symbol (average of point-by-point for block)
    num_symbols = len(rx_pilot_points) // no_pulses_per_symbol
    if num_symbols == 0:
        st.error("Signal too short for one Qc symbol.")
        return

    estimated_phases = []
    for i in range(num_symbols):
        start = i * no_pulses_per_symbol
        end = start + no_pulses_per_symbol

        rx_chunk = rx_pilot_points[start:end]
        tx_chunk = tx_pilot_points[start:end]

        ref_phase = np.angle(tx_chunk)
        measured_phase = np.angle(rx_chunk)
        est_phase_noise = ref_phase - measured_phase
        est_phase_noise = np.unwrap(est_phase_noise) if np.any(np.abs(est_phase_noise) > np.pi) else est_phase_noise
        avg_est = np.mean(est_phase_noise)
        estimated_phases.append(avg_est)

    st.success(f"Phase estimated for {num_symbols} Qc symbol(s).")

    # Step 6: Apply phase correction to full signal per block
    samples_per_symbol = no_pulses_per_symbol * num_samples_per_pulse
    full_corrected_signal = np.zeros_like(freq_compensated_signal, dtype=np.complex128)

    for i in range(num_symbols):
        start_full = i * samples_per_symbol
        end_full = min(start_full + samples_per_symbol, len(freq_compensated_signal))
        phase_corr = estimated_phases[i]
        full_corrected_signal[start_full:end_full] = freq_compensated_signal[start_full:end_full] * np.exp(1j * phase_corr)

    # Handle remaining samples
    if end_full < len(freq_compensated_signal):
        full_corrected_signal[end_full:] = freq_compensated_signal[end_full:] * np.exp(1j * estimated_phases[-1])

    # Step 7: Plot before/after
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    ax1.scatter(np.real(demod_current), np.imag(demod_current), alpha=0.5, s=5)
    ax1.set_title('Before Corrections')
    ax1.set_xlabel('I'); ax1.set_ylabel('Q'); ax1.grid(True); ax1.set_aspect('equal')

    ax2.scatter(np.real(full_corrected_signal), np.imag(full_corrected_signal), alpha=0.5, s=5)
    ax2.set_title('After Freq + Phase Corrections')
    ax2.set_xlabel('I'); ax2.set_ylabel('Q'); ax2.grid(True); ax2.set_aspect('equal')

    plt.tight_layout()
    st.pyplot(fig)

# --- Streamlit UI ---
st.title("Optical Signal Correction for CVQKD")

st.sidebar.header("Configuration")
uploaded_file_i = st.sidebar.file_uploader("Upload I data (CSV)", type="csv")
uploaded_file_q = st.sidebar.file_uploader("Upload Q data (CSV)", type="csv")

st.sidebar.subheader("Signal Parameters")
Fs = st.sidebar.number_input("Sampling Frequency (Fs)", value=2.5e9, step=1e6, format="%.2e")
pilot_freq = st.sidebar.number_input("Pilot Frequency (Hz)", value=500e6, step=1e6, format="%.2e")
qc_symbol_rate = st.sidebar.number_input("Qc Symbol Rate (Hz)", value=100e6, step=1e6, format="%.2e")

# Tunable sample position
num_samples_per_pulse_calc = int(Fs / pilot_freq) if pilot_freq != 0 else 1
sample_position = st.sidebar.slider("Sample Position in Pulse", min_value=1, max_value=num_samples_per_pulse_calc, value=3)

analysis_type = st.sidebar.selectbox(
    "Choose Analysis",
    ["FFT", "Combine FFT", "Scatter Plot", "Frequency Correction", "Phase Correction", "Full Signal Correction"]
)

process_button = st.sidebar.button("Process")

# --- Main Logic ---
if process_button:
    if uploaded_file_i is not None and uploaded_file_q is not None:
        try:
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
                plot_phase_correction(data_i, data_q, Fs, pilot_freq, qc_symbol_rate, sample_position)
            elif analysis_type == "Full Signal Correction":
                plot_full_correction(data_i, data_q, Fs, pilot_freq, qc_symbol_rate, sample_position)

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.error("Please ensure the CSV files are in the correct format (e.g., from a Tektronix oscilloscope, with data in the 5th column and 4 header rows).")
    else:
        st.warning("Please upload both I and Q data files to begin.")