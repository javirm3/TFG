import numpy as np
import matplotlib.pyplot as plt

# Time axis (ms)
t = np.linspace(0, 5, 1000)

# Define key points for a smoother action potential
key_times = np.array([0, 1, 1.1, 1.5, 1.9, 2.5, 3.5, 5])
key_voltages = np.array([-70, -70, -55, 30, -30, -80, -70, -70])

# Interpolate linearly between key points
V_raw = np.interp(t, key_times, key_voltages)

# Create a Gaussian smoothing kernel
def gaussian_kernel(size, sigma):
    x = np.linspace(-size // 2, size // 2, size)
    kernel = np.exp(-x**2 / (2 * sigma**2))
    return kernel / np.sum(kernel)

kernel = gaussian_kernel(size=101, sigma=10)

# Smooth the signal
V_smooth = np.convolve(V_raw, kernel, mode='same')

# Plot the action potential
plt.figure(figsize=(10, 5))
plt.plot(t, V_smooth, color='red', linewidth=2)

# Plot resting potential line
V_rest = -70
plt.axhline(V_rest, color='gray', linestyle='--', linewidth=1)

# Plot threshold potential (at -55 mV)
threshold = -55
plt.axhline(threshold, color='black', linestyle='--', linewidth=1)

# Find exact points for annotation
# Resting annotation: choose a point on the flat region, e.g., t = 0.5 ms
t_rest, V_rest_val = 0.5, V_rest+1

# Threshold crossing: first index in rising phase where V_smooth >= -55
indices_rising = np.where((t >= 0.9) & (t <= 1.5))[0]
crossing_indices = indices_rising[np.where(V_smooth[indices_rising] >= threshold)[0]]
cross_idx = crossing_indices[0]
t_thresh, V_thresh_val = t[cross_idx]+0.2, V_smooth[cross_idx]

# Depolarization annotation: point shortly above threshold
t_depol, V_depol_val = t[cross_idx + 5], V_smooth[cross_idx + 5]

# Peak annotation: maximum value around t = 1.5 ms
peak_idx = np.argmax(V_smooth)
t_peak, V_peak_val = t[peak_idx], V_smooth[peak_idx]

# Repolarization annotation: choose mid-descent point, e.g., time ≈ 1.8 ms
t_repol = 1.8
V_repol_val = V_smooth[np.abs(t - t_repol).argmin()]

# Hyperpolarization annotation: lowest point in the interval [2 ms, 3 ms]
hy_idx_range = np.where((t >= 2) & (t <= 3))[0]
min_hyper_idx = hy_idx_range[np.argmin(V_smooth[hy_idx_range])]
t_hyper, V_hyper_val = t[min_hyper_idx], V_smooth[min_hyper_idx]

# Annotations with arrows exactly touching the curve
plt.annotate(
    'Resting Potential\n(−70 mV)',
    xy=(t_rest, V_rest_val),
    xytext=(t_rest, V_rest_val + 15),
    ha='center',
    color='gray',
    arrowprops=dict(arrowstyle='->', color='gray')
)

plt.annotate(
    'Threshold\n(−55 mV)',
    xy=(t_thresh, V_thresh_val),
    xytext=(t_thresh, V_thresh_val - 20),
    ha='center',
    color='black',
    arrowprops=dict(arrowstyle='->', color='black')
)

plt.annotate(
    'Depolarization',
    xy=(t_depol, V_depol_val),
    xytext=(t_depol, V_depol_val + 20),
    ha='center',
    arrowprops=dict(arrowstyle='->', color='black')
)

plt.annotate(
    'Peak (~+30 mV)',
    xy=(t_peak, V_peak_val),
    xytext=(t_peak + 0.5, V_peak_val + 5),
    ha='left',
    arrowprops=dict(arrowstyle='->', color='black')
)

plt.annotate(
    'Repolarization',
    xy=(t_repol, V_repol_val),
    xytext=(t_repol + 0.8, V_repol_val - 5),
    ha='left',
    arrowprops=dict(arrowstyle='->', color='black')
)

plt.annotate(
    'Hyperpolarization',
    xy=(t_hyper, V_hyper_val),
    xytext=(t_hyper + 0.8, V_hyper_val - 5),
    ha='left',
    arrowprops=dict(arrowstyle='->', color='black')
)

# Labels in English
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (mV)')
plt.title('Action Potential: Detailed Phases')
plt.ylim(-90, 50)
plt.xlim(0, 5)
plt.grid(False)
plt.tight_layout()
plt.savefig('action_potential_detailed.png', dpi=300)
plt.show()