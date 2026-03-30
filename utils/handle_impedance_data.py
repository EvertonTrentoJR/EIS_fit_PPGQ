from utils import file_utils, linKK
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#_______________________________________________________________________________________________________________________
# READ DATA

spec_obj = file_utils.read('./data/data_ppgq_test/12_EIS_persulfato.csv')
freqs = spec_obj.freq
z_real = spec_obj.Z_real
z_imag = -spec_obj.Z_imag #imaginary impedance is negative

#_______________________________________________________________________________________________________________________
# PLOT DATA

fig = plt.figure(figsize=(8,6) )
plt.plot(z_real, z_imag, 'b-')
plt.title('Nyquist Impedance')
plt.ylabel("Z'")
plt.xlabel("Z''")
plt.grid(True)
plt.tight_layout()

Z_real = np.asarray(z_real, dtype=float)
Z_imag = np.asarray(z_imag, dtype=float)
Z_mag = np.sqrt(Z_real**2 + Z_imag**2)
Z_phase = np.degrees(np.arctan2(Z_imag, Z_real))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
fig.suptitle('Impedance Data')
# Bode magnitude
ax1.plot(freqs, Z_mag, color='blue')
ax1.set_xscale('log')
ax1.set_ylabel('|Z| (Ohm)')
ax1.set_title('Bode Plot')
ax1.grid(True)
# Phase
ax2.plot(freqs, Z_phase, color='red')
ax2.set_xscale('log')
ax2.set_ylabel('Phase (deg)')
ax2.set_xlabel('Frequency (Hz)')
ax2.grid(True)
plt.tight_layout()
# plt.show()

#_______________________________________________________________________________________________________________________
# validate eis data - by Linear Krammers Kroning

linkk_obj = linKK.LinearKramersKronig(spec_obj, c=0.1, max_iter=50, add_capacitor=True, verbose=False)

if linkk_obj.chi_square > 1e-2:
    raise ValueError(f'Linear Kramers-Kronig test failed: x² = {linkk_obj.chi_square}')

#_______________________________________________________________________________________________________________________
