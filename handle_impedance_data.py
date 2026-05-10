from utils import file_utils, linKK, ECM_utils, fitting_utils
import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#_______________________________________________________________________________________________________________________
# READ DATA .CSV

'''spec_obj = file_utils.read('./data/17 - EIS ferro(III) 5,0e-6 - DC 0,6 V.csv')
freqs = spec_obj.freq
z_real = spec_obj.Z_real
z_imag = -spec_obj.Z_imag #imaginary impedance is negative

#_______________________
# PLOT DATA
fig = plt.figure(figsize=(8,6))
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
ax2.set_title('Phase Plot')
ax2.grid(True)
plt.tight_layout()
plt.show()'''
#_______________________________________________________________________________________________________________________
# READ DATA .XLSX
spec_obj = file_utils.read('./data/Dados Everton.xlsx')
freqs = spec_obj.freq
z_real = spec_obj.Z_real
z_imag = -spec_obj.Z_imag #imaginary impedance is negative

#_______________________
# PLOT DATA
'''label=["65 10 umolL - Pb S25", "65 10 umolL - Cd S15", "65 10 umolL - Hg S8", "65 Tampão Hh S1"]
fig = plt.figure(figsize=(8,6))
for i in range(len(z_real[:,0])):
    plt.plot(z_real[i,:], z_imag[i,:], label=label[i])
plt.title('Nyquist Impedance')
plt.ylabel("Z'")
plt.xlabel("Z''")
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()

Z_real = np.asarray(z_real, dtype=float)
Z_imag = np.asarray(z_imag, dtype=float)
Z_mag = np.sqrt(Z_real**2 + Z_imag**2)
Z_phase = np.degrees(np.arctan2(Z_imag, Z_real))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
fig.suptitle('Impedance Data')

# Bode magnitude
for i in range(len(Z_mag[:,0])):
    ax1.plot(freqs, Z_mag[i,:], label=label[i])
ax1.set_xscale('log')
ax1.set_ylabel('|Z| (Ohm)')
ax1.set_title('Bode Plot')
ax1.legend(loc='upper right')
ax1.grid(True)

# Phase
for i in range(len(Z_phase[:,0])):
    ax2.plot(freqs, Z_phase[i,:], label=label[i])
ax2.set_xscale('log')
ax2.set_ylabel('Phase (deg)')
ax2.set_xlabel('Frequency (Hz)')
ax2.set_title('Phase Plot')
ax2.grid(True)
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()

#_______________________________________________________________________________________________________________________
# Validate EIS data - by Linear Krammers Kroning

linkk_obj = linKK.LinearKramersKronig(spec_obj, c=0.1, max_iter=50, add_capacitor=True, verbose=True)
if linkk_obj.chi_square > 1e-2:
    raise ValueError(f'Linear Kramers-Kronig test failed: x² = {linkk_obj.chi_square}')
'''
#_______________________________________________________________________________________________________________________
# Defining the fitting parameters from the choosen Electric Circuit Model (ECM)
# write the circuit as text
# circuit = "(R//(R+L+C))+(C//W)"

circuit= "(R1 +(R2//CPE1)+(R3//CPE2))" # prof paula dados everton .xlsx
ECM_Params = ECM_utils.CircuitParams(circuit)

print("Circuit:")
print(circuit)
print("Parameter names:")
print(ECM_Params.param_names[:])

# print symnbols Z(omega) + circuit
#_______________________________________________________________________________________________________________________
# Evaluate the frequency response of the impedance for the ECM

# define parameter values
# params = {
#     "R1": 100.0,
#     "R2": 300.0,
#     "L1":1e-6,
#     "C1": 1e-6,
#     "C2": 1e-6,
#     "Q1": 2e-4,
#     "alpha1": 0.85,
#     "W1": 10.0
# }

# params ={
#     "R1":100.0,
#     "R2": 100.0,
#     "Q1": 2e-4 ,
#     "alpha1": 0.8,
#     "R3" : 100.0,
#     "Q2":2e-4 ,
#     "alpha2": 0.8,
# }

param_value = np.array([100.0,100.0,2e-4 ,0.8,100.0,2e-4 ,0.8,])

ECM_Z = ECM_utils.CircuitEvaluate(freqs, ECM_Params, param_value, verbose=True)

#_______________________________________________________________________________________________________________________
# fitting data
# method = "BFGS" or "NLLS"

ECM_fit = fitting_utils.Circuit_fitting(spec_obj,freqs,ECM_Params)

initial_params = np.array([1,1,1,1,1,1,1])
scaling_array = np.array([1e3, 1e-7, 1e6, 1e-2, 1e4, 1e-1, 1])
ECM_fit_params = ECM_fit.fit_circuit(initial_params, scaling_array, method="BFGS", verbose=True)

#________________________
# Plot fitting result
# plot
fig, ax = plt.subplots()
leg = []
ax.scatter(fit_obj.z_meas_real, fit_obj.z_meas_imag, marker='o', color="tab:blue")
leg.append('ice measured')
ax.plot(fit_params.opt_fit.real, -fit_params.opt_fit.imag, color="tab:orange")
leg.append('Longo2020')
x1, x2, y1, y2 = -1000, 20000, 1000, 12000
axins = ax.inset_axes([0.5, 0.18, 0.4, 0.4],
                      xlim=(x1, x2), ylim=(y1, y2))
axins.scatter(fit_obj.z_meas_real, fit_obj.z_meas_imag, marker='o', color="tab:blue")
axins.plot(fit_params.opt_fit.real, -fit_params.opt_fit.imag, color="tab:orange")
#axins.grid()
ax.indicate_inset_zoom(axins, edgecolor="black", linewidth=1.5)
plt.xlabel("Z'")
plt.ylabel("Z''")
plt.legend(leg)
plt.grid()
plt.show()

plt.figure()
plt.subplot(1,2,1)
leg = []
plt.scatter(np.log10(spec_ice_obj.freqs), np.abs(fit_obj.z_meas))
leg.append('ice measured')
plt.plot(np.log10(spec_ice_obj.freqs), np.abs(fit_params.opt_fit), color="tab:orange")
leg.append('Longo2020')
plt.ylabel("|Z|")
plt.xlabel("log(Frequency)")
plt.legend(leg)
plt.grid()

plt.subplot(1,2,2)
leg = []
plt.scatter(np.log10(spec_ice_obj.freqs), -np.angle(fit_obj.z_meas.astype('complex')))
leg.append('ice measured')
plt.plot(np.log10(spec_ice_obj.freqs), -np.angle(fit_params.opt_fit), color="tab:orange")
leg.append('Longo2020')
plt.ylabel("-∠Z")
plt.xlabel("log(Frequency)")
plt.legend(leg)
plt.grid()

# _______________________________________________________________________________________________________________________
# Evaluate Prof. Paula new tests

# test 65 - 10 umolL - Pb S25
# paramsPbS25 ={
#     "R1":100.0,
#     "R2": 100.0,
#     "Q1": 2e-4 ,
#     "alpha1": 0.8,
#     "R3" : 100.0,
#     "Q2":2e-4 ,
#     "alpha2": 0.8,
# }

# test 65 - 10umolL - Cd S15
# paramsCdS15 ={
#     "R1":207.9,
#     "R2": 6104.0,
#     "Q1": 121.4e-6 ,
#     "alpha1": 0.772,
#     "R3" : 2044.0,
#     "Q2":29.97e-6 ,
#     "alpha2": 0.658,
# }

# test 65 - 10umolL - Hg S8
# paramsHgS8 ={
#     "R1":170.9,
#     "R2": 4627.0,
#     "Q1": 277.6e-6 ,
#     "alpha1": 0.56,
#     "R3" : 3919.0,
#     "Q2":26.31e-6 ,
#     "alpha2": 0.716,
# }

# test 65 - tampão - Hg S1
# paramsHgS1 ={
#     "R1":100.0,
#     "R2": 100.0,
#     "Q1": 2e-4 ,
#     "alpha1": 0.8,
#     "R3" : 100.0,
#     "Q2":2e-4 ,
#     "alpha2": 0.8,
# }

# ECM_Z_PbS25 = ECM_utils.CircuitEvaluate(freqs, paramsPbS25, ECM_Params.tree)
# ECM_Z_CdS15 = ECM_utils.CircuitEvaluate(freqs, paramsCdS15, ECM_Params.tree)
# ECM_Z_HgS8 = ECM_utils.CircuitEvaluate(freqs, paramsHgS8, ECM_Params.tree)
# ECM_Z_HgS1 = ECM_utils.CircuitEvaluate(freqs, paramsHgS1, ECM_Params.tree)