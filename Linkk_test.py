import matplotlib
matplotlib.use('TkAgg')
from utils import file_utils, linKK, ECM_utils, fitting_utils
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

#  __________________ load data_________________
folder = Path("data/ferroIII/")
# folder = Path("data/cadmioII/")
# folder = Path("data/dicromato/")

spec_obj = {}
for file in sorted(folder.glob("*.csv")):
    spec_obj[file.stem] = file_utils.read(file, flip=True)

# ____________________ plot data ________________
# fig, ax = plt.subplots(figsize=(12, 9))
# colors = plt.cm.Greys(np.linspace(0.25, 0.95, len(spec_obj)))
#
# for i, (name, data) in enumerate(spec_obj.items()):
#     z_real = data.Z_real
#     z_imag = -data.Z_imag
#
#     ax.plot(z_real, z_imag, '-', color=colors[i], linewidth=1.5, label=name)
#
# ax.set_title('Nyquist Impedance')
# ax.set_xlabel(r"$Z'$ ($\Omega$)")
# ax.set_ylabel(r"$-Z''$ ($\Omega$)")
# ax.grid(True, alpha=0.3)
#
# # Legend below the plot
# ax.legend(
#     loc='upper center',
#     bbox_to_anchor=(0.5, -0.15),
#     ncol=4,
#     fontsize=8,
#     frameon=False
# )
#
# plt.tight_layout()
# plt.show()
#
# # # _______________________
# # # PLOT DATA - NYQUIST OVERLAY BY EXCITATION VOLTAGE
# #
# excitations = ["0,1 V", "0,3 V", "0,6 V"]
#
# fig, axes = plt.subplots(1, 3, figsize=(22, 9))
#
# for ax, excitation in zip(axes, excitations):
#
#     # Filter datasets by excitation voltage in filename
#     filtered_data = {
#         name: data
#         for name, data in spec_obj.items()
#         if excitation in name
#     }
#
#     colors = plt.cm.turbo(np.linspace(0, 1, len(spec_obj)))
#
#     for i, (name, data) in enumerate(filtered_data.items()):
#         z_real = data.Z_real
#         z_imag = -data.Z_imag
#
#         ax.plot(
#             z_real, z_imag,
#             '-',
#             color=colors[i],
#             linewidth=1.5,
#             label=name
#         )
#
#     ax.set_title(f'Excitation: {excitation}')
#     ax.set_xlabel(r"$Z'$ ($\Omega$)")
#     ax.set_ylabel(r"$-Z''$ ($\Omega$)")
#     ax.grid(True, alpha=0.3)
#
#     # Legend below each subplot
#     ax.legend(
#         loc='upper center',
#         bbox_to_anchor=(0.5, -0.18),
#         ncol=2,
#         fontsize=7,
#         frameon=False
#     )
#
#     print(f"{excitation}: {len(filtered_data)} datasets")
#
# fig.suptitle("Nyquist Impedance", fontsize=16)
# fig.subplots_adjust(
#     left=0.06,
#     right=0.98,
#     top=0.90,
#     bottom=0.43,
#     wspace=0.25
# )
# plt.show()

#_______________________________________________________________________________________________________________________
# Validate EIS data - by Linear Krammers Kroning
linkk_dataAnalyses = {}
for i, (name, data) in enumerate(spec_obj.items()):
    linkk_obj = linKK.LinearKramersKronig(data, c=0.1, max_iter=50, add_capacitor=True, verbose=True)

    # if linkk_obj.chi_square > 1e-2:
    #     raise ValueError(f'Linear Kramers-Kronig test failed: x² = {linkk_obj.chi_square}')

    linkk_dataAnalyses[name] = linkk_obj