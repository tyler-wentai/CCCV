import matplotlib.pyplot as plt


### Onset_Count_Global_NINO3_square4.csv STRONG
# psi_threshold = [1.350, 1.800, 2.250, 2.700, 3.150, 3.600, 4.050]
# dmi_effect = [0.0628, 0.0779, 0.0733, 0.0839, 0.1011, 0.1491, 0.1455]
# dmi_lb = [-0.0296, -0.0316, -0.0453, -0.0476, -0.0352, 0.0010, -0.0133]
# dmi_ub = [0.1542, 0.1891, 0.1910, 0.2114, 0.2363, 0.2945, 0.3014]

### Onset_Count_Global_NINO3_square4.csv WEAK
# psi_threshold = [0.450, 0.900, 1.350, 1.800]
# dmi_effect = [0.3135, 0.1406, 0.1621, 0.1186]
# dmi_lb = [0.0775, -0.0201, 0.0437, 0.0152]
# dmi_ub = [0.5396, 0.2938, 0.2789, 0.2192]

### Onset_Count_Global_NINO3_square2.csv STRONG
# psi_threshold = [1.350, 1.800, 2.250, 2.700, 3.150, 3.600, 4.050]
# dmi_effect = [0.0840, 0.1078, 0.0834, 0.0551, 0.0613, 0.1274, 0.1524]
# dmi_lb = [-0.0086, 0.0027, -0.0359, -0.0792, -0.0771, -0.0191, -0.0098]
# dmi_ub = [0.1752, 0.2113, 0.1987, 0.1872, 0.1931, 0.2659, 0.3086]

### Onset_Count_Global_NINO3_square2.csv WEAK
# psi_threshold = [0.450, 0.900, 1.350, 1.800]
# dmi_effect = [0.2265, 0.0959, 0.1328, 0.0975]
# dmi_lb = [-0.0507, -0.0816, 0.0132, -0.0066]
# dmi_ub = [0.4930, 0.2598, 0.2472, 0.1988]


### Onset_Binary_GlobalState_NINO3.csv STRONG
# psi_threshold = [1.350, 1.800, 2.250, 2.700, 3.150, 3.600, 4.050]
# dmi_effect = [0.0051, 0.0055, 0.0048, 0.0050, 0.0058, 0.0065, 0.0071]
# dmi_lb = [0.0009, 0.0010, -0.0007, -0.0008, -0.0008, -0.0006, 0.0001]
# dmi_ub = [0.0094, 0.0101, 0.0103, 0.0109, 0.0124, 0.0136, 0.0142]

### Onset_Binary_GlobalState_NINO3.csv WEAK
# psi_threshold = [0.450, 0.900, 1.350]
# dmi_effect = [-0.0005, 0.0014, 0.0041]
# dmi_lb = [-0.0056, -0.0032, -0.0016]
# dmi_ub = [0.0047, 0.0061, 0.0098]


### Onset_Count_Global_DMI_square4.csv STRONG
# psi_threshold = [1.350, 1.575, 1.800, 2.025, 2.250, 2.475]
# dmi_effect = [0.1636, 0.8126, 0.7110, 0.9786, 1.1758, 1.2344]
# dmi_lb = [-0.3886, 0.1228, -0.0339, 0.1222, 0.2075, 0.2187]
# dmi_ub = [0.6913, 1.4874, 1.4338, 1.8509, 2.1384, 2.2472]

### Onset_Count_Global_DMI_square4.csv WEAK
# psi_threshold = [0.450, 0.900, 1.350]
# dmi_effect = [0.0329, 0.1567, 0.1687]
# dmi_lb = [-0.3710, -0.1046, -0.0661]
# dmi_ub = [0.4446, 0.4068, 0.3920]

# Calculate asymmetric error bars
lower_errors = [e - lb for e, lb in zip(dmi_effect, dmi_lb)]
upper_errors = [ub - e for e, ub in zip(dmi_effect, dmi_ub)]
asymmetric_error = [lower_errors, upper_errors]

# Plot
plt.figure(figsize=(5, 4))
plt.errorbar(psi_threshold, dmi_effect, yerr=asymmetric_error, fmt='o', color='red')
plt.xlabel(r'Teleconnection threshold ($\Psi_i > $ threshold)')
plt.ylabel(r'ENSO Effect (ACR per °C)')
plt.axhline(0, linewidth=1.0, linestyle='--', color='k')
plt.ylim(-0.1, 0.6)
plt.xticks(psi_threshold)

plt.title('NINO3, Grids 2°, Strong Teleconnections')

plt.tight_layout()
plt.savefig('/Users/tylerbagwell/Desktop/justin_slidedeck/NINO3aggpoisson_GRID2_stronglyteleconnected.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()
