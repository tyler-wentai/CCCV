import matplotlib.pyplot as plt

# Data provided by the user
### Onset_Count_Global_DMI_square4.csv STRONG
psi_threshold = [1.350, 1.575, 1.800, 2.025, 2.250, 2.475]
dmi_effect = [0.1636, 0.8126, 0.7110, 0.9786, 1.1758, 1.2344]
dmi_lb = [-0.3886, 0.1228, -0.0339, 0.1222, 0.2075, 0.2187]
dmi_ub = [0.6913, 1.4874, 1.4338, 1.8509, 2.1384, 2.2472]

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
plt.ylabel(r'DMI Effect (count per °C)')
plt.axhline(0, linewidth=1.0, linestyle='--', color='k')
plt.ylim(-0.5, 2.5)

plt.tight_layout()
plt.savefig('/Users/tylerbagwell/Desktop/justin_slidedeck/DMIaggcount_highlyteleconnected.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()
