"""Plot the directly calculated Mn3Sn path and export its numerical energies."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run", type=Path, required=True)
parser.add_argument("--reader", type=Path, help="Directory containing vaspwfc.py; required without --cached.")
parser.add_argument("--cached", action="store_true", help="Plot the saved band_energies.npz without WAVECAR.")
parser.add_argument("--output", type=Path, help="Destination directory; defaults to RUN.")
parser.add_argument("--font-family", default="DejaVu Serif")
parser.add_argument("--font-dir", type=Path, help="Optional directory containing .ttf fonts.")
args = parser.parse_args()
run = args.run.resolve()
out = args.output.resolve() if args.output else run
out.mkdir(parents=True, exist_ok=True)
data = json.loads((run / "provenance.json").read_text())
n = data["path_point_count"]
if args.cached:
    saved = np.load(run / "band_energies.npz")
    kpoints, distance, energies = saved["kpoints"], saved["distance"], saved["energies_relative_eV"]
else:
    if args.reader is None:
        parser.error("--reader is required unless --cached is selected.")
    sys.path.insert(0, str(args.reader.resolve()))
    from vaspwfc import vaspwfc
    wave = vaspwfc(str(run / "bands/WAVECAR"), lsorbit=True)
    reciprocal = 2*np.pi*np.linalg.inv(wave._Acell).T
    kpoints = wave._kvecs[:n]
    cartesian = kpoints @ reciprocal
    distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(cartesian, axis=0), axis=1))]
    energies = wave._bands[0, :n] - data["source_fermi_eV"]
if args.font_dir:
    for font in args.font_dir.glob("*.ttf"):
        font_manager.fontManager.addfont(str(font))
font_path = font_manager.findfont(args.font_family, fallback_to_default=False)
rcParams.update({"font.family": "serif", "font.serif": [args.font_family],
                 "font.size": 30, "svg.fonttype": "none", "pdf.fonttype": 42,
                 "ps.fonttype": 42, "mathtext.default": "regular"})
fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(distance, energies, color="#202020", linewidth=0.8)
ticks = data["path_tick_indices_zero_based"]
for t in ticks:
    ax.axvline(distance[t], color="black", linewidth=0.8)
ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
ax.set_xticks(distance[ticks], [r"$\Gamma$" if s == 'G' else s for s in data["path"]])
ax.set_ylabel(r"$E-E_F$ (eV)")
ax.set_ylim(-5, 3)
ax.set_xlim(distance[0], distance[-1])
ax.tick_params(axis="both", labelsize=30)
validation_path = run / "analysis_probe/validation.json"
if validation_path.is_file():
    validation = json.loads(validation_path.read_text())
    for dim, color, label in [(2, "#1f77b4", "A: 2D corep"), (4, "#d62728", "A: 4D corep")]:
        values = [b["energy_relative_eV"] for b in validation["blocks"]
                  if b["k_index"] == 4 and b["dimension"] == dim and b["passed"]
                  and -5 <= b["energy_relative_eV"] <= 3]
        ax.scatter([distance[ticks[4]]] * len(values), values, s=36,
                   facecolors="white", edgecolors=color, linewidths=1.2, zorder=4, label=label)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2,
              fontsize=18, frameon=False, borderaxespad=0)
fig.tight_layout()
for suffix in ["png", "svg", "pdf"]:
    fig.savefig(out / ("Mn3Sn_bands." + suffix), dpi=220)
np.savez_compressed(out / "band_energies.npz", kpoints=kpoints, distance=distance,
                    energies_relative_eV=energies, fermi_eV=data["source_fermi_eV"], tick_indices=ticks)
np.savetxt(out / "band_energies.csv", np.c_[distance, kpoints, energies], delimiter=",",
           header="distance_inv_A,k1,k2,k3," + ",".join(f"band_{i+1}_eV" for i in range(energies.shape[1])), comments="")
(out / "plot_metadata.json").write_text(json.dumps({"font": font_path, "points": n,
            "bands": int(energies.shape[1]), "fermi_eV": data["source_fermi_eV"],
            "path": data["path"], "general_point_excluded_from_path": True}, indent=2) + "\n")
print("Wrote Mn3Sn_bands.png/.svg/.pdf and band energies.")
