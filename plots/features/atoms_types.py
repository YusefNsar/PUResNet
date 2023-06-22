import sys
from matplotlib.colors import BoundaryNorm, ListedColormap

sys.path.append("/home/yusef/Development/FCI/PUResNet")

from openbabel import pybel
import matplotlib.pyplot as plt
from model.mol_3d_grid import Mol3DGrid

mol = next(pybel.readfile("mol2", "test/1a2n_1/protein.mol2"))

mol_grid = Mol3DGrid()
grid = mol_grid.setMol(mol).transform()

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

xs = mol_grid.coords[:, 0]
ys = mol_grid.coords[:, 1]
zs = mol_grid.coords[:, 2]

# Create a segmented colormap with discrete colors
cmap = ListedColormap(
    [
        "#984ea3",
        "#4daf4a",
        "#377eb8",
        "#e41a1c",
        "#ff7f00",
        "#ffff33",
        "#a65628",
        "#f781bf",
        "#999999",
    ]
)

# Set the boundaries between color regions
norm = BoundaryNorm([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], cmap.N)

colors = []
for i in range(len(xs)):
    atom_type = list(mol_grid.features[i][0:9]).index(1.0)
    colors.append(atom_type)

scatter = ax.scatter(
    xs, ys, zs, s=50, alpha=0.9, c=colors, edgecolors=None, cmap=cmap, norm=norm
)
cbar = plt.colorbar(scatter, ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8])
cbar.ax.set_yticklabels(["B", "C", "N", "O", "P", "S", "Se", "Halogen", "Metal"])

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.show()
