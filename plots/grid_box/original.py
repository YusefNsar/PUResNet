import sys

sys.path.append("/home/yusef/Development/FCI/PUResNet")

from openbabel import pybel
import matplotlib.pyplot as plt

mol = next(pybel.readfile("mol2", "test/1a2n_1/protein.mol2"))

ogxs = []
ogys = []
ogzs = []

for atom in mol.atoms:
    x, y, z = atom.coords
    ogxs.append(x)
    ogys.append(y)
    ogzs.append(z)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(ogxs, ogys, ogzs, s=50, alpha=0.6, edgecolors="w")

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.show()
