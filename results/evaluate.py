# iterate over the folder

# check every pocket

import os

from metrics import (
    create_3d_grid,
    intersection_over_union,
    intersection_over_lig,
    coordinates,
    get_DVO,
    get_PLI,
    get_DCC,
)

input_path = "/home/yusef/development/collage/PUResNet/BU48/bench2"
res_dcc = []
res_dvo = []
res_pli = []

for mol_name in os.listdir(input_path):
    print(mol_name)
    mol_folder = os.path.join(input_path, mol_name)

    mol_path = os.path.join(mol_folder, "protein.mol2")
    site_path = os.path.join(mol_folder, "site.mol2")
    ligand_path = os.path.join(mol_folder, "ligand.mol2")
    pred_path = os.path.join(mol_folder, "pocket0.mol2")

    res_dcc.append(get_DCC(site_path, pred_path))
    if res_dcc[-1] == 35:
        res_dcc.pop()
        continue
    print(res_dcc[-1])
    res_dvo.append(get_DVO(site_path, pred_path))
    print(res_dvo[-1])
    res_pli.append(get_PLI(ligand_path, pred_path))
    print(res_pli[-1])

import numpy as np

# print(res_dcc, res_dvo, res_pli)

results = np.asarray([res_dcc, res_dvo, res_pli])
print((results[0] < 4).sum() / len(results[0]) * 100)
print(results[0].mean())
print(results[1].mean())
print(results[2].mean())

np.savetxt(
    "data.csv", np.transpose(results), delimiter=",", fmt="%.4f", header="dcc,dvo,pli"
)
