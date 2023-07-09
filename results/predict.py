# init model runner with weights

# iterate over protein folders in bu48

# use protien.mol2 file to predict pred_site.mol

# compare it to site from protein.mol2 substructure
import sys

sys.path.append("/home/yusef/development/collage/PUResNet/")

from model.model_runner import ModelRunner
from cli.program import Program
import os
from typing import List
from openbabel import pybel

pybel.ob.obErrorLog.StopLogging()


mr = ModelRunner("/home/yusef/development/collage/PUResNet/weights.h5", True)


def predict_multi(input_path):
    for mol_name in os.listdir(input_path):
        mol_folder = os.path.join(input_path, mol_name)

        mol_path = os.path.join(mol_folder, "protein.mol2")
        file_format = "mol2"

        mol = next(pybel.readfile(file_format, mol_path))
        print("before")
        pockets = mr.predictBindingSites(mol)
        print("after")
        save_pockets(pockets, mol_folder, file_format)


# HELPERS
def make_output_folder(output_folder_path, input_file_path):
    protein_name = os.path.basename(input_file_path)
    o_path = os.path.join(output_folder_path, protein_name)
    if not os.path.exists(o_path):
        os.mkdir(o_path)
    return o_path


def save_pockets(
    pockets: List[pybel.Molecule], mol_output_path: str, save_format: str
) -> None:
    for i in range(len(pockets)):
        pocket_file_name = f"pocket{i}.{save_format}"
        pocket_file_path = os.path.join(mol_output_path, pocket_file_name)

        print(pocket_file_path)
        pockets[i].write(save_format, pocket_file_path, overwrite=True)


path = "/home/yusef/development/collage/PUResNet/BU48/bench2"


predict_multi(path)
