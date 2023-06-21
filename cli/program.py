import os
import sys
from typing import List
from openbabel import pybel
import tensorflow as tf

from cli.args_handler import ArgsHandler, Args
from model.model_runner import ModelRunner

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)


class Program:
    args: Args
    model: ModelRunner

    def __init__(self):
        self.args = ArgsHandler.parse()
        ArgsHandler.validate(self.args)

        self.model = ModelRunner()

        is_single_file = self.args.multi == 0

        if is_single_file:
            self.predict_single()
        else:
            self.predict_multi()

    # PREDICTION
    def predict_single(self):
        mol = next(pybel.readfile(self.args.file_format, self.args.input_path))
        o_path = self.make_output_folder(self.args.output_path, self.args.input_path)

        pockets = self.model.predictBindingSites(mol)
        self.save_pockets(pockets, o_path)

    def predict_multi(self):
        for mol_name in os.listdir(self.args.input_path):
            mol_path = os.path.join(self.args.input_path, mol_name)

            mol = next(pybel.readfile(self.args.file_format, mol_path))
            o_path = self.make_output_folder(self.args.output_path, mol_path)

            pockets = self.model.predictBindingSites(mol)
            self.save_pockets(pockets, o_path)

    # HELPERS
    def make_output_folder(self, output_folder_path, input_folder_path):
        protein_name = os.path.basename(input_folder_path)
        o_path = os.path.join(output_folder_path, protein_name)
        if not os.path.exists(o_path):
            os.mkdir(o_path)
        return o_path

    def save_pockets(self, pockets: List[pybel.Molecule], mol_output_path: str) -> None:
        save_format = self.args.output_format
        for i in range(len(pockets)):
            pocket_file_name = f"pocket{i}.{save_format}"
            pocket_file_path = os.path.join(mol_output_path, pocket_file_name)

            pockets[i].write(save_format, pocket_file_path)
