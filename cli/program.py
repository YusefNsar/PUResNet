import os
import sys
from openbabel import pybel
import tensorflow as tf

from args_handler import ArgsHandler, Args
from ResNet import PUResNet

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)


class Program:
    args: Args
    model: PUResNet

    def __init__(self):
        self.args = ArgsHandler.parse()
        ArgsHandler.validate(self.args)

        self.model = PUResNet()
        self.model.load_weights(
            "/home/yusef/development/collage/PUResNet/whole_trained_model1.hdf"
        )

        is_single_file = self.args.multi == 0

        if is_single_file:
            self.predict_single()
        else:
            self.predict_multi()

    # PREDICTION
    def predict_single(self):
        mol = next(pybel.readfile(self.args.file_format, self.args.input_path))
        o_path = self.make_output_folder(self.args.output_path, self.args.input_path)
        self.model.save_pocket_mol2(mol, o_path, self.args.output_format)

    def predict_multi(self):
        for mol_name in os.listdir(self.args.input_path):
            mol_path = os.path.join(self.args.input_path, mol_name)
            mol = next(pybel.readfile(self.args.file_format, mol_path))
            o_path = self.make_output_folder(self.args.output_path, mol_path)
            self.model.save_pocket_mol2(mol, o_path, self.args.output_format)

    # HELPERS
    def make_output_folder(self, output_folder_path, input_folder_path):
        protein_name = os.path.basename(input_folder_path)
        o_path = os.path.join(output_folder_path, protein_name)
        if not os.path.exists(o_path):
            os.mkdir(o_path)
        return o_path


Program()
