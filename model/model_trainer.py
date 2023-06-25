import os
from typing import List, Tuple
from openbabel import pybel
import pickle
import numpy as np
from tensorflow.python.keras.layers import Input as KerasTensor

from utils.feature_extractor import FeatureExtractor
from utils.mol_3d_grid import Mol3DGrid
from model.PUResNet import PUResNet


class ModelTrainer:
    def __init__(self):
        self.train_data: np.ndarray = None
        self.mol_grid = Mol3DGrid(max_dist=35.0, scale=0.5)
        d = self.mol_grid.box_size
        f = len(self.mol_grid.fe.FEATURE_NAMES)

        self.model = PUResNet(d, f)
        self.model.compile(
            optimizer="Adam", loss="binary_crossentropy", metrics=["acc"]
        )
        self.load_training_data()
        X, y = self.preprocess_train_data()
        self.model.fit(X, y, batch_size=5)
        self.model.save_weights("weights/1.hdf")

        pass

    # def train_model(self) -> None:
    #     proteins, proteins_sites = self.save_training_data("final_data")

    #     self.model.train_model(proteins, proteins_sites)

    def load_training_data(self):
        with open("train_data[0:1004].pickle", "rb") as f:
            # [rows[x[pcoords, pfeats], scoords]
            self.train_data = pickle.load(f)

    def save_training_data(
        self,
        training_data_path,
    ) -> Tuple[List[pybel.Molecule], List[pybel.Molecule]]:
        # get list of proteins files storted by name
        protein_names = os.listdir(training_data_path)
        protein_names.sort()

        train_data = []
        fe = FeatureExtractor()

        # determine section of train data to save in this session
        start = 0
        end = 1004

        for i in range(start, end):
            protein_name = protein_names[i]

            path = os.path.join(training_data_path, protein_name)
            mol_path = path + "/protein.mol2"
            site_path = path + "/site.mol2"

            protein = next(pybel.readfile("mol2", mol_path))
            site = next(pybel.readfile("mol2", site_path))

            protein_coords, protein_features = fe.get_feature(protein)
            site_coords = fe.get_all_coords(site)

            train_data_row = [[protein_coords, protein_features], site_coords]
            train_data.append(train_data_row)

        with open(f"train_data[{start}:{end}].pickle", "wb") as f:
            pickle.dump(train_data, f)

        self.train_data = train_data

    def train_model(
        self, proteins: List[pybel.Molecule], proteins_sites: List[pybel.Molecule]
    ):
        # iterate over data
        # create metrics for training

        X = map(proteins, lambda p: self.mol_grid.setMol(p).transform())
        y = map(proteins_sites, lambda ps: self.mol_grid.setMol(ps).transform())
        print(X.shape, y.shape, X[0], y[0])

        pass

    def success_rate(self, y_true: KerasTensor, y_pred: KerasTensor) -> float:
        y_true

        pass

    def preprocess_train_data(self):
        X = []
        y = []
        for row in self.train_data:
            coords, feats = row[0]
            coords, feats = np.array(coords), np.array(feats)
            grid = self.mol_grid.setMolAsCoords(coords, feats).transform()
            X.append(grid)

            site_coords = np.array(row[1])
            grid = self.mol_grid.setMolAsCoords(site_coords).transform()
            y.append(grid)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        print(X.shape, y.shape)

        return X, y
