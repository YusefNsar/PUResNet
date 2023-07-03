import os
from typing import List, Tuple
from openbabel import pybel
import pickle
import numpy as np
from sklearn.model_selection import KFold
from keras.utils import Sequence

from utils.feature_extractor import FeatureExtractor
from utils.mol_3d_grid import Mol3DGrid
from model.PUResNet import PUResNet

# from PUResNet.utils.feature_extractor import FeatureExtractor
# from PUResNet.utils.mol_3d_grid import Mol3DGrid
# from PUResNet.model.PUResNet import PUResNet


class ModelTrainer:
    def __init__(self):
        self.proteins_data: np.ndarray = None
        self.mol_grid = Mol3DGrid(max_dist=35.0, scale=0.5)
        d = self.mol_grid.box_size
        f = len(self.mol_grid.fe.FEATURE_NAMES)
        self.model = PUResNet(d, f)

        pass

    def train_model(self) -> None:
        print("Compiling model...")
        self.model.compile(
            optimizer="Adam",
            loss=get_tversky_loss(alpha=0.7),
            metrics=["acc"],
        )

        print("Loading Data...")
        self.load_training_data()

        print("Starting Training...")
        batch_size = 5
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        folds_results = []
        folds_evaluations = []
        for i, (train_index, test_index) in enumerate(kf.split(self.proteins_data)):
            print(f"Fold #{i+1} => train:{len(train_index)} & test:{len(test_index)}.")

            # spliting data
            Xy_train = []
            for i in train_index:
                Xy_train.append(self.proteins_data[i])
            Xy_test = []
            for i in test_index:
                Xy_test.append(self.proteins_data[i])

            train_data_generator = ProteinsGridsGenerator(
                Xy_train, batch_size, self.mol_grid
            )
            test_data_generator = ProteinsGridsGenerator(
                Xy_test, batch_size, self.mol_grid
            )

            fold_results = self.model.fit_generator(
                generator=train_data_generator,
                epochs=1,
                validation_data=test_data_generator,
            )

            evaluations = self.model.evaluate_generator(generator=test_data_generator)

            folds_results.append(fold_results.history)
            folds_evaluations.append(evaluations)

        print("Training is complete.")

        print("Saving results")
        self.model.save_weights("weights/3.h5", overwrite=True)

        with open(f"folds_results.pickle", "wb") as f:
            pickle.dump(folds_results, f)
        with open(f"folds_evaluations.pickle", "wb") as f:
            pickle.dump(folds_evaluations, f)

        print("Done.")

    def load_training_data(self):
        with open("train_data.pickle", "rb") as f:
            # [rows[x[pcoords, pfeats], scoords]
            self.proteins_data = pickle.load(f)

    def save_training_data(
        self,
        training_data_path="/content/final_data",
    ) -> Tuple[List[pybel.Molecule], List[pybel.Molecule]]:
        # get list of proteins files storted by name
        protein_names = os.listdir(training_data_path)
        protein_names.sort()

        proteins_data = []
        fe = FeatureExtractor()

        # determine section of train data to save in this session
        start = 0
        end = len(protein_names)

        for i in range(start, end):
            protein_name = protein_names[i]
            print("Protein", protein_name, "#", i)

            path = os.path.join(training_data_path, protein_name)
            mol_path = path + "/protein.mol2"
            site_path = path + "/site.mol2"
            ligand_path = path + "/ligand.mol2"

            protein = next(pybel.readfile("mol2", mol_path))
            site = next(pybel.readfile("mol2", site_path))
            ligand = next(pybel.readfile("mol2", ligand_path))

            protein_coords, protein_features = fe.get_feature(protein)
            site_coords = fe.get_all_coords(site)
            ligand_coords = fe.get_all_coords(ligand)

            proteins_data_row = [
                [protein_coords, protein_features],
                site_coords,
                ligand_coords,
            ]
            proteins_data.append(proteins_data_row)

        with open(f"train_data.pickle", "wb") as f:
            pickle.dump(proteins_data, f)

        self.proteins_data = proteins_data

    # def train_model(
    #     self, proteins: List[pybel.Molecule], proteins_sites: List[pybel.Molecule]
    # ):
    #     # iterate over data
    #     # create metrics for training

    #     X = map(proteins, lambda p: self.mol_grid.setMol(p).transform())
    #     y = map(proteins_sites, lambda ps: self.mol_grid.setMol(ps).transform())
    #     print(X.shape, y.shape, X[0], y[0])

    #     pass

    def success_rate(self, y_true, y_pred) -> float:
        y_true

        pass

    def preprocess_train_data(self):
        X = []
        y = []
        for row in self.proteins_data:
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


import tensorflow as tf
from keras import backend as K


# history = pd.DataFrame(model.history.history)
# history.to_csv(os.path.join(args.output, 'history.csv'))
def get_tversky_loss(alpha=0.5, smooth=1e-6):
    """
    Tversky loss function for 3D image segmentation model.

    :param alpha: Weight of false positives (float)
    :param beta: Weight of false negatives (float)
    :param smooth: Smoothing factor (float)
    :return: Tversky loss (float)
    """

    beta = 1 - alpha

    def tversky_loss(y_true, y_pred):
        """
        :param y_true: Ground truth segmentation mask (tensor)
        :param y_pred: Predicted segmentation mask (tensor)
        """

        # Reshape the inputs to 2D arrays
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)

        y_pred = K.round(y_pred)

        # Calculate true positives, false positives, and false negatives
        true_positives = K.sum(y_true * y_pred)
        false_positives = K.sum((1 - y_true) * y_pred)
        false_negatives = K.sum(y_true * (1 - y_pred))

        # Calculate Tversky index
        tversky_index = (true_positives + smooth) / (
            true_positives + alpha * false_positives + beta * false_negatives + smooth
        )

        # Calculate Tversky loss
        return 1 - tversky_index

    return tversky_loss


class ProteinsGridsGenerator(Sequence):
    def __init__(self, proteins_data, batch_size, mol_grid):
        self.proteins_data = proteins_data
        self.mol_grid = mol_grid
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil(len(self.proteins_data) / float(self.batch_size))).astype(
            np.int
        )

    def __getitem__(self, idx):
        batch_data = self.proteins_data[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]

        batch_x = []
        batch_y = []
        for row in batch_data:
            coords, feats = row[0]
            grid = self.mol_grid.setMolAsCoords(coords, feats).transform()
            batch_x.append(grid)

            site_coords = np.asarray(row[1])
            grid = self.mol_grid.setMolAsCoords(site_coords).transform()
            batch_y.append(grid)

        batch_x = np.asarray(batch_x, dtype=np.float32)
        batch_y = np.asarray(batch_y, dtype=np.float32)

        return batch_x, batch_y


mt = ModelTrainer()
# mt.save_training_data()
#!cp /content/train_data.pickle /content/drive/MyDrive/GP/train_data.pickle
mt.train_model()
#!cp /content/weights/2.h5 /content/drive/MyDrive/GP/Results/test_2.h5
