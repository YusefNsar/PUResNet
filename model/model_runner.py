from feature_extractor import FeatureExtractor
from openbabel import pybel

class ModelRunner:
    def __init__(self):
      self.__fe = FeatureExtractor()
      pass

    def predictBindingSites(self, mol: pybel.Molecule):
        
        # get mol features
        coords, features = self.__fe.get_feature(mol)

        # make input 4d points
        # run on prediction model
        # transform 4d points back to features
        # transform features back to a mol object
