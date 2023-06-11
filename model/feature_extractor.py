from openbabel import pybel
import numpy as np


class FeatureExtractor:
    def __init__(self, atoms_props_feats) -> None:
        """Atoms"""
        self.atoms_props_feats = atoms_props_feats
        pass

    def get_feature(self, protein_mol: pybel.Molecule):
        if not isinstance(protein_mol, pybel.Molecule):
            raise TypeError('mol should be a pybel.Molecule object, got %s '
                            'instead' % type(protein_mol))

        coords = []
        features = []
        heavy_atoms = []

        for i, atom in enumerate(protein_mol):
            # ignore hydrogens and dummy atoms (they have atomicnum set to 0)
            if atom.atomicnum > 1:
                heavy_atoms.append(i)
                coords.append(atom.coords)

                features.append(np.concatenate((
                    self.encode_num(atom.atomicnum),
                    [atom.__getattribute__(prop) for prop in self.atoms_props_feats],
                    [func(atom) for func in self.CALLABLES],
                )))

        coords = np.array(coords, dtype=np.float32)
        features = np.array(features, dtype=np.float32)
        if self.save_molecule_codes:
            features = np.hstack((features,
                                  molcode * np.ones((len(features), 1))))
        features = np.hstack([features,
                              self.find_smarts(protein_mol)[heavy_atoms]])

        if np.isnan(features).any():
            raise RuntimeError('Got NaN when calculating features')

        return coords, features
