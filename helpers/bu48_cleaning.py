from openbabel import pybel, openbabel
import os


def extract_sites(file_path):
    file = open(file_path, "r")
    lines = file.readlines()

    # get lines of site records
    sites_lines = []
    for line in lines:
        if line[:4] == "SITE":
            sites_lines.append(line)
        elif len(sites_lines) > 0:
            break

    # get sites residues
    sites_residues = []
    for line in sites_lines:
        sites_residues.extend([line[18:27], line[29:38], line[40:49], line[51:60]])

    # get site atoms
    sites_atoms = []
    for line in lines:
        # skip non-atom lines
        if line[:6] != "ATOM  ":
            continue

        # check if atom is from a site res
        atom_res = line[17:26]
        if atom_res not in sites_residues:
            continue

        # site atom found
        atom_coords_str = filter(lambda x: x is not "", line[30:54].split(" "))

        atom_coords = tuple(map(float, atom_coords_str))

        sites_atoms.append(atom_coords)

    # print(len(sites_atoms), sites_atoms)

    mol = openbabel.OBMol()
    for x, y, z in sites_atoms:
        a = mol.NewAtom()
        a.SetVector(float(x), float(y), float(z))

    # convert to pybel molecule and save
    p_mol = pybel.Molecule(mol)
    folder = os.path.dirname(file_path)
    out_path = os.path.join(folder, "site.mol2")
    p_mol.write("mol2", out_path, overwrite=True)


def wirte_sites():
    path = "/home/yusef/development/collage/PUResNet/BU48/bench2"

    for mol_name in os.listdir(path):
        mol_path = os.path.join(path, mol_name, f"pdb{mol_name}.ent")
        print(mol_path)
        extract_sites(mol_path)


import shutil


def delete_empty_sites():
    path = "/home/yusef/development/collage/PUResNet/BU48/bench2"

    for mol_name in os.listdir(path):
        mol_folder = os.path.join(path, mol_name)
        site_path = os.path.join(mol_folder, "site.mol2")

        f = open(site_path, "r")
        f.readline()
        f.readline()
        print(site_path)
        mol_atoms_len = f.readline()
        if " 0 0 0 0 0" in mol_atoms_len:
            shutil.rmtree(mol_folder)
        # print(site_path)
        # extract_sites(site_path)
