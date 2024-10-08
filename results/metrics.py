import numpy as np
from openbabel import pybel
import openbabel as ob
import math

pybel.ob.obErrorLog.StopLogging()


class BindingPocket:
    def __init__(self, occupied_cells):
        self.occupied_cells = occupied_cells


def create_3d_grid(pocket, resolution):
    min_coords = np.min(pocket.occupied_cells, axis=0)
    shifted_coords = pocket.occupied_cells - min_coords

    max_coords = np.max(shifted_coords, axis=0)
    grid_shape = np.ceil((max_coords) / resolution).astype(int) + 1
    grid = np.zeros(grid_shape, dtype=bool)

    for cell in shifted_coords:
        cell_idx = np.floor(cell / resolution).astype(int)
        grid[tuple(cell_idx)] = True
    return grid


def intersection_over_union(pocket1, pocket2, resolution):
    grid1 = create_3d_grid(pocket1, resolution)
    grid2 = create_3d_grid(pocket2, resolution)

    common_grid_shape = np.maximum(grid1.shape, grid2.shape)

    grid1_padded = np.pad(
        grid1,
        [(0, int(common_grid_shape[i] - grid1.shape[i])) for i in range(3)],
        mode="constant",
        constant_values=False,
    )

    grid2_padded = np.pad(
        grid2,
        [(0, int(common_grid_shape[i] - grid2.shape[i])) for i in range(3)],
        mode="constant",
        constant_values=False,
    )

    intersection_grid = np.logical_and(grid1_padded, grid2_padded)
    union_grid = np.logical_or(grid1_padded, grid2_padded)

    intersection_volume = np.sum(intersection_grid) * resolution**3
    union_volume = np.sum(union_grid) * resolution**3

    return intersection_volume / union_volume


def intersection_over_lig(lig, pocket, resolution):
    grid1 = create_3d_grid(lig, resolution)
    grid2 = create_3d_grid(pocket, resolution)

    common_grid_shape = np.maximum(grid1.shape, grid2.shape)

    grid1_padded = np.pad(
        grid1,
        [(0, int(common_grid_shape[i] - grid1.shape[i])) for i in range(3)],
        mode="constant",
        constant_values=False,
    )

    grid2_padded = np.pad(
        grid2,
        [(0, int(common_grid_shape[i] - grid2.shape[i])) for i in range(3)],
        mode="constant",
        constant_values=False,
    )

    intersection_grid = np.logical_and(grid1_padded, grid2_padded)

    intersection_volume = np.sum(intersection_grid) * resolution**3
    lig_volume = np.sum(grid1_padded) * resolution**3

    return intersection_volume / lig_volume


def coordinates(pdb_file):
    molecule = next(pybel.readfile(pdb_file.split(".")[-1], pdb_file))
    ligand_coords = [atom.coords for atom in molecule.atoms]
    return np.array(ligand_coords)


def get_DVO(pkt1, pkt2, resolution=1):
    pocket1_coords = coordinates(pkt1)
    pocket2_coords = coordinates(pkt2)
    pocket1 = BindingPocket(pocket1_coords)
    pocket2 = BindingPocket(pocket2_coords)
    return intersection_over_union(pocket1, pocket2, resolution)


def get_PLI(lig, pkt, resolution=1):
    lig_coords = coordinates(lig)
    pkt_coords = coordinates(pkt)
    ligand = BindingPocket(lig_coords)
    pocket = BindingPocket(pkt_coords)
    return intersection_over_lig(ligand, pocket, resolution)


def get_DCC(lig, pkt):
    lig_coords = coordinates(lig)
    pkt_coords = coordinates(pkt)
    center_lig = [0] * 3
    center_pkt = [0] * 3
    if len(pkt_coords) == 0:
        return 35
    for i in lig_coords:
        center_lig[0] += i[0]
        center_lig[1] += i[1]
        center_lig[2] += i[2]
    center_lig[0] /= len(lig_coords)
    center_lig[1] /= len(lig_coords)
    center_lig[2] /= len(lig_coords)

    for i in pkt_coords:
        center_pkt[0] += i[0]
        center_pkt[1] += i[1]
        center_pkt[2] += i[2]
    center_pkt[0] /= len(pkt_coords)
    center_pkt[1] /= len(pkt_coords)
    center_pkt[2] /= len(pkt_coords)

    dist = (
        math.pow((center_pkt[0] - center_lig[0]), 2)
        + math.pow((center_pkt[1] - center_lig[1]), 2)
        + math.pow((center_pkt[2] - center_lig[2]), 2)
    )
    dist = math.sqrt(dist)

    return dist


site = "/home/yusef/development/collage/PUResNet/BU48/bench2/1gcg/site.mol2"
pred = "/home/yusef/development/collage/PUResNet/BU48/bench2/1gcg/pocket0.mol2"

print(get_DCC(site, pred))
