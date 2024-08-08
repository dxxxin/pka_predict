import torch

from atom import Atom
from util import *


AA_ATOMS = {
    "ASP": ["OD1", "OD2"],
    "CTR": ["OD1", "OXT"],
    "CYS": ["SG"],
    "GLU": ["OE1", "OE2"],
    "HIS": ["ND1", "NE2"],
    "LYS": ["NZ"],
    "NTR": ["N"],
    "TYR": ["OH"],
}

ATOM_OHE = [
    "N",
    "NZ_LYS",
    "NH_ARG",
    "O_AMIDE",
    "OH_TYR",
    "NE2_HIS",
    "SD_MET",
    "SG_CYS",
    "OG_SER",
    "O",
    "NE1_TRP",
    "OG1_THR",
    "O_COOH",
    "NE_ARG",
    "N_AMIDE",
    "ND1_HIS",
    "S_SEC",
    "O_SEC"
]

RES_OHE = ["CTR", "CYS", "TYR", "GLU", "HIS", "ASP", "LYS", "NTR"]


class Residue:
    def __init__(self, protein, chain, resname, resnumb, number):

        self.protein = protein
        self.chain = chain
        self.resname = resname
        self.resnumb = resnumb
        self.number = number

        self.atoms = {}

        self.ca = ''
        self.ca_xyz = ()
        self.ca_dists = []

        self.pssm = []

        self.env_tensor_size = 250
        self.ohe_size = len(ATOM_OHE)

        self.graph_features = torch.zeros(
            self.env_tensor_size, self.ohe_size, dtype=torch.float32
        )
        self.edge_index=[]
        self.edge_weight=[]
        self.features=[]


        self.env_dists = []
        self.env_anames = []
        self.env_resnames = []
        self.env_resnum=[]

        self.env_oheclasses = []
        self.dists_sorted = []
        self.atoms_sorted = []
        self.resnumb_sorted = []


        self.env_tensor = torch.zeros(
            self.env_tensor_size, 1, self.ohe_size, dtype=torch.float32
        )
        self.ohe_resname = torch.zeros(1, 1, len(RES_OHE), dtype=torch.float32)
        #self.input_layer = []
        self.input_layer = torch.zeros(
            self.env_tensor_size, self.ohe_size, dtype=torch.float32
        )
        self.pka=-1

    def add_atom(self, aname, anumb, x, y, z, resnumb, resname):
        atom = Atom(aname, anumb, x, y, z, resnumb, resname)
        self.atoms[anumb] = atom

    def iter_atoms(self):
        for atom in self.atoms.values():
            yield atom

    def calc_cutoff_atoms(self, cutoff_dist: float):
        cutoff_sq = cutoff_dist ** 2

        site_ref_atoms = []
        for atom in self.iter_atoms():
            if atom.aname in AA_ATOMS[self.resname]:
                site_ref_atoms.append(atom)

        for atom in self.protein.iter_atoms():
            min_dist = 99999
            for residue_atom in site_ref_atoms:
                if atom in self.atoms.values():
                    continue

                sqdist = residue_atom.calc_sqdist(atom)

                if sqdist < min_dist:
                    min_dist = sqdist

            if min_dist < cutoff_sq:
                self.env_anames.append(atom.aname)
                self.env_resnames.append(atom.resname)
                self.env_resnum.append(atom.resnumb)
                self.env_dists.append(min_dist ** 0.5)

    def encode_atoms(self):
        for aname, resname in zip(self.env_anames, self.env_resnames):
            aname = aname.strip()
            resname = resname.strip()

            if aname in AA_ATOMS["CTR"]:
                resname = "CTR"

            if aname in ("N", "O"):
                if resname != "NTR":
                    new_aname = aname
                else:
                    new_aname = "NZ_LYS"
            elif resname in ("CTR", "ASP", "GLU"):
                new_aname = "O_COOH"
            elif resname in ("GLN", "ASN"):
                new_aname = f"{aname[0]}_AMIDE"
            elif resname == "ARG" and aname in ("NH1", "NH2"):
                new_aname = "NH_ARG"
            else:
                new_aname = f"{aname}_{resname}"

            self.env_oheclasses.append(new_aname)

    def input_sort(self):
        for dist, atom, numb in sorted(zip(self.env_dists, self.env_oheclasses, self.env_resnum)):
            self.dists_sorted.append(dist)
            self.atoms_sorted.append(atom)
            self.resnumb_sorted.append(numb)

    def apply_ohe(self):
        for li, letter in enumerate(self.atoms_sorted):
            if li == self.env_tensor_size:
                break

            ohe_index = ATOM_OHE.index(letter)

            r = self.dists_sorted[li]

            self.env_tensor[li][0][ohe_index] = 1 / (r ** 2)
            #print(self.protein.all_residues['A'][self.resnumb_sorted[li]].number,self.protein.all_residues['A'][self.resnumb_sorted[li]].pssm)
            #pssm = torch.Tensor(self.protein.all_residues['A'][self.resnumb_sorted[li]].pssm * (1 / (r ** 2)))
            #self.input_layer[li]=torch.cat((self.env_tensor[li][0],pssm), axis=0)
            self.input_layer[li] =self.env_tensor[li][0]

    def encode_input(self):
        self.encode_atoms()
        self.input_sort()
        self.apply_ohe()


