import numpy as np
import torch

from util import *
from residue import Residue



PK_MODS = {
    "ASP": 3.79,
    "CTR": 2.90,
    "CYS": 8.67,
    "GLU": 4.20,
    "HIS": 6.74,
    "LYS": 10.46,
    "NTR": 7.99,
    "TYR": 9.59,
}


class Protein:
    def __init__(self, pdb_path, pdb_f, pssm_path):

        self.pdb_f = pdb_path+pdb_f
        self.pssm_path = pssm_path
        self.pssm_f = pdb_f.replace('.pdb','')

        self.pssm = {}

        self.tit_residues = {}
        self.all_residues = {}

        self.read_pdb()

    def iter_atoms(self):
        for residue in self.iter_residues():
            for atom in residue.iter_atoms():
                yield atom

    def iter_residues(self, titrable_only=False):
        if titrable_only:
            residues = self.tit_residues
        else:
            residues = self.all_residues
        for chain in sorted(residues.keys()):
            chain_residues = residues[chain]
            for resnumb in sorted(chain_residues.keys()):
                residue = chain_residues[resnumb]
                yield residue

    def read_pdb(self):
        """Removes all atoms from pdb_f that are not Nitrogens, Sulfurs, or Oxygens"""
        tit_aas = PK_MODS.keys()

        with open(self.pdb_f) as f:
            count = 0
            for line in f:
                if line.startswith("ATOM "):
                    line_cols = self.read_pdb_line(line)
                    (
                        aname,
                        anumb,
                        b,
                        resname,
                        chain,
                        resnumb,
                        x,
                        y,
                        z,
                        icode,
                    ) = line_cols

                    if b not in (" ", "A") or icode != " ":
                        continue

                    if aname[0] not in "NOS" and aname != 'CA':
                        continue

                    if chain !='A':
                        continue

                    if chain not in self.all_residues.keys():
                        self.all_residues[chain] = {}

                    if resnumb not in self.all_residues[chain].keys():
                        count += 1
                        new_res = Residue(self, chain, resname, resnumb, count)
                        self.all_residues[chain][resnumb] = new_res
                    else:
                        new_res = self.all_residues[chain][resnumb]

                    if aname == 'CA':
                        self.all_residues[chain][resnumb].ca=resnumb
                        self.all_residues[chain][resnumb].ca_xyz=(x,y,z)
                        continue
                    #aname, anumb, x, y, z, resnumb, residue
                    new_res.add_atom(aname, anumb, x, y, z, resnumb, resname)

                    if resname in tit_aas:
                        if chain not in self.tit_residues.keys():
                            self.tit_residues[chain] = {}

                        if resnumb not in self.tit_residues[chain]:
                            self.tit_residues[chain][resnumb] = new_res

    def align_pka(self, pka, file_name):
        for residue in self.iter_residues(titrable_only=True):
            name=file_name+residue.chain+str(residue.resnumb)+residue.resname
            if name in pka.keys():
                residue.pka=float(pka[name])

    def read_pssm(self):
        file_name = self.pssm_f
        try:
            file_name=file_name.lower()
        except:
            pass

        if file_name not in os.listdir(self.pssm_path) and self.pssm_f not in os.listdir(self.pssm_path):
            self.pssm = -1
            return -1

        try:
            fd = open(self.pssm_path + file_name)
        except:
            fd = open(self.pssm_path + self.pssm_f)
        file = fd.readlines()
        fd.close()
        for i in range(len(file[3:])):
            if file[3+i] == '\n':
                break
            temp = file[3+i][:-1].split()
            self.pssm[str(temp[0])] = np.array(temp[2:22]).astype(int)
        #print(self.pssm)
        return 0

    def align_pssm(self, length=20):
        if self.pssm==-1:
            return -1
        for residues in self.all_residues.values():
            for residue, v in residues.items():
                v.pssm = self.pssm[str(v.number)]
        for residues in self.all_residues.values():
            for residue, v in residues.items():
                if len(v.pssm)<length:
                    v.pssm = [0]*length

    @staticmethod
    def read_pdb_line(line: str) -> tuple:
        aname = line[12:16].strip()
        anumb = int(line[5:11].strip())
        b = line[16]
        resname = line[17:21].strip()
        chain = line[21]
        resnumb = int(line[22:26])
        icode = line[26]
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
        return (aname, anumb, b, resname, chain, resnumb, x, y, z, icode)

    def apply_cutoff(self, cutoff_dist=15):

        # TODO speed up by doing distance matrix a priori in cython

        for residue in self.iter_residues(titrable_only=True):
            if residue.pka ==-1:
                continue
            residue.calc_cutoff_atoms(cutoff_dist)
            residue.encode_input()

    def creat_graph(self, max_distance=15, max_stack=250):
        for residue in self.iter_residues(titrable_only=True):
            xyz = []
            edge_index1=[]
            edge_index2=[]

            chain=residue.chain
            neighbor = residue.resnumb_sorted[:max_stack]

            index={}
            index_reverse={}
            for num in range(len(neighbor)):
                resnum = neighbor[num]
                xyz.append(self.all_residues[chain][resnum].ca_xyz)
                if str(resnum) not in index.keys():
                    index[str(resnum)] = num
                index_reverse[str(num)] = resnum

            if len(xyz)==0:
                continue

            pair_distance = cdist(xyz, xyz, 'euclidean')
            pair_distance[pair_distance > max_distance] = 0

            for x in range(len(pair_distance)):
                for y in range(len(pair_distance)):
                    if pair_distance[x][y] != 0:
                        edge_index1.append(index[str(index_reverse[str(x)])])
                        edge_index2.append(index[str(index_reverse[str(y)])])
                        residue.edge_weight.append(1 / pair_distance[x][y])

            residue.edge_index = [edge_index1, edge_index2]


    def predict_pkas(self, model, device, loss_func, optimizer):
        predictions_tr = torch.Tensor()
        labels_tr = torch.Tensor()

        for residue in self.iter_residues(titrable_only=True):
            if len(residue.input_layer)==0 or len(residue.edge_weight)==0 or len(residue.edge_index)==0:
                continue
            x = residue.input_layer
            edge_weight = torch.tensor(residue.edge_weight, dtype=torch.float)
            edge_index = torch.tensor(residue.edge_index, dtype=torch.int64)
            pssm = torch.tensor(residue.pssm, dtype=torch.int64)
            pka = torch.tensor([residue.pka]).float()


            if len(x)==0 or len(edge_index)==0 or len(edge_weight)==0:
                continue

            optimizer.zero_grad()

            dpks = model(x.to(device), edge_index.to(device), edge_weight.to(device), residue.ohe_resname.to(device), pssm.to(device))

            predictions_tr = torch.cat((predictions_tr, dpks), 0)
            labels_tr = torch.cat((labels_tr, pka), 0)

        return predictions_tr, labels_tr

    def test(self,model, device):
        result=[]
        for residue in self.iter_residues(titrable_only=True):
            if residue.pka == -1:
                continue

            x = residue.input_layer
            edge_weight = torch.tensor(residue.edge_weight, dtype=torch.float)
            edge_index = torch.tensor(residue.edge_index, dtype=torch.int64)
            pssm = torch.tensor(residue.pssm, dtype=torch.int64)
            x = torch.tensor(x)
            pka = torch.tensor(residue.pka).float()
            dpks = model(x.to(device), edge_index.to(device), edge_weight.to(device), residue.ohe_resname.to(device),pssm.to(device))
            #print(pka,dpks)

            result.append([dpks.detach().numpy(),pka.detach().numpy()])
        return result