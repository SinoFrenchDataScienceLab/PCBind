tankbind_src_folder_path = "../tankbind/"
import sys
sys.path.insert(0, tankbind_src_folder_path)

import json
from Bio.PDB import PDBParser
import rdkit.Chem as Chem
from feature_utils import get_protein_feature
from feature_utils import extract_torchdrug_feature_from_mol
import os
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)
from data import TankBind_prediction
import logging
from torch_geometric.loader import DataLoader
from tqdm import tqdm    # pip install tqdm if fails.
from model import get_model
from generation_utils import get_LAS_distance_constraint_mask, get_info_pred_distance, write_with_new_coords

#prepare dir
pre = "./7kac/"
pdir = f"{pre}/PDBs/"
pdb = '7kac'
os.system(f"mkdir -p {pdir}")


parser = PDBParser(QUIET=True)



#protein feature
proteinFile = f"{pre}/{pdb}_protein.pdb"
parser = PDBParser(QUIET=True)
s = parser.get_structure("x", proteinFile)
res_list = list(s.get_residues())
protein_dict = {}
protein_dict[pdb] = get_protein_feature(res_list)


#compound feature
ligandName="%s_ligand"%(pdb)
compound_dict = {}
ligandFile=f"{pre}/{ligandName}.sdf"
for mol in Chem.SDMolSupplier(ligandFile):
    if mol==None:
        raise Exception
    compound_dict[pdb+f"_{ligandName}"+"_rdkit"] = extract_torchdrug_feature_from_mol(mol, has_LAS_mask=True)


#p2r
pdb_list = [pdb]
ds = f"{pre}/protein_list.ds"
with open(ds, "w") as out:
    for pdb in pdb_list:
        out.write(f"./{pdb}_protein.pdb\n")

p2rank = "bash /home/jovyan/TankBind/p2rank_2.3/prank"
cmd = f"{p2rank} predict {ds} -o {pre}/p2rank -threads 1"
os.system(cmd)

info = []
for pdb in pdb_list:
    for compound_name in list(compound_dict.keys()):
        # use protein center as the block center.
        com = ",".join([str(a.round(3)) for a in protein_dict[pdb][0].mean(axis=0).numpy()])
        info.append([pdb, compound_name, "protein_center", com])

        p2rankFile = f"{pre}/p2rank/{pdb}_protein.pdb_predictions.csv"
        pocket = pd.read_csv(p2rankFile)
        pocket.columns = pocket.columns.str.strip()
        pocket_coms = pocket[['center_x', 'center_y', 'center_z']].values
        for ith_pocket, com in enumerate(pocket_coms):
            com = ",".join([str(a.round(3)) for a in com])
            info.append([pdb, compound_name, f"pocket_{ith_pocket+1}", com])
info = pd.DataFrame(info, columns=['protein_name', 'compound_name', 'pocket_name', 'pocket_com'])
print(info)





#model choose pocket

dataset_path = f"{pre}/{pdb}_dataset/"
os.system(f"rm -r {dataset_path}")
os.system(f"mkdir -p {dataset_path}")
dataset = TankBind_prediction(dataset_path, data=info, protein_dict=protein_dict, compound_dict=compound_dict)

dataset_path = f"{pre}/{pdb}_dataset/"
dataset = TankBind_prediction(dataset_path)


# from utils import *


batch_size = 5
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device= 'cpu'
logging.basicConfig(level=logging.INFO)
model = get_model(0, logging, device)
# re-dock model
# modelFile = "../saved_models/re_dock.pt"
# self-dock model
model_path="../saved_models/"
model_file_name="self_dock"  #baseline_epoch_149 weighted_contact_loss_and_change_ratio_of_two_loss_epoch_92 weighted_contact_loss_epoch_132
model_file_name="baseline_epoch_149"
model_file_name="weighted_contact_loss_and_change_ratio_of_two_loss_epoch_92"
model_file_name="weighted_contact_loss_epoch_132"


modelFile = "%s/%s.pt"%(model_path,model_file_name)



model.load_state_dict(torch.load(modelFile, map_location=device))
_ = model.eval()

data_loader = DataLoader(dataset, batch_size=batch_size, follow_batch=['x', 'y', 'compound_pair'], shuffle=False, num_workers=8)
affinity_pred_list = []
y_pred_list = []
for data in tqdm(data_loader):
    data = data.to(device)
    y_pred, affinity_pred = model(data)
    affinity_pred_list.append(affinity_pred.detach().cpu())
    for i in range(data.y_batch.max() + 1):
        y_pred_list.append((y_pred[data['y_batch'] == i]).detach().cpu())

affinity_pred_list = torch.cat(affinity_pred_list)


info = dataset.data
info['affinity'] = affinity_pred_list

info.to_csv(f"{pre}/info_with_predicted_affinity.csv")


chosen = info.loc[info.groupby(['protein_name', 'compound_name'],sort=False)['affinity'].agg('idxmax')].reset_index()
print(chosen)


#from predicted interaction distance map to sdf¶

import matplotlib.pyplot as plt





device = 'cpu'
for i, line in chosen.iterrows():
    idx = line['index']
    pocket_name = line['pocket_name']
    compound_name = line['compound_name']
    ligandName = compound_name.split("_")[1]
    coords = dataset[idx].coords.to(device)
    protein_nodes_xyz = dataset[idx].node_xyz.to(device)
    n_compound = coords.shape[0]
    n_protein = protein_nodes_xyz.shape[0]
    y_pred = y_pred_list[idx].reshape(n_protein, n_compound).to(device)
    y_true = dataset[idx].dis_map.reshape(n_protein, n_compound).to(device)


    with open("%s/%s_rmsd_ana.txt"%(pre,model_file_name),"w") as fp_rmsd_out:

        for range_i,range_j in [(0,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,10),(10,11)]:

            y_pred_cutoff=y_pred[(y_true >=range_i) &( y_true<range_j)]
            y_true_cutoff=y_true[(y_true >=range_i) & (y_true<range_j)]
            cutoff_num=y_true_cutoff.shape[0]
            cutoff_rmsd=torch.sqrt(torch.sum((y_pred_cutoff-y_true_cutoff)**2)/cutoff_num)
            print(json.dumps({"cutoff_rmsd>=%s<%s"%(range_i,range_j):float(cutoff_rmsd),
                              "cutoff_num>=%s<%s"%(range_i,range_j):float(cutoff_num)}),
                  file=fp_rmsd_out)

    np.savetxt("%s/%s_%s_y_pred.csv"%(pre,model_file_name,ligandName), y_pred.numpy(), delimiter=",")
    np.savetxt("%s/%s_%s_y_true.csv" % (pre,model_file_name,ligandName),y_true.numpy() , delimiter=",")


    compound_pair_dis_constraint = torch.cdist(coords, coords)
    rdkitMolFile = ligandFile
    mol = Chem.MolFromMolFile(rdkitMolFile)


    LAS_distance_constraint_mask = get_LAS_distance_constraint_mask(mol).bool()
    info = get_info_pred_distance(coords, y_pred, protein_nodes_xyz, compound_pair_dis_constraint,
                                  LAS_distance_constraint_mask=LAS_distance_constraint_mask,
                                  n_repeat=1, show_progress=False)

    result_folder = f'{pre}/{pdb}_result/'
    os.system(f'mkdir -p {result_folder}')
    # toFile = f'{result_folder}/{ligandName}_{pocket_name}_tankbind.sdf'
    toFile = f'{result_folder}/{ligandName}_%s_tankbind.sdf'%(model_file_name)
    # print(toFile)
    new_coords = info.sort_values("loss")['coords'].iloc[0].astype(np.double)
    write_with_new_coords(mol, new_coords, toFile)