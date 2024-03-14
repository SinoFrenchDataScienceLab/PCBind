#!/usr/bin/env python
# coding: utf-8

# # overview
# 
# We start from the raw PDBbind dataset downloaded from http://www.pdbbind.org.cn/download.php
# 
# 1. filter out those unable to process using RDKit.
# 
# 2. Process the protein by only preserving the chains that with at least one atom within 10Å from any atom of the ligand.
# 
# 3. Use p2rank to segment protein into blocks.
# 
# 4. extract protein and ligand features.
# 
# 5. construct the training and test dataset.
# 

# In[1]:


# test = info.query("group == 'test'").reset_index(drop=True)
# test_pdb_list = info.query("group == 'test'").protein_name.unique()


# In[2]:


tankbind_src_folder_path = "../tankbind/"
import sys
sys.path.insert(0, tankbind_src_folder_path)


# In[3]:


import pandas as pd
import numpy as np
from tqdm import tqdm


# # process the raw PDBbind dataset.

# In[4]:


from utils import read_pdbbind_data


# In[5]:


# raw PDBbind dataset could be downloaded from http://www.pdbbind.org.cn/download.php
pre = "/home/jovyan/pdbbind2020/v2020-all-PL"
df_pdb_id = pd.read_csv(f'{pre}/index/INDEX_general_PL_name.2020', sep="  ", comment='#', header=None, names=['pdb', 'year', 'uid', 'd', 'e','f','g','h','i','j','k','l','m','n','o'], engine='python')
df_pdb_id = df_pdb_id[['pdb','uid']]
data = read_pdbbind_data(f'{pre}/index/INDEX_general_PL_data.2020')
data = data.merge(df_pdb_id, on=['pdb'])


# # ligand file should be readable by RDKit.

# In[6]:


from feature_utils import read_mol


# In[7]:


from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
pdb_list = []
probem_list = []
for pdb in tqdm(data.pdb):
    sdf_fileName = f"{pre}/{pdb}/{pdb}_ligand.sdf"
    mol2_fileName = f"{pre}/{pdb}/{pdb}_ligand.mol2"
    mol, problem = read_mol(sdf_fileName, mol2_fileName)
    if problem:
        probem_list.append(pdb)
        continue
    if pdb=="2r1w":
        continue
    pdb_list.append(pdb)


# In[8]:


data = data.query("pdb in @pdb_list").reset_index(drop=True)


# In[9]:


data.shape


# ### for ease of RMSD evaluation later, we renumber the atom index to be consistent with the smiles

# In[10]:


from feature_utils import write_renumbered_sdf
import os


# In[11]:


toFolder = f"{pre}/renumber_atom_index_same_as_smiles"
os.system(f"mkdir -p {toFolder}")
for pdb in tqdm(pdb_list):
    sdf_fileName = f"{pre}/{pdb}/{pdb}_ligand.sdf"
    mol2_fileName = f"{pre}/{pdb}/{pdb}_ligand.mol2"
    toFile = f"{toFolder}/{pdb}.sdf"
    write_renumbered_sdf(toFile, sdf_fileName, mol2_fileName)


# # process PDBbind proteins, removing extra chains, cutoff 10A

# In[12]:


toFolder = f"{pre}/protein_remove_extra_chains_10A/"
os.system(f"mkdir -p {toFolder}")


# In[13]:


input_ = []
cutoff = 10
for pdb in data.pdb.values:
    pdbFile = f"{pre}/{pdb}/{pdb}_protein.pdb"
    ligandFile = f"{pre}/renumber_atom_index_same_as_smiles/{pdb}.sdf"
    toFile = f"{toFolder}/{pdb}_protein.pdb"
    x = (pdbFile, ligandFile, cutoff, toFile)
    input_.append(x)


# In[14]:


from feature_utils import select_chain_within_cutoff_to_ligand_v2


# In[15]:


import mlcrate as mlc
import os
pool = mlc.SuperPool(64)
pool.pool.restart()
_ = pool.map(select_chain_within_cutoff_to_ligand_v2,input_)
pool.exit()


# In[16]:


# previously, I found that 2r1w has no chain near the ligand.
data = data.query("pdb != '2r1w'").reset_index(drop=True)


# # p2rank segmentation

# In[17]:


p2rank_prediction_folder = f"{pre}/p2rank_protein_remove_extra_chains_10A"
os.system(f"mkdir -p {p2rank_prediction_folder}")
ds = f"{p2rank_prediction_folder}/protein_list.ds"
with open(ds, "w") as out:
    for pdb in data.pdb.values:
        out.write(f"../protein_remove_extra_chains_10A/{pdb}_protein.pdb\n")


# In[18]:


# # takes about 30 minutes.
# p2rank = "bash /home/jovyan/TankBind/p2rank_2.3/prank"
# cmd = f"{p2rank} predict {ds} -o {p2rank_prediction_folder}/p2rank -threads 16"
# os.system(cmd)


# In[19]:


data.to_csv(f"{pre}/data.csv")


# In[20]:


pdb_list = data.pdb.values


# In[21]:


tankbind_data_path = f"{pre}/tankbind_data"
name_list = pdb_list
d_list = []

for name in tqdm(name_list):
    p2rankFile = f"{pre}/p2rank_protein_remove_extra_chains_10A/p2rank/{name}_protein.pdb_predictions.csv"
    d = pd.read_csv(p2rankFile)
    d.columns = d.columns.str.strip()
    d_list.append(d.assign(name=name))
d = pd.concat(d_list).reset_index(drop=True)
d.reset_index(drop=True).to_feather(f"{tankbind_data_path}/p2rank_result.feather")


# In[22]:


d = pd.read_feather(f"{tankbind_data_path}/p2rank_result.feather")


# In[23]:


pockets_dict = {}
for name in tqdm(name_list):
    pockets_dict[name] = d[d.name == name].reset_index(drop=True)


# In[ ]:





# # protein feature

# In[24]:


from feature_utils import get_protein_feature


# In[25]:


input_ = []
protein_embedding_folder = f"{tankbind_data_path}/gvp_protein_embedding"
os.system(f"mkdir -p {protein_embedding_folder}")
for pdb in pdb_list:
    proteinFile = f"{pre}/protein_remove_extra_chains_10A/{pdb}_protein.pdb"
    toFile = f"{protein_embedding_folder}/{pdb}.pt"
    x = (pdb, proteinFile, toFile)
    input_.append(x)


# In[26]:


from Bio.PDB import PDBParser
from feature_utils import get_clean_res_list
import torch
torch.set_num_threads(1)

def batch_run(x):
    protein_dict = {}
    pdb, proteinFile, toFile = x
    parser = PDBParser(QUIET=True)
    s = parser.get_structure(pdb, proteinFile)
    res_list = get_clean_res_list(s.get_residues(), verbose=False, ensure_ca_exist=True)
    protein_dict[pdb] = get_protein_feature(res_list)
    torch.save(protein_dict, toFile)


# In[27]:


import mlcrate as mlc
import os
pool = mlc.SuperPool(64)
pool.pool.restart()
_ = pool.map(batch_run,input_)
pool.exit()


# In[1]:


import torch
protein_dict = {}
for pdb in tqdm(pdb_list):
    protein_dict.update(torch.load(f"{protein_embedding_folder}/{pdb}.pt"))


# In[ ]:





# # Compound Features

# In[ ]:


from feature_utils import extract_torchdrug_feature_from_mol
compound_dict = {}
skip_pdb_list = []
for pdb in tqdm(pdb_list):
    mol, _ = read_mol(f"{pre}/renumber_atom_index_same_as_smiles/{pdb}.sdf", None)
    # extract features from sdf.
    try:
        compound_dict[pdb] = extract_torchdrug_feature_from_mol(mol, has_LAS_mask=True)  # self-dock set has_LAS_mask to true
    except Exception as e:
        print(e)
        skip_pdb_list.append(pdb)
        print(pdb)


# In[ ]:


torch.save(compound_dict, f"{tankbind_data_path}/compound_torchdrug_features.pt")


# In[ ]:


skip_pdb_list


# In[ ]:


data = data.query("pdb not in @skip_pdb_list").reset_index(drop=True)


# # construct dataset.

# In[ ]:


# we use the time-split defined in EquiBind paper.
# https://github.com/HannesStark/EquiBind/tree/main/data
valid = np.loadtxt("/home/jovyan/TankBind/equbind/timesplit_no_lig_overlap_val", dtype=str)
test = np.loadtxt("/home/jovyan/TankBind/equbind/timesplit_test", dtype=str)
def assign_group(pdb, valid=valid, test=test):
    if pdb in valid:
        return 'valid'
    if pdb in test:
        return 'test'
    return 'train'

data['group'] = data.pdb.map(assign_group)


# In[ ]:


data.value_counts("group")


# In[ ]:


data['name'] = data['pdb']


# In[ ]:


info = []
err_pdb_list=[]
for i, line in tqdm(data.iterrows(), total=data.shape[0]):
    pdb = line['pdb']
    uid = line['uid']
    # smiles = line['smiles']
    smiles = ""
    affinity = line['affinity']
    group = line['group']

    compound_name = line['name']
    protein_name = line['name']
    try:
        pocket = pockets_dict[pdb].head(10)
    except:
        err_pdb_list.append(pdb)
        continue
    pocket.columns = pocket.columns.str.strip()
    pocket_coms = pocket[['center_x', 'center_y', 'center_z']].values
    # native block.
    info.append([protein_name, compound_name, pdb, smiles, affinity, uid, None, True, False, group])
    # protein center as a block.
    protein_com = protein_dict[protein_name][0].numpy().mean(axis=0).astype(float).reshape(1, 3)
    info.append([protein_name, compound_name, pdb+"_c", smiles, affinity, uid, protein_com, False, False, group])
    
    for idx, pocket_line in pocket.iterrows():
        pdb_idx = f"{pdb}_{idx}"
        info.append([protein_name, compound_name, pdb_idx, smiles, affinity, uid, pocket_coms[idx].reshape(1, 3), False, False, group])
info = pd.DataFrame(info, columns=['protein_name', 'compound_name', 'pdb', 'smiles', 'affinity', 'uid', 'pocket_com', 
                                   'use_compound_com', 'use_whole_protein',
                                  'group'])
print(len(info))


# In[ ]:


len(err_pdb_list)


# In[ ]:


len(pockets_dict)


# In[ ]:


info.shape


# In[ ]:


from data import TankBindDataSet
import os


# In[ ]:


toFilePre = f"{pre}/dataset"
os.system(f"mkdir -p {toFilePre}")
dataset = TankBindDataSet(toFilePre, data=info, protein_dict=protein_dict, compound_dict=compound_dict)


# In[ ]:


dataset = TankBindDataSet(toFilePre)


# In[ ]:


t = []
data = dataset.data
pre_pdb = None
for i, line in tqdm(data.iterrows(), total=data.shape[0]):
    pdb = line['compound_name']
    d = dataset[i]
    p_length = d['node_xyz'].shape[0]
    c_length = d['coords'].shape[0]
    y_length = d['y'].shape[0]
    num_contact = (d.y > 0).sum()
    t.append([i, pdb, p_length, c_length, y_length, num_contact])



# In[ ]:


# data = data.drop(['p_length', 'c_length', 'y_length', 'num_contact'], axis=1)


# In[ ]:


t = pd.DataFrame(t, columns=['index', 'pdb' ,'p_length', 'c_length', 'y_length', 'num_contact'])
t['num_contact'] = t['num_contact'].apply(lambda x: x.item())


# In[ ]:


data = pd.concat([data, t[['p_length', 'c_length', 'y_length', 'num_contact']]], axis=1)


# In[ ]:


native_num_contact = data.query("use_compound_com").set_index("protein_name")['num_contact'].to_dict()
data['native_num_contact'] = data.protein_name.map(native_num_contact)
# data['fract_of_native_contact'] = data['num_contact'] / data['native_num_contact']


# In[ ]:


torch.save(data, f"{toFilePre}/processed/data.pt")


# In[ ]:


import torch
info = torch.load(f"{toFilePre}/processed/data.pt")


# In[ ]:


test = info.query("group == 'test'").reset_index(drop=True)
test_pdb_list = info.query("group == 'test'").protein_name.unique()


# In[ ]:


test = info.query("group == 'test'").reset_index(drop=True)
test_pdb_list = info.query("group == 'test'").protein_name.unique()


# In[ ]:


subset_protein_dict = {}
for pdb in tqdm(test_pdb_list):
    subset_protein_dict[pdb] = protein_dict[pdb]


# In[ ]:


subset_compound_dict = {}
for pdb in tqdm(test_pdb_list):
    subset_compound_dict[pdb] = compound_dict[pdb]


# In[ ]:


toFilePre = f"{pre}/test_dataset"
os.system(f"mkdir -p {toFilePre}")
dataset = TankBindDataSet(toFilePre, data=test, protein_dict=subset_protein_dict, compound_dict=subset_compound_dict)


# In[ ]:


def canonical_smiles(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))

