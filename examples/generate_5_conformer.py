from rdkit.Chem import AllChem, AddHs
from rdkit import Chem
import torch
tankbind_src_folder_path = "../tankbind/"
import sys
sys.path.insert(0, tankbind_src_folder_path)
import warnings
import pandas as pd
import numpy as np
from tqdm import tqdm
pre = "/home/jovyan/data"

def generate_5_conformer(mol):
    candicate_conformation_num = 5
    ps = AllChem.ETKDGv2()
    ps.numThreads = 10
    candicate_conformation_id_list = list(AllChem.EmbedMultipleConfs(mol,numConfs=candicate_conformation_num, params = ps))
    if len(candicate_conformation_id_list) != 0:
        for i in candicate_conformation_id_list:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=500, confId=i)
    return candicate_conformation_id_list

def inference_preprocessing(mol, pdb_id):
    ligands_list = []
    # print('Reading molecules and generating local structures with RDKit')
    try:
        mol.RemoveAllConformers()
        mol = AddHs(mol)
        candicate_conformation_id_list = generate_5_conformer(mol)
        mol = Chem.RemoveHs(mol)
        return [mol.GetConformer(i).GetPositions() for i in candicate_conformation_id_list]
    except Exception as e:
        print('Failed to read molecule ', pdb_id, ' We are skipping it. The reason is the exception: ', e)
        return ligands_list

from feature_utils import read_mol

t = []
data = torch.load('/home/jovyan/torsional/dataset-all/torsional/train_dataset/processed/data.pt')
pdb_list = data.compound_name.unique().tolist()
error_pdb_list = [] #一个小分子构象都生成不了
lack_pdb_list = [] #无法生成5个小分子构象，但可以生成构象
for pdb in tqdm(pdb_list):
    mol, _ = read_mol(f"{pre}/renumber_atom_index_same_as_smiles/{pdb}.sdf", None)
    candicate_conf_pos = inference_preprocessing(mol, pdb)
    if len(candicate_conf_pos) == 0:
        error_pdb_list.append(pdb)
    elif len(candicate_conf_pos) < 5:
        for i in range(5 - len(candicate_conf_pos)):
            candicate_conf_pos.append(None)
        lack_pdb_list.append(pdb)
    t.append([pdb, candicate_conf_pos])
    # from IPython import embed
    # embed()


torch.save(t, 'candicate_conf_pos_v1.pt')
torch.save(error_pdb_list, 'candicate_conf_pos_error_v1.pt')
torch.save(lack_pdb_list, 'candicate_conf_pos_lack_v1.pt')

#第一次没成功的分子处理：不使用AddHs
pdb_list = error_pdb_list
error_pdb_list_1 = [] #一个小分子构象都生成不了
lack_pdb_list_1 = [] #无法生成5个小分子构象，但可以生成构象
for pdb in tqdm(pdb_list):
    mol, _ = read_mol(f"{pre}/renumber_atom_index_same_as_smiles/{pdb}.sdf", None)
    mol=AllChem.RemoveHs(mol)
    AllChem.MolToSmiles(mol)
    candicate_conformation_num = 5
    ps = AllChem.ETKDGv2()
    ps.numThreads = 5
    try:
        candicate_conformation_id_list = list(AllChem.EmbedMultipleConfs(mol,numConfs=candicate_conformation_num, params = ps))
        candicate_conf_pos = [mol.GetConformer(i).GetPositions() for i in candicate_conformation_id_list]
    except Exception as e:
        print('Failed to read molecule ', pdb, ' We are skipping it. The reason is the exception: ', e)
        candicate_conf_pos = []
    if len(candicate_conf_pos) == 0:
        error_pdb_list_1.append(pdb)
    elif len(candicate_conf_pos) < 5:
        for i in range(5 - len(candicate_conf_pos)):
            candicate_conf_pos.append(None)
        lack_pdb_list_1.append(pdb)
    t.append([pdb, candicate_conf_pos])

torch.save(t, 'candicate_conf_pos_v2.pt')
torch.save(error_pdb_list_1, 'candicate_conf_pos_error_v2.pt')
torch.save(lack_pdb_list_1, 'candicate_conf_pos_lack_v2.pt')

#第二次没成功的分子处理：使用useRandomCoords
pdb_list = error_pdb_list_1
error_pdb_list_2 = [] #一个小分子构象都生成不了
lack_pdb_list_2 = [] #无法生成5个小分子构象，但可以生成构象
for pdb in tqdm(pdb_list):
    mol, _ = read_mol(f"{pre}/renumber_atom_index_same_as_smiles/{pdb}.sdf", None)
    try:
        mol=AllChem.AddHs(mol)
        AllChem.EmbedMolecule(mol,useRandomCoords=True)
        AllChem.MMFFOptimizeMolecule(mol)
        mol=AllChem.RemoveHs(mol)
        candicate_conf_pos = [mol.GetConformer().GetPositions()]
    except Exception as e:
        print('Failed to read molecule ', pdb, ' We are skipping it. The reason is the exception: ', e)
        candicate_conf_pos = []
    if len(candicate_conf_pos) == 0:
        candicate_conf_pos = [None] * 5
        error_pdb_list_2.append(pdb)
    elif len(candicate_conf_pos) < 5:
        for i in range(5 - len(candicate_conf_pos)):
            candicate_conf_pos.append(None)
        lack_pdb_list_2.append(pdb)
    t.append([pdb, candicate_conf_pos])

torch.save(t, 'candicate_conf_pos_v3.pt')
torch.save(error_pdb_list_2, 'candicate_conf_pos_error_v3.pt')
torch.save(lack_pdb_list_2, 'candicate_conf_pos_lack_v3.pt')

error_pdb_list = error_pdb_list_2

pre = '/home/jovyan/torsional/dataset-all/torsional'
info = pd.read_csv(f"{pre}/test_dataset/apr23_testset_pdbbind_gvp_pocket_radius20_info.csv", index_col=0)
info_va = pd.read_csv(f"{pre}/valid_dataset/apr23_validset_pdbbind_gvp_pocket_radius20_info.csv", index_col=0)
info =  info.drop(index = info[info['compound_name'].isin(error_pdb_list)].index.tolist())
info_va =  info_va.drop(index = info_va[info_va['compound_name'].isin(error_pdb_list)].index.tolist())
info.to_csv(f"{pre}/test_dataset/apr23_testset_pdbbind_gvp_pocket_radius20_info.csv")
info_va.to_csv(f"{pre}/valid_dataset/apr23_validset_pdbbind_gvp_pocket_radius20_info.csv")

t = pd.DataFrame(t, columns=['compound_name' ,'candicate_conf_pos'])
data_test = torch.load('/home/jovyan/torsional/dataset-all/torsional/test_dataset/processed/data.pt')
data_valid = torch.load('/home/jovyan/torsional/dataset-all/torsional/valid_dataset/processed/data.pt')
data_train = torch.load('/home/jovyan/torsional/dataset-all/torsional/train_dataset/processed/data.pt')
print('len(data_train):',len(data_train), 'len(data_valid):',len(data_valid), 'len(data_test):',len(data_test))
data_test =  data_test.drop(index = data_test[data_test['compound_name'].isin(error_pdb_list)].index.tolist())
data_valid =  data_valid.drop(index = data_valid[data_valid['compound_name'].isin(error_pdb_list)].index.tolist())
data_train =  data_train.drop(index = data_train[data_train['compound_name'].isin(error_pdb_list)].index.tolist())
print('len(data_train):',len(data_train), 'len(data_valid):',len(data_valid), 'len(data_test):',len(data_test))

data_mol = []
from spyrmsd import rmsd, molecule
pdb_list = data_train.compound_name.unique().tolist()
for pdb in tqdm(pdb_list):
    mol, _ = read_mol(f"{pre}/renumber_atom_index_same_as_smiles/{pdb}.sdf", None)
    try:
        mol = molecule.Molecule.from_rdkit(mol)
        data_mol.append([pdb, mol.atomicnums, mol.adjacency_matrix])
    except Exception as e:
        print('Failed to generate ', pdb, ' We are skipping it. The reason is the exception: ', e)
data_mol = pd.DataFrame(data_mol, columns=['compound_name' ,'atomicnums', 'adjacency_matrix'])
data_train = pd.merge(data_train, data_mol,how="left" ,on='compound_name' )
data_test = pd.merge(data_test, data_mol,how="left" ,on='compound_name' )
data_valid = pd.merge(data_valid, data_mol,how="left" ,on='compound_name' )

result_train = pd.merge(data_train, t,how="left" ,on='compound_name' )
result_test = pd.merge(data_test, t,how="left" ,on='compound_name' )
result_valid = pd.merge(data_valid, t,how="left" ,on='compound_name' )
data_test = result_test
data_valid = result_valid
data_train = result_train
torch.save(data_train, '/home/jovyan/torsional/dataset-all/torsional/train_dataset/processed/data.pt')
torch.save(data_test, '/home/jovyan/torsional/dataset-all/torsional/test_dataset/processed/data.pt')
torch.save(data_valid, '/home/jovyan/torsional/dataset-all/torsional/valid_dataset/processed/data.pt')