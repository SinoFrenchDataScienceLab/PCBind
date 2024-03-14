
from select import select
import torch
from metrics import *
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import scipy.spatial
from torch_geometric.data import Data
from torch_geometric.data import HeteroData
import torch.nn.functional as F
from tqdm import tqdm
import torchmetrics
import math
from torch_geometric.utils import to_networkx, degree
import networkx as nx
import time
def read_pdbbind_data(fileName):
    with open(fileName) as f:
        a = f.readlines()
    info = []
    for line in a:
        if line[0] == '#':
            continue
        lines, ligand = line.split('//')
        pdb, resolution, year, affinity, raw = lines.strip().split('  ')
        ligand = ligand.strip().split('(')[1].split(')')[0]
        # print(lines, ligand)
        info.append([pdb, resolution, year, affinity, raw, ligand])
    info = pd.DataFrame(info, columns=['pdb', 'resolution', 'year', 'affinity', 'raw', 'ligand'])
    info.year = info.year.astype(int)
    info.affinity = info.affinity.astype(float)
    return info

def compute_dis_between_two_vector(a, b):
    return (((a - b)**2).sum())**0.5

def get_protein_edge_features_and_index(protein_edge_index, protein_edge_s, protein_edge_v, keepNode):
    # protein
    input_edge_list = []
    input_protein_edge_feature_idx = []
    new_node_index = np.cumsum(keepNode) - 1
    keepEdge = keepNode[protein_edge_index].min(axis=0)
    new_edge_inex = new_node_index[protein_edge_index]
    input_edge_idx = torch.tensor(new_edge_inex[:, keepEdge], dtype=torch.long)
    input_protein_edge_s = protein_edge_s[keepEdge]
    input_protein_edge_v = protein_edge_v[keepEdge]
    return input_edge_idx, input_protein_edge_s, input_protein_edge_v

def get_keepNode(com, protein_node_xyz, n_node, pocket_radius, use_whole_protein, 
                     use_compound_com_as_pocket, add_noise_to_com, chosen_pocket_com):
    if use_whole_protein:
        keepNode = np.ones(n_node, dtype=bool)
    else:
        keepNode = np.zeros(n_node, dtype=bool)
        # extract node based on compound COM.
        if use_compound_com_as_pocket:
            if add_noise_to_com:
                com = com + add_noise_to_com * (2 * np.random.rand(*com.shape) - 1)
            for i, node in enumerate(protein_node_xyz):
                dis = compute_dis_between_two_vector(node, com)
                keepNode[i] = dis < pocket_radius

    if chosen_pocket_com is not None:
        another_keepNode = np.zeros(n_node, dtype=bool)
        for a_com in chosen_pocket_com:
            if add_noise_to_com:
                a_com = a_com + add_noise_to_com * (2 * np.random.rand(*a_com.shape) - 1)
            for i, node in enumerate(protein_node_xyz):
                dis = compute_dis_between_two_vector(node, a_com)
                another_keepNode[i] |= dis < pocket_radius
        keepNode |= another_keepNode
    return keepNode


def construct_data_from_graph_gvp(protein_node_xyz, protein_seq, protein_node_s, 
                                  protein_node_v, protein_edge_index, protein_edge_s, protein_edge_v,
                                 coords, compound_node_features, input_atom_edge_list, 
                                 input_atom_edge_attr_list, includeDisMap=True, contactCutoff=8.0, pocket_radius=20, interactionThresholdDistance=10, compoundMode=1, 
                                 add_noise_to_com=None, use_whole_protein=False, use_compound_com_as_pocket=True, chosen_pocket_com=None):
    n_node = protein_node_xyz.shape[0]
    n_compound_node = coords.shape[0]
    # centroid instead of com. 
    com = coords.mean(axis=0)
    keepNode = get_keepNode(com, protein_node_xyz.numpy(), n_node, pocket_radius, use_whole_protein, 
                             use_compound_com_as_pocket, add_noise_to_com, chosen_pocket_com)

    if keepNode.sum() < 5:
        # if only include less than 5 residues, simply add first 100 residues.
        keepNode[:100] = True
    input_node_xyz = protein_node_xyz[keepNode]
    input_edge_idx, input_protein_edge_s, input_protein_edge_v = get_protein_edge_features_and_index(protein_edge_index, protein_edge_s, protein_edge_v, keepNode)

    # construct graph data.
    data = HeteroData()
    data_compound = HeteroData() #只考虑小分子的边，要么edge_mask维度会有问题

    # only if your ligand is real this y_contact is meaningful.
    dis_map = scipy.spatial.distance.cdist(input_node_xyz.cpu().numpy(), coords)
    y_contact = dis_map < contactCutoff
    if includeDisMap:
        # treat all distance above 10A as the same.
        dis_map[dis_map>interactionThresholdDistance] = interactionThresholdDistance
        data.dis_map_mask = torch.tensor(dis_map<10, dtype=torch.bool).flatten()
        data.dis_map = torch.tensor(dis_map, dtype=torch.float).flatten()

    # additional information. keep records.
    data.node_xyz = input_node_xyz
    data.coords = torch.tensor(coords, dtype=torch.float)
    data.y = torch.tensor(y_contact, dtype=torch.float).flatten()
    data.seq = protein_seq[keepNode]
    data['protein'].node_s = protein_node_s[keepNode] # [num_protein_nodes, num_protein_feautre]
    data['protein'].node_v = protein_node_v[keepNode]
    data['protein', 'p2p', 'protein'].edge_index = input_edge_idx
    data['protein', 'p2p', 'protein'].edge_s = input_protein_edge_s
    data['protein', 'p2p', 'protein'].edge_v = input_protein_edge_v

    if compoundMode == 0:
        data['compound'].x = torch.tensor(compound_node_features, dtype=torch.bool)  # [num_compound_nodes, num_compound_feature]
        data['compound', 'c2c', 'compound'].edge_index = torch.tensor(input_atom_edge_list, dtype=torch.long).t().contiguous()
        c2c = torch.tensor(input_atom_edge_attr_list, dtype=torch.long)
        data['compound', 'c2c', 'compound'].edge_attr = F.one_hot(c2c-1, num_classes=1)  # [num_edges, num_edge_features]
    elif compoundMode == 1:
        data['compound'].x = compound_node_features
        data['compound', 'c2c', 'compound'].edge_index = input_atom_edge_list[:,:2].long().t().contiguous()
        data['compound', 'c2c', 'compound'].edge_weight = torch.ones(input_atom_edge_list.shape[0])
        data['compound', 'c2c', 'compound'].edge_attr = input_atom_edge_attr_list
        ##new
        data['compound'].pos = torch.tensor(coords, dtype=torch.float)
        get_lig_graph(data_compound, data)
        edge_mask, mask_rotate = get_transformation_mask(data_compound)
        data['compound'].edge_mask = torch.tensor(edge_mask)
        data['compound'].mask_rotate = mask_rotate

        #data['compound'].rotate_bond_index=data['compound', 'c2c', 'compound'].edge_index.T[edge_mask]
        data['compound'].rotate_bond_num=len(mask_rotate)
    #放到data下，follow_batch要加上
    data.compound_compound_edge_attr=data['compound', 'c2c', 'compound'].edge_attr

    return data, input_node_xyz, keepNode


def get_lig_graph(data_compound, data):
    data_compound['compound'].x = data['compound'].x
    data_compound['compound'].pos = data['compound'].pos
    data_compound['compound', 'c2c', 'compound'].edge_index = data['compound', 'c2c', 'compound'].edge_index
    data_compound['compound', 'c2c', 'compound'].edge_attr = data['compound', 'c2c', 'compound'].edge_attr
    return

def get_transformation_mask(pyg_data):
    G = to_networkx(pyg_data.to_homogeneous(), to_undirected=False)
    to_rotate = []
    edges = pyg_data['compound', 'c2c', 'compound'].edge_index.T.numpy()
    for i in range(0, edges.shape[0], 2):
        assert edges[i, 0] == edges[i+1, 1]

        G2 = G.to_undirected()
        G2.remove_edge(*edges[i])
        if not nx.is_connected(G2):
            l = list(sorted(nx.connected_components(G2), key=len)[0])
            if len(l) > 1:
                if edges[i, 0] in l:
                    to_rotate.append([])
                    to_rotate.append(l)
                else:
                    to_rotate.append(l)
                    to_rotate.append([])
                continue
        to_rotate.append([])
        to_rotate.append([])

    mask_edges = np.asarray([0 if len(l) == 0 else 1 for l in to_rotate], dtype=bool)
    mask_rotate = np.zeros((np.sum(mask_edges), len(G.nodes())), dtype=bool)
    idx = 0
    for i in range(len(G.edges())):
        if mask_edges[i]:
            mask_rotate[idx][np.asarray(to_rotate[i], dtype=int)] = True
            idx += 1

    return mask_edges, mask_rotate

def my_affinity_criterion(y_pred, y, mask, decoy_gap=1.0):
    affinity_loss = torch.zeros(y_pred.shape).to(y_pred.device)
    affinity_loss[mask] = (((y_pred - y)**2)[mask])
    affinity_loss[~mask] = (((y_pred - (y - decoy_gap)).relu())**2)[~mask]
    return affinity_loss.mean()

def evaulate(data_loader, model, criterion, device, saveFileName=None):
    y_list = []
    y_pred_list = []
    batch_loss = 0.0
    for data in data_loader:
        data = data.to(device)
        y_pred = model(data)
        with torch.no_grad():
            loss = criterion(y_pred, data.y)
        batch_loss += len(y_pred)*loss.item()
        y_list.append(data.y)
        y_pred_list.append(y_pred.sigmoid().detach())
        # torch.cuda.empty_cache()
    y = torch.cat(y_list)
    y_pred = torch.cat(y_pred_list)
    metrics = {"loss":batch_loss/len(y_pred)}
    metrics.update(myMetric(y_pred, y))
    if saveFileName:
        torch.save((y, y_pred), saveFileName)
    return metrics

def is_ligand_pocket(pdb):
    if len(pdb) == 4:
        return True
    else:
        return False
    

def select_pocket_by_predicted_affinity(info):
    info['is_ligand_pocket'] = info.pdb.apply(lambda x:is_ligand_pocket(x))
    pdb_to_num_contact = info.query("is_ligand_pocket").set_index("pdb")['num_contact'].to_dict()
    info['base_pdb'] = info.pdb.apply(lambda x: x.split("_")[0])
    info['native_num_contact'] = info.base_pdb.apply(lambda x: pdb_to_num_contact[x])
    info['cover_contact_ratio'] = info['num_contact'] / info['native_num_contact']
    use_whole_protein_list = set(info.base_pdb.unique()) - set(info.query("not is_ligand_pocket").base_pdb)
    # assume we don't know the true ligand binding site.
    selected_A = info.query("not is_ligand_pocket or (base_pdb in @use_whole_protein_list)").sort_values(['base_pdb', 
                      'affinity_pred_A']).groupby("base_pdb").tail(1).sort_values("index").reset_index(drop=True)
    selected_B = info.query("not is_ligand_pocket or (base_pdb in @use_whole_protein_list)").sort_values(['base_pdb', 
                      'affinity_pred_B']).groupby("base_pdb").tail(1).sort_values("index").reset_index(drop=True)
    return selected_A, selected_B


def compute_numpy_rmse(x, y):
    return np.sqrt(((x - y)**2).mean())

def extract_list_from_prediction(info, y, y_pred, selected=None, smiles_to_mol_dict=None, coords_generated_from_smiles=False):
    idx = 0
    y_list = []
    y_pred_list = []
    for i, line in info.iterrows():
        n_protein = line['p_length']
        n_compound = line['c_length']
        n_y = n_protein * n_compound
        if selected is None or (i in selected['index'].values):
            y_pred_list.append(y_pred[idx:idx+n_y].reshape(n_protein, n_compound))
            y_list.append(y[idx:idx+n_y].reshape(n_protein, n_compound))
        idx += n_y
    d = (y_list, y_pred_list)
    return d

def weighted_rmsd_loss(y_pred, y_true, mode=0):
    if mode == 0:
        return torch.mean(100 * (1 / (y_true ** 2)) * (y_pred - y_true) ** 2) ##TODO 修改contact loss与affinity loss权重，与之前scale匹配
    elif mode == 1:
        import math
        return torch.mean(math.exp(10) / torch.exp(y_true) * (y_pred - y_true) ** 2) 
    elif mode == 2:
        import math
        return torch.mean(math.pow(2, 10) / torch.pow(2, y_true) * (y_pred - y_true) ** 2) 
    elif mode == 3 or 4 or 5:
        return torch.mean(10 ** mode * (1 / (y_true ** mode)) * (y_pred - y_true) ** 2) 
    else:
        raise ValueError(f'invalid mode number:{mode}')


def cut_off_rmsd(y_pred,y_true,cut_off=5):
    y_pred_cutoff = y_pred[y_true < cut_off]
    y_true_cutoff = y_true[y_true < cut_off]
    cutoff_rmsd = torch.mean((y_pred_cutoff - y_true_cutoff) ** 2)
    return cutoff_rmsd


def evaluate_with_affinity(data_loader,
                           model,
                           contact_criterion,
                           affinity_criterion,
                           relative_k,
                           device,
                           pred_dis=False,
                           info=None,
                           saveFileName=None,
                           use_y_mask=False,
                           opt_torsion_dict=None,
                           skip_y_metrics_evaluation=False):
    y_list = []
    y_pred_list = []
    affinity_list = []
    real_y_mask_list = []
    p_length_list = []
    c_length_list = []
    affinity_A_pred_list = []
    affinity_B_pred_list = []
    tr_pred_list = []
    rot_pred_list = []
    tor_pred_list = []
    rmsd_pred_list = []
    prmsd_pred_list = []
    data_new_pos_batched_list = []
    epoch_loss_affinity_A = 0.0
    epoch_loss_affinity_B = 0.0
    epoch_loss_rmsd = 0.0
    epoch_loss_prmsd = 0.0
    epoch_loss_contact = 0.0
    epoch_loss_contact_5A = 0.0
    epoch_num_nan_contact_5A = 0
    epoch_loss_contact_10A = 0.0
    epoch_num_nan_contact_10A = 0
    epoch_affinity_B_recycling_1_loss = 0
    epoch_affinity_B_recycling_2_loss = 0
    epoch_affinity_B_recycling_3_loss = 0

    epoch_rmsd_recycling_0_loss=0
    epoch_rmsd_recycling_1_loss=0
    epoch_rmsd_recycling_2_loss=0
    epoch_rmsd_recycling_3_loss=0
    epoch_rmsd_recycling_4_loss=0
    epoch_rmsd_recycling_5_loss=0
    epoch_rmsd_recycling_6_loss=0
    epoch_rmsd_recycling_7_loss=0
    epoch_rmsd_recycling_8_loss=0
    epoch_rmsd_recycling_19_loss=0
    epoch_rmsd_recycling_39_loss=0


    epoch_tr_loss =0
    epoch_rot_loss =0
    epoch_tor_loss=0
    epoch_tr_loss_recy_0, epoch_rot_loss_recy_0, epoch_tor_loss_recy_0 = 0, 0, 0
    epoch_tr_loss_recy_1, epoch_rot_loss_recy_1, epoch_tor_loss_recy_1 = 0, 0, 0
    for data in tqdm(data_loader):
        protein_ptr = data['protein']['ptr']
        p_length_list += [int(protein_ptr[ptr] - protein_ptr[ptr-1]) for ptr in range(1, len(protein_ptr))]
        compound_ptr = data['compound']['ptr']
        c_length_list += [int(compound_ptr[ptr] - compound_ptr[ptr-1]) for ptr in range(1, len(compound_ptr))]

        data = data.to(device)
        sample_num = len(data.pdb)

        affinity_pred_A, affinity_pred_B_list, prmsd_list, pred_result_list= model(data)

        y = data.y
        dis_map = data.dis_map
        dis_map_mask = data.dis_map_mask
        y_pred = pred_result_list[-1][4] #pred_result_list:(tr_pred, rot_pred, torsion_pred_batched,next_candicate_conf_pos_batched, next_candicate_dis_matrix,current_candicate_conf_pos_batched)
        data_groundtruth_pos_batched = data['compound'].pos.split(degree(data['compound'].batch, dtype=torch.long).tolist())
        # 记录每个样本的学习信息
        _data_new_pos_batched_list = []
        for i in range(sample_num):  # data_new_pos_batched_list=[[]]*sample_num  #这么写会有严重的bug  所有[]其实都指向了一个[]
            _data_new_pos_batched_list.append([])
        candicate_conf_pos_batched = pred_result_list[0][5]  # 初始坐标
        for i in range(len(candicate_conf_pos_batched)):
            _data_new_pos_batched_list[i].append(candicate_conf_pos_batched[i].detach().cpu().numpy().tolist())
        for pred_result in pred_result_list:  # 每个pred_result是一个迭代轮次的batch中的全部样本
            # data_new_pos_batched:  bs*pos..
            next_candicate_conf_pos_batched = pred_result[3]
            for i in range(len(next_candicate_conf_pos_batched)):
                _data_new_pos_batched_list[i].append(next_candicate_conf_pos_batched[i].detach().cpu().numpy().tolist())
        data_new_pos_batched_list.extend(_data_new_pos_batched_list)
        # rmsd_loss
        rmsd_list = []
        for pred_result in pred_result_list:
            next_candicate_conf_pos_batched = pred_result[3]
            tmp_list = []
            for i in range(len(data_groundtruth_pos_batched)):
                tmp_rmsd = RMSD(next_candicate_conf_pos_batched[i], data_groundtruth_pos_batched[i])
                tmp_list.append(tmp_rmsd)
            rmsd_list.append(torch.tensor(tmp_list, dtype=torch.float).to(y_pred.device))

        candicate_conf_pos_batched = pred_result_list[0][5]  # 初始坐标
        rmsd_recycling_0_loss = 0
        for i in range(len(data_groundtruth_pos_batched)):
            tmp_rmsd = RMSD(candicate_conf_pos_batched[i], data_groundtruth_pos_batched[i])
            rmsd_recycling_0_loss += tmp_rmsd

        rmsd_recycling_0_loss = torch.tensor(rmsd_recycling_0_loss / len(data_groundtruth_pos_batched)).to(y_pred.device)
        rmsd_recycling_1_loss = torch.mean(rmsd_list[0]) if len(rmsd_list) >= 1 else torch.tensor([0]).to(y_pred.device)
        rmsd_recycling_2_loss = torch.mean(rmsd_list[1]) if len(rmsd_list) >= 2 else torch.tensor([0]).to(y_pred.device)
        rmsd_recycling_3_loss = torch.mean(rmsd_list[2]) if len(rmsd_list) >= 3 else torch.tensor([0]).to(y_pred.device)
        rmsd_recycling_4_loss = torch.mean(rmsd_list[3]) if len(rmsd_list) >= 4 else torch.tensor([0]).to(y_pred.device)
        rmsd_recycling_5_loss = torch.mean(rmsd_list[4]) if len(rmsd_list) >= 5 else torch.tensor([0]).to(y_pred.device)
        rmsd_recycling_6_loss = torch.mean(rmsd_list[5]) if len(rmsd_list) >= 6 else torch.tensor([0]).to(y_pred.device)
        rmsd_recycling_7_loss = torch.mean(rmsd_list[6]) if len(rmsd_list) >= 7 else torch.tensor([0]).to(y_pred.device)
        rmsd_recycling_8_loss = torch.mean(rmsd_list[7]) if len(rmsd_list) >= 8 else torch.tensor([0]).to(y_pred.device)
        rmsd_recycling_19_loss = torch.mean(rmsd_list[18]) if len(rmsd_list) >= 19 else torch.tensor([0]).to(y_pred.device)
        rmsd_recycling_39_loss = torch.mean(rmsd_list[38]) if len(rmsd_list) >= 39 else torch.tensor([0]).to(y_pred.device)


        if use_y_mask:
            y_pred = y_pred[data.real_y_mask]
            y = y[data.real_y_mask]
            dis_map = dis_map[data.real_y_mask]
            dis_map_mask = dis_map_mask[data.equivalent_native_y_mask]
            y_pred = y_pred[dis_map_mask]
            dis_map = dis_map[dis_map_mask]
            for i in range(len(prmsd_list)):
                prmsd_list[i] = prmsd_list[i][data.is_equivalent_native_pocket]
        with torch.no_grad():
            if pred_dis:

                # tr,rot,tor loss
                tr_loss = 0
                rot_loss = 0
                tor_loss = 0
                tr_loss_recy_0, rot_loss_recy_0, tor_loss_recy_0 = 0, 0, 0
                tr_loss_recy_1, rot_loss_recy_1, tor_loss_recy_1 = 0, 0, 0
                tmp_cnt = 0
                tor_last = []
                for _ in range(len(data_groundtruth_pos_batched)):
                    tor_last.append([])
                for recycling_num, pred_result in enumerate(pred_result_list):

                    tr_pred, rot_pred, torsion_pred_batched, _, _, current_candicate_conf_pos_batched = pred_result
                    compound_edge_index_batched = data['compound', 'compound'].edge_index.T.split(degree(data.compound_compound_edge_attr_batch, dtype=torch.long).tolist())
                    compound_rotate_edge_mask_batched = data['compound'].edge_mask.split(degree(data.compound_compound_edge_attr_batch, dtype=torch.long).tolist())
                    ligand_atom_sizes = degree(data['compound'].batch, dtype=torch.long).tolist()
                    for i in range(len(data_groundtruth_pos_batched)):
                        rotate_edge_index = compound_edge_index_batched[i][compound_rotate_edge_mask_batched[i]] - sum(
                            ligand_atom_sizes[:i])  # 把edge_id 从batch计数转换为样本内部计数
                       
                        OptimizeConformer_obj = OptimizeConformer(current_pos=current_candicate_conf_pos_batched[i],
                                                                ground_truth_pos=data_groundtruth_pos_batched[i],
                                                                rotate_edge_index=rotate_edge_index,
                                                                mask_rotate=data['compound'].mask_rotate[i])
                        if recycling_num == 0:
                            if data.pdb[i][:4] not in opt_torsion_dict.keys():
                                ttt = time.time()
                                opt_tr,opt_rotate, opt_torsion, opt_rmsd=OptimizeConformer_obj.run(maxiter=50)
                                opt_torsion_dict[data.pdb[i][:4]] = opt_torsion
                                tor_last[i] = torsion_pred_batched[i]
                                # print(tmp_cnt, opt_rmsd,time.time()-ttt)
                            else:
                                opt_torsion = opt_torsion_dict[data.pdb[i][:4]]
                                opt_rmsd, opt_R, opt_tr = OptimizeConformer_obj.apply_torsion(opt_torsion if opt_torsion is None else  opt_torsion.detach().cpu().numpy())
                                opt_rotate = matrix_to_axis_angle(opt_R).float()
                                opt_tr = opt_tr.T[0]
                                tor_last[i] = torsion_pred_batched[i]
                        else:
                            opt_torsion = opt_torsion_dict[data.pdb[i][:4]]
                            if opt_torsion is not None:
                                opt_torsion = opt_torsion - tor_last[i].detach().cpu() #opt_tor2 = opt_tor1 - tor_pred
                            _, opt_R, opt_tr = OptimizeConformer_obj.apply_torsion(opt_torsion if opt_torsion is None else  opt_torsion.detach().cpu().numpy())
                            opt_rotate = matrix_to_axis_angle(opt_R).float()
                            opt_tr = opt_tr.T[0]
                            if torsion_pred_batched[i] is not None:
                                tor_last[i] = tor_last[i] + torsion_pred_batched[i] % (math.pi * 2)#累加tor_pred
                        if recycling_num == 0:
                            tr_loss_recy_0 += torch.sqrt(F.mse_loss(tr_pred[i],opt_tr))
                            rot_pred_norm = torch.norm(rot_pred[i], p=2, dim=-1, keepdim=True)
                            rot_loss_recy_0 += torch.sqrt(F.mse_loss(rot_pred[i]/rot_pred_norm * (rot_pred_norm % (math.pi * 2)),opt_rotate)).item()
                            if opt_torsion is not None:
                                tor_loss_recy_0 += torch.sqrt(F.mse_loss(torsion_pred_batched[i], opt_torsion.to(torsion_pred_batched[i].device)))
                        elif recycling_num == 1:
                            tr_loss_recy_1 += torch.sqrt(F.mse_loss(tr_pred[i],opt_tr))
                            rot_pred_norm = torch.norm(rot_pred[i], p=2, dim=-1, keepdim=True)
                            rot_loss_recy_1 += torch.sqrt(F.mse_loss(rot_pred[i]/rot_pred_norm * (rot_pred_norm % (math.pi * 2)),opt_rotate)).item()
                            if opt_torsion is not None:
                                tor_loss_recy_1 += torch.sqrt(F.mse_loss(torsion_pred_batched[i], opt_torsion.to(torsion_pred_batched[i].device)))
                        tr_loss += torch.sqrt(F.mse_loss(tr_pred[i], opt_tr))
                        rot_pred_norm = torch.norm(rot_pred[i], p=2, dim=-1, keepdim=True)
                        rot_loss += torch.sqrt(F.mse_loss(rot_pred[i]/rot_pred_norm * (rot_pred_norm % (math.pi * 2)),opt_rotate)).item()
                        if opt_torsion is not None:
                            tor_loss += torch.sqrt(F.mse_loss(torsion_pred_batched[i], opt_torsion.to(torsion_pred_batched[i].device)))
                        tmp_cnt += 1
                tr_loss=tr_loss/tmp_cnt
                rot_loss=rot_loss/tmp_cnt
                tor_loss=tor_loss/tmp_cnt
                tr_loss_recy_0 = tr_loss_recy_0/len(data_groundtruth_pos_batched)
                rot_loss_recy_0 = rot_loss_recy_0/len(data_groundtruth_pos_batched)
                tor_loss_recy_0 = tor_loss_recy_0/len(data_groundtruth_pos_batched)
                tr_loss_recy_1 = tr_loss_recy_1/len(data_groundtruth_pos_batched)
                rot_loss_recy_1 = rot_loss_recy_1/len(data_groundtruth_pos_batched)
                tor_loss_recy_1 = tor_loss_recy_1/len(data_groundtruth_pos_batched)

                prmsd_loss = torch.stack([contact_criterion(rmsd_list[i], prmsd_list[i]) for i in range(len(prmsd_list))]).mean() if len(prmsd_list) > 0 else torch.tensor([0]).to(y_pred.device)
                rmsd_loss = torch.stack(rmsd_list[1:]).mean() if len(rmsd_list) > 1 else torch.tensor([0]).to(y_pred.device)
                contact_loss = contact_criterion(y_pred, dis_map) if len(dis_map) > 0 else torch.tensor([0]).to(dis_map.device)
                if math.isnan(cut_off_rmsd(y_pred, dis_map, cut_off=5)):
                    epoch_num_nan_contact_5A += len(y_pred)
                    contact_loss_cat_off_rmsd_5 = torch.zeros(1).to(y_pred.device)[0]
                else:
                    contact_loss_cat_off_rmsd_5 = cut_off_rmsd(y_pred, dis_map, cut_off=5)
                if math.isnan(cut_off_rmsd(y_pred, dis_map, cut_off=10)):
                    epoch_num_nan_contact_10A += len(y_pred)
                    contact_loss_cat_off_rmsd_10 = torch.zeros(1).to(y_pred.device)[0]
                else:
                    contact_loss_cat_off_rmsd_10 = cut_off_rmsd(y_pred, dis_map, cut_off=10)
            else:
                prmsd_loss = torch.stack([contact_criterion(rmsd_list[i], prmsd_list[i]) for i in range(len(prmsd_list))]).mean() if len(prmsd_list) > 0 else torch.tensor([0]).to(y_pred.device)
                rmsd_loss = torch.stack(rmsd_list[1:]).mean() if len(rmsd_list) > 1 else torch.tensor([0]).to(y_pred.device)
                y_pred = y_pred.sigmoid()
            affinity_loss_A = relative_k * affinity_criterion(affinity_pred_A, data.affinity)
            affinity_loss_B = relative_k * torch.stack([affinity_criterion(affinity_pred_B_list[i], data.affinity) for i in range(len(affinity_pred_B_list))],0).mean()
            affinity_loss_B_recycling_1 = affinity_criterion(affinity_pred_B_list[0], data.affinity) if len(affinity_pred_B_list) >= 1 else torch.tensor([0]).to(y_pred.device)
            affinity_loss_B_recycling_2 = affinity_criterion(affinity_pred_B_list[1], data.affinity) if len(affinity_pred_B_list) >= 2 else torch.tensor([0]).to(y_pred.device)
            affinity_loss_B_recycling_3 = affinity_criterion(affinity_pred_B_list[2], data.affinity) if len(affinity_pred_B_list) >= 3 else torch.tensor([0]).to(y_pred.device)
            # loss = contact_loss + affinity_loss ## unused-drop
        epoch_loss_contact += len(y_pred) * contact_loss.item()
        epoch_loss_contact_5A += len(y_pred) * contact_loss_cat_off_rmsd_5.item()
        epoch_loss_contact_10A += len(y_pred) * contact_loss_cat_off_rmsd_10.item()
        epoch_loss_affinity_A += len(affinity_pred_A) * affinity_loss_A.item()
        epoch_loss_affinity_B += len(affinity_pred_B_list[0]) * affinity_loss_B.item()
        epoch_loss_rmsd += len(rmsd_list[0]) * rmsd_loss.item()
        epoch_loss_prmsd += len(prmsd_list[0]) * prmsd_loss.item()
        epoch_affinity_B_recycling_1_loss += len(affinity_pred_B_list[0]) * affinity_loss_B_recycling_1.item()
        epoch_affinity_B_recycling_2_loss += len(affinity_pred_B_list[0]) * affinity_loss_B_recycling_2.item()
        epoch_affinity_B_recycling_3_loss += len(affinity_pred_B_list[0]) * affinity_loss_B_recycling_3.item()

        epoch_rmsd_recycling_0_loss += len(rmsd_list[0]) * rmsd_recycling_0_loss.item()
        epoch_rmsd_recycling_1_loss += len(rmsd_list[0]) * rmsd_recycling_1_loss.item()
        epoch_rmsd_recycling_2_loss += len(rmsd_list[0]) * rmsd_recycling_2_loss.item()
        epoch_rmsd_recycling_3_loss += len(rmsd_list[0]) * rmsd_recycling_3_loss.item()
        epoch_rmsd_recycling_4_loss += len(rmsd_list[0]) * rmsd_recycling_4_loss.item()
        epoch_rmsd_recycling_5_loss += len(rmsd_list[0]) * rmsd_recycling_5_loss.item()
        epoch_rmsd_recycling_6_loss += len(rmsd_list[0]) * rmsd_recycling_6_loss.item()
        epoch_rmsd_recycling_7_loss += len(rmsd_list[0]) * rmsd_recycling_7_loss.item()
        epoch_rmsd_recycling_8_loss += len(rmsd_list[0]) * rmsd_recycling_8_loss.item()
        epoch_rmsd_recycling_19_loss += len(rmsd_list[0]) * rmsd_recycling_19_loss.item()
        epoch_rmsd_recycling_39_loss += len(rmsd_list[0]) * rmsd_recycling_39_loss.item()

        epoch_tr_loss += len(rmsd_list[0]) * tr_loss.item()
        epoch_rot_loss += len(rmsd_list[0]) * rot_loss
        epoch_tor_loss += len(rmsd_list[0]) * tor_loss.item() if torch.is_tensor(tor_loss) else len(rmsd_list[0]) * tor_loss
        epoch_tr_loss_recy_0 += len(rmsd_list[0]) * tr_loss_recy_0.item()
        epoch_rot_loss_recy_0 += len(rmsd_list[0]) * rot_loss_recy_0
        epoch_tor_loss_recy_0 += len(rmsd_list[0]) * tor_loss_recy_0.item() if torch.is_tensor(tor_loss_recy_0) else len(rmsd_list[0]) * tor_loss_recy_0
        epoch_tr_loss_recy_1 += len(rmsd_list[0]) * tr_loss_recy_1.item()
        epoch_rot_loss_recy_1 += len(rmsd_list[0]) * rot_loss_recy_1
        epoch_tor_loss_recy_1 += len(rmsd_list[0]) * tor_loss_recy_1.item() if torch.is_tensor(tor_loss_recy_1) else len(rmsd_list[0]) * tor_loss_recy_1
        y_list.append(y)
        y_pred_list.append(y_pred.detach())
        affinity_list.append(data.affinity)
        affinity_A_pred_list.append(affinity_pred_A.detach())
        affinity_B_pred_list.append(affinity_pred_B_list[-1].detach()) #只取最后一个pred做pearson， TODO
        rmsd_pred_list.append(rmsd_list[-1].detach())
        prmsd_pred_list.append(prmsd_list[-1].detach())
        tr_pred_list.append(pred_result_list[-1][0].detach())
        rot_pred_list.append(pred_result_list[-1][1].detach())

        real_y_mask_list.append(data.real_y_mask)
        # torch.cuda.empty_cache()
    y = torch.cat(y_list)
    y_pred = torch.cat(y_pred_list)
    if pred_dis:
        y_pred = torch.clip(1 - (y_pred / 10.0), min=1e-6, max=0.99999)
        # we define 8A as the cutoff for contact, therefore, contact_threshold will be 1 - 8/10 = 0.2
        threshold = 0.2

    real_y_mask = torch.cat(real_y_mask_list)
    affinity = torch.cat(affinity_list)
    affinity_pred_A = torch.cat(affinity_A_pred_list)
    affinity_pred_B = torch.cat(affinity_B_pred_list)
    RMSD_pred = torch.cat(rmsd_pred_list)
    PRMSD_pred = torch.cat(prmsd_pred_list)
    TR_pred = torch.cat(tr_pred_list)
    ROT_pred = torch.cat(rot_pred_list)
    if saveFileName:
        torch.save((y, y_pred, affinity, affinity_pred_A, affinity_pred_B, RMSD_pred, PRMSD_pred), saveFileName)
    metrics = {
        "loss": epoch_loss_rmsd / len(RMSD_pred) + epoch_loss_affinity_A / len(affinity_pred_A) + epoch_loss_affinity_B / len(affinity_pred_B), 
        "loss_affinity_A": epoch_loss_affinity_A / len(affinity_pred_A),
        "loss_affinity_B": epoch_loss_affinity_B / len(affinity_pred_B),
        "loss_rmsd": epoch_loss_rmsd / len(RMSD_pred),
        "loss_prmsd": epoch_loss_prmsd / len(PRMSD_pred),
        "loss_contact": epoch_loss_contact / len(y_pred),
        "loss_contact_5A": epoch_loss_contact_5A / (len(y_pred) - epoch_num_nan_contact_5A),
        "loss_contact_10A": epoch_loss_contact_10A / (len(y_pred) - epoch_num_nan_contact_10A),
        "epoch_rmsd_recycling_0_loss":epoch_rmsd_recycling_0_loss/len(RMSD_pred),
        "epoch_rmsd_recycling_1_loss": epoch_rmsd_recycling_1_loss / len(RMSD_pred),
        "epoch_rmsd_recycling_2_loss": epoch_rmsd_recycling_2_loss / len(RMSD_pred),
        "epoch_rmsd_recycling_2_loss_2A_ratio": (RMSD_pred < 2).sum() / len(RMSD_pred),
        "epoch_rmsd_recycling_2_loss_5A_ratio": (RMSD_pred < 5).sum() / len(RMSD_pred),
        "epoch_rmsd_recycling_2_25": RMSD_pred.quantile(0.25),
        "epoch_rmsd_recycling_2_50": RMSD_pred.quantile(0.5),
        "epoch_rmsd_recycling_2_75": RMSD_pred.quantile(0.75),
        "epoch_rmsd_recycling_3_loss": epoch_rmsd_recycling_3_loss / len(RMSD_pred),
        "epoch_rmsd_recycling_4_loss": epoch_rmsd_recycling_4_loss / len(RMSD_pred),
        "epoch_rmsd_recycling_5_loss": epoch_rmsd_recycling_5_loss / len(RMSD_pred),
        "epoch_rmsd_recycling_6_loss": epoch_rmsd_recycling_6_loss / len(RMSD_pred),
        "epoch_rmsd_recycling_7_loss": epoch_rmsd_recycling_7_loss / len(RMSD_pred),
        "epoch_rmsd_recycling_8_loss": epoch_rmsd_recycling_8_loss / len(RMSD_pred),
        "epoch_rmsd_recycling_19_loss": epoch_rmsd_recycling_19_loss / len(RMSD_pred),
        "epoch_rmsd_recycling_39_loss": epoch_rmsd_recycling_39_loss / len(RMSD_pred),
        "epoch_tr_loss":epoch_tr_loss / len(RMSD_pred),
        "epoch_rot_loss": epoch_rot_loss / len(RMSD_pred),
        "epoch_tor_loss": epoch_tor_loss / len(RMSD_pred),
        "epoch_tr_loss_recy_0":epoch_tr_loss_recy_0 / len(RMSD_pred),
        "epoch_rot_loss_recy_0":epoch_rot_loss_recy_0 / len(RMSD_pred),
        "epoch_tor_loss_recy_0":epoch_tor_loss_recy_0 / len(RMSD_pred),
        "epoch_tr_loss_recy_1":epoch_tr_loss_recy_1 / len(RMSD_pred),
        "epoch_rot_loss_recy_1":epoch_rot_loss_recy_1 / len(RMSD_pred),
        "epoch_tor_loss_recy_1":epoch_tor_loss_recy_1 / len(RMSD_pred),
        "loss_affinity_B_recycling_1": epoch_affinity_B_recycling_1_loss / len(affinity_pred_B), 
        "loss_affinity_B_recycling_2": epoch_affinity_B_recycling_2_loss / len(affinity_pred_B),
        "loss_affinity_B_recycling_3": epoch_affinity_B_recycling_3_loss / len(affinity_pred_B), 

    }

    if info is not None:
        # print(affinity, affinity_pred)
        try:
            info['affinity'] = affinity.cpu().numpy()
        except:
            import pdb
            pdb.set_trace()
        info['affinity_pred_A'] = affinity_pred_A.cpu().numpy()
        info['affinity_pred_B'] = affinity_pred_B.cpu().numpy()
        info['rmsd_pred'] = RMSD_pred.cpu().numpy()
        info['prmsd_pred'] = PRMSD_pred.cpu().numpy()
        info['tr_pred_0'] = TR_pred[:, 0].cpu().numpy()
        info['tr_pred_1'] = TR_pred[:, 1].cpu().numpy()
        info['tr_pred_2'] = TR_pred[:, 2].cpu().numpy()
        info['rot_pred_0'] = ROT_pred[:, 0].cpu().numpy()
        info['rot_pred_1'] = ROT_pred[:, 1].cpu().numpy()
        info['rot_pred_2'] = ROT_pred[:, 2].cpu().numpy()
        info['candicate_conf_pos'] = data_new_pos_batched_list
        # selected_A, selected_B = select_pocket_by_predicted_affinity(info) #真口袋用不上排序选最好的，但是后面全部口袋时要用上 TODO
        selected_A = selected_B = info
        # selected_A = selected_B = info.sort_values(['compound_name', 'rmsd_pred']).groupby("compound_name").head(1).sort_values("index").reset_index(drop=True) ##将test集合改成最优的rmsd，查看模型上限 TODO
        result = {}
        real_affinity = 'real_affinity' if 'real_affinity' in selected_A.columns else 'affinity'
        # result['Pearson'] = selected['affinity'].corr(selected['affinity_pred'])
        result['Pearson_A'] = selected_A[real_affinity].corr(selected_A['affinity_pred_A'])
        result['Pearson_B'] = selected_B[real_affinity].corr(selected_B['affinity_pred_B'])
        result['RMSE_A'] = compute_numpy_rmse(selected_A[real_affinity], selected_A['affinity_pred_A'])
        result['RMSE_B'] = compute_numpy_rmse(selected_B[real_affinity], selected_B['affinity_pred_B'])
        # result['loss_rmsd'] = selected_A['rmsd_pred'].mean()

        native_y = y[real_y_mask].bool()
        native_y_pred = y_pred[real_y_mask]
        native_auroc = torchmetrics.functional.auroc(native_y_pred, native_y)
        result['native_auroc'] = native_auroc

        info['p_length'] = p_length_list
        info['c_length'] = c_length_list
        # y_list, y_pred_list = extract_list_from_prediction(info, y.cpu(), y_pred.cpu(), selected=selected, smiles_to_mol_dict=None, coords_generated_from_smiles=False)
        # selected_y = torch.cat([y.flatten() for y in y_list]).long()
        # selected_y_pred = torch.cat([y_pred.flatten() for y_pred in y_pred_list])
        # selected_auroc = torchmetrics.functional.auroc(selected_y_pred, selected_y)
        # result['selected_auroc'] = selected_auroc

        # for i in [90, 80, 50]:
        #     # cover ratio, CR.
        #     result[f'CR_{i}'] = (selected.cover_contact_ratio > i / 100).sum() / len(selected)
        metrics.update(result)
    # if not skip_y_metrics_evaluation:
    #     metrics.update(myMetric(y_pred, y, threshold=threshold))
    # metrics.update(affinity_metrics(affinity_pred_A, affinity))
    return metrics, info, opt_torsion_dict

def evaluate_affinity_only(data_loader, model, criterion, affinity_criterion, relative_k, device, info=None, saveFileName=None, use_y_mask=False):
    y_list = []
    y_pred_list = []
    affinity_list = []
    affinity_pred_list = []
    batch_loss = 0.0
    affinity_batch_loss = 0.0
    for data in tqdm(data_loader):
        data = data.to(device)
        y_pred, affinity_pred = model(data)
        y = data.y
        if use_y_mask:
            y_pred = y_pred[data.real_y_mask]
            y = y[data.real_y_mask]
        with torch.no_grad():
            contact_loss = criterion(y_pred, y) if len(y) > 0 else torch.tensor([0]).to(y.device)
            affinity_loss = relative_k * affinity_criterion(affinity_pred, data.affinity)
            loss = contact_loss + affinity_loss
        batch_loss += len(y_pred)*contact_loss.item()
        affinity_batch_loss += len(affinity_pred)*affinity_loss.item()

        affinity_list.append(data.affinity)
        affinity_pred_list.append(affinity_pred.detach())
        # torch.cuda.empty_cache()

    affinity = torch.cat(affinity_list)
    affinity_pred = torch.cat(affinity_pred_list)
    if saveFileName:
        torch.save((None, None, affinity, affinity_pred), saveFileName)
    metrics = {"loss":affinity_batch_loss/len(affinity_pred)}
    if info is not None:
        # print(affinity, affinity_pred)
        info['affinity'] = affinity.cpu().numpy()
        info['affinity_pred'] = affinity_pred.cpu().numpy()
        selected = select_pocket_by_predicted_affinity(info)
        result = {}
        real_affinity = 'real_affinity' if 'real_affinity' in selected.columns else 'affinity'
        # result['Pearson'] = selected['affinity'].corr(selected['affinity_pred'])
        result['Pearson'] = selected[real_affinity].corr(selected['affinity_pred'])
        result['RMSE'] = compute_numpy_rmse(selected[real_affinity], selected['affinity_pred'])
        for i in [90, 80, 50]:
            # cover ratio, CR.
            result[f'CR_{i}'] = (selected.cover_contact_ratio > i / 100).sum() / len(selected)
        metrics.update(result)

    metrics.update(affinity_metrics(affinity_pred, affinity))

    return metrics



#以下为获得每轮recycling最佳旋转/平移/扭转的方法

from torsion_geometry import modify_conformer_torsion_angles_np,rigid_transform_Kabsch_3D_torch,matrix_to_axis_angle
from scipy.optimize import differential_evolution

class OptimizeConformer:
    def __init__(self, current_pos, ground_truth_pos, rotate_edge_index, mask_rotate):
        """
        计算从初始构象到真值构象的 rot,tr,tor,用于构建recycling 中的真值标签
        :param candicate_pos:
        :param ground_truth_pos:
        :param rotate_edge_index:
        :param mask_rotate
        """
        self.candidate_pos=current_pos
        self.ground_truth_pos=ground_truth_pos
        self.rotate_edge_index=rotate_edge_index
        self.mask_rotate=mask_rotate
        self.rotate_bond_num=len(self.mask_rotate)

    def apply_torsion(self,torsion):
        """

        :param torsion: 输入是numpy.ndarray
        :return:
        """
        if torsion is not None:

            new_pos = modify_conformer_torsion_angles_np(self.candidate_pos.detach().cpu().numpy(),
                                                         self.rotate_edge_index.detach().cpu().numpy(),
                                                         self.mask_rotate,
                                                          torsion)
            new_pos=torch.from_numpy(new_pos).to(self.candidate_pos.device)
        else:
            new_pos=self.candidate_pos


        #lig_center=np.mean(new_pos, axis=0, )
        lig_center = torch.mean(new_pos, dim=0, keepdim=True)

        R_, t = rigid_transform_Kabsch_3D_torch(
            (new_pos-lig_center).T,
            (self.ground_truth_pos-lig_center).T)

        aligned_new_pos = torch.mm((new_pos - lig_center), R_.T) + t.T + lig_center
        #aligned_new_pos = np.dot((new_pos-lig_center) , R.T) + t.T+lig_center


        # RMSD 计算公式https://cloud.tencent.com/developer/article/1668887
        #rmsd = np.sqrt(np.sum((aligned_new_pos.cpu().numpy() - self.ground_truth_pos.cpu().numpy()) ** 2)/len(aligned_new_pos)) #
        rmsd = RMSD(aligned_new_pos, self.ground_truth_pos)
        return rmsd,R_, t

    def score_conformation(self, torsion):

        score=self.apply_torsion(torsion)[0]
        return score


    def run(self ,popsize=15,maxiter=500,mutation=(0.5, 1), recombination=0.8,seed=0):
        """

        :param popsize:
        :param maxiter:
        :param mutation:
        :param recombination:
        :param seed:
        :return:
         opt_tr:
         opt_rotate:
         opt_torsion:
         opt_rmsd: 当前构象优化到真实构象后剩余的rmsd，这个值越小代表返回的 opt_t,tor,rot 越准确
        """

        max_bound = [np.pi] * self.rotate_bond_num
        min_bound = [-np.pi] * self.rotate_bond_num
        bounds = (min_bound, max_bound)
        bounds = list(zip(bounds[0], bounds[1]))

        # Optimize conformations
        if self.rotate_bond_num!=0:
            result = differential_evolution(self.score_conformation, bounds,
                                            maxiter=maxiter, popsize=popsize,
                                            mutation=mutation, recombination=recombination, disp=False, seed=seed)
            opt_torsion = torch.from_numpy(result["x"]).to(self.candidate_pos.device).float()
            opt_rmsd,opt_R, opt_tr = self.apply_torsion(result["x"])
        else:
            opt_rmsd,opt_R, opt_tr = self.apply_torsion(None)
            opt_torsion = None

        opt_rotate = matrix_to_axis_angle(opt_R).float()
        opt_tr = opt_tr.T[0]
        return opt_tr,opt_rotate,opt_torsion,opt_rmsd


def RMSD(pos1,pos2):
    """
    for tensor
    :param pos1: n*3
    :param pos2: n*3
    :return:
    """
    rmsd=torch.sqrt(F.mse_loss(pos1, pos2, reduction="sum") / len(pos1)).item()
    return rmsd