import numpy as np
import pandas as pd
import sys
import Bio.PDB
from collections import defaultdict
# Check if the code is running in a Jupyter notebook
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

import fm
import torch
import math
from Bio.PDB import PDBParser
from torch_geometric.utils import add_self_loops, to_undirected, remove_self_loops, coalesce
import warnings
from freesasa import *
import freesasa
import os
from pathlib import Path
import shutil
import subprocess

biopython_parser = PDBParser()


def RNA_init(
    seqs,
    pdb_files=None,
    fa_files=None,
    pdbcodes=None,
    device=None,
    rhofold_path=None,
):
    """
    Encode RNA graph features.

    :param seqs: List of RNA sequences
    :param pdb_files: List of RNA PDB file paths (optional)
    :param fa_files: List of RNA FASTA file paths (optional)
    :param pdbcodes: List of PDB code identifiers
    :param rhofold_path: Path to the local RhoFold installation
    :return: Dictionary of RNA graph features keyed by PDB code
    """
    if device is None:
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

    device = str(device)

    result_dict = {}

    # RhoFold is required only when no RNA structure is provided
    if not pdb_files:
        if rhofold_path is None:
            raise ValueError(
                "No RNA structure was provided. Please specify "
                "'rhofold_path' in the configuration file."
            )

        rhofold_path = Path(rhofold_path).expanduser().resolve()

        if not rhofold_path.is_dir():
            raise FileNotFoundError(
                f"RhoFold directory not found: {rhofold_path}"
            )

        print(f"Using RhoFold from: {rhofold_path}")

    # ==================== Load the RNA-FM model ====================

    RNA_FM_URL = (
        "https://huggingface.co/cuhkaih/rnafm/resolve/main/"
        "RNA-FM_pretrained.pth?download=true"
    )
    os.makedirs('pretrained', exist_ok=True)
    rna_fm_path = Path("pretrained/RNA-FM_pretrained.pth")

    # Download the model if the weight file does not exist or is empty
    if not rna_fm_path.is_file() or rna_fm_path.stat().st_size == 0:
        print(f"RNA-FM weight not found: {rna_fm_path}")
        print("Downloading the RNA-FM pretrained weight...")

        # Check whether wget is available
        if shutil.which("wget") is None:
            raise RuntimeError(
                "wget was not found. Install it with: "
                "sudo apt install wget"
            )

        # Create the pretrained directory if it does not exist
        rna_fm_path.parent.mkdir(parents=True, exist_ok=True)

        # Download the model weight with resume support
        subprocess.run(
            [
                "wget",
                "-c",
                "--show-progress",
                "--tries=10",
                "--timeout=30",
                "-O",
                str(rna_fm_path),
                RNA_FM_URL,
            ],
            check=True,
        )

        # Verify that the downloaded file exists and is not empty
        if (
            not rna_fm_path.is_file()
            or rna_fm_path.stat().st_size == 0
        ):
            raise RuntimeError(
                f"RNA-FM weight download failed: {rna_fm_path}"
            )

        print(
            "RNA-FM weight downloaded successfully: "
            f"{rna_fm_path}"
        )
    else:
        print(f"Using existing RNA-FM weight: {rna_fm_path}")

    # Convert the Path object to a string for model loading
    rna_fm_path = str(rna_fm_path)

    # Load the RNA-FM model from the local weight file
    model, alphabet = fm.pretrained.rna_fm_t12(
        model_location=rna_fm_path
    )

    # Create the sequence batch converter
    batch_converter = alphabet.get_batch_converter()

    # Disable dropout for deterministic inference
    model.eval()
    ###############################################################


    for idx, seq in enumerate(tqdm(seqs)):
        pdbcode = pdbcodes[idx]  # 1C0A_C_A for native, 1C0A_C_A_model_1 for docking decoys
        pdbname = fa_files[idx].split('/')[-1].split('.')[0] 
        print(pdbname)
        # print(edge_index.shape)
        # print(edge_weight.shape)
        fa_file = fa_files[idx]
        out_path = os.path.dirname(fa_file)

        # print(out_path)
        out_file = os.path.join(out_path, f"{pdbname}_feature.pt" if pdb_files else f"{pdbname}_feature_seq.pt")
        
        if not os.path.exists(out_file):

            if pdb_files is None:   ## RNA tertiary structures are unavailable
                output_path = os.path.join(out_path, 'rhofold_model')
                os.makedirs(output_path, exist_ok=True)
                pdb_file = os.path.join(output_path, 'relaxed_1000_model.pdb')
                if not os.path.exists(pdb_file):
                    predict_struct_cmd = f'python {rhofold_path}/inference.py --input_fas {fa_file} --device {device} --single_seq_pred True --output_dir {output_path} --ckpt {rhofold_path}/pretrained/RhoFold_pretrained.pt'
                    os.system(predict_struct_cmd)

                structure = Bio.PDB.PDBParser().get_structure('', pdb_file)
                pdb_struc = structure

                options = {'skip-unknown': False}  # guess unknown atoms, do not skip
                freesasa_structure = Structure(pdb_file, None, options)

                pred_ss = get_mxfold2_ss(fa_file)



            else:   ## RNA tertiary structures are available
                pdb_file = pdb_files[idx]
                fa_file = fa_files[idx]
                structure = Bio.PDB.PDBParser().get_structure('', pdb_file)
                pdb_struc = structure

                options = {'skip-unknown': False}  # guess unknown atoms, do not skip
                freesasa_structure = Structure(pdb_file, None, options)

                pred_ss = get_mxfold2_ss(fa_file)

            # print(pdb_file)
            # calculate SASA from Bio.PDB structure #
            SASA_result = freesasa.calc(freesasa_structure)

            sasa_result = pd.DataFrame(columns=['number', 'total_sasa', 'polar_sasa', 'apolar_sasa'])
            ii = 0
            for chain_id, residues in SASA_result.residueAreas().items():
                for residue_id, residue_area in sorted(residues.items(), key=lambda x: str(x[0])):
                    total_sasa = residue_area.total
                    polar_sasa = residue_area.polar
                    apolar_sasa = residue_area.apolar
                    new_row = {'number': ii, 'total_sasa': total_sasa, 'polar_sasa': polar_sasa, 'apolar_sasa': apolar_sasa}
                    sasa_result.loc[len(sasa_result)] = new_row
                    ii += 1
            sasa_result.iloc[:, 1:] = sasa_result.iloc[:, 1:].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
            # print(sasa_result)

            # calculate LN norm for each nucleotide in RNA #
            LN_result = cal_RNA_LN(pdb_file)
            LN_result.iloc[:, 1:] = LN_result.iloc[:, 1:].apply(lambda x: (x - x.min()) / (x.max() - x.min()))


            # print(LN_result)

            seq_feat = RNA_feature(seq, sasa_result, LN_result, pred_ss)
            # print('seq_feat:', seq_feat.shape)
            xyz_coord = get_atom_coord(pdb_struc, atom='C4\'')
            # print('xyz_coord:', xyz_coord.shape)
            assert xyz_coord.shape[0] == seq_feat.shape[0], f"{pdb_file}, the number of xyz_coord {xyz_coord.shape[0]} and seq_feat {seq_feat.shape[0]} do not match"

            data = [
                ("RNA1", seq),
            ]
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            # print(batch_tokens)
            # Extract embeddings (on CPU)
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[12])
            token_repr = results["representations"][12][0][1:-1]
            # print(token_repr)


            pdb_struc = pdb_struc[0]
            edge_index, edge_weight = contact_map(pdb=pdb_struc)

            result_dict[pdbcode] = {
                'seq': seq,
                'seq_feat': torch.from_numpy(seq_feat),
                'xyz_coord': torch.from_numpy(xyz_coord),
                'token_representation': token_repr.half(),
                'num_nodes': len(seq),
                'num_pos': torch.arange(len(seq)).reshape(-1, 1),
                'edge_index': edge_index,
                'edge_weight': edge_weight,
            }
        
            torch.save(result_dict[pdbcode], out_file)

        else:
            # Load the file for validation
            result_subdict = torch.load(out_file)
            # print(result_dict)

            result_dict[pdbcode] = result_subdict

    return result_dict

# normalize
def dic_normalize(dic):
    # print(dic)
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    # print(max_value)
    interval = float(max_value) - float(min_value)
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    dic['X'] = (max_value + min_value) / 2.0
    return dic

RNA_nuc_table = ['A','U','C','G','X']
RNA_ss_table = ['(', '.', ')', '[', ']']
RNA_nuc_pur_table = ['A', 'G']
RNA_nuc_pyr_table = ['U', 'C']


nuc_weight_table = {'A': 329.2, 'U': 306.2, 'C': 305.2, 'G': 345.2}



nuc_weight_table = dic_normalize(nuc_weight_table)
# nuc_ss_table = dic_normalize(nuc_ss_table)

def nuc_features(nuc, sasa_fea, ln_fea):
    nuc_property1 = [1 if nuc in RNA_nuc_pur_table else 0, 1 if nuc in RNA_nuc_pyr_table else 0]

    # nuc_ss = nuc_ss_dict[nuc]
    # nuc_property2 = [nuc_weight_table[nuc], nuc_ss_table[nuc_ss]]
    nuc_property2 = [nuc_weight_table[nuc]] + [x for x in sasa_fea] + [x for x in ln_fea]
    # print(np.array(nuc_property1 + nuc_property2).shape)
    return np.array(nuc_property1 + nuc_property2)

# one ont encoding
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        # print(x)
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    '''Maps inputs not in the allowable set to the last element.'''
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def RNA_feature(RNA_seq, sasa_result, LN_result, pred_ss):

    RNA_hot = np.zeros((len(RNA_seq), len(RNA_nuc_table)))
    RNA_ss_hot = np.zeros((len(RNA_seq), len(RNA_ss_table)))
    RNA_property = np.zeros((len(RNA_seq), 11))
    # if pdb_files is None:
    #     RNA_ss_dict = get_RNA_ss(pdb_file=pdb_file)
    # else:
    #     RNA_ss_dict = get_RNA_ss(pdb_file=pdb_files[idx])

    for i in range(len(RNA_seq)):
        # if 'X' in RNA_seq:
        #     print(RNA_seq)
        RNA_hot[i,] = one_of_k_encoding(RNA_seq[i], RNA_nuc_table)
        RNA_ss_hot[i,] = one_of_k_encoding(pred_ss[i], RNA_ss_table)
        RNA_property[i,] = nuc_features(RNA_seq[i], sasa_result.iloc[i, 1:].to_numpy(), LN_result.iloc[i, 1:].to_numpy())
    return np.concatenate((RNA_hot, RNA_ss_hot, RNA_property), axis=1)


def contact_map(contact_map_proba=None, pdb=None, contact_threshold=0.5, distance_cutoff=8):

    if pdb is None:
        num_residues = contact_map_proba.shape[0]
        prot_contact_adj = (contact_map_proba >= contact_threshold).long()
        edge_index = prot_contact_adj.nonzero(as_tuple=False).t().contiguous()
        row, col = edge_index
        edge_weight = contact_map_proba[row, col].float()

    else:
        chains = []
        for chain in pdb:
            chains.append(chain.get_id())
        contact_map_proba = calc_dist_matrix(pdb)
        contact_map_proba = -contact_map_proba/distance_cutoff + 1  # distance_cutoff = 0, dist 0 = 1
        contact_map_proba = torch.tensor(contact_map_proba)

        num_residues = contact_map_proba.shape[0]
        prot_contact_adj = (contact_map_proba >= contact_threshold).long()
        edge_index = prot_contact_adj.nonzero(as_tuple=False).t().contiguous()
        row, col = edge_index
        edge_weight = contact_map_proba[row, col].float()

    ############### CONNECT ISOLATED NODES - Prevent Disconnected Residues ######################
    seq_edge_head1 = torch.stack([torch.arange(num_residues)[:-1], (torch.arange(num_residues) + 1)[:-1]])
    seq_edge_tail1 = torch.stack([(torch.arange(num_residues))[1:], (torch.arange(num_residues) - 1)[1:]])
    seq_edge_weight1 = torch.ones(seq_edge_head1.size(1) + seq_edge_tail1.size(1)) * contact_threshold
    edge_index = torch.cat([edge_index, seq_edge_head1, seq_edge_tail1], dim=-1)
    edge_weight = torch.cat([edge_weight, seq_edge_weight1], dim=-1)

    seq_edge_head2 = torch.stack([torch.arange(num_residues)[:-2], (torch.arange(num_residues) + 2)[:-2]])
    seq_edge_tail2 = torch.stack([(torch.arange(num_residues))[2:], (torch.arange(num_residues) - 2)[2:]])
    seq_edge_weight2 = torch.ones(seq_edge_head2.size(1) + seq_edge_tail2.size(1)) * contact_threshold
    edge_index = torch.cat([edge_index, seq_edge_head2, seq_edge_tail2], dim=-1)
    edge_weight = torch.cat([edge_weight, seq_edge_weight2], dim=-1)
    ############### CONNECT ISOLATED NODES - Prevent Disconnected Residues ######################

    edge_index, edge_weight = coalesce(edge_index, edge_weight, reduce='max')
    edge_index, edge_weight = to_undirected(edge_index, edge_weight, reduce='max')
    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
    edge_index, edge_weight = add_self_loops(edge_index, edge_weight, fill_value=1)

    return edge_index, edge_weight

def calc_residue_dist(residue_one, residue_two):
    """Returns the shortest distance between non-hydrogen heavy atoms of two residues."""
    min_distance = float('inf')
    for atom_one in residue_one.get_atoms():
        if atom_one.element == 'H':  # skip hydrogen atoms
            continue
        for atom_two in residue_two.get_atoms():
            if atom_two.element == 'H':  # skip hydrogen atoms
                continue
            # Calculate the distance between the two atoms
            diff_vector = atom_one.coord - atom_two.coord
            distance = np.sqrt(np.sum(diff_vector * diff_vector))
            min_distance = min(min_distance, distance)
    return min_distance

def calc_dist_matrix(structure):
    """
    Returns a distance matrix of all residues in a structure.
    This includes residues across multiple chains.
    """
    residues = []
    for chain in structure.get_chains():
        residues.extend(list(chain.get_residues()))  

    n = len(residues)
    dist_matrix = np.zeros((n, n), dtype=float)

    for i, residue_one in enumerate(residues):
        for j, residue_two in enumerate(residues):
            if j >= i:  # use symmetry to avoid redundant calculations
                dist_matrix[i, j] = calc_residue_dist(residue_one, residue_two)
                dist_matrix[j, i] = dist_matrix[i, j]  # symmetric matrix
    return dist_matrix  # return distance matrix


def cal_RNA_LN(infile):
    """
    Processes a PDB file to calculate distance-weighted properties.

    Args:
        infile (str): Path to the PDB input file.

    Returns:
        pd.DataFrame: A DataFrame containing the calculated distances.
    """
    # Parse PDB file and filter relevant atoms
    pdb = defaultdict(list)
    with open(infile, 'r') as f:
        num = 0
        for idx, line in enumerate(f):
            if idx > 0:
                if line.startswith('ATOM'):
                    resi_num = line[22:27].strip()
                    atom = line[13:15].strip()
                    atom_13 = line[13:14]
                    if resi_num != resi_num_prev:
                        num += 1
                    resi_num_prev = resi_num

                    if atom == 'C3' or atom_13 in {'C', 'O', 'N', 'P'}:
                        pdb[num].append(line)
            else:
                if line.startswith('ATOM'):
                    resi_num = line[22:27].strip()
                    resi_num_prev = resi_num
                    atom = line[13:15].strip()
                    atom_13 = line[13:14]
                    if atom == 'C3' or atom_13 in {'C', 'O', 'N', 'P'}:
                        pdb[num].append(line)

    # Extract C3 atoms
    pdb_c3 = []
    for num in sorted(pdb, key=lambda x: int(x)):
        found_c3 = False
        for element_info in pdb[num]:
            element = element_info[13:15].strip()
            if element == 'C3':
                pdb_c3.append(element_info)
                found_c3 = True
                break
        if not found_c3:
            pdb_c3.append(pdb[num][0])

    # Calculate pairwise distances
    all_distance = []
    coordinates = {}
    # print(len(pdb_c3))
    for num, line in enumerate(pdb_c3):
        # num = line[22:27].strip()
        x, y, z = map(float, [line[30:38], line[38:46], line[46:54]])
        coordinates[num] = (x, y, z)

    for num1, coord1 in coordinates.items():
        for num2, coord2 in coordinates.items():
            if int(num1) > int(num2):
                dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(coord1, coord2)))
                all_distance.append(dist)

    # Sort distances and calculate mean thresholds
    all_distance.sort()
    mean = len(all_distance) // 4
    thresholds = [0, mean, mean * 2, mean * 3, len(all_distance) - 1]

    # Compute distance-weighted properties
    hash_map = defaultdict(lambda: defaultdict(dict))
    for num1, coord1 in coordinates.items():
        for num2, coord2 in coordinates.items():
            if abs(int(num1) - int(num2)) > 1:
                dist_squared = sum((a - b) ** 2 for a, b in zip(coord1, coord2))
                for threshold in thresholds:
                    cda = all_distance[threshold]
                    if cda != 0:
                        weight = math.exp(-dist_squared / (cda ** 2))
                    else:
                        weight = 1
                    hash_map[cda][num1][num2] = weight

    # Calculate distances and create DataFrame
    data = []
    for num1, coord1 in coordinates.items():
        x, y, z = coord1
        row = [num1]
        for threshold in thresholds:
            cda = all_distance[threshold]
            sumx = sumy = sumz = sumb = 0
            for num2, coord2 in coordinates.items():
                if abs(int(num1) - int(num2)) > 1:
                    x1, y1, z1 = coord2
                    weight = hash_map[cda][num1].get(num2, 0)
                    sumx += x1 * weight
                    sumy += y1 * weight
                    sumz += z1 * weight
                    sumb += weight
            meanx, meany, meanz = sumx / sumb, sumy / sumb, sumz / sumb
            distance = math.sqrt((x - meanx) ** 2 + (y - meany) ** 2 + (z - meanz) ** 2)
            row.append(distance)
        data.append(row)

    columns = ['residue_number'] + ['LN_0', 'LN_1/4', 'LN_1/2', 'LN_3', '1']
    df = pd.DataFrame(data, columns=columns)
    return df

def get_mxfold2_ss(fa_file):
    import subprocess
    cmd = f'mxfold2 predict {fa_file}'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    pred_ss = result.stdout.split('\n')[2].split(' ')[0]

    return pred_ss


def get_atom_coord(pdb_struc, atom=None):
    """
    Get the coordinates of the specified atom in the PDB structure.
    :param pdb_struc: The PDB structure object.
    :param atom: The atom name to extract coordinates for (default is None).
    :return: A numpy array of shape (num_atoms, 3) containing the coordinates.
    """
    coords = []
    for model in pdb_struc:
        for chain in model:
            for residue in chain:
                target_atom = atom
                if not residue.has_id(target_atom):
                    if target_atom != 'C4\'' and residue.has_id('C5\''):
                        target_atom = 'C5\''
                    elif not residue.has_id('C5\'') and residue.has_id('C4'):
                        target_atom = 'C4'
                    # choose the first atom in the residue if the specified atom is not found
                    else:
                        first_atom = list(residue.get_atoms())[0]
                        # print(first_atom)
                        target_atom = first_atom.get_name()
                if residue.has_id(target_atom):
                    coords.append(residue[target_atom].get_coord())
                
                # print(target_atom)
                # print(next(iter(residue)))
                # print(residue, target_atom, residue[target_atom].get_coord())
    return np.array(coords)