import numpy as np
import pandas as pd
import os
import sys
import torch
from skimage.transform import resize
# Check if the code is running in a Jupyter notebook
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


def resize_image_skimage(image, target_size):
    # adjust the image size to the target size using skimage's resize function
    resized_image = resize(image, target_size, order=0, anti_aliasing=True)
    return resized_image


def RP_init(RNA_seqs, prot_seqs, dist_files=None, dist_cutoff=None, target_size=None, seq_files=None, pdbcodes=None):

    result_dict = {}
    for idx, file in enumerate(tqdm(dist_files)):
        pdbcode = pdbcodes[idx]  # 1C0A_C_A for native
        # print(pdbcode)
        # print(file)

        out_path = os.path.dirname(file)
        out_file = os.path.join(out_path, f'{pdbcode}_lbl.pt')
        
        if not os.path.exists(out_file):

            pdbname = dist_files[idx].split("/")[-1].split(".")[0]
            seq_file = seq_files[idx]
            path = os.path.join(*seq_file.split("/")[:-1])

            residue_map_file = os.path.join(f'/{path}', 'residue_mapping.csv')
            nuc_map_file = os.path.join(f'/{path}', 'nucleotide_mapping.csv')
            conmap_truth = create_conmap_truth(file, residue_map_file, nuc_map_file)
            
            bins = np.arange(2.5, 20.5, 0.5)  # example：[2.5, 3.0, 3.5, ..., 20.0]

            conmap_truth_lbl = np.digitize(conmap_truth, bins, right=True)

            min_val = np.min(conmap_truth)
            max_val = np.max(conmap_truth)

            conmap_truth_std = np.where(conmap_truth <= dist_cutoff, 1, 0)  # binary [0, 1]

            target_size = target_size
            conmap_truth_resize = resize_image_skimage(conmap_truth_std, target_size)

            RNA_site_truth = create_RNA_site_truth(file, nuc_map_file, dist_cutoff=dist_cutoff)
            prot_site_truth = create_prot_site_truth(file, residue_map_file, dist_cutoff=dist_cutoff)
            
            
            RNA_seq = RNA_seqs[idx]
            prot_seq = prot_seqs[idx]

            result_dict[pdbcode] = {
                'RNA seq': RNA_seq,
                'prot_seq': prot_seq,
                'conmap_truth_origin': conmap_truth,
                'conmap_truth': conmap_truth_std, 
                'conmap_truth_resize': conmap_truth_resize,
                'conmap_truth_lbl': conmap_truth_lbl,
                'RNA_site_truth': RNA_site_truth,
                'prot_site_truth': prot_site_truth, 
                'pdbname': pdbcode,
            }
            torch.save(result_dict[pdbcode], out_file)
        else:
            # load the file for verification

            result_subdict = torch.load(out_file)
            result_subdict['pdbname'] = pdbcode
            # print(result_dict)

            result_dict[pdbcode] = result_subdict

            
    return result_dict


def create_prot_site_truth(distance_file, residue_map_file, dist_cutoff):
    """
    Generate a two-dimensional matrix from the two input files: prot_site_truth

    Parameters:
        distance_file (str): The file path (.txt) containing distance data.
        residue_map_file (str): The file path (.csv) for residue mapping.

    Returns:
        np.ndarray: The generated two-dimensional matrix prot_site_truth.
    """

    # resulting in a DataFrame of distance with columns: residue, nuc, dist
    data_df = pd.read_csv(distance_file, sep='\t', header=None)
    data_df = data_df.rename(columns={0: 'residue', 1: 'nuc', 2: 'dist'})

    # residue -> residue_number
    residue_map_df = pd.read_csv(residue_map_file)
    dict_residue_to_number = dict(zip(residue_map_df['residue'], residue_map_df['residue_number']))


    # Map residue and nucleotide to numeric IDs
    data_df['residue_number'] = data_df['residue'].map(dict_residue_to_number)

    # determine the size of the matrix
    max_residue = data_df['residue_number'].max()

    # Initialize the matrix
    prot_site_truth = np.full((max_residue, 1), 0)

    # Fill the matrix
    for _, row in data_df.iterrows():
        residue_idx = row['residue_number'] - 1  # Convert to 0 index
        # print(residue_idx)
        # print(row['dist'])
        if row['dist'] <= dist_cutoff:
            prot_site_truth[residue_idx, 0] = 1

    return prot_site_truth

def create_RNA_site_truth(distance_file, nuc_map_file, dist_cutoff):
    """
    Generate a two-dimensional matrix from the two input files: RNA_site_truth

    Parameters:
        distance_file (str): The file path (.txt) containing distance data.
        nuc_map_file (str): The file path (.csv) for nucleotide mapping.

    Returns:
        np.ndarray: The generated two-dimensional matrix RNA_site_truth.
    """
    # Read the distance data file
    data_df = pd.read_csv(distance_file, sep='\t', header=None)
    data_df = data_df.rename(columns={0: 'residue', 1: 'nuc', 2: 'dist'})

    # Read the mapping file and create a mapping dictionary
    nuc_map_df = pd.read_csv(nuc_map_file)
    dict_nuc_to_number = dict(zip(nuc_map_df['nucleotide'], nuc_map_df['nucleotide_number']))

    # Map nucleotide to numeric IDs
    data_df['nucleotide_number'] = data_df['nuc'].map(dict_nuc_to_number)

    # determine the size of the matrix
    max_nucleotide = data_df['nucleotide_number'].max()

    # Initialize the matrix
    RNA_site_truth = np.full((max_nucleotide, 1), 0)

    # Fill the matrix
    for _, row in data_df.iterrows():
        nucleotide_idx = row['nucleotide_number'] - 1  # Convert to 0 index
        if row['dist'] <= dist_cutoff:
            RNA_site_truth[nucleotide_idx, 0] = 1

    return RNA_site_truth

def create_conmap_truth(distance_file, residue_map_file, nuc_map_file):
    """
    Generate a two-dimensional matrix from the three input files: conmap_truth.

    Parameters:
        distance_file (str): The file path (.txt) containing distance data.
        residue_map_file (str): The file path (.csv) for residue mapping.
        nuc_map_file (str): The file path (.csv) for nucleotide mapping.

    Returns:
        np.ndarray: The generated two-dimensional matrix conmap_truth.
    """
    # Read the distance data file
    data_df = pd.read_csv(distance_file, sep='\t', header=None)
    data_df = data_df.rename(columns={0: 'residue', 1: 'nuc', 2: 'dist'})

    # Read the mapping file and create a mapping dictionary
    residue_map_df = pd.read_csv(residue_map_file)
    dict_residue_to_number = dict(zip(residue_map_df['residue'], residue_map_df['residue_number']))

    nuc_map_df = pd.read_csv(nuc_map_file)
    dict_nuc_to_number = dict(zip(nuc_map_df['nucleotide'], nuc_map_df['nucleotide_number']))

    # Map residue and nucleotide to numeric IDs
    data_df['residue_number'] = data_df['residue'].map(dict_residue_to_number)
    data_df['nucleotide_number'] = data_df['nuc'].map(dict_nuc_to_number)

    # determine the size of the matrix
    max_residue = int(data_df['residue_number'].max())
    # print(f'max_residue: {max_residue}')
    max_nucleotide = int(data_df['nucleotide_number'].max())
    # print(f'max_nucleotide: {max_nucleotide}')
    # Initialize the matrix
    conmap_truth = np.full((max_nucleotide, max_residue), np.nan)

    # Fill the matrix
    for _, row in data_df.iterrows():
        # print(row['nucleotide_number'])
        nucleotide_idx = int(row['nucleotide_number']) - 1  # Convert to 0 index
        residue_idx = int(row['residue_number']) - 1  # Convert to 0 index
        conmap_truth[nucleotide_idx, residue_idx] = row['dist']

    return conmap_truth

def distance_to_k(v):
    if v <= 4:
        lbl = 0
    elif v <= 8:
        lbl = 1
    elif v <= 12:
        lbl = 2
    elif v <= 16:
        lbl = 3
    elif v <= 20:
        lbl = 4
    elif v <= 24:
        lbl = 5
    elif v <= 28:
        lbl = 6
    elif v <= 32:
        lbl = 7
    elif v <= 36:
        lbl = 8
    elif v <= 40:
        lbl = 9
    elif v <= 44:
        lbl = 10
    elif v <= 48:
        lbl = 11
    elif v <= 52:
        lbl = 12
    elif v <= 56:
        lbl = 13
    else:
        lbl = 14

    return lbl

def distance_to_37(v):
    if v <= 2.5:
        lbl = 0
    elif v <= 3.0:
        lbl = 1
    elif v <= 3.5:
        lbl = 2
    elif v <= 4.0:
        lbl = 3
    elif v <= 4.5:
        lbl = 4
    elif v <= 5.0:
        lbl = 5
    elif v <= 5.5:
        lbl = 6
    elif v <= 6.0:
        lbl = 7
    elif v <= 6.5:
        lbl = 8
    elif v <= 7.0:
        lbl = 9
    elif v <= 7.5:
        lbl = 10
    elif v <= 8.0:
        lbl = 11
    elif v <= 8.5:
        lbl = 12
    elif v <= 9.0:
        lbl = 13
    elif v <= 9.5:
        lbl = 14
    elif v <= 10.0:
        lbl = 15
    elif v <= 10.5:
        lbl = 16
    elif v <= 11.0:
        lbl = 17
    elif v <= 11.5:
        lbl = 18
    elif v <= 12.0:
        lbl = 19
    elif v <= 12.5:
        lbl = 20
    elif v <= 13.0:
        lbl = 21
    elif v <= 13.5:
        lbl = 22
    elif v <= 14.0:
        lbl = 23
    elif v <= 14.5:
        lbl = 24
    elif v <= 15.0:
        lbl = 25
    elif v <= 15.5:
        lbl = 26
    elif v <= 16.0:
        lbl = 27
    elif v <= 16.5:
        lbl = 28
    elif v <= 17.0:
        lbl = 29
    elif v <= 17.5:
        lbl = 30
    elif v <= 18.0:
        lbl = 31
    elif v <= 18.5:
        lbl = 32
    elif v <= 19.0:
        lbl = 33
    elif v <= 19.5:
        lbl = 34
    elif v <= 20.0:
        lbl = 35
    else:
        lbl = 36

    return lbl