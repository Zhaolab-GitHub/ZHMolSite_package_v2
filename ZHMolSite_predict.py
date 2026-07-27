#!/usr/bin/env python
# coding: utf-8
"""
RNA-Protein Binding Nucleotides Prediction Script
-----------------------------------------
This script runs RNA-protein binding nucleotides prediction using a trained model.
It supports both JSON-based configuration and command-line argument overrides.

Author: Haoquan Liu, Yunjie Zhao
Date: 2025-12-20
"""

import os
import json
import pickle
import argparse
import torch
import pandas as pd
from utils.protein_init import *
from utils.RNA_init import *
from utils.dataset import *
from utils.Predictor import Predictor
from utils.utils import DataLoader

# -----------------------------
# Argument Parser and Config Loader
# -----------------------------

def parse_args():
    """
    Parse command-line arguments and optionally merge them with a JSON config file.
    Command-line arguments take precedence over JSON file values.
    """

    parser = argparse.ArgumentParser(description="Run RNA-protein binding nucleotides prediction with JSON config")
    
    # JSON configuration file
    parser.add_argument('--config', type=str, default='config.json', help='Path to JSON config file')
    
     # Optional overrides for config values
    parser.add_argument('--work_dir', type=str, help='Override work_dir')
    parser.add_argument('--project_name', type=str, help='Override project name')
    parser.add_argument('--RNA_fasta_file', type=str, help='Override RNA fasta file')
    parser.add_argument('--prot_fasta_file', type=str, help='Override protein fasta file')
    parser.add_argument('--RNA_pdb_file', type=str, help='Override RNA pdb file')
    parser.add_argument('--prot_pdb_file', type=str, help='Override protein pdb file')
    parser.add_argument('--model_path', type=str, help='Override model path')
    parser.add_argument('--RNA_thresh', type=float, help='Override threshold value')

    args = parser.parse_args()

    # Load configuration from JSON file
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Overwrite JSON config with command-line arguments (if provided)
    for k, v in vars(args).items():
        if v is not None and k != 'config':
            config[k] = v

    return config

# -----------------------------
# FASTA Parser
# -----------------------------
def parse_fasta(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    header = lines[0].strip()
    sequence = ''.join(line.strip() for line in lines[1:])
    return header, sequence

# -----------------------------
# Main Prediction Pipeline
# -----------------------------
def main():
    # Load configuration (from JSON + CLI)
    config = parse_args()
    
    # Extract parameters
    data_path = config['work_dir']
    project_name = config['project_name']
    RNA_fasta_file = config['RNA_fasta_file']
    prot_fasta_file = config['prot_fasta_file']
    RNA_pdb_file = config['RNA_pdb_file']
    prot_pdb_file = config['prot_pdb_file']
    model_path = config['model_path']
    RNA_thresh = config['RNA_thresh']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -----------------------------
    # Load sequences from FASTA
    # -----------------------------
    _, RNA_fa = parse_fasta(RNA_fasta_file)
    _, prot_fa = parse_fasta(prot_fasta_file)

    # Build input DataFrame for dataset creation
    records = [{
        'PDB_code': project_name,
        'RNA_seq': RNA_fa,
        'prot_seq': prot_fa,
        'RNA_PDB': RNA_pdb_file,
        'protein_PDB': prot_pdb_file,
        'RNA_fa': RNA_fasta_file,
        'protein_fa': prot_fasta_file
    }]
    test_df = pd.DataFrame(records)
    # print(test_df)

    # -----------------------------
    # Initialize or Load RNA/Protein Graphs
    # -----------------------------
    RNA_path = os.path.join(f'{data_path}', f'RNA_{project_name}.pt')
    protein_path = os.path.join(f'{data_path}', f'protein_{project_name}.pt')

    # Load or create RNA graph
    if os.path.exists(RNA_path):
        print('Loading RNA Graph data...')
        RNA_dict = torch.load(RNA_path)
    else:
        print('Initialising RNA Sequence to RNA Graph...')
        rhofold_path = config.get("rhofold_path")
        RNA_pdb_files = (
            [RNA_pdb_file]
            if RNA_pdb_file is not None
            else None
        )
        RNA_dict = RNA_init([RNA_fa], pdb_files=RNA_pdb_files,
                            fa_files=[RNA_fasta_file], pdbcodes=[project_name], rhofold_path=rhofold_path)
        torch.save(RNA_dict, RNA_path)

    # Load or create protein graph
    if os.path.exists(protein_path):
        print('Loading Protein Graph data...')
        prot_dict = torch.load(protein_path)
    else:
        print('Initialising Protein Sequence to Protein Graph...')
        prot_pdb_files = (
            [prot_pdb_file]
            if prot_pdb_file is not None
            else None
        )
        prot_dict = protein_init([prot_fa], seq_files=[prot_fasta_file],
                                 pdb_files=prot_pdb_files, pdbcodes=[project_name])
        torch.save(prot_dict, protein_path)

    # -----------------------------
    # Create Dataset and DataLoader
    # -----------------------------
    test_dataset = RNAProteinMoleculeDataset(test_df, RNA_dict, prot_dict, device=device)
    test_loader = DataLoader(test_dataset, batch_size=1, follow_batch=['RNA_node_aa', 'prot_node_aa'])

    # -----------------------------
    # Load Pre-trained Model
    # -----------------------------
    print(f"Loading model from: {model_path}")

    model = torch.load(
        model_path,
        map_location=device,
    )

    model = model.to(device)
    model.device = torch.device(device)
    model.eval()

    predictor = Predictor(model=model)

    # -----------------------------
    # Run Inference
    # -----------------------------
    for data in test_loader:
        data = data.to(device)
        test_id = project_name

        # Run prediction
        RNA_final_scores, RNA_s, prot_s = predictor.predict(data=data)

        RNA_final_scores_cpu = RNA_final_scores.cpu().numpy()

        # Create prediction DataFrame
        pred_score_df = pd.DataFrame(RNA_final_scores_cpu, columns=['pred_score'])
        RNA_pred_labels = (RNA_final_scores_cpu >= RNA_thresh).astype(int)
        pred_df = pd.DataFrame(RNA_pred_labels, columns=['pred_label'])
        pred_df = pd.concat([pred_score_df, pred_df], axis=1)

        # Save results to CSV
        pred_df_file = f'{data_path}/{test_id}_pred.csv'
        pred_df.to_csv(pred_df_file, index=False)
        print(f"Prediction saved to: {pred_df_file}")
        # print(pred_df)

        # print(RNA_s)
        # print(RNA_s[0].shape)
        RNA_cluster_file = f'{data_path}/{test_id}_RNA_K.pkl'
        with open(RNA_cluster_file, 'wb') as f:
            pickle.dump(RNA_s, f)

        prot_cluster_file = f'{data_path}/{test_id}_prot_K.pkl'
        with open(prot_cluster_file, 'wb') as f:
            pickle.dump(prot_s, f)


# -----------------------------
# Entry Point
# -----------------------------
if __name__ == '__main__':
    main()
