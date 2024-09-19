import os
import time
import argparse
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import random
import time
import pickle
from tqdm import tqdm
import esm
import faiss
from Bio.Align.Applications import ClustalwCommandline
np.set_printoptions(threshold=np.inf)
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Set, Mapping, MutableMapping
import re
import subprocess


def count_list_files(path, extension):
    count = 0
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension) and file.find('gcn')==-1:
                count += 1
                file_name_with_path = os.path.join(root, file)
                file_list.append(file_name_with_path)
    return count, file_list

def remove_strings_with_gcn(strings):
    filtered_strings = [string for string in strings if 'gcn' not in string.lower()]
    return filtered_strings

def parse_fasta(fasta_string: str) -> Tuple[Sequence[str], Sequence[str]]:
    sequences = []
    descriptions = []
    index = -1
    for line in fasta_string.splitlines():
        line = line.strip()
        if line.startswith('>'):
            index += 1
            descriptions.append(line[1:])  # Remove the '>' at the beginning.
            sequences.append('')
            continue
        elif not line:
            continue  # Skip blank lines.
        sequences[index] += line

    return sequences, descriptions

def main(args):
    pkl_data_path = args.uniref_pkl_path
    pkl_extension = '.pkl'
    pkl_file_num, pkl_file_list = count_list_files(pkl_data_path, pkl_extension)
    pkl_file_list = remove_strings_with_gcn(pkl_file_list)
    pkl_file_num = len(pkl_file_list)
    print("file list generated")
    print("file_num: " + str(pkl_file_num))
    print("file_list_example: " + str(pkl_file_list[0]))

    model_esm, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    print("Model loaded")
    print("alphabet loaded")
    batch_converter = alphabet.get_batch_converter()
    model_esm.eval() # disables dropout for deterministic results

    device = torch.device('cpu')
    print("Device: " + str(device))
    model_esm = model_esm.to(device)

    class LightningModel(pl.LightningModule):
        def __init__(self):
            super(LightningModel, self).__init__()
            self.model_MLP1 = nn.Linear(1280, 1280)

        def forward(self, x):
            return self.model_MLP1(x)

        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = nn.MSELoss()(y_hat, y)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=0.001)

    # Contrastive model checkpoint loading
    model = LightningModel.load_from_checkpoint(args.checkpoint_path)
    model.eval()
    model = model.to(device)

    multimer_id = args.multimer_id
    #if not os.path.isdir("/Code_backup/casp16/sequences/" + multimer_id):
    #    os.makedirs("/Code_backup/casp16/sequences/" + multimer_id)
    input_fasta_path = args.target_path

    oligomer_state = args.oligomer_state
    counts = [int(match.group(1)) for match in re.finditer(r'(\d+)', oligomer_state)]
    target_name = multimer_id
    with open(input_fasta_path) as f:
        input_fasta_str = f.read()
    input_seqs, input_descs = parse_fasta(input_fasta_str)
    
    initial_line = "#"
    initial_line2 = ""
    for i, seq in enumerate(input_seqs):
        initial_line += str(len(seq)) + ","
        initial_line2 += str(counts[i]) + ","
    initial_line2 = initial_line2[:-1] + "\n" + ">"
    initial_line = initial_line[:-1] + "\t" + initial_line2
    query = ""
    for n, i in enumerate(input_seqs):
        initial_line += str(n + 101) + "\t"
        query += i
    
    target_list = []
    with open(args.target_path, "r") as f:
        seq_list_lines = f.readlines()
        for i in range(0, len(seq_list_lines), 2):
            target_list.append((seq_list_lines[i][1:].split()[0].replace("\n", ""), seq_list_lines[i + 1].replace("\n", "")))
    cnt = 0
    for gt_ind in range(len(target_list)):
        cnt += 1
        temp_target = target_list[gt_ind]
        batch_labels, batch_strs, batch_tokens = batch_converter([temp_target])
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

        with torch.no_grad():
            results = model_esm(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]

        sequence_representations = []
        for i, tokens_len in enumerate(batch_lens):
            sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))

        with torch.no_grad():
            output = model(torch.stack(sequence_representations).float()).detach().cpu().numpy()
            
        searching_indices = []

        #for pkl_file in tqdm(pkl_file_list[:1]):
        for pkl_file in tqdm(pkl_file_list):
            whole_features = []
            whole_sequences = []
            with open(pkl_file, 'rb') as fp:
                f = pickle.load(fp)
            for i in range(len(f)):
                whole_features.append(f[i]['contrastive_representation'])
                whole_sequences.append(f[i]['sequence'])
            whole_features = np.vstack(whole_features)
            
            index = faiss.IndexFlatL2(whole_features.shape[1])
            index.add(whole_features)
            D, I = index.search(output, args.searched_MSA_num) 
            for i in range(args.searched_MSA_num):
                searching_indices.append((whole_sequences[I[0][i]], D[0][i]))
                

        print("Sequences num: " + str(len(searching_indices)))
        
        searching_indices = sorted(searching_indices, key=lambda x: x[1])[:args.searched_MSA_num]
        I = [[item[0] for item in searching_indices]]
        D = [[item[1] for item in searching_indices]]

        save_path = f'{args.output_path}{temp_target[0]}/alignments/{temp_target[0]}/output_not_aligned' + str(101 + gt_ind) + '.fasta'
            
        directory = os.path.dirname(save_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
                
        with open(save_path, 'w') as file:
            file.write(">" + temp_target[0] + "\n")
            file.write(temp_target[1] + "\n")
            for ind, j in enumerate(I[0]):
                file.write(">" + temp_target[0] + "_" + str(ind) + "\n")
                if whole_sequences[ind][-1] == '\n':
                    file.write(j.replace('-', ''))
                else:
                    file.write(j.replace('-', '') + "\n")
                        
        #Align 시작
        input_path = f'{args.output_path}{temp_target[0]}/alignments/{temp_target[0]}/output_not_aligned' + str(101 + gt_ind) + '.fasta'
        with open(input_path,"r") as fs:
            temp_file = fs.readlines()
            for i in range(len(temp_file)):
                if i % 2 == 0:
                    temp_file[i] = ">" + temp_target[0] + "_" + str(gt_ind) + "_" + str(i // 2) + "\n"
                    
        with open(input_path,"w") as fs:
            fs.writelines(temp_file)
        
        temp_target = target_list[gt_ind]

        #clustalw_command_line = ClustalwCommandline("clustalw", infile=input_path, outfile=f"{args.output_path}/{temp_target[0]}/alignments/{temp_target[0]}/output_aligned" + str(101 + gt_ind) + ".fasta", outorder="INPUT", output="FASTA")
        #clustalw_command_line()
        aligned_fasta = f"{args.output_path}{temp_target[0]}/alignments/{temp_target[0]}/output_aligned" + str(101 + gt_ind) + ".fasta"
        os.system(f"kalign -i {input_path} -f afa -o {aligned_fasta}")
        
        
        save_path = f"{args.output_path}{temp_target[0]}/alignments/{temp_target[0]}/MSA_custom" + str(101 + gt_ind) + ".a3m"

        #Align 되어있는 파일을 query 기준으로 gap 제거하여 a3m 파일로 변환
        msa_seq = []
        with open(aligned_fasta,"r") as fs:
            cluster = fs.readlines()

        temp_st = ""
        for i, st in enumerate(cluster):
            flag = False
            if st[0] == '>':
                flag = True
                
            if flag:
                if i != 0:
                    msa_seq.append(temp_st.replace("\n", ""))
                temp_st = ""
                msa_seq.append(st.replace("\n", ""))
            else:
                temp_st += st.replace("\n", "")
        msa_seq.append(temp_st.replace("\n", ""))

        def align_strings(reference_str, target_str):
            aligned_str = ""
            indices_to_keep = [i for i, char in enumerate(reference_str) if char != "-"]
            for index in indices_to_keep:
                aligned_str += target_str[index]

            return aligned_str

        ref = msa_seq[1]
        for i in range(1, len(msa_seq), 2):
            msa_seq[i] = align_strings(ref, msa_seq[i])

        msa = ""
        for i in msa_seq:
            msa += i + "\n"
        text_file = open(save_path, "w")
        text_file.write(msa)
        text_file.close()
    
    length_list = []

    for t in range(len(input_seqs)) : 
        length_list.append(len(input_seqs[t]))
    merging = ""
    for t in range(len(input_seqs)):
        msa_write_ind = str(t + 101)
        save_path = f"{args.output_path}{temp_target[0]}/alignments/{temp_target[0]}/MSA_custom" + msa_write_ind + ".a3m"
        with open(save_path) as fs:
            cluster = fs.readlines()
            
        for i, sent in enumerate(cluster):
            if i % 2 == 0:
                merging += sent
            else:
                for j in range(0, t):
                    merging += "-" * length_list[j]
                merging += sent.replace("\n", "")
                for j in range(t + 1, len(input_seqs)):
                    merging += "-" * length_list[j]
                merging += "\n"
    merging = initial_line + "\n" + query + "\n" + merging
    text_file = open(save_path.rsplit("/", 1)[0] + "/" + multimer_id + "MSA_merge.a3m", "w")
    text_file.write(merging)
    text_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #MSA search 원하는 fasta file
    parser.add_argument('--target_path', type=str, default="/MSA_search_pipeline/casp16/sequences/T1298.fasta")
    #Search 해온 sequences와 align 결과를 저장할 폴더
    parser.add_argument('--output_path', type=str, default="/MSA_search_pipeline/casp16/CLsearch/")
    #MSA search 학습한 network checkpoint
    parser.add_argument('--checkpoint_path', type=str, default="/MSA_search_pipeline/remove_gap_final_check/esm-epoch=39.ckpt")
    #UniRef50에 대해서 representation을 전처리해둔 데이터베이스 위치, uniref_preprocesses.py 통해서 전처리 가능
    parser.add_argument('--uniref_pkl_path', type=str, default="/protein_data/UniRef50/pkl/")
    #CASP 기준 타겟의 id
    parser.add_argument('--multimer_id', type=str, default="T1298")
    #unit들이 이루고 있는 상태 A chain 2개, B chain 2개면 A2B2로 입력
    parser.add_argument('--oligomer_state', type=str, default="A1")
    #search 해오는 sequence 수 지정
    parser.add_argument('--searched_MSA_num', type=int, default=128)
    args = parser.parse_args()
    main(args) 