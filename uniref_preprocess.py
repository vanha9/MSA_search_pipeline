import esm
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import pickle
import re
import torch
from tqdm import tqdm
from torch.nn.parameter import Parameter

model_esm, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
print("Model loaded")
print("alphabet loaded")
batch_converter = alphabet.get_batch_converter()
model_esm.eval() # disables dropout for deterministic results

device = torch.device('cuda:0')
print("Device: " + str(device))
model_esm = model_esm.to(device)


# checkpoint를 불러오기 위한 임시 pytorch lightning model 정의
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
model = LightningModel.load_from_checkpoint("remove_gap_final_check/esm-epoch=39.ckpt")
model.eval()
model = model.to(device)


import pickle
import re
import torch

# 딕셔너리를 저장할 변수 초기화


# 입력 FASTA 파일 경로 설정
for temp_cnt in tqdm(range(1, 68)):
    fasta_dict = {}
    #uniref50 파일은 100만개의 서열씩 68개의 파일로 분할되어 있습니다.
    #해당 파일들을 순서대로 읽어와 전처리하여 pickle 파일을 각각 생성합니다.
    fasta_file = "/protein_data/UniRef50/tmp/2313001781436220779/UniRef50_" + str(temp_cnt) + ".fasta"
    pickle_file = "/protein_data/UniRef50/pkl/UniRef50_" + str(temp_cnt) + ".pkl"
    
    # 현재 처리 중인 UniProt ID 초기화
    current_uniprot_id = None

    # TaxName, TaxID, RepID pattern 설정
    pattern = r"Tax=(.*?)\sTaxID=(\d+)\sRepID=(\w+)"

    # 입력 FASTA 파일을 열고 처리
    count = 0
    sequence = ""
    tax = None
    taxid = None
    repid = None
    descrip = None

    #100만개의 서열이 저장된 fasta 파일 하나를 읽어와 전처리를 시작합니다. fasta 형태이기 때문에 > format일때 서열의 description으로 시작합니다.
    with open(fasta_file, "r") as f:
        for line in f:
            if line.startswith('>'):
                if sequence == "":
                    descrip = line.replace('\n', '')
                else:
                    #시작이 >이고 sequence가 비어있지 않으니, 다음 서열을 읽기 시작합니다. 따라서 지금까지 읽어왔던 서열의 전처리를 진행합니다.
                    if count % 100 == 0:
                        progress = (count / 1000000) * 100
                        print(f"진행률: {progress:.2f}% 완료 => {count} / 1000000", end='\r')

                    count += 1  # sequence_count
                    # 매 update_interval마다 진행률 업데이트
                    
                    #description 통해서 종 정보 저장
                    match = re.search(pattern, descrip)
                    if match:
                        tax, taxid, repid = match.groups()
                    else:
                        print("crashed")
                        print(line)
                        break
                    
                    #서열의 길이가 1000이 넘어가는 경우 1000으로 crop 합니다.
                    if len(sequence) > 1000:
                        sequence = sequence[:1000]
                    sequence = sequence.replace("\n", "")
                    

                    # esm embedding
                    sequence = [(repid, sequence)]
                    batch_labels, batch_strs, batch_tokens = batch_converter(sequence)
                    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
                    
                    with torch.no_grad():
                        results = model_esm(batch_tokens.to(device), repr_layers=[33], return_contacts=True)
                    token_representations = results["representations"][33]

                    # Generate per-sequence representations via averaging
                    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
                    sequence_representations = []
                    for i, tokens_len in enumerate(batch_lens):
                        sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))
                    esm_representation = sequence_representations[0].cpu().numpy()
                    #print(esm_representation)

                    # contrastive embedding
                    with torch.no_grad():
                        output = model(torch.stack(sequence_representations).float().to(device)).detach().cpu().numpy()
                    contrastive_representation = torch.tensor(output[0])
                    #print(contrastive_embedding)
                    
                    
                    
                    fasta_dict[count-1] = {
                        'tax': tax,
                        'taxid': taxid,
                        'repid': repid,
                        'sequence' : sequence[0][1],
                        'contrastive_representation': contrastive_representation
                    }
                    sequence = ""
                    descrip = line.replace('\n', '')

                    # if count == 100:
                    #     break
            #line이 >로 시작하지 않는다는 것을 서열을 읽어온다는 의미입니다.
            else:
                sequence += line.replace('\n', '')

        if sequence:    # 마지막 시퀀스
            if count % 100 == 0:
                progress = (count / 1000000) * 100
                print(f"진행률: {progress:.2f}% 완료 => {count} / 1000000", end='\r')

            count += 1  # sequence_count
            # 매 update_interval마다 진행률 업데이트
            
            match = re.search(pattern, descrip)
            if match:
                tax, taxid, repid = match.groups()
            else:
                print("crashed")
                print(line)
                break

            #print("length: " + str(len(sequence)))
            if len(sequence) > 1000:
                sequence = sequence[:1000]
            sequence = sequence.replace("\n", "")

            # esm embedding
            sequence = [(repid, sequence)]
            batch_labels, batch_strs, batch_tokens = batch_converter(sequence)
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
            
            with torch.no_grad():
                results = model_esm(batch_tokens.to(device), repr_layers=[33], return_contacts=True)
            token_representations = results["representations"][33]
            contact = results["contacts"]

            # Generate per-sequence representations via averaging
            # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
            sequence_representations = []
            for i, tokens_len in enumerate(batch_lens):
                sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))
            esm_representation = sequence_representations[0].cpu().numpy()

                    #print(esm_representation)
                    # contrastive embedding
            with torch.no_grad():
                output = model(torch.stack(sequence_representations).float().to(device)).detach().cpu().numpy()
            contrastive_representation = torch.tensor(output[0])
            
            
                    
            fasta_dict[count-1] = {
                    'tax': tax,
                    'taxid': taxid,
                    'repid': repid,
                    'sequence' : sequence[0][1],
                    'contrastive_representation': contrastive_representation
                }

    with open(pickle_file, 'wb') as outfile:
        pickle.dump(fasta_dict, outfile)

