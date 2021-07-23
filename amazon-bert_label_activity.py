import time

import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch import cuda

from sklearn.preprocessing import MultiLabelBinarizer

import transformers
from transformers import BertTokenizer

device = 'cuda' if cuda.is_available() else 'cpu'

class AmazonDataset :
    def __init__(self, 
                 dataframe, 
                 labels, 
                 MAX_LEN=300) :
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.data = dataframe.text
        self.labels = labels
        self.MAX_LEN=MAX_LEN
    def __len__(self) :
        return len(self.data)
    def __getitem__(self, index):
        text = str(self.data[index])
        text = " ".join(text.split())
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.MAX_LEN,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]
        target = self.labels[index].toarray()[0]
        target = torch.from_numpy(target).type(torch.FloatTensor)
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': target
        }
    def tokenized_length_statistics(self) :
        lengths = []
        for index in range(1000) :
            print(index)
            text = str(self.data[index])
            text = " ".join(text.split())
            length_current = len(self.tokenizer(text)["input_ids"])
            lengths += [length_current]
        print("Mean :", np.mean(np.array(lengths), axis=0))
        print("STD :", np.std(np.array(lengths), axis=0))
        return None

def get_data_loders() :
    df_trn = pd.read_csv("./data/amazoncat-13k/bert-data/trn.csv")
    df_tst = pd.read_csv("./data/amazoncat-13k/bert-data/tst.csv")

    labels_trn = np.load("./data/amazoncat-13k/bert-data/train_labels.npy", allow_pickle=True)
    labels_tst = np.load("./data/amazoncat-13k/bert-data/test_labels.npy", allow_pickle=True)
    
    MLB = MultiLabelBinarizer(sparse_output=True)
    labels_trn = MLB.fit_transform(labels_trn)
    labels_tst = MLB.fit_transform(labels_tst)
    
    trn_set = AmazonDataset(df_trn, labels_trn)

    trn_params = {'batch_size': 8,
                  'shuffle': True,
                  'num_workers': 7}   
    trn_loader = DataLoader(trn_set, **trn_params)
    
    tst_set = AmazonDataset(df_trn, labels_trn)
    tst_params = {'batch_size': 8,
                  'shuffle': True,
                  'num_workers': 7}   
    tst_loader = DataLoader(tst_set, **tst_params)
    return trn_loader, tst_loader

class BERTRegression(torch.nn.Module):
    def __init__(self):
        super(BERTRegression, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.l2 = torch.nn.Linear(768, 512)
        self.l3 = torch.nn.Linear(512, 512)
        self.l4 = torch.nn.Linear(512, 1)
    def forward(self, 
                ids, 
                mask, 
                token_type_ids):
        _, output_1 = self.l1(ids, 
                              attention_mask=mask, 
                              token_type_ids=token_type_ids, 
                              return_dict=False)
        
        output_2 = self.l2(output_1)
        output_2 = torch.nn.ReLU()(output_2)
        
        output_3 = self.l3(output_2)
        output_3 = torch.nn.ReLU()(output_3)
        
        output = self.l4(output_3)
        return output

def train_step(model, 
               trn_loader, 
               tst_loader,
               optimizer):
    model_path = "./checkpoints/amazoncat13k-label_activity.torch"
    loss_fn = lambda outputs, targets : torch.nn.MSELoss()(outputs, targets)
    model.train()
    for idx, data in enumerate(trn_loader):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.float)
                
        targets = torch.sum(targets, dim=1)
        targets = targets[:, None]

        outputs = model(ids, mask, token_type_ids)

        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)

        if idx%200 == 0 :
            print("idx :", idx, 
                  "l2 loss :", loss.item())

        if idx%10000 == 0 :
            validate(model, 
                     tst_loader, 
                     10000)
            torch.save(model.state_dict(), model_path)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return None

def validate(model, 
             tst_loader, 
             c_max) -> None :
    start = time.time()
    model.eval()
    print("Validating...")
    with torch.no_grad() :
        counter = 0
        loss_cumulated = 0
        for _, data in enumerate(tst_loader) :
            if counter == c_max :
                break
            counter += 1
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            targets = torch.sum(targets, dim=1)
            targets = targets[:, None]

            outputs = model(ids, mask, token_type_ids)

            loss_cumulated += torch.nn.MSELoss()(outputs, targets)
    end = time.time()
    print("Validation loss :", loss_cumulated / c_max, 
          "time for val loop:", end - start)
    model.train()
    return None

def train() -> None :
    trn_loader, tst_loader = get_data_loders()
   
    model = BERTRegression()
    model.to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-05)

    for _ in range(5) :
        train_step(model, 
                   trn_loader,
                   tst_loader,
                   optimizer)
    return None

def main() -> int :

    train()

    return 0

if __name__ == '__main__' :
    main()