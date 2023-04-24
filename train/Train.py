from model.Model import STSBertModel
from data_processing.Dataset import DataSequence, collate_fn

from transformers import BertTokenizer, XLMRobertaTokenizer
from transformers.models.deberta_v2 import DebertaV2TokenizerFast

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from tqdm import tqdm
import pandas as pd

from datetime import datetime
import yaml

models = {'bert': 'bert-base-uncased',
          'xlmroberta': 'xlm-roberta-base',
          'debertav3': 'microsoft/deberta-v3-base'}

model_tokenizer = {'bert-base-uncased': BertTokenizer,
                   'xlm-roberta-base': XLMRobertaTokenizer,
                   'microsoft/deberta-v3-base': DebertaV2TokenizerFast}


def train_and_save(lm, train_path):
    with open('./config/model.yml', 'r') as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    bs = cfg['batch_size']
    epochs = cfg['epochs']
    learning_rate = float(cfg['learning_rate'])

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model_name = models[lm]
    tokenizer = model_tokenizer[model_name].from_pretrained(model_name)
    dataset = pd.read_csv(train_path)

    model = STSBertModel(model_name)
    model.to(device)

    criterion = nn.CosineEmbeddingLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    train_dataset = DataSequence(dataset, tokenizer)
    train_dataloader = DataLoader(train_dataset, num_workers=4, batch_size=bs, shuffle=True)
    train_loop(model, train_dataloader, device, epochs, optimizer, criterion)


def train_loop(model, dataloader, device, epochs, optimizer, criterion):
    now = datetime.now()
    best_loss = 1000
    lossess = []

    for i in range(epochs):
        model.train()

        total_loss_train = 0.0

        for train_data, train_label in tqdm(dataloader):

            train_label = torch.tensor(train_label).to(device)
            train_data['input_ids'] = train_data['input_ids'].to(device)
            train_data['attention_mask'] = train_data['attention_mask'].to(device)
            if 'token_type_ids' in train_data.keys():
                del train_data['token_type_ids']

            train_data = collate_fn(train_data)

            optimizer.zero_grad()

            output = [torch.as_tensor(model(feature)['sentence_embedding'], dtype=torch.float) for feature in
                      train_data]

            output_1 = torch.stack([out[0] for out in output])
            output_2 = torch.stack([out[1] for out in output])

            loss = criterion(output_1, output_2, train_label)
            total_loss_train += loss.item()
            loss.backward()
            optimizer.step()

        print(f'Epochs: {i + 1} | Loss: {loss.item()}')

        lossess.append(loss.item())

        if loss.item() < best_loss:
            best_loss = loss.item()
            save_path = f'./checkpoints/{model.model_name} : {now.strftime("%m-%d-%Y-%H-%M-%S")}'
            torch.save(model, save_path)
            print(f'Best model saved to! {save_path}')
