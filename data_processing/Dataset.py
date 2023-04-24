import torch


class DataSequence(torch.utils.data.Dataset):

    def __init__(self, dataset, tokenizer):
        self.data = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text_cat = [str(row['first']), str(row['second'])]

        tokenized = self.tokenizer(text_cat, padding='max_length', max_length=128, truncation=True,
                                   return_tensors="pt")
        return tokenized, 1.0


def collate_fn(texts):
    num_texts = len(texts['input_ids'])
    features = list()
    for i in range(num_texts):
        features.append({'input_ids': texts['input_ids'][i], 'attention_mask': texts['attention_mask'][i]})

    return features
