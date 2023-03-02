from torch import nn
from transformers import Trainer
from DataProcessing import create_dataset, FewShotDataset, DataLoader
from sklearn.model_selection import train_test_split


class DistilTrainer:
    def __init__(self,
                 index,
                 queries,
                 model,
                 pairs_num,
                 training_args,
                 optimizer,
                 scheduler,
                 callbacks):
        self.index = index
        self.pairs_num = pairs_num
        self.model = model
        self.queries = queries
        self.training_args = training_args
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.callbacks = callbacks

    def __process_data(self):
        dataset = create_dataset(self.index, self.pairs_num)
        train, val = train_test_split(dataset, test_size=0.2, random_state=42)
        self.train_dataset = FewShotDataset(train)
        self.val_dataset = FewShotDataset(val)

        self.train_loader = DataLoader(self.train_dataset,
                                       shuffle=True)

        self.val_loader = DataLoader(self.val_dataset,
                                     shuffle=True)

    def train(self):
        self.__process_data()

        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=self.train_loader.collate_fn,
            optimizers=(self.optimizer, self.scheduler),
            callbacks=self.callbacks
        )

        trainer.train()

        return self.model
