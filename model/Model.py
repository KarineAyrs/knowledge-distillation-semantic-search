import torch
from sentence_transformers import SentenceTransformer, models


class STSBertModel(torch.nn.Module):

    def __init__(self, model_name):
        super(STSBertModel, self).__init__()
        self.model_name = model_name

        word_embedding_model = models.Transformer(model_name, max_seq_length=128)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        self.sts_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    def forward(self, input_data):
        output = self.sts_model(input_data)

        return output

    def encode(self, input_data, convert_to_tensor):
        return self.sts_model.encode(input_data, convert_to_tensor)
