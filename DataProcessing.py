import openai
import pandas as pd
from dotenv import load_dotenv
import os
from torch.utils.data import DataLoader, Dataset


def create_dataset(index,
                   pairs_num=5,
                   max_tokens=3000,
                   temperature=0):
    load_dotenv()
    openai.api_key = os.getenv('OPENAI_KEY')

    text = openai.Completion.create(
        model='text-davinci-003',
        prompt=f'Создай список из {pairs_num} пар вопросов и ответов по данному тексту:{index}Список:',
        max_tokens=3000,
        temperature=0

    )

    qa = text['choices'][0]['text']

    questions = []
    answers = []

    for el in qa.split('\n'):
        if el == '':
            continue
        else:
            if el[0].isdigit():
                questions.append(el[3:])
            else:
                answers.append(el[7:])

    return pd.DataFrame({'question': questions, 'answer': answers})


class FewShotDataset(Dataset):
    def __init__(self, data):
        pass
