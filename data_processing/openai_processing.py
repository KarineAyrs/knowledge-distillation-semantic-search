import openai
import pandas as pd
import os

PAIRS_NUM = 5
MAX_TOKENS = 500
TEMPERATURE = 0.9


def __create_pairs(doc):
    openai.api_key = os.getenv('OPENAI_KEY')
    text = openai.Completion.create(
        model='text-davinci-003',
        prompt=f'Generate numerated list of {PAIRS_NUM} pairs of right \
            search queries for text. Queries in pair should be similar by meaning and separated by ":". \
            Text: {doc}\nList:',
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE

    )
    pairs = []
    generated_pairs = text['choices'][0]['text']
    for el in generated_pairs.split('\n'):
        if el == '':
            continue
        else:
            if el[0].isdigit():
                splitted = el[3:].split(':')

                if len(splitted) == 1:
                    splitted.append('')
                elif len(splitted) == 0:
                    splitted.append('')
                    splitted.append('')

                pairs.append([splitted[0].strip(), splitted[1].strip()])

    return pairs


def __create_dataset(data, text_file):
    to_csv_samples = {'first': [], 'second': []}
    for i, ind in enumerate(data.index.tolist()):
        pairs = __create_pairs(data.loc[ind]['docs'])

        for pair in pairs:
            to_csv_samples['first'].append(pair[0])
            to_csv_samples['second'].append(pair[1])

            text_file.write(f'{ind} : {pair[0]} : {pair[1]}\n')

        print(f'created pairs for doc {i + 1} / {len(data)}\n')


def create_openai_samples(docs_path):
    docs = pd.read_csv(docs_path)

    with open('./data/openai_samples.txt', 'a') as f:
        __create_dataset(docs, f)


def openai_samples_to_csv():
    with open('./data/openai_samples.txt') as f:
        lines = f.read()
        lines = lines.split('\n')
        good_lines = []
        bad_lines = []

        for line in lines:
            splitted = line.split(',')
            if len(splitted) == 3:
                good_lines.append(':'.join(splitted))
            else:
                if line.find('\"') != -1:
                    q_splitted = line.split('\"')
                    print(q_splitted)
                    try:
                        good_lines.append(':'.join([q_splitted[0].replace(',', ''), q_splitted[1], q_splitted[3]]))
                    except IndexError:
                        bad_lines.append(q_splitted)

        d = {'ids': [], 'first': [], 'second': []}

        for line in good_lines:
            splitted = line.split(':')
            d['ids'].append(int(splitted[0]))
            d['first'].append(splitted[1].replace('\"', ''))
            d['second'].append(splitted[2].replace('\"', ''))

        pd.DataFrame.from_dict(d).to_csv('./data/openai_train.csv', index=False)
