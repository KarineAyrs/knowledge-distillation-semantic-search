import pandas as pd
import re

from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain


def __create_dataset(data, text_file, model_chain):
    for i, ind in enumerate(data.index.tolist()):
        doc = data.loc[ind]['docs'].encode('ascii', errors='ignore').decode('utf-8')

        answer = model_chain.run(doc)
        answer = answer.replace('\"', '').replace('\n', '')
        answer = re.split('\d*\.\s', answer)[1:]

        n = len(answer)

        for j in range(n // 2):
            text_file.write(f'{i + 1} $:$ {answer[j]} $:$ {answer[n - j - 1]}\n')

        print(f'created pairs for doc {ind}/{len(data)}\n')


def create_llama_samples(llm_path, docs_path):
    template = """

        Question: Which are the ten most relevant searching queries in English for this text: {question}? List them.
        Answer:

        """

    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm = LlamaCpp(model_path=llm_path)
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    docs = pd.read_csv(docs_path)

    save_path = './data/llama_samples.txt'
    with open(save_path, 'a') as f:
        __create_dataset(docs, f, llm_chain)


def llama_samples_to_csv():
    print('converting to csv...\n')
    ids = []
    first = []
    second = []

    path = './data/llama_samples.txt'

    with open(path) as f:
        lines = f.readlines()

        for i, line in enumerate(lines):

            line = re.sub(r'http\S+', '', line, flags=re.MULTILINE)
            line = re.sub(r"[\([{<>})\]\/\"\']", "", line)

            splitted_line = line.split('$:$')

            for j in range(len(splitted_line)):
                splitted_line[j] = splitted_line[j].strip()
                splitted_line[j] = splitted_line[j].replace('\n', '')

            if splitted_line[1] != '' and splitted_line[2] != '':
                ids.append(int(splitted_line[0]))
                first.append(splitted_line[1])
                second.append(splitted_line[2])

                print(splitted_line)

    d = {'ids': ids, 'first': first, 'second': second}

    save_path = './data/llama_train.csv'
    train = pd.DataFrame.from_dict(d)
    train.to_csv(save_path, index=False)
