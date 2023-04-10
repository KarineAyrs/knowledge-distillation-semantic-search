import pandas as pd
import re

from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain


def create_dataset(data, text_file, model_chain):
    for i, ind in enumerate(data.index.tolist()):
        doc = data.loc[ind]['snippet'].encode('ascii', errors='ignore').decode('utf-8')

        answer = model_chain.run(doc)
        answer = answer.replace('\"', '').replace('\n', '')
        answer = re.split('\d*\.\s', answer)[1:]

        n = len(answer)

        for j in range(n // 2):
            text_file.write(f'{i+1} $:$ {answer[j]} $:$ {answer[n - j - 1]}\n')

        print(f'created pairs for doc {ind}/{len(data)}\n')


if __name__ == "__main__":
    template = """

    Question: Which are the ten most relevant searching queries in English for this text: {question}? List them.
    Answer:

    """

    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm = LlamaCpp(model_path='./gpt4all-lora-quantized-new.bin')
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    flights = pd.read_csv('flights.csv')
    flights_ranks = flights[['searchTerms', 'rank', 'snippet']]

    with open('data/llama_samples.txt', 'a') as f:
        create_dataset(flights_ranks, f, llm_chain)
