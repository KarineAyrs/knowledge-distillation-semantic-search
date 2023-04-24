from data_processing.openai_processing import create_openai_samples, openai_samples_to_csv
from data_processing.llama_processing import create_llama_samples, llama_samples_to_csv
from train.Train import train_and_save


def pipeline(llm, lm, llm_path='./model/gpt4all-lora-quantized-new.bin', docs_path='./data/docs.csv'):
    train_csv = ''
    print('generating data...\n')
    if llm == 'openai':
        create_openai_samples(docs_path)
        openai_samples_to_csv()
        train_csv = './data/openai_train.csv'

    else:
        create_llama_samples(llm_path, docs_path)
        llama_samples_to_csv()
        train_csv = './data/llama_train_test.csv'

    print('start training...\n')
    train_and_save(lm, train_csv)
