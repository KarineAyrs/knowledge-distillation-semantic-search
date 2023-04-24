import argparse
import os
import pandas as pd
from pipeline import pipeline


def exit_on_error(msg):
    print(msg)
    exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Welcome to KDSS!\nPlease choose LLM for distillation! If openai is chosen, do not
        forget to add OPENAI_KEY=<your key> to env! In case of alpaca download model by instructions in README.md!
        \nThere are 3 models are available for tuning now: BERT, XLM-RoBERTa, DeBERTa V3''')
    action_choices = ['openai', 'alpaca']
    models_choices = ['bert', 'xlmroberta', 'debertav3']
    parser.add_argument('--large_model', action='store', choices=action_choices, dest='large_model',
                        help='openai - text-davinci-003, alpaca - Alpaca(LoRa)',
                        required=True)

    parser.add_argument('--alpaca_path', action='store', dest='alpaca_path',
                        help='Path to Alpaca(LoRa). See README.md for download instructions',
                        nargs='?', default='./model/gpt4all-lora-quantized-new.bin'
                        )
    parser.add_argument('--model', action='store', choices=models_choices, dest='model',
                        help='''There are 3 models are available for tuning now: bert - BERT,
                        xlmroberta - XLM-RoBERTa, debertav3 - DeBERTa V3''',
                        required=True)
    parser.add_argument('--path_to_docs', action='store', dest='path_to_docs',
                        nargs='?', default='./data/docs.csv',
                        help='path to docs to finetune model. File should be in csv format with required column `docs`')

    args = parser.parse_args()

    current_dir = os.getcwd()

    llm, lm, llm_path, docs_path = args.large_model, args.model, args.alpaca_path, args.path_to_docs

    if llm == 'openai' and os.getenv('OPENAI_KEY') == '':
        exit_on_error('Please provide your valid OPENAI_KEY!')

    if llm == 'alpaca' and not os.path.isfile(llm_path):
        exit_on_error('Please provide valid path to alpaca model!')

    if not os.path.isfile(docs_path):
        exit_on_error('Please provide valid path to documents!')

    if not docs_path.lower().endswith('.csv'):
        exit_on_error('Please provide CSV file!')

    docs = pd.read_csv(docs_path)
    if 'docs' not in docs.columns:
        exit_on_error('Please provide CSV with `docs` column!')

    pipeline(llm, lm, llm_path, docs_path)
