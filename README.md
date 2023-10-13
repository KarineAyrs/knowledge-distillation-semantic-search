# Knowledge Distillation based Semantic Search
KDSS is the framework that allows to use knowledge of large language models to train smaller models.




![kdss_resized](https://user-images.githubusercontent.com/52883493/230601804-aaea074b-237f-43c4-8885-dfa7104e9262.png)

## KDSS Usage
- Fine-tuned model will be saved in folder `checkpoints`. Then you can use it to construct embeddings!
- `main.py [-h] --large_model {openai,alpaca} [--alpaca_path [ALPACA_PATH]] --model {bert,xlmroberta,debertav3} [--path_to_docs [PATH_TO_DOCS]]`
```
  Welcome to KDSS! Please choose LLM for distillation! If openai is chosen, do not 
  forget to add OPENAI_KEY=<your key> to env! In case of alpaca download model by 
  instructions in README.md! There are 3 models are available for tuning now: 
  BERT, XLM-RoBERTa, DeBERTa V3

optional arguments:
  -h, --help            show this help message and exit
  --large_model {openai,alpaca}
                        openai - text-davinci-003, alpaca - Alpaca(LoRa)
  --alpaca_path [ALPACA_PATH]
                        Path to Alpaca(LoRa). See README.md for download instructions
  --model {bert,xlmroberta,debertav3}
                        There are 3 models are available for tuning now: bert - BERT, xlmroberta - XLM-RoBERTa, debertav3 - DeBERTa V3
  --path_to_docs [PATH_TO_DOCS]
                        path to docs to finetune model. File should be in csv format with required column `docs` 
```

- Provide path to `docs.csv` or paste it into `data` folder, where you can find example of correct `docs.csv` file
- Also, provide path or paste Alpaca `gpt4all-lora-quantized.bin` model into `model` folder.

## Run example

- For example, this command 
`python3 main.py --large_model alpaca --model bert` will do following:
  - alpaca is used as LLM and path to model is set to default path `./model/gpt4all-lora-quantized-new.bin`
  - path to docs is also default `./data/docs.csv`
  - LoRa generates 5 semantic similar pairs for each document, resulting file `./data/llama_samples.txt`
  - generated samples are cleaned and saved to csv `./data/llama_train.csv`
  - Then, BERT (`bert-base-uncased`) is trained on `./data/llama_train.csv` and trained model is saved in `./checkpoints/`



## Data examples 
- In `data` folder you can find examples of 
  - `docs.csv` - documents for generating synthetic data
  - `<model>_samples.txt` - generated synthetic data
  - `<model>_train.csv` - cleaned generated synthetic data in csv format

## ALPACA (LoRa)
### Obtain
- Obtain model and paste it to the `model` folder!
- Get model `gpt4all-lora-quantized.bin` from https://github.com/nomic-ai/gpt4all#try-it-yourself
  - If necessary, get tokenizer from https://github.com/juncongmoo/pyllama :
    - `pip install pyllama -U`
    - `python -m llama.download --model_size 7B`
- Run `python3 migrate-ggml-2023-03-30-pr613.py models/gpt4all-7B/gpt4all-lora-quantized.bin models/gpt4all-7B/gpt4all-lora-quantized-new.bin`
  - `migrate-ggml-2023-03-30-pr613.py` from  https://github.com/ggerganov/llama.cpp/blob/master/migrate-ggml-2023-03-30-pr613.py
### ALPACA (LoRa) usage 
- Install `pip install pyllamacpp`
 ```python
from pyllamacpp.model import Model
 
model = Model(ggml_model='./models/gpt4all-lora-quantized-new.bin', n_ctx=512)

generated_text = model.generate("Once upon a time, ", n_predict=55)
print(generated_text)
```
