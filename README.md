# Knowledge Distillation based Semantic Search
![kdss_resized](https://user-images.githubusercontent.com/52883493/230601804-aaea074b-237f-43c4-8885-dfa7104e9262.png)

# Table of contents
- [LLaMa](https://github.com/KarineAyrs/knowledge-distillation-semantic-search##LLaMA)


## LLaMA
### Obtain
- Get model `gpt4all-lora-quantized.bin` from https://github.com/nomic-ai/gpt4all#try-it-yourself
  - If necessary, get tokenizer from https://github.com/juncongmoo/pyllama :
    - `pip install pyllama -U`
    - `python -m llama.download --model_size 7B`
- Run `python3 migrate-ggml-2023-03-30-pr613.py models/gpt4all-7B/gpt4all-lora-quantized.bin models/gpt4all-7B/gpt4all-lora-quantized-new.bin`
  - `migrate-ggml-2023-03-30-pr613.py` from  https://github.com/ggerganov/llama.cpp/blob/master/migrate-ggml-2023-03-30-pr613.py
### LLaMA usage 
- Install `pip install pyllamacpp`
 ```python
from pyllamacpp.model import Model
 
model = Model(ggml_model='./models/gpt4all-lora-quantized-new.bin', n_ctx=512)

generated_text = model.generate("Once upon a time, ", n_predict=55)
print(generated_text)
```
