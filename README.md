# Knowledge Distillation based Semantic Search
![](../../study_wr_tex/slides301122/images/kdss.png)


## Obtain LLaMA

- Get model `gpt4all-lora-quantized.bin` from https://github.com/nomic-ai/gpt4all#try-it-yourself
  - If necessary, get tokenizer from https://github.com/juncongmoo/pyllama
- Run `python3 migrate-ggml-2023-03-30-pr613.py models/gpt4all-7B/gpt4all-lora-quantized.bin models/gpt4all-7B/gpt4all-lora-quantized-new.bin`
  - `migrate-ggml-2023-03-30-pr613.py` from  https://github.com/ggerganov/llama.cpp/blob/master/migrate-ggml-2023-03-30-pr613.py
## LLaMA usage 
- Install `pip install pyllamacpp`
 ```
 from pyllamacpp.model import Model
model = Model(ggml_model='./models/gpt4all-lora-quantized-new.bin', n_ctx=512)
generated_text = model.generate("Once upon a time, ", n_predict=55)
print(generated_text)
```