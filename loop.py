from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch
import pandas as pd
import sys
# load arg argument.
model_name = sys.argv[1]
#model_name = "/model-weights/Llama-2-13b-hf"
print(model_name)


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )


pipe = pipeline(
    "text-generation", 
    model=model, 
    tokenizer = tokenizer, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)

session_ids = []
item_names = []
responses = []
conditions = []
for item in ["tire", "pants", "shoe", "table", "bottle"]:
    for i in range(0, 10):
        print(f'{item}: {str(i)}')

        prompt = f'For this task, you have to come up with original and creative uses for {item}. The goal is to come up with creative ideas, which are ideas that strike people as clever, unusual, interesting, uncommon, humorous, innovative, or different. Your ideas don\'t have to be practical or realistic; they can be silly or strange, even, so long as they are creative uses rather than ordinary uses. You can type in FIVE ideas, but creative quality is more important than quantity. It\'s better to have a few really good ideas than a lot of uncreative ones.'

        sequences = pipe(
            prompt,
            do_sample=True,
            temperature=0.5, 
            max_new_tokens=250, 
            top_k=50, 
            top_p=0.95,
            num_return_sequences=1,
        )
        session_ids.append(i)
        item_names.append(item)
        responses.append(sequences[0]['generated_text'])
        conditions.append('base-input-prompt-only')
        df = pd.DataFrame(data={'session_ids': session_ids, 'item_names': item_names, 'responses': responses, 'conditions': conditions})
        # save to csv.
        df.to_csv(f'{model_name.split("/")[2]}.csv', index=False)