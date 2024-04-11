from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline, LlamaForQuestionAnswering, AutoModelForQuestionAnswering
import torch
import pandas as pd
import sys
# load arg argument.
#model_name = sys.argv[1]
# model_name = "/model-weights/gemma-7b-it"

base_models = ["/model-weights/Llama-2-7b-hf", "/model-weights/Llama-2-13b-hf", "/model-weights/Mistral-7B-v0.1", "/model-weights/gemma-7b"]
model_names = ["/model-weights/Llama-2-7b-hf", "/model-weights/Llama-2-13b-hf", "/model-weights/Mistral-7B-v0.1", "/model-weights/gemma-7b"]

# model_name = model_names[0]

# spilit_by = "<end_of_turn>" if model_name == "/model-weights/gemma-7b-it" else "[/INST]"
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
# )

# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         quantization_config=bnb_config,
#         torch_dtype=torch.bfloat16,
#         device_map="auto",
#         trust_remote_code=True,
#     )


# pipe = pipeline(
#     "text-generation", 
#     model=model, 
#     tokenizer = tokenizer, 
#     torch_dtype=torch.bfloat16, 
#     device_map="auto"
# )


# # chat = [
# # {"role": "user", "content": f'For this task, you have to come up with original and creative uses for bottle. The goal is to come up with creative ideas, which are ideas that strike people as clever, unusual, interesting, uncommon, humorous, innovative, or different. Your ideas don\'t have to be practical or realistic; they can be silly or strange, even, so long as they are creative uses rather than ordinary uses. You can type in FIVE ideas, but creative quality is more important than quantity. It\'s better to have a few really good ideas than a lot of uncreative ones.\n'}
# # ]
# # prompt = tokenizer.apply_chat_template(chat, tokenize=False)

# prompt = f'One April 6 2024, Joseph and John exchanged one text message. Joseph said: "For this task, you have to come up with original and creative uses for bottle. The goal is to come up with creative ideas, which are ideas that strike people as clever, unusual, interesting, uncommon, humorous, innovative, or different. Your ideas don\'t have to be practical or realistic; they can be silly or strange, even, so long as they are creative uses rather than ordinary uses. You can type in FIVE ideas, but creative quality is more important than quantity. It\'s better to have a few really good ideas than a lot of uncreative ones". Then, John gave a list of ideas:.\n'

# sequences = pipe(
#     prompt,
#     do_sample=True,
#     temperature=0.5, 
#     max_new_tokens=512, 
#     top_k=50, 
#     top_p=0.95,
#     num_return_sequences=1,
# )

# print(sequences[0]['generated_text'])


for model_name in model_names:
    spilit_by = "<end_of_turn>" if model_name == "/model-weights/gemma-7b-it" else "[/INST]"
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

            chat = [
            {"role": "user", "content": f'For this task, you have to come up with original and creative uses for {item}. The goal is to come up with creative ideas, which are ideas that strike people as clever, unusual, interesting, uncommon, humorous, innovative, or different. Your ideas don\'t have to be practical or realistic; they can be silly or strange, even, so long as they are creative uses rather than ordinary uses. You can type in FIVE ideas, but creative quality is more important than quantity. It\'s better to have a few really good ideas than a lot of uncreative ones.\n'}
            ]
            prompt = tokenizer.apply_chat_template(chat, tokenize=False)
            if model_name in base_models:
                prompt = f'One April 6 2024, Joseph and John exchanged one text message. Joseph said: "For this task, you have to come up with original and creative uses for {item}. The goal is to come up with creative ideas, which are ideas that strike people as clever, unusual, interesting, uncommon, humorous, innovative, or different. Your ideas don\'t have to be practical or realistic; they can be silly or strange, even, so long as they are creative uses rather than ordinary uses. You can type in FIVE ideas, but creative quality is more important than quantity. It\'s better to have a few really good ideas than a lot of uncreative ones". Then, John gave a list of ideas:.\n'

            sequences = pipe(
                prompt,
                do_sample=True,
                temperature=0.5, 
                max_new_tokens=512, 
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
            df.to_csv(f'idea1/{model_name.split("/")[2]}.csv', index=False)