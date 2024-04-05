from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline, LlamaForQuestionAnswering, AutoModelForQuestionAnswering
import torch
import pandas as pd
import sys
from helpers import cut_off_text, remove_substring, generate, B_INST, E_INST, B_SYS, E_SYS, DEFAULT_SYSTEM_PROMPT, get_prompt, SYSTEM_PROMPT, generate_response
# load arg argument.
#model_name = sys.argv[1]
model_names = ["/model-weights/Llama-2-7b-chat-hf", "/model-weights/Mistral-7B-Instruct-v0.1"]


def run(model_name):
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

            sequences = pipe(
                prompt,
                do_sample=True,
                temperature=0.5, 
                max_new_tokens=512, 
                top_k=50, 
                top_p=0.95,
                num_return_sequences=1,
            )

            bot_feedback = {"role": "assistant", "content": sequences[0]['generated_text'].split('[/INST]')[1]}
            chat.append(bot_feedback)
            chat.append({"role": "user", "content": f'Can you give suggestions on how to improve the overall creativity of the list? This could include the originality of each idea, the diversity of the ideas in the list, and the number of ideas. Don’t fix the list, just give suggestions.\n'})
            prompt = tokenizer.apply_chat_template(chat, tokenize=False)

            sequences = pipe(
                prompt,
                do_sample=True,
                temperature=0.5, 
                max_new_tokens=512, 
                top_k=50, 
                top_p=0.95,
                num_return_sequences=1,
            )

            bot_feedback = {"role": "assistant", "content": sequences[0]['generated_text'].split('[/INST]')[-1]}
            chat.append(bot_feedback)
            chat.append({"role": "user", "content": f'Now update the list based on the suggestions.\n'})
            prompt = tokenizer.apply_chat_template(chat, tokenize=False)
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
            df.to_csv(f'{model_name.split("/")[2]}_idea2.csv', index=False)

for model_name in model_names:
    run(model_name)