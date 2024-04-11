from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline, LlamaForQuestionAnswering, AutoModelForQuestionAnswering
import torch
import pandas as pd
import sys
# load arg argument.
# model_names = ["/model-weights/Llama-2-7b-chat-hf", "/model-weights/Mistral-7B-Instruct-v0.1", "/model-weights/gemma-7b-it", "/model-weights/Llama-2-13b-chat-hf"]

# model_names = ["/model-weights/gemma-7b-it", "/model-weights/Llama-2-13b-chat-hf"]
agent_1_model_name = "/model-weights/Llama-2-7b-chat-hf"
agent_2_model_name = "/model-weights/Mistral-7B-Instruct-v0.1"
agent_1_bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
agent_2_bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
agent_1_tokenizer = AutoTokenizer.from_pretrained(agent_1_model_name)
agent_1_model = AutoModelForCausalLM.from_pretrained(
        agent_1_model_name,
        quantization_config=agent_1_bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
agent_2_tokenizer = AutoTokenizer.from_pretrained(agent_2_model_name)
agent_2_model = AutoModelForCausalLM.from_pretrained(
        agent_2_model_name,
        quantization_config=agent_2_bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

agent_1_pipe = pipeline(
    "text-generation", 
    model=agent_1_model, 
    tokenizer = agent_1_tokenizer, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)


agent_2_pipe = pipeline(
    "text-generation", 
    model=agent_2_model, 
    tokenizer = agent_2_tokenizer, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)
session_ids = []
item_names = []
responses = []
conditions = []
for item in ["tire", "pants", "shoe", "table", "bottle"]:
    for i in range(0, 10):
        spilit_by = "[/INST]"

        chat = [
        {"role": "user", "content": f'For this task, you have to come up with original and creative uses for {item}. The goal is to come up with creative ideas, which are ideas that strike people as clever, unusual, interesting, uncommon, humorous, innovative, or different. Your ideas don\'t have to be practical or realistic; they can be silly or strange, even, so long as they are creative uses rather than ordinary uses. You can type in FIVE ideas, but creative quality is more important than quantity. It\'s better to have a few really good ideas than a lot of uncreative ones.\n'}
        ]
        prompt = agent_1_tokenizer.apply_chat_template(chat, tokenize=False)
        sequences = agent_1_pipe(
            prompt,
            do_sample=True,
            temperature=0.5, 
            max_new_tokens=512, 
            top_k=50, 
            top_p=0.95,
            num_return_sequences=1,
        )

        model_1_first_idea = sequences[0]['generated_text'].split(spilit_by)[-1]


        chat_for_feedback = [
            {"role": "user", "content": model_1_first_idea}, 
            {"role": "user", "content": f'Can you give suggestions on how to improve the overall creativity of the list? This could include the originality of each idea, the diversity of the ideas in the list, and the number of ideas. Don’t fix the list, just give suggestions.\n'}
        ]
        prompt = agent_2_tokenizer.apply_chat_template(chat, tokenize=False)
        sequences = agent_2_pipe(
            prompt,
            do_sample=True,
            temperature=0.5, 
            max_new_tokens=512, 
            top_k=50, 
            top_p=0.95,
            num_return_sequences=1,
        )

        agent_2_feedback = sequences[0]['generated_text'].split(spilit_by)[-1]
        chat_for_refined_list = [{"role": "user", "content": f'For this task, you have to come up with original and creative uses for {item}. The goal is to come up with creative ideas, which are ideas that strike people as clever, unusual, interesting, uncommon, humorous, innovative, or different. Your ideas don\'t have to be practical or realistic; they can be silly or strange, even, so long as they are creative uses rather than ordinary uses. You can type in FIVE ideas, but creative quality is more important than quantity. It\'s better to have a few really good ideas than a lot of uncreative ones.\n'},{"role": "assistant", "content": model_1_first_idea}, {"role": "user", "content": f'Here are some suggestions to improve the overall creativity of the list: {agent_2_feedback}'},{"role": "user", "content": f'Now update the list based on the suggestions.\n'}]

        prompt = agent_1_tokenizer.apply_chat_template(chat, tokenize=False)
        sequences = agent_1_pipe(
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
        df.to_csv(f'multi_agents/{agent_1_model_name.split("/")[2]}.csv', index=False)