from langchain.prompts import ChatPromptTemplate, PromptTemplate, FewShotChatMessagePromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from vllm import LLM, SamplingParams
from langchain_community.llms import Ollama
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
import sys
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from langchain.prompts import ChatPromptTemplate, PromptTemplate, FewShotChatMessagePromptTemplate
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import json
import requests
import os
import re
import time

model_map = {
    'qwen2-7b': {
        'api': 'http://10.100.30.240:1222/generate',
        'tokenizer': 'Qwen/Qwen2-7B'
    },
    'qwen2-72b': {
        'api': 'http://10.100.30.240:1224/generate',
        'tokenizer': 'Qwen/Qwen2-72B'
    },
    'llama3-8b': {
        'api': 'http://10.100.30.240:1223/generate',
        'tokenizer': 'meta-llama/Meta-Llama-3-8B-Instruct'
    },
    'llama3-70b': {
        'api': 'http://10.100.30.239:1225/generate',
        'tokenizer': 'meta-llama/Meta-Llama-3-70B-Instruct'
    },
}

def get_model_list():
    return list(model_map.keys())

def get_format_instructions(parser):
    reduced_schema = {k: v for k, v in parser.pydantic_object.schema().items()}
    print(reduced_schema)
    if "title" in reduced_schema:
        del reduced_schema["title"]
    if "type" in reduced_schema:
        del reduced_schema["type"]

    example = {}
    for key, property in reduced_schema['properties'].items():
        description = f'<{property["description"]}>'
        if property['type'] == 'array':
            example[key] = [description]
        else:
            example[key] = description
    schema_str = json.dumps(example, ensure_ascii=False)

    return f'Выведи результат в блоке json следуя следующему формату:\n```json\n{schema_str}\n```'

def call_api(api_url, prompt, max_tokens=1536, top_k=50, top_p=0.95, temperature=1):
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature
    }
    
    response = requests.post(api_url, json=payload)
    
    if response.status_code == 200:
        print(f"response = {response.json()}")
        return response.json()
    else:
        raise Exception(f"API call failed with status code {response.status_code}: {response.text}")

def chat_template_messages(template, args):
    messages = template.invoke(args).to_messages()
    result = []
    for message in messages:
        role = message.type
        if role == 'human':
            role = 'user'
        if role == 'ai':
            role = 'assistant'
        if role != 'user' and role != 'assistant' and role != 'system':
            raise Exception(f'Unsupported role {role}')
        
        result.append({'role': role, 'content': message.content})

    return result

def use_chat_template(template, args, tokenizer):
    messages = chat_template_messages(template, args)
    if "format_instructions" in args.keys():
        messages[-1]['content'] += f'\n\n{args["format_instructions"]}'
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False, add_generation_prompt=True
    )

overall_time = 0
num_calls = 0

class ChatChain:
    def __init__(self, template, tokenizer, api_url) -> None:
        self.chat_template = template
        self.api_url = api_url
        self.tokenizer = tokenizer

    def invoke(self, args):
        global overall_time, num_calls
        prompt = use_chat_template(self.chat_template, args, self.tokenizer)
        print(f"prompt = {prompt}")
        start_time = time.time()
        response = call_api(self.api_url, prompt)
        end_time = time.time()
        num_calls += 1
        overall_time += (end_time - start_time)
        print(f'API call times: {(overall_time / num_calls):.2f}s')
        return response
    
def str_to_json(output):
    if not isinstance(output, str):
        raise ValueError("output should be a string")

    # Попытка преобразовать строку в JSON
    try:
        # Убираем переносы строк, если они есть
        output = output.replace("\n", "")
        # Регулярное выражение для извлечения JSON данных
        match = re.search(r'```json\{(.*?)\}```|```json\{(.*?)\}```|```json\{(.*?)\}|\{(.*?)\}', output)

        if match:
            # Извлекаем JSON строку из регулярного выражения
            json_string = match.group(1) or match.group(2) or match.group(3) or match.group(4)
            
            # Преобразование строки в JSON объект
            json_object = json.loads(f'{{{json_string}}}')
            
            return json_object
        else:
            print("JSON не найден")
            raise Exception('JSON could not be found')
    except json.JSONDecodeError as e:
        print(f"Ошибка декодирования output в output_json: {e}")
        raise Exception('JSON could not be parsed')
