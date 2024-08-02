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
from utils import model_map, get_format_instructions, ChatChain, str_to_json

system_message = "Ты профессиональный экзаменатор с глубоким знанием предмета. Ты должен отвечать на Русском языке."
question = "Вопрос: Какие из предложенных действий вы выполните ? Расположите их в правильной последовательности"

class Description(BaseModel):
    case_desc: str = Field(description="Описание кейса")
    
class GeneratedQuestion(BaseModel):
    right_steps: list[str] = Field(description="Сгенерированные правильные шаги")
    
class Distractors(BaseModel):
    negative_steps: list[str] = Field(description="Сгенерированные неправильные шаги")
    
class ExamQuestion(BaseModel):
    question: str
    correct_answer: str
    distractors: list[str] = Field(default_factory=list)

    def to_string(this):
        return f"{this.question}\n\n# Правильный ответ:\n{this.correct_answer}"

    def to_string_with_distractors(this):
        return f"{this.question}\n\n# Правильный ответ:\n{this.correct_answer}\n\n# Дистракторы:\n - {this.distractors[0]}\n - {this.distractors[1]}\n - {this.distractors[2]}"

class ExamQuestionType2(BaseModel):
    case_name: str
    competence: str
    
    def to_string_case_name(this):
        return f"Название кейса: {this.case_name}"
    
    def to_string_competence(this):
        return f"Описание компетенции: {this.competence}"


parser_desc = PydanticOutputParser(pydantic_object=Description)
parser_generation = PydanticOutputParser(pydantic_object=GeneratedQuestion)
parser_distractors = PydanticOutputParser(pydantic_object=Distractors)

few_shot_examples = {
    'desc': [
        {
            'case_name': "Кейс №1.              «Анализ данных об НЛО»",
            "competence": "№360  «Использует подходящие техники и инструменты языка программирования для написания структурированного программного кода»",
            'output': '##Логические размышления:\n\n1. ###Чтение входных данных:\n   - Читаем число \( N \), указывающее количество точек.\n   - Считываем \( N \) пар координат (широта и долгота) и сохраняем их в список.\n\n2. ###Сгруппировать точки:\n   - Используем словарь для подсчета количества повторений каждой точки. Ключом будет кортеж координат (широта, долгота), а значением — количество повторений этой точки.\n\n3. ###Упорядочить точки:\n   - Преобразуем словарь в список кортежей, где каждый кортеж содержит координаты точки и количество её повторений.\n   - Сортируем список сначала по убыванию количества повторений, затем по возрастанию широты, и, наконец, по возрастанию долготы.\n\n4. ###Вывод результата:\n   - Печатаем точки в указанном порядке, каждая точка на новой строке в формате "широта долгота количество_повторений".\n\n```json\n{\n  "case_desc": "Вы разрабатываете сервис - анализатор координат мест обнаружения НЛО. Входом стандартного ввода программы является натуральное число \( N \) (количество точек в журнале), а затем в точности \( N \) точек, идущих с новой строки каждая, заданных парой координат (32-битные целые неотрицательные широта и долгота). Необходимо учесть, что объём обрабатываемых данных значительный (до \( 10^8 \) записей о событиях), он может быть размещён в оперативной памяти сервера, но алгоритмы и структуры данных для его обработки должны быть достаточно эффективными (с точки зрения вычислительной сложности и объёмов требуемой памяти). \n\nВ журнале точки могут повторяться более одного раза. Программа должна сгруппировать точки с одинаковыми координатами и упорядочить точки по убыванию количества повторений. Если две точки имеют одинаковое количество повторений, то точки упорядочиваются по возрастанию широты, а при совпадении широты - по возрастанию долготы. Результат необходимо напечатать на стандартный вывод."\n}\n```',
            'format_instructions': get_format_instructions(parser_desc)
        },
    ],
    'generation': [
        {
            "competence": "№360  «Использует подходящие техники и инструменты языка программирования для написания структурированного программного кода»",
            "case_desc": "Вы разрабатываете сервис - анализатор координат мест обнаружения НЛО. Входом стандартного ввода программы является натуральное число \( N \) (количество точек в журнале), а затем в точности \( N \) точек, идущих с новой строки каждая, заданных парой координат (32-битные целые неотрицательные широта и долгота). Необходимо учесть, что объём обрабатываемых данных значительный (до \( 10^8 \) записей о событиях), он может быть размещён в оперативной памяти сервера, но алгоритмы и структуры данных для его обработки должны быть достаточно эффективными (с точки зрения вычислительной сложности и объёмов требуемой памяти). \n\nВ журнале точки могут повторяться более одного раза. Программа должна сгруппировать точки с одинаковыми координатами и упорядочить точки по убыванию количества повторений. Если две точки имеют одинаковое количество повторений, то точки упорядочиваются по возрастанию широты, а при совпадении широты - по возрастанию долготы. Результат необходимо напечатать на стандартный вывод.",
            "question": "Вопрос: Какие из предложенных действий вы выполните ? Расположите их в правильной последовательности",
            "output": """```json\n{\n"right_steps": [\n
                "Начало",
                "Подключу необходимые заголовочные файлы стандартной библиотеки.",
                "Объявлю вспомогательную структуру для подсчёта повторов точек:\nstruct PointCnt : public Point {\n    unsigned cnt;\n    /* необходимые операторы, конструкторы */\n};",
                "Объявлю структуру, описывающую координатную точку:\nstruct Point {\n    unsigned x;\n    unsigned y;\n    /* необходимые операторы, конструкторы */\n};",
                "Объявлю функцию main, считаю значение параметра N,\nпроверю что введено ненулевое значение параметра N.",
                "Объявлю вектор точек, std::vector<Point> points(n);\nСчитаю точки из стандартного ввода.",
                "Отсортирую вектор точек: std::sort(points.begin(), points.end())",
                "Объявлю вектор точек с количеством повторов: std::vector<PointCnt> points_cnt;\nНаполню points_cnt на основе содержимого контейнера points.\nОтсортирую контейнер points_cnt:\nstd::sort(points_cnt.begin(), points_cnt.end(),\n        [] (const PointCnt& p1, const PointCnt& p2) {\n            return p1.cnt > p2.cnt ? true : (p1 < p2);\n        }\n    );",
                "Напечатаю содержимое points_cnt на стандартный вывод",
                "Конец",
                ]\n}\n```""",
            'format_instructions': get_format_instructions(parser_generation)
        }
    ],
    'distractors': [
        {
            "competence": "№360  «Использует подходящие техники и инструменты языка программирования для написания структурированного программного кода»",
            "case_desc": "Вы разрабатываете сервис - анализатор координат мест обнаружения НЛО. Входом стандартного ввода программы является натуральное число \( N \) (количество точек в журнале), а затем в точности \( N \) точек, идущих с новой строки каждая, заданных парой координат (32-битные целые неотрицательные широта и долгота). Необходимо учесть, что объём обрабатываемых данных значительный (до \( 10^8 \) записей о событиях), он может быть размещён в оперативной памяти сервера, но алгоритмы и структуры данных для его обработки должны быть достаточно эффективными (с точки зрения вычислительной сложности и объёмов требуемой памяти). \n\nВ журнале точки могут повторяться более одного раза. Программа должна сгруппировать точки с одинаковыми координатами и упорядочить точки по убыванию количества повторений. Если две точки имеют одинаковое количество повторений, то точки упорядочиваются по возрастанию широты, а при совпадении широты - по возрастанию долготы. Результат необходимо напечатать на стандартный вывод.",
            "question": "Вопрос: Какие из предложенных действий вы выполните ? Расположите их в правильной последовательности",
            "correct_steps": " - Начало\n - Подключу необходимые заголовочные файлы стандартной библиотеки.\n - Объявлю вспомогательную структуру для подсчёта повторов точек:\\nstruct PointCnt : public Point {\\n    unsigned cnt;\\n    /* необходимые операторы, конструкторы */\\n};\n - Объявлю структуру, описывающую координатную точку:\\nstruct Point {\\n    unsigned x;\\n    unsigned y;\\n    /* необходимые операторы, конструкторы */\\n};\n - Объявлю функцию main, считаю значение параметра N,\\nпроверю что введено ненулевое значение параметра N.\n - Объявлю вектор точек, std::vector<Point> points(n);\\nСчитаю точки из стандартного ввода.\n - Отсортирую вектор точек: std::sort(points.begin(), points.end())\n - Объявлю вектор точек с количеством повторов: std::vector<PointCnt> points_cnt;\\nНаполню points_cnt на основе содержимого контейнера points.\\nОтсортирую контейнер points_cnt:\\nstd::sort(points_cnt.begin(), points_cnt.end(),\\n        [] (const PointCnt& p1, const PointCnt& p2) {\\n            return p1.cnt > p2.cnt ? true : (p1 < p2);\\n        }\\n    );\n - Напечатаю содержимое points_cnt на стандартный вывод\n - Конец",
            "output": """```json\n{\n"negative_steps": [\n
                "Объявлю множество точек, std::unordered_map<Point, unsigned> points(n); \nСчитаю точки из стандратного ввода.",
                "Объявлю карту точек с количеством повторов: std::map<int, PointCnt> points_cnt;\nНаполню points_cnt на основе содержимого контейнера points.",
                "Объявлю множество точек, std::map<Point, PointsCnt> points; \nСчитаю точки из стандратного ввода.",
                "Объявлю струтуру, описывающую координатную точку:\nstruct Point {\n    unsigned char x, y;\n    /* необходимые операторы, конструкторы */\n};\""
                ]\n}\n```""",
            'format_instructions': get_format_instructions(parser_distractors)
        },
    ],
}

def get_model(primary_model='qwen2-72b', num_examples=1):
    few_shot = True
    if num_examples <= 0:
        few_shot = False
    if primary_model not in model_map:
        raise Exception(f'{primary_model} is not a valid or supported model')
    model_info = model_map[primary_model]
    api_url = model_info['api']
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_info['tokenizer'], trust_remote_code=True)

    desc_prompt_few_shot = ChatPromptTemplate.from_messages(
        [
            ("human", "# Название кейса: {case_name}\n\n Описание компетенции на развитие которого направлена наша задача: {competence} \n\nТщательно проанализируй требуемую компетенцию и составь на её основе и на основе названия кейса сам кейс - его описание.\n\n{format_instructions}"),
            ("ai", "{output}")
        ]
    )
    
    desc_few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=desc_prompt_few_shot,
        examples=few_shot_examples['desc'][:num_examples],
    )
    
    desc_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            *([desc_few_shot_prompt] if few_shot else []),
            ("human", "# Название кейса: {case_name}\n\n Описание компетенции на развитие которого направлена наша задача: {competence} \n\nТщательно проанализируй требуемую компетенцию и составь на её основе и на основе названия кейса сам кейс - его описание."),
        ]
    )

    desc_chain = ChatChain(desc_prompt_template, tokenizer, api_url)
    
    def generate_desc(case_name, competence):
        desc_args = {
            "case_name": case_name,
            "competence": competence,
            "format_instructions": get_format_instructions(parser_desc)
        }
        print(f"desc_args = {desc_args}")
        while True:
            try:
                desc_result = desc_chain.invoke(desc_args)
                desc_json = str_to_json(desc_result)
                desc = desc_json['case_desc']
                
                print(f"desc = {desc}")
                return desc
            except:
                pass

    def get_generation_prompt():
        generation_prompt_few_shot = ChatPromptTemplate.from_messages(
            [
                ("human", "# Описание компетенции: {competence}\n\n# Описание кейса: {case_desc}\n```\n# Вопрос: {question}\n```\n\n\nТы должен сгенерировать правильные шаги в правильной последовательности Для начала тщательно проанализируй тему и используй цепочку размышлений и напиши свой анализ. После этого, сгенерируй правильные шаги. Ты не должен повторять или перефразировать существующие вопросы. Ты должен сгенерировать уникальные шаги. Не генерируй варианты ответа, только правильный ответ. Не добавляй букву или номер к ответу. \n\n{format_instructions}"),
                ("ai", "{output}")
            ]
        )

        generation_few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=generation_prompt_few_shot,
            examples=few_shot_examples['generation'][:num_examples],
        )

        generation_prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                *([generation_few_shot_prompt] if few_shot else []),
                ("human", "# Описание компетенции: {competence}\n\n# Описание кейса: {case_desc}\n```\n# Вопрос: {question}\n```\n\n\nТы должен сгенерировать правильные шаги в правильной последовательности. Для начала тщательно проанализируй тему и используй цепочку размышлений и напиши свой анализ. После этого, сгенерируй правильные шаги. Ты должен сгенерировать уникальные шаги. Не добавляй букву или номер к ответу."),
            ]
        )

        return ChatChain(generation_prompt_template, tokenizer, api_url)
    
    def get_distractors_prompt():
        # Step 5: Generate 3 distractors
        distractors_few_shot = ChatPromptTemplate.from_messages(
            [
                ("human", "# Описание компетенции: {competence}\n\n# Описание кейса: ```\n{case_desc}\n```\n# Вопрос: {question}\n# Правильные шаги в правильной последовательности: ```\n{correct_steps}\n```\n\n\nТы должен сгенерировать правдоподобные, но неправильные шаги (дистракторы), похожие на правильные. Для начала тщательно проанализируй тему и используй цепочку размышлений и напиши свой анализ. После этого, сгенерируй неправильные шаги (дистракторы). Ты должен сгенерировать уникальные шаги. Ты должен сгенерировать неочевидные (правдоподобные) неправильные шаги (дистракторы). Не добавляй букву или номер к ответу.\n\n{format_instructions}"),
                ("ai", "{output}")
            ]
        )

        distractors_few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=distractors_few_shot,
            examples=few_shot_examples['distractors'][:num_examples],
        )

        distractors_prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                *([distractors_few_shot_prompt] if few_shot else []),
                ("human", "# Описание компетенции: {competence}\n\n# Описание кейса: ```\n{case_desc}\n```\n# Вопрос: {question}\n# Правильные шаги в правильной последовательности: ```\n{correct_steps}\n```\n\n\nТы должен сгенерировать правдоподобные, но неправильные шаги (дистракторы), похожие на правильные. Для начала тщательно проанализируй тему и используй цепочку размышлений и напиши свой анализ. После этого, сгенерируй неправильные шаги (дистракторы). Ты должен сгенерировать уникальные шаги. Ты должен сгенерировать неочевидные (правдоподобные) неправильные шаги (дистракторы). Не добавляй букву или номер к ответу."),
            ]
        )

        distractors_chain = ChatChain(distractors_prompt_template, tokenizer, api_url)
        return distractors_chain
    
    def generate_steps(competence, case_desc):
        print('Started generating steps')
        generation_chain = get_generation_prompt()
        generation_args = {
            "competence": competence,
            "case_desc": case_desc,
            "question": question,
            "format_instructions": get_format_instructions(parser_generation)
        }
        print('Start generation of right steps')
        while True:
            try:
                generated_steps_right = generation_chain.invoke(generation_args)
                generated_steps_right_json = str_to_json(generated_steps_right)
                generated_steps_right = generated_steps_right_json['right_steps']
                break
            except Exception as e:
                print('Repeat attempt generation', e)
                pass
        
        print(f"generated_steps_right = {generated_steps_right}")
        
        # Шаг 3: Генерация дистракторов
        distractors_chain = get_distractors_prompt()
        distractors_args = {
            "competence": competence,
            "case_desc": case_desc,
            "question": question,
            "correct_steps": '\n'.join([f' - {step}' for step in generated_steps_right]),
            "format_instructions": get_format_instructions(parser_distractors)
        }
        print(f"distractors_args = {distractors_args}")
        while True:
            try:
                distractors_result = distractors_chain.invoke(distractors_args)
                distractors_json = str_to_json(distractors_result)
                break
            except Exception as e:
                print('Repeat attempt distractors', e)
                pass
        
        print(f"distractors_json = {distractors_json}")
        
        return {
            "generated_steps_right": generated_steps_right,
            "distractors": distractors_json
        }

    return generate_desc, generate_steps
