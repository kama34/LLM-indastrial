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
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import json
import requests
import os
import re

system_message_generation = "Ты профессиональный экзаменатор с глубоким знанием предмета. Твоя задача, написание различных вопросов для студентческого экзамена. Ты должен отвечать на Русском языке."
system_message_validation = "Ты эксперт по предмету и профессор с многолетним опытом. Твоя задача - проверять экзаминационные задания на правильность. Ты должен отвечать на Русском языке."
system_message_distractors = "Ты профессиональный экзаменатор с опытом создания дистракторов. Ты должен отвечать на Русском языке."


class QuestionTheme(BaseModel):
    question_theme: str = Field(description="Тема вопроса")
    
class GeneratedQuestion(BaseModel):
    question: str = Field(description="Сгенерированный вопрос")
    correct_answer: str = Field(description="Правильный ответ")
    
class Distractors(BaseModel):
    distractor_1: str = Field(description="Первый дистрактор")
    distractor_2: str = Field(description="Второй дистрактор")
    distractor_3: str = Field(description="Третий дистрактор")


parser_theme = PydanticOutputParser(pydantic_object=QuestionTheme)
parser_generation = PydanticOutputParser(pydantic_object=GeneratedQuestion)
parser_distractors = PydanticOutputParser(pydantic_object=Distractors)


def get_format_instructions(parser):
    reduced_schema = {k: v for k, v in parser.pydantic_object.schema().items()}
    if "title" in reduced_schema:
        del reduced_schema["title"]
    if "type" in reduced_schema:
        del reduced_schema["type"]

    example = {}
    for key, property in reduced_schema['properties'].items():
        description = f'<{property["description"]}>'
        example[key] = description
    schema_str = json.dumps(example, ensure_ascii=False)

    return f'Выведи результат в блоке json следуя следующему формату:\n```json\n{schema_str}\n```'

few_shot_examples = {
    'theme': [
        {
            'reference_question': '# Вопрос:\nПри реализации метода ближайших соседей, скорее всего будет фигурировать следующее:\n\n# Правильный ответ:\nKD-дерево\n\n# Дистракторы:\n - Алгоритм Дейкстры\n - Поиск A*\n - Красно-чёрное дерево',
            'output': '## Анализ вопроса\n\n### Цепочка размышлений:\n\n1. **Понимание метода ближайших соседей**:\n Метод ближайших соседей (K-Nearest Neighbors, KNN) – это алгоритм классификации и регрессии, который основывается на нахождении ближайших объектов в многомерном пространстве. Он используется в задачах машинного обучения и анализа данных.\n\n2. **Основные концепции метода ближайших соседей**:\n - Для поиска ближайших соседей необходимо эффективно организовать данные в пространстве.\n - Простое решение – перебор всех точек, но это неэффективно для больших наборов данных.\n\n3. **Структуры данных для ускорения поиска**:\n Чтобы ускорить процесс поиска ближайших соседей, используются специализированные структуры данных:\n - KD-дерево (K-Dimensional Tree) – одна из наиболее распространенных структур данных для этой задачи. Оно позволяет разбивать пространство на меньшие подпространства, что существенно ускоряет поиск.\n - Другие структуры данных, такие как R-дерево или Ball-дерево, также могут использоваться, но они менее распространены для KNN.\n\n4. **Анализ дистракторов**:\n - **Алгоритм Дейкстры**: Используется для нахождения кратчайших путей в графах, не имеет прямого отношения к поиску ближайших соседей в пространстве.\n - **Поиск A\\***: Используется для нахождения оптимальных путей, особенно в пространстве состояний, также не связан непосредственно с KNN.\n - **Красно-чёрное дерево**: Самобалансирующееся бинарное дерево поиска, используется для хранения отсортированных данных и неэффективно для многомерного поиска ближайших соседей.\n\n### Итоговый анализ:\nМетод ближайших соседей требует эффективной организации данных для быстрого поиска. KD-дерево – оптимальный выбор для таких задач, так как оно значительно сокращает время поиска по сравнению с перебором всех точек.\n\n## Описание общей темы вопроса\n\n### Общая тема:\n**Структуры данных и алгоритмы для эффективного поиска в многомерных пространствах**'
        }
    ],
    'format_theme': [
        {
            'reference_question': '# Вопрос:\nПри реализации метода ближайших соседей, скорее всего будет фигурировать следующее:\n\n# Правильный ответ:\nKD-дерево\n\n# Дистракторы:\n - Алгоритм Дейкстры\n - Поиск A*\n - Красно-чёрное дерево',
            'response': '## Анализ вопроса\n\n### Цепочка размышлений:\n\n1. **Понимание метода ближайших соседей**:\n   Метод ближайших соседей (K-Nearest Neighbors, KNN) – это алгоритм классификации и регрессии, который основывается на нахождении ближайших объектов в многомерном пространстве. Он используется в задачах машинного обучения и анализа данных.\n\n2. **Основные концепции метода ближайших соседей**:\n   - Для поиска ближайших соседей необходимо эффективно организовать данные в пространстве.\n   - Простое решение – перебор всех точек, но это неэффективно для больших наборов данных.\n\n3. **Структуры данных для ускорения поиска**:\n   Чтобы ускорить процесс поиска ближайших соседей, используются специализированные структуры данных:\n   - KD-дерево (K-Dimensional Tree) – одна из наиболее распространенных структур данных для этой задачи. Оно позволяет разбивать пространство на меньшие подпространства, что существенно ускоряет поиск.\n   - Другие структуры данных, такие как R-дерево или Ball-дерево, также могут использоваться, но они менее распространены для KNN.\n\n4. **Анализ дистракторов**:\n   - **Алгоритм Дейкстры**: Используется для нахождения кратчайших путей в графах, не имеет прямого отношения к поиску ближайших соседей в пространстве.\n   - **Поиск A\\***: Используется для нахождения оптимальных путей, особенно в пространстве состояний, также не связан непосредственно с KNN.\n   - **Красно-чёрное дерево**: Самобалансирующееся бинарное дерево поиска, используется для хранения отсортированных данных и неэффективно для многомерного поиска ближайших соседей.\n\n### Итоговый анализ:\nМетод ближайших соседей требует эффективной организации данных для быстрого поиска. KD-дерево – оптимальный выбор для таких задач, так как оно значительно сокращает время поиска по сравнению с перебором всех точек.\n\n## Описание общей темы вопроса\n\n### Общая тема:\n**Структуры данных и алгоритмы для эффективного поиска в многомерных пространствах**',
            'format_instructions': get_format_instructions(parser_theme),
            'output': '```json\n{\n  "question_theme": "Структуры данных и алгоритмы для эффективного поиска в многомерных пространствах"\n}\n```'
        }
    ],
    'generation': [
        {
            'theme': 'Структуры данных и алгоритмы для эффективного поиска в многомерных пространствах',
            'reference_question': '# Вопрос:\nПри реализации метода ближайших соседей, скорее всего будет фигурировать следующее:\n\n# Правильный ответ:\nKD-дерево\n\n# Дистракторы:\n - Алгоритм Дейкстры\n - Поиск A*\n - Красно-чёрное дерево',
            'output': '# Анализ темы:\n\nМногомерные пространства и алгоритмы для эффективного поиска в них представляют собой важную область в компьютерных науках, особенно в задачах машинного обучения, компьютерного зрения и геоинформационных систем. Одним из ключевых методов для решения задач поиска в многомерных пространствах является метод ближайших соседей (K-Nearest Neighbors, KNN). Для эффективного поиска ближайших соседей часто используются специализированные структуры данных. Вот некоторые из них:\n\n1. **KD-дерево (k-dimensional tree)** - структура данных для поиска ближайших соседей в многомерном пространстве. Она рекурсивно делит пространство на подпространства, что позволяет быстро искать точки, ближайшие к заданной точке.\n2. **R-дерево** - структура данных, используемая для индексирования многомерных данных, таких как географические координаты. Она особенно полезна для поиска диапазонов и ближайших соседей.\n3. **Octree** - структура данных, которая делит трехмерное пространство на восемь подпространств. Используется для различных задач, включая графику и моделирование физических систем.\n4. **Grid-based методы** - простая структура данных, где пространство делится на фиксированные ячейки или сетку. Это позволяет быстро искать точки в ближайших ячейках.\n\nТеперь, исходя из анализа, я сгенерирую уникальный вопрос и правильный ответ по данной теме.\n\n# Вопрос:\n\nКакую структуру данных следует использовать для эффективного индексирования и поиска ближайших соседей в многомерных данных, где необходимо часто выполнять обновления данных?\n\n# Правильный ответ:\n\nR-дерево'
        }
    ],
    'format_generation': [
        {
            'response': '# Анализ темы:\n\nМногомерные пространства и алгоритмы для эффективного поиска в них представляют собой важную область в компьютерных науках, особенно в задачах машинного обучения, компьютерного зрения и геоинформационных систем. Одним из ключевых методов для решения задач поиска в многомерных пространствах является метод ближайших соседей (K-Nearest Neighbors, KNN). Для эффективного поиска ближайших соседей часто используются специализированные структуры данных. Вот некоторые из них:\n\n1. **KD-дерево (k-dimensional tree)** - структура данных для поиска ближайших соседей в многомерном пространстве. Она рекурсивно делит пространство на подпространства, что позволяет быстро искать точки, ближайшие к заданной точке.\n2. **R-дерево** - структура данных, используемая для индексирования многомерных данных, таких как географические координаты. Она особенно полезна для поиска диапазонов и ближайших соседей.\n3. **Octree** - структура данных, которая делит трехмерное пространство на восемь подпространств. Используется для различных задач, включая графику и моделирование физических систем.\n4. **Grid-based методы** - простая структура данных, где пространство делится на фиксированные ячейки или сетку. Это позволяет быстро искать точки в ближайших ячейках.\n\nТеперь, исходя из анализа, я сгенерирую уникальный вопрос и правильный ответ по данной теме.\n\n# Вопрос:\n\nКакую структуру данных следует использовать для эффективного индексирования и поиска ближайших соседей в многомерных данных, где необходимо часто выполнять обновления данных?\n\n# Правильный ответ:\n\nR-дерево',
            'format_instructions': get_format_instructions(parser_generation),
            'output': '```json\n{\n  "question": "Какую структуру данных следует использовать для эффективного индексирования и поиска ближайших соседей в многомерных данных, где необходимо часто выполнять обновления данных?",\n  "correct_answer": "R-дерево"\n}\n```'
        }
    ],
    'validation': [
        {
            'generated_question': 'Какую структуру данных следует использовать для эффективного индексирования и поиска ближайших соседей в многомерных данных, где необходимо часто выполнять обновления данных?',
            'correct_answer': 'R-дерево',
            'output': '## Анализ правильности ответа\n\n### Цепочка размышлений:\n\n1. **Тип задачи**:\n   - Вопрос касается структур данных, используемых для индексирования и поиска ближайших соседей в многомерных данных.\n   - Указывается, что данные необходимо часто обновлять.\n\n2. **Критерии для оценки структур данных**:\n   - **Эффективность индексирования**: Структура должна поддерживать быстрое извлечение данных по запросу.\n   - **Поиск ближайших соседей**: Важна возможность быстро находить ближайшие элементы в многомерном пространстве.\n   - **Частые обновления**: Структура должна эффективно обрабатывать вставки, удаления и модификации данных.\n\n3. **Анализ R-дерева**:\n   - **Эффективность индексирования**: R-дерево разделяет пространство на минимальные прямоугольники, которые содержат точки данных, что позволяет эффективно индексировать данные.\n   - **Поиск ближайших соседей**: R-дерево хорошо справляется с поиском ближайших соседей благодаря своей иерархической структуре и возможности ограничивать поиск малыми частями пространства.\n   - **Частые обновления**: R-деревья разработаны с учетом частых обновлений данных. Они поддерживают эффективные операции вставки и удаления, сохраняя при этом сбалансированную структуру.\n\n4. **Альтернативные структуры данных**:\n   - **k-d дерево**: Подходит для поиска ближайших соседей, но менее эффективно при частых обновлениях.\n   - **Quad-дерево**: Используется для разбиения двухмерного пространства и также менее эффективно при обновлениях.\n   - **Ball-дерево, Cover-tree**: Более сложные структуры, которые могут быть менее эффективны для частых обновлений.\n\n### Заключение:\n- **Смысл вопроса**: Вопрос сформулирован корректно и имеет практическое значение в области обработки и поиска в многомерных данных.\n- **Правильность ответа**: Ответ \"R-дерево\" является правильным. Эта структура данных предназначена для эффективного индексирования, поиска ближайших соседей и частых обновлений в многомерных данных.\n\n### Объяснение, почему ответ верен:\nR-дерево (Rectangular Tree) — это структура данных, разработанная для эффективного управления и поиска в многомерных пространственных данных. Она разбивает пространство на иерархию прямоугольников и поддерживает операции вставки, удаления и поиска, оставаясь сбалансированной. Это делает R-дерево особенно подходящим для задач, где данные часто обновляются, и требуется быстрое индексирование и поиск ближайших соседей.\n\nТаким образом, ответ \"R-дерево\" является верным и соответствует требованиям, указанным в вопросе.'
        }
    ],
    'distractors': [
        {
            'theme': 'Структуры данных и алгоритмы для эффективного поиска в многомерных пространствах',
            'generated_question': 'Какую структуру данных следует использовать для эффективного индексирования и поиска ближайших соседей в многомерных данных, где необходимо часто выполнять обновления данных?',
            'correct_answer': 'R-дерево',
            'output': 'Для анализа данного вопроса необходимо рассмотреть требования и характеристики эффективной структуры данных для индексирования и поиска ближайших соседей в многомерных данных. R-дерево является подходящей структурой данных, поскольку оно оптимизировано для работы с многомерными данными и поддерживает эффективные обновления.\n\nЧтобы создать правдоподобные, но неправильные ответы, нужно выбрать структуры данных, которые кажутся подходящими для многомерных данных, но имеют значительные недостатки в частых обновлениях или поиске ближайших соседей. \n\nВот цепочка размышлений для анализа:\n\n1. k-d дерево (k-dimensional tree) часто используется для поиска ближайших соседей в многомерных данных, но оно плохо справляется с частыми обновлениями, так как каждая вставка или удаление может потребовать перестройки дерева.\n2. Octree используется для работы с трехмерными данными и хорошо подходит для статических наборов данных, но имеет проблемы с производительностью при частых обновлениях.\n3. Quad-дерево (Quad tree) используется для работы с двухмерными пространствами и подходит для статических данных. В многомерных пространствах и при частых обновлениях его эффективность значительно падает.\n\nСгенерируем 3 неправильных ответа на основе этого анализа:\n\n# Дистракторы:\n - k-d дерево (k-dimensional tree)\n - Octree\n - Quad-дерево (Quad tree)'
        }
    ],
    'format_distractors': [
        {
            'response': 'Для анализа данного вопроса необходимо рассмотреть требования и характеристики эффективной структуры данных для индексирования и поиска ближайших соседей в многомерных данных. R-дерево является подходящей структурой данных, поскольку оно оптимизировано для работы с многомерными данными и поддерживает эффективные обновления.\n\nЧтобы создать правдоподобные, но неправильные ответы, нужно выбрать структуры данных, которые кажутся подходящими для многомерных данных, но имеют значительные недостатки в частых обновлениях или поиске ближайших соседей. \n\nВот цепочка размышлений для анализа:\n\n1. k-d дерево (k-dimensional tree) часто используется для поиска ближайших соседей в многомерных данных, но оно плохо справляется с частыми обновлениями, так как каждая вставка или удаление может потребовать перестройки дерева.\n2. Octree используется для работы с трехмерными данными и хорошо подходит для статических наборов данных, но имеет проблемы с производительностью при частых обновлениях.\n3. Quad-дерево (Quad tree) используется для работы с двухмерными пространствами и подходит для статических данных. В многомерных пространствах и при частых обновлениях его эффективность значительно падает.\n\nСгенерируем 3 неправильных ответа на основе этого анализа:\n\n# Дистракторы:\n - k-d дерево (k-dimensional tree)\n - Octree\n - Quad-дерево (Quad tree)',
            'format_instructions': get_format_instructions(parser_distractors),
            'output': '```json\n{\n  "distractor_1": "k-d дерево (k-dimensional tree)",\n  "distractor_2": "Octree",\n  "distractor_3": "Quad-дерево (Quad tree)"\n}\n```'
        }
    ],
}

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
        print("AAAAAAAAAAAA")
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
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False, add_generation_prompt=True
    )

class ChatChain:
    def __init__(self, template, tokenizer, api_url) -> None:
        self.chat_template = template
        self.api_url = api_url
        self.tokenizer = tokenizer

    def invoke(self, args):
        prompt = use_chat_template(self.chat_template, args, self.tokenizer)
        print(f"prompt = {prompt}")
        prompt += "\n\n"
        if "format_instructions" in args.keys():
            prompt += args["format_instructions"]
            print(f"prompt with format_instructions = {prompt}")
        response = call_api(self.api_url, prompt)
        return response

def get_model(primary_model, few_shot=False, api_url="http://10.100.30.240:1222/generate"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    llm = LLM(
        model=primary_model,
        trust_remote_code=True,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.9
    )
    
    tokenizer = llm.get_tokenizer()  # Оставляем как есть, если требуется

    theme_prompt_few_shot = ChatPromptTemplate.from_messages(
        [
            ("human", "Вопрос:\n{reference_question}\n\nТщательно проанализируй пример и используй цепочку размышлений и напиши свой анализ вопроса и общей темы. Тема не должна быть слишком специфичной, а более общей. После этого, опиши общую тему вопроса."),
            ("ai", "{output}")
        ]
    )
    
    theme_few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=theme_prompt_few_shot,
        examples=few_shot_examples['theme'],
    )
    
    theme_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_message_validation),
            *([theme_few_shot_prompt] if few_shot else []),
            ("human", "Вопрос: {reference_question}\nТщательно проанализируй пример и используй цепочку размышлений и напиши свой анализ вопроса и общей темы. Тема не должна быть слишком специфичной, а более общей. После этого, опиши общую тему вопроса.")
        ]
    )

    theme_chain = ChatChain(theme_prompt_template, tokenizer, api_url)

    def get_generation_prompt(existing_questions):
        generation_prompt_few_shot = ChatPromptTemplate.from_messages(
            [
                ("human", "# Тема вопроса: {theme}\n\n# Пример вопроса по теме:\n```{reference_question}```\n\n# Существующие вопросы:\n1. ```{reference_question}```\n\n\nТы должен сгенерировать вопрос и правильный ответ к нему по данной теме для студенческого экзамена. Не генерируй вопросы похожие на существующие!. Для начала тщательно проанализируй тему и используй цепочку размышлений и напиши свой анализ. После этого, сгенерируй вопрос и правильный ответ по этой общей теме. Ты не должен повторять или перефразировать существующие вопросы. Ты должен сгенерировать уникальный вопрос и один правильный вариант ответа, без других вариантов. Не генерируй варианты ответа, только правильный ответ. Не добавляй букву или номер к ответу. Ты должен генерировать уникальные вопросы и ответы, достаточно сильно отличающиеся от существующих вопросов, но на ту же общую тему."),
                ("ai", "{output}")
            ]
        )

        generation_few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=generation_prompt_few_shot,
            examples=few_shot_examples['generation'],
        )

        generation_prompt_template = ChatPromptTemplate.from_messages(
            [
                *([generation_few_shot_prompt] if few_shot else []),
                ("system", system_message_generation),
                ("human", "# Тема вопроса: {theme}\n\n# Пример вопроса по теме:\n```{reference_question}```\n\n# Существующие вопросы:\n1. ```{reference_question}```" + "\n\n\nТы должен сгенерировать вопрос и правильный ответ к нему по данной теме для студенческого экзамена. Не генерируй вопросы похожие на существующие!. Для начала тщательно проанализируй тему и используй цепочку размышлений и напиши свой анализ. После этого, сгенерируй вопрос и правильный ответ по этой общей теме. Ты не должен повторять или перефразировать существующие вопросы. Ты должен сгенерировать уникальный вопрос и один правильный вариант ответа, без других вариантов. Не генерируй варианты ответа, только правильный ответ. Не добавляй букву или номер к ответу. Ты должен генерировать уникальные вопросы и ответы, достаточно сильно отличающиеся от существующих вопросов, но на ту же общую тему."),
            ]
        )

        return ChatChain(generation_prompt_template, tokenizer, api_url)
    
    def get_distractors_prompt():
        # Step 5: Generate 3 distractors
        distractors_few_shot = ChatPromptTemplate.from_messages(
            [
                ("human", "# Тема вопроса: {theme}\n# Вопрос по теме: {generated_question}\n# Правильный ответ: {correct_answer}\n\nСоздай 3 правдоподобных, но неправильных ответа (дистракторов) для данного вопроса. Для начала проанализируй вопрос и используй цепочку размышлений. После этого, сгенерируй 3 неправильных ответа (дистрактора). В итоге, после анализа и рассуждений, ты должен сгенерировать 3 неправильных ответа в формате:\n# Дистракторы:\n - <неправильный ответ 1>\n - <неправильный ответ 2>\n - <неправильный ответ 3>.\n\nНе добавляй номера или буквы к ответам."),
                ("ai", "{output}")
            ]
        )

        distractors_few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=distractors_few_shot,
            examples=few_shot_examples['distractors'],
        )

        distractors_prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", system_message_distractors),
                *([distractors_few_shot_prompt] if few_shot else []),
                ("human", "# Тема вопроса: {theme}\n# Вопрос по теме: {generated_question}\n# Правильный ответ: {correct_answer}\n\nСоздай 3 правдоподобных, но неправильных ответа (дистракторов) для данного вопроса. Для начала проанализируй вопрос и используй цепочку размышлений. После этого, сгенерируй 3 неправильных ответа (дистрактора). В итоге, после анализа и рассуждений, ты должен сгенерировать 3 неправильных ответа в формате:\n# Дистракторы:\n - <неправильный ответ 1>\n - <неправильный ответ 2>\n - <неправильный ответ 3>.\n\nНе добавляй номера или буквы к ответам."),
            ]
        )

        distractors_chain = ChatChain(distractors_prompt_template, tokenizer, api_url)
        return distractors_chain

    return theme_chain, get_distractors_prompt, get_generation_prompt

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
    except json.JSONDecodeError as e:
        print(f"Ошибка декодирования output в output_json: {e}")
        raise 

def generate_exam_question(theme, reference_question, existing_questions, api_url="http://10.100.30.240:1222/generate"):
    theme_chain, get_distractors_prompt, get_generation_prompt = get_model(
        primary_model="IlyaGusev/saiga_llama3_8b",
        few_shot=False,
        api_url=api_url
    )
    
    # Шаг 1: Сгенерировать тему
    theme_args = {
        "reference_question": reference_question,
        "format_instructions": get_format_instructions(parser_theme)
    }
    print(f"theme_args = {theme_args}")
    theme_result = theme_chain.invoke(theme_args)
    theme_json = str_to_json(theme_result)
    theme = theme_json['question_theme']
    
    print(f"theme = {theme}")

    # Шаг 2: Сгенерировать вопрос по теме
    generation_chain = get_generation_prompt(existing_questions)
    generation_args = {
        "theme": theme,
        "reference_question": reference_question,
        "existing_questions": existing_questions,
        "format_instructions": get_format_instructions(parser_generation)
    }
    generated_question_result = generation_chain.invoke(generation_args)
    generated_question_json = str_to_json(generated_question_result)

    generated_question = generated_question_json['question']
    correct_answer = generated_question_json['correct_answer']
    
    print(f"generated_question = {generated_question}")
    print(f"correct_answer = {correct_answer}")

    # Шаг 3: Генерация дистракторов
    #validation_args = {
    #    "generated_question": generated_question,
    #    "correct_answer": correct_answer
    #}
    #distractors = validation_chain.invoke(validation_args)
    
    # Шаг 3: Генерация дистракторов
    distractors_chain = get_distractors_prompt()
    distractors_args = {
        "correct_answer": correct_answer,
        "theme": theme,
        "generated_question": generated_question,
        "format_instructions": get_format_instructions(parser_distractors)
    }
    print(f"distractors_args = {distractors_args}")
    distractors_result = distractors_chain.invoke(distractors_args)
    distractors_json = str_to_json(distractors_result)
    
    print(f"distractors_json = {distractors_json}")
    
    return {
        "theme": theme,
        "generated_question": generated_question,
        "correct_answer": correct_answer,
        "distractors": distractors_json
    }

existing_questions = ["Пример существующего вопроса 1", "Пример существующего вопроса 2"]

result = generate_exam_question(
    "Пример темы",
    "Пример существующего вопроса",
    existing_questions,
    api_url="http://10.100.30.240:1222/generate"
)

print(result)