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

debug = True


def log_debug(var, val):
    if debug:
        print(f"[Debug] {var}: {val}\n")

system_message_generation = "Ты профессиональный экзаменатор с глубоким знанием предмета. Твоя задача, написание различных вопросов для студентческого экзамена. Ты должен отвечать на Русском языке."
system_message_validation = "Ты эксперт по предмету и профессор с многолетним опытом. Твоя задача - проверять экзаминационные задания на правильность. Ты должен отвечать на Русском языке."
system_message_summary = "Ты опытный экзаменатор, подводящий итоги проверки. Ты должен отвечать на Русском языке."
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


class ExamQuestion(BaseModel):
    question: str
    correct_answer: str
    distractors: list[str] = Field(default_factory=list)

    def to_string(this):
        return f"# Вопрос:\n{this.question}\n\n# Правильный ответ:\n{this.correct_answer}"

    def to_string_with_distractors(this):
        return f"# Вопрос:\n{this.question}\n\n# Правильный ответ:\n{this.correct_answer}\n\n# Дистракторы:\n - {this.distractors[0]}\n - {this.distractors[1]}\n - {this.distractors[2]}"


# Set up PydanticOutputParsers
parser_generation = PydanticOutputParser(pydantic_object=GeneratedQuestion)
parser_theme = PydanticOutputParser(pydantic_object=QuestionTheme)
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

def get_model_list():
    return [
        'qwen2',
        'llama3',
        'saiga'
    ]

def get_model(primary_model, few_shot=False):
    llm = LLM(
        model="IlyaGusev/saiga_llama3_8b",
        trust_remote_code=True,
        enable_prefix_caching=True
    )

    tokenizer = llm.get_tokenizer()

    sampleParams = SamplingParams(
        max_tokens=1536,
        top_k=30,
        top_p=0.95,
        temperature=0.9,
        stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    )

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

    def use_chat_template(template, args):
        messages = chat_template_messages(template, args)
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False, add_generation_prompt=True
        )
    
    class ChatChain:
        def __init__(self, template) -> None:
            self.chat_template = template

        def invoke(self, args):
            prompt = use_chat_template(self.chat_template, args)
            outputs = llm.generate(
                [prompt],
                sampling_params=sampleParams
            )
            response = outputs[0].outputs[0].text
            return response
    
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

    theme_chain = ChatChain(theme_prompt_template)

    format_theme_prompt_few_shot = ChatPromptTemplate.from_messages(
        [
            ("human", "Вопрос: {reference_question}\nТщательно проанализируй пример и используй цепочку размышлений и напиши свой анализ вопроса и общей темы. Тема не должна быть слишком специфичной, а более общей. После этого, опиши общую тему вопроса."),
            ("ai", "{response}"),
            ("human",
             "Подведи итог анализа темы вопроса и отформатируй сгенерированный ответ (тему) на русском языке в формате JSON. Ты должен строго следовать инструкциям и вывести только JSON и ничего больше. Инструкции для формата:\n{format_instructions}"),
            ("ai", "{output}")
        ]
    )

    format_theme_few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=format_theme_prompt_few_shot,
        examples=few_shot_examples['format_theme'],
    )

    format_theme_prompt = ChatPromptTemplate.from_messages(
        [
            *([format_theme_few_shot_prompt] if few_shot else []),
            ("human", "Вопрос: {reference_question}\nТщательно проанализируй пример и используй цепочку размышлений и напиши свой анализ вопроса и общей темы. Тема не должна быть слишком специфичной, а более общей. После этого, опиши общую тему вопроса."),
            ("ai", "{response}"),
            ("human",
             "Подведи итог анализа темы вопроса и отформатируй сгенерированный ответ (тему) на русском языке в формате JSON. Ты должен строго следовать инструкциям и вывести только JSON и ничего больше. Инструкции для формата:\n{format_instructions}")
        ]
    )

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
                ("human", "# Тема вопроса: {theme}\n\n# Пример вопроса по теме:\n```{reference_question}```\n\n# Существующие вопросы:\n1. ```{reference_question}```" + "\n".join([f"{i+2}. ```{question.to_string()}```" for i, question in enumerate(
                    existing_questions)]) + "\n\n\nТы должен сгенерировать вопрос и правильный ответ к нему по данной теме для студенческого экзамена. Не генерируй вопросы похожие на существующие!. Для начала тщательно проанализируй тему и используй цепочку размышлений и напиши свой анализ. После этого, сгенерируй вопрос и правильный ответ по этой общей теме. Ты не должен повторять или перефразировать существующие вопросы. Ты должен сгенерировать уникальный вопрос и один правильный вариант ответа, без других вариантов. Не генерируй варианты ответа, только правильный ответ. Не добавляй букву или номер к ответу. Ты должен генерировать уникальные вопросы и ответы, достаточно сильно отличающиеся от существующих вопросов, но на ту же общую тему."),
            ]
        )

        return ChatChain(generation_prompt_template)

    format_generation_few_shot = ChatPromptTemplate.from_messages(
        [
            ("human",
             "Отформатируй сгенерированный вопрос и правильный ответ к нему в формате JSON. Ты дожен взять именно сгенерированный вопрос и ответ, а не вопрос из приимера.\n{format_instructions}\nТекст: {response}\n"),
            ("ai", "{output}")
        ]
    )

    format_generation_few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=format_generation_few_shot,
        examples=few_shot_examples['format_generation'],
    )

    format_generation_prompt = ChatPromptTemplate.from_messages(
        [
            *([format_generation_few_shot_prompt] if few_shot else []),
            ("human",
             "Отформатируй сгенерированный вопрос и правильный ответ к нему в формате JSON. Ты дожен взять именно сгенерированный вопрос и ответ, а не вопрос из приимера.\n{format_instructions}\nТекст: {response}\n"),
        ]
    )

    # Step 2: Validation of question and correct answer
    validation_few_shot = ChatPromptTemplate.from_messages(
        [
            ("human", "# Вопрос: {generated_question}\n# Правильный ответ: {correct_answer}\n\nСначало тщательно проанализируй правильность ответа, используя цепочку размышлений, а потом скажи, имеет ли вопрос смысл и правильный ли ответ для этого вопроса, и наконец объясни, почему этот ответ верный или нет."),
            ("ai", "{output}")
        ]
    )

    validation_few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=validation_few_shot,
        examples=few_shot_examples['validation'],
    )

    validation_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_message_validation),
            *([validation_few_shot_prompt] if few_shot else []),
            ("human", "# Вопрос: {generated_question}\n# Правильный ответ: {correct_answer}\n\nСначало тщательно проанализируй правильность ответа, используя цепочку размышлений, а потом скажи, имеет ли вопрос смысл и правильный ли ответ для этого вопроса, и наконец объясни, почему этот ответ верный или нет.")
        ]
    )

    validation_chain = ChatChain(validation_prompt_template)

    # Step 3: Summarize the validation answer
    summary_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_message_summary),
            ("human",
             "Следующий вопрос и ответ к нему были отданы на проверку:\n# Вопрос: {generated_question}\n# Правильный ответ: {correct_answer}\n\nРезультат проверки:\n{validation_answer}\n\nПодведи итог проверки вопроса и правильного ответа одним словом: 'правильно' или 'неправильно'. Ты должен ответить только одно слово 'правильно' или 'неправильно'")
        ]
    )

    summary_chain = ChatChain(summary_prompt_template)

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

        distractors_chain = ChatChain(distractors_prompt_template)
        return distractors_chain

    # Format the distractors output
    format_distractors_few_shot = ChatPromptTemplate.from_messages(
        [
            ("human",
             "Отформатируй 3 неправильных ответа (дистракторов) из текста в формате JSON.\n\nТекст: {response}\n\nИнструкции для форматирования:\n{format_instructions}"),
            ("ai", "{output}")
        ]
    )

    format_distractors_few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=format_distractors_few_shot,
        examples=few_shot_examples['format_distractors'],
    )

    format_distractors_prompt = ChatPromptTemplate.from_messages(
        [
            *([format_distractors_few_shot_prompt] if few_shot else []),
            ("human",
             "Отформатируй 3 неправильных ответа (дистракторов) из текста в формате JSON.\n\nТекст: {response}\n\nИнструкции для форматирования:\n{format_instructions}"),
        ]
    )

    def generate_theme(reference_question):
        theme_response = theme_chain.invoke(
                    {"reference_question": reference_question})
        log_debug("theme_response", theme_response)
        while True:
            try:
                formatted_theme = ChatChain(format_theme_prompt).invoke(
                    {"reference_question": reference_question, "response": theme_response, "format_instructions": get_format_instructions(parser_theme)})
                log_debug("formatted_theme", formatted_theme)
                theme = parser_theme.invoke(formatted_theme)
                log_debug("theme", theme)
                break
            except Exception as e:
                print(e)
                continue
        return theme

    # Main function to generate and validate exam questions
    def generate_exam_question(theme, reference_question, existing_questions: List[ExamQuestion]):
        attempts = 0
        while attempts <= 6:
            # Step 1: Generate a similar question and correct answer
            generation_chain = get_generation_prompt(
                existing_questions)
            generated_response = generation_chain.invoke(
                {"theme": theme.question_theme, "reference_question": reference_question})
            log_debug("generated_response", generated_response)
            while True:
                try:
                    formatted_generation = ChatChain(format_generation_prompt).invoke(
                        {"response": generated_response, "format_instructions": get_format_instructions(parser_generation)})
                    log_debug("formatted_generation", formatted_generation)
                    generated_question = parser_generation.invoke(
                        formatted_generation)
                    log_debug("generated_question", generated_question)
                    break
                except:
                    continue

            # Step 2: Validate the question and correct answer
            validation_response = validation_chain.invoke(
                {"generated_question": generated_question.question, "correct_answer": generated_question.correct_answer})
            log_debug("validation_response", validation_response)

            # Step 3: Summarize the validation answer
            summary_response = summary_chain.invoke({"generated_question": generated_question.question,
                                                    "correct_answer": generated_question.correct_answer, "validation_answer": validation_response})
            log_debug("summary_response", summary_response)

            # Step 4: If validation result is "incorrect", return to step 1
            if "неправильно" in summary_response.strip().lower() or "правильно" not in summary_response.strip().lower():
                attempts += 1
            else:
                break

        # Step 5: Generate 3 distractors
        distractors_chain = get_distractors_prompt()
        distractors_response = distractors_chain.invoke(
            {"theme": theme.question_theme, "generated_question": generated_question.question, "correct_answer": generated_question.correct_answer})
        log_debug("distractors_response", distractors_response)
        while True:
            try:
                formatted_distractors = ChatChain(format_distractors_prompt).invoke(
                    {"response": distractors_response, "format_instructions": get_format_instructions(parser_distractors)})
                log_debug("formatted_distractors", formatted_distractors)
                distractors = parser_distractors.invoke(formatted_distractors)
                log_debug("distractors", distractors)
                break
            except:
                continue

        # Step 6: Structure the output
        exam_question = ExamQuestion(
            question=generated_question.question,
            correct_answer=generated_question.correct_answer,
            distractors=[distractors.distractor_1,
                         distractors.distractor_2, distractors.distractor_3]
        )

        return exam_question

    return generate_theme, generate_exam_question
