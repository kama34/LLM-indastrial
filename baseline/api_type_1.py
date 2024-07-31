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

system_message_generation = "Ты профессиональный экзаменатор с глубоким знанием предмета. Твоя задача, написание различных вопросов для студентческого экзамена. Ты должен отвечать на Русском языке."
system_message_distractors = "Ты профессиональный экзаменатор с глубоким знанием предмета. Ты должен отвечать на Русском языке."


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
        return f"{this.question}\n\n# Правильный ответ:\n{this.correct_answer}"

    def to_string_with_distractors(this):
        return f"{this.question}\n\n# Правильный ответ:\n{this.correct_answer}\n\n# Дистракторы:\n - {this.distractors[0]}\n - {this.distractors[1]}\n - {this.distractors[2]}"


parser_theme = PydanticOutputParser(pydantic_object=QuestionTheme)
parser_generation = PydanticOutputParser(pydantic_object=GeneratedQuestion)
parser_distractors = PydanticOutputParser(pydantic_object=Distractors)

model_map = {
    'qwen2-72b': {
        'api': 'http://10.100.30.240:1224/generate',
        'tokenizer': 'Qwen/Qwen2-72B'
    },
    'qwen2-7b': {
        'api': 'http://10.100.30.240:1222/generate',
        'tokenizer': 'Qwen/Qwen2-7B'
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
    return model_map.keys()

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
            'reference_question': 'При реализации метода ближайших соседей, скорее всего будет фигурировать следующее:\n\n# Правильный ответ:\nKD-дерево\n\n# Дистракторы:\n - Алгоритм Дейкстры\n - Поиск A*\n - Красно-чёрное дерево',
            'output': '## Анализ вопроса\n\n### Цепочка размышлений:\n\n1. **Понимание метода ближайших соседей**:\n Метод ближайших соседей (K-Nearest Neighbors, KNN) – это алгоритм классификации и регрессии, который основывается на нахождении ближайших объектов в многомерном пространстве. Он используется в задачах машинного обучения и анализа данных.\n\n2. **Основные концепции метода ближайших соседей**:\n - Для поиска ближайших соседей необходимо эффективно организовать данные в пространстве.\n - Простое решение – перебор всех точек, но это неэффективно для больших наборов данных.\n\n3. **Структуры данных для ускорения поиска**:\n Чтобы ускорить процесс поиска ближайших соседей, используются специализированные структуры данных:\n - KD-дерево (K-Dimensional Tree) – одна из наиболее распространенных структур данных для этой задачи. Оно позволяет разбивать пространство на меньшие подпространства, что существенно ускоряет поиск.\n - Другие структуры данных, такие как R-дерево или Ball-дерево, также могут использоваться, но они менее распространены для KNN.\n\n4. **Анализ дистракторов**:\n - **Алгоритм Дейкстры**: Используется для нахождения кратчайших путей в графах, не имеет прямого отношения к поиску ближайших соседей в пространстве.\n - **Поиск A\\***: Используется для нахождения оптимальных путей, особенно в пространстве состояний, также не связан непосредственно с KNN.\n - **Красно-чёрное дерево**: Самобалансирующееся бинарное дерево поиска, используется для хранения отсортированных данных и неэффективно для многомерного поиска ближайших соседей.\n\n### Итоговый анализ:\nМетод ближайших соседей требует эффективной организации данных для быстрого поиска. KD-дерево – оптимальный выбор для таких задач, так как оно значительно сокращает время поиска по сравнению с перебором всех точек.\n\n## Описание общей темы вопроса\n\n### Общая тема:\n**Структуры данных и алгоритмы для эффективного поиска в многомерных пространствах**\n\n```json\n{\n  "question_theme": "Структуры данных и алгоритмы для эффективного поиска в многомерных пространствах"\n}\n```',
            'format_instructions': get_format_instructions(parser_theme)
        },
        {
            'reference_question': 'Что из перечисленного не считается общепринятой метрикой для задач бинарной классификации?\n\n# Правильный ответ:\nBrier score\n\n# Дистракторы:\n - Recall\n - Accuracy\n - Precision',
            'output': '## Анализ вопроса\n\n### Цепочка размышлений:\n\n1. **Понимание задач бинарной классификации**:\n   Бинарная классификация — это задача машинного обучения, где каждый объект необходимо отнести к одному из двух классов. Оценка качества модели является ключевым этапом в процессе обучения.\n\n2. **Основные метрики для оценки качества модели**:\n   В задачах бинарной классификации существуют несколько общепринятых метрик для оценки качества модели, среди которых:\n   - **Recall (чувствительность)**: показывает, какую долю положительных примеров модель смогла правильно определить.\n   - **Accuracy (точность)**: общее количество правильно классифицированных примеров (как положительных, так и отрицательных) относительно общего числа примеров.\n   - **Precision (точность предсказаний)**: показывает, какую долю предсказанных моделью положительных примеров действительно являются положительными.\n\n3. **Анализ метрик**:\n   - **Recall, Accuracy, Precision**: эти метрики широко используются и общеприняты в задачах бинарной классификации, так как они помогают понять различные аспекты работы модели, включая точность и полноту предсказаний.\n   - **Brier score**: это метрика, которая оценивает точность прогнозов вероятностей. Она меньше известна и используется, поскольку не является стандартом для бинарной классификации, но может быть полезна в других контекстах, например, для оценки вероятностных прогнозов.\n\n4. **Анализ дистракторов**:\n   - **Recall, Accuracy, Precision**: все эти метрики являются общепринятыми и широко используются для оценки моделей бинарной классификации.\n   - **Brier score**: не является стандартной метрикой для задач бинарной классификации, поэтому его можно считать верным ответом на вопрос.\n\n### Итоговый анализ:\nДля задач бинарной классификации существуют общепринятые метрики, такие как Recall, Accuracy и Precision, которые оценивают различные аспекты качества модели. Brier score, несмотря на свою полезность в других контекстах, не считается общепринятой метрикой для данной задачи.\n\n## Описание общей темы вопроса\n\n### Общая тема:\n**Метрики и способы оценки качества моделей в машинном обучении**\n\n```json\n{\n  \"question_theme\": \"Метрики и способы оценки качества моделей в машинном обучении\"\n}\n```',
            'format_instructions': get_format_instructions(parser_theme)
        },
        {
            'reference_question': 'Логистическая регрессия это модель, которая основана на функции...?\n\n# Правильный ответ:\nSigmoid\n\n# Дистракторы:\n - TanH\n - SoftPlus\n - ReLU',
            'output': '## Анализ вопроса\n\n### Цепочка размышлений:\n\n1. **Понимание логистической регрессии**:\n   Логистическая регрессия – это статистический метод, который используется для решения задач бинарной классификации. Модель прогнозирует вероятность того, что наблюдение принадлежит к одному из двух классов.\n\n2. **Основные концепции логистической регрессии**:\n   - Основная идея заключается в том, что линейная комбинация входных признаков (веса и признаки) преобразуется в вероятность с использованием логистической функции.\n   - Логистическая функция также известна как сигмоидная функция (Sigmoid function).\n\n3. **Функция активации для логистической регрессии**:\n   - Функция активации Sigmoid (сигмоидная функция) принимает на вход любое вещественное число и преобразует его в значение в диапазоне от 0 до 1, что идеально подходит для моделирования вероятностей.\n\n4. **Анализ дистракторов**:\n   - **TanH**: Гиперболический тангенс также является функцией активации, но он преобразует значения в диапазон от -1 до 1, что не соответствует требованиям логистической регрессии.\n   - **SoftPlus**: Это гладкая версия ReLU, которая также используется в нейронных сетях, но не имеет отношения к логистической регрессии.\n   - **ReLU**: Rectified Linear Unit, популярная функция активации в глубоких нейронных сетях, но не используется в контексте логистической регрессии.\n\n### Итоговый анализ:\nЛогистическая регрессия использует сигмоидную функцию для преобразования линейной комбинации признаков в вероятность. Sigmoid является ключевым элементом этой модели, что делает его правильным ответом.\n\n## Описание общей темы вопроса\n\n### Общая тема:\n**Функции активации и их применение в различных моделях машинного обучения**\n\n```json\n{\n  \"question_theme\": \"Функции активации и их применение в различных моделях машинного обучения\"\n}\n```',
            'format_instructions': get_format_instructions(parser_theme)
        },
    ],
    'generation': [
        {
            'theme': 'Структуры данных и алгоритмы для эффективного поиска в многомерных пространствах',
            'reference_question': 'При реализации метода ближайших соседей, скорее всего будет фигурировать следующее:\n\n# Правильный ответ:\nKD-дерево\n\n# Дистракторы:\n - Алгоритм Дейкстры\n - Поиск A*\n - Красно-чёрное дерево',
            'output': '# Анализ темы:\n\nМногомерные пространства и алгоритмы для эффективного поиска в них представляют собой важную область в компьютерных науках, особенно в задачах машинного обучения, компьютерного зрения и геоинформационных систем. Одним из ключевых методов для решения задач поиска в многомерных пространствах является метод ближайших соседей (K-Nearest Neighbors, KNN). Для эффективного поиска ближайших соседей часто используются специализированные структуры данных. Вот некоторые из них:\n\n1. **KD-дерево (k-dimensional tree)** - структура данных для поиска ближайших соседей в многомерном пространстве. Она рекурсивно делит пространство на подпространства, что позволяет быстро искать точки, ближайшие к заданной точке.\n2. **R-дерево** - структура данных, используемая для индексирования многомерных данных, таких как географические координаты. Она особенно полезна для поиска диапазонов и ближайших соседей.\n3. **Octree** - структура данных, которая делит трехмерное пространство на восемь подпространств. Используется для различных задач, включая графику и моделирование физических систем.\n4. **Grid-based методы** - простая структура данных, где пространство делится на фиксированные ячейки или сетку. Это позволяет быстро искать точки в ближайших ячейках.\n\nТеперь, исходя из анализа, я сгенерирую уникальный вопрос и правильный ответ по данной теме.\n\n# Вопрос:\n\nКакую структуру данных следует использовать для эффективного индексирования и поиска ближайших соседей в многомерных данных, где необходимо часто выполнять обновления данных?\n\n# Правильный ответ:\n\nR-дерево\n\n```json\n{\n  "question": "Какую структуру данных следует использовать для эффективного индексирования и поиска ближайших соседей в многомерных данных, где необходимо часто выполнять обновления данных?",\n  "correct_answer": "R-дерево"\n}\n```',
            'format_instructions': get_format_instructions(parser_generation)
        },
        {
            'theme': 'Метрики и способы оценки качества моделей в машинном обучении',
            'reference_question': 'Что из перечисленного не считается общепринятой метрикой для задач бинарной классификации?\n\n# Правильный ответ:\nBrier score\n\n# Дистракторы:\n - Recall\n - Accuracy\n - Precision',
            'output': '# Анализ:\n\nМетрики и способы оценки качества моделей в машинном обучении являются важным аспектом, так как правильная оценка позволяет определить эффективность модели и её пригодность для решения конкретных задач. В задачах классификации, регрессии и кластеризации используются различные метрики, такие как точность (Accuracy), полнота (Recall), точность (Precision), F1-мера (F1 Score), среднеквадратичная ошибка (MSE), средняя абсолютная ошибка (MAE) и другие. Метрики подбираются в зависимости от конкретной задачи, и каждая из них фокусируется на разных аспектах предсказательной способности модели.\n\nНапример, для задач бинарной классификации широко применяются такие метрики, как AUC-ROC, F1-Score, точность и полнота. Для задач регрессии, напротив, используются метрики, такие как среднеквадратичная ошибка и средняя абсолютная ошибка. Метрики для кластеризации, такие как коэффициент силуэта и индекс Дэвиса-Болдина, позволяют оценивать качество разбиения данных на кластеры.\n\nВажно понимать, что выбор метрики может существенно повлиять на восприятие качества модели. Например, в задачах с несбалансированными классами метрика точности может быть обманчиво высокой, тогда как другие метрики, такие как F1-Score или AUC-ROC, предоставляют более точную картину.\n\nНа основе этого анализа, сформулируем уникальный вопрос, связанный с выбором и применением метрик для оценки моделей машинного обучения.\n\n# Вопрос:\n\nКакая метрика чаще всего используется для оценки качества модели регрессии, предсказывающей непрерывные значения?\n\n# Правильный ответ:\n\nСреднеквадратичная ошибка (MSE)\n\n```json\n{\n  \"question\": \"Какая метрика чаще всего используется для оценки качества модели регрессии, предсказывающей непрерывные значения?\",\n  \"correct_answer\": \"Среднеквадратичная ошибка (MSE)\"\n}\n```',
            'format_instructions': get_format_instructions(parser_generation)
        },
        {
            'theme': 'Функции активации и их применение в различных моделях машинного обучения',
            'reference_question': 'Логистическая регрессия это модель, которая основана на функции...?\n\n# Правильный ответ:\nSigmoid\n\n# Дистракторы:\n - TanH\n - SoftPlus\n - ReLU',
            'output': '# Анализ темы:\n\nАнализ временных рядов — это область статистики и машинного обучения, которая занимается изучением данных, собранных в определённые моменты времени. Это может быть полезно для прогнозирования будущих значений на основе исторических данных. Основные методы включают:\n\n1. **ARIMA (AutoRegressive Integrated Moving Average)** — один из классических методов для моделирования временных рядов, который учитывает автокорреляцию данных.\n2. **SARIMA (Seasonal ARIMA)** — расширение ARIMA, которое добавляет компонент для сезонных данных.\n3. **Exponential Smoothing (Экспоненциальное сглаживание)** — метод, который учитывает тренды и сезонные компоненты для прогнозирования временных рядов.\n4. **LSTM (Long Short-Term Memory)** — рекуррентная нейронная сеть, которая способна обучаться на последовательностях данных и использовать их для долгосрочного прогнозирования.\n5. **Prophet** — метод, разработанный Facebook, который упрощает прогнозирование временных рядов с трендами и сезонностью.\n\nПри выборе метода для анализа временных рядов важно учитывать характер данных, такие как наличие тренда, сезонности и шума.\n\nТеперь, исходя из анализа, я сгенерирую уникальный вопрос и правильный ответ по данной теме.\n\n# Вопрос:\n\nКакой метод анализа временных рядов наиболее подходит для моделирования данных с явным сезонным компонентом и трендом, требующим учёта как короткосрочных, так и долгосрочных зависимостей?\n\n# Правильный ответ:\n\nSARIMA\n\n```json\n{\n  \"question\": \"Какой метод анализа временных рядов наиболее подходит для моделирования данных с явным сезонным компонентом и трендом, требующим учёта как короткосрочных, так и долгосрочных зависимостей?\",\n  \"correct_answer\": \"SARIMA\"\n}\n```',
            'format_instructions': get_format_instructions(parser_generation)
        }
    ],
    'distractors': [
        {
            'theme': 'Структуры данных и алгоритмы для эффективного поиска в многомерных пространствах',
            'generated_question': 'Какую структуру данных следует использовать для эффективного индексирования и поиска ближайших соседей в многомерных данных, где необходимо часто выполнять обновления данных?',
            'correct_answer': 'R-дерево',
            'output': '# Анализ:\n\nДля анализа данного вопроса необходимо рассмотреть требования и характеристики эффективной структуры данных для индексирования и поиска ближайших соседей в многомерных данных. R-дерево является подходящей структурой данных, поскольку оно оптимизировано для работы с многомерными данными и поддерживает эффективные обновления.\n\nЧтобы создать правдоподобные, но неправильные ответы, нужно выбрать структуры данных, которые кажутся подходящими для многомерных данных, но имеют значительные недостатки в частых обновлениях или поиске ближайших соседей. \n\nВот цепочка размышлений для анализа:\n\n1. k-d дерево (k-dimensional tree) часто используется для поиска ближайших соседей в многомерных данных, но оно плохо справляется с частыми обновлениями, так как каждая вставка или удаление может потребовать перестройки дерева.\n2. Octree используется для работы с трехмерными данными и хорошо подходит для статических наборов данных, но имеет проблемы с производительностью при частых обновлениях.\n3. Quad-дерево (Quad tree) используется для работы с двухмерными пространствами и подходит для статических данных. В многомерных пространствах и при частых обновлениях его эффективность значительно падает.\n\nСгенерируем 3 неправильных ответа на основе этого анализа:\n\n# Дистракторы:\n - k-d дерево (k-dimensional tree)\n - Octree\n - Quad-дерево (Quad tree)\n\n```json\n{\n  "distractor_1": "k-d дерево (k-dimensional tree)",\n  "distractor_2": "Octree",\n  "distractor_3": "Quad-дерево (Quad tree)"\n}\n```',
            'format_instructions': get_format_instructions(parser_distractors)
        },
        {
            'theme': 'Метрики и способы оценки качества моделей в машинном обучении',
            'generated_question': 'Какая метрика чаще всего используется для оценки качества модели регрессии, предсказывающей непрерывные значения?',
            'correct_answer': 'Среднеквадратичная ошибка (MSE)',
            'output': '# Анализ:\n\nДля анализа этого вопроса необходимо учитывать, что MSE (Среднеквадратичная ошибка) — это стандартная метрика, которая используется для оценки качества модели регрессии, так как она измеряет средний квадрат ошибок между предсказанными и фактическими значениями. \n\nЧтобы создать правдоподобные, но неправильные ответы, нужно выбрать метрики, которые также используются в машинном обучении, но не являются основными для оценки регрессионных моделей, предсказывающих непрерывные значения.\n\nВот цепочка размышлений для анализа:\n\n1. **ROC AUC (Area Under the Receiver Operating Characteristic Curve)** — эта метрика используется для оценки моделей классификации, а не регрессии.\n2. **F1-мера** — метрика, также применяемая для классификации, оценивающая баланс между точностью и полнотой. Неприменима для регрессии.\n3. **Кросс-энтропия** — обычно используется для оценки моделей классификации, особенно в задачах с несколькими классами.\n\nЭти метрики могут показаться знакомыми, но не подходят для оценки моделей регрессии, что делает их хорошими дистракторами.\n\nСгенерируем 3 неправильных ответа на основе этого анализа:\n\n# Дистракторы:\n - ROC AUC\n - F1-мера\n - Кросс-энтропия\n\n```json\n{\n  \"distractor_1\": \"ROC AUC\",\n  \"distractor_2\": \"F1-мера\",\n  \"distractor_3\": \"Кросс-энтропия\"\n}\n```',
            'format_instructions': get_format_instructions(parser_distractors)
        },
        {
            'theme': 'Функции активации и их применение в различных моделях машинного обучения',
            'generated_question': 'Какой метод анализа временных рядов наиболее подходит для моделирования данных с явным сезонным компонентом и трендом, требующим учёта как короткосрочных, так и долгосрочных зависимостей?',
            'correct_answer': 'SARIMA',
            'output': '# Анализ:\n\nВопрос связан с моделированием временных рядов, особенно с учётом сезонности и трендов, где важно улавливать как краткосрочные, так и долгосрочные зависимости. SARIMA (Seasonal Autoregressive Integrated Moving Average) является расширением ARIMA, включающим сезонные компоненты, что делает его подходящим выбором для таких задач.\n\nДля создания правдоподобных, но неправильных ответов, нужно выбрать методы анализа временных рядов, которые также применяются для работы с временными рядами, но не обеспечивают такого уровня учёта сезонности и долгосрочных зависимостей, как SARIMA:\n\n1. ARIMA (Autoregressive Integrated Moving Average) — этот метод хорошо подходит для моделирования данных без явной сезонности, но в его базовой форме нет встроенной поддержки сезонных компонентов.\n2. Holt-Winters — метод экспоненциального сглаживания, который используется для сезонных временных рядов, но не подходит для учёта сложных долгосрочных зависимостей.\n3. LSTM (Long Short-Term Memory) — нейронная сеть, которая может улавливать как краткосрочные, так и долгосрочные зависимости, но не предназначена специально для работы с временными рядами с явной сезонностью и трендами в том виде, как это делает SARIMA.\n\nСгенерируем 3 неправильных ответа на основе этого анализа:\n\n# Дистракторы:\n- ARIMA\n- Holt-Winters\n- LSTM\n\n```json\n{\n  \"distractor_1\": \"ARIMA\",\n  \"distractor_2\": \"Holt-Winters\",\n  \"distractor_3\": \"LSTM\"\n}\n```',
            'format_instructions': get_format_instructions(parser_distractors)
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

def get_model(primary_model='qwen2-72b', num_examples=3):
    few_shot = True
    if num_examples <= 0:
        few_shot = False
    if primary_model not in model_map:
        raise Exception(f'{primary_model} is not a valid or supported model')
    model_info = model_map[primary_model]
    api_url = model_info['api']
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_info['tokenizer'], trust_remote_code=True)

    theme_prompt_few_shot = ChatPromptTemplate.from_messages(
        [
            ("human", "# Вопрос: {reference_question}\n\nТщательно проанализируй пример и используй цепочку размышлений и напиши свой анализ вопроса и общей темы. Тема не должна быть слишком специфичной, а более общей. После этого, опиши общую тему вопроса.\n\n{format_instructions}"),
            ("ai", "{output}")
        ]
    )
    
    theme_few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=theme_prompt_few_shot,
        examples=few_shot_examples['theme'][:num_examples],
    )
    
    theme_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_message_distractors),
            *([theme_few_shot_prompt] if few_shot else []),
            ("human", "# Вопрос: {reference_question}\nТщательно проанализируй пример и используй цепочку размышлений и напиши свой анализ вопроса и общей темы. Тема не должна быть слишком специфичной, а более общей. После этого, опиши общую тему вопроса.")
        ]
    )

    theme_chain = ChatChain(theme_prompt_template, tokenizer, api_url)
    
    def generate_theme(reference_question):
        # Шаг 1: Сгенерировать тему
        theme_args = {
            "reference_question": reference_question,
            "format_instructions": get_format_instructions(parser_theme)
        }
        print(f"theme_args = {theme_args}")
        while True:
            try:
                theme_result = theme_chain.invoke(theme_args)
                theme_json = str_to_json(theme_result)
                theme = theme_json['question_theme']
                
                print(f"theme = {theme}")
                return theme
            except:
                pass

    def get_generation_prompt(existing_questions):
        generation_prompt_few_shot = ChatPromptTemplate.from_messages(
            [
                ("human", "# Тема вопроса: {theme}\n\n# Пример вопроса по теме:\n```\n# Вопрос: {reference_question}\n```\n\n# Существующие вопросы:\n1. ```\n# Вопрос: {reference_question}\n```\n\n\nТы должен сгенерировать вопрос и правильный ответ к нему по данной теме для студенческого экзамена. Не генерируй вопросы похожие на существующие!. Для начала тщательно проанализируй тему и используй цепочку размышлений и напиши свой анализ. После этого, сгенерируй вопрос и правильный ответ по этой общей теме. Ты не должен повторять или перефразировать существующие вопросы. Ты должен сгенерировать уникальный вопрос и один правильный вариант ответа, без других вариантов. Не генерируй варианты ответа, только правильный ответ. Не добавляй букву или номер к ответу. Ты должен генерировать уникальные вопросы и ответы, достаточно сильно отличающиеся от существующих вопросов, но на ту же общую тему.\n\n{format_instructions}"),
                ("ai", "{output}")
            ]
        )

        generation_few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=generation_prompt_few_shot,
            examples=few_shot_examples['generation'][:num_examples],
        )

        generation_prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", system_message_generation),
                *([generation_few_shot_prompt] if few_shot else []),
                ("human", "# Тема вопроса: {theme}\n\n# Пример вопроса по теме:\n```\n# Вопрос: {reference_question}\n```\n\n# Существующие вопросы:\n1. ```\n# Вопрос: {reference_question}\n```" + "\n\n\nТы должен сгенерировать вопрос и правильный ответ к нему по данной теме для студенческого экзамена. Не генерируй вопросы похожие на существующие!. Для начала тщательно проанализируй тему и используй цепочку размышлений и напиши свой анализ. После этого, сгенерируй вопрос и правильный ответ по этой общей теме. Ты не должен повторять или перефразировать существующие вопросы. Ты должен сгенерировать уникальный вопрос и один правильный вариант ответа, без других вариантов. Не генерируй варианты ответа, только правильный ответ. Не добавляй букву или номер к ответу. Ты должен генерировать уникальные вопросы и ответы, достаточно сильно отличающиеся от существующих вопросов, но на ту же общую тему."),
            ]
        )

        return ChatChain(generation_prompt_template, tokenizer, api_url)
    
    def get_distractors_prompt():
        # Step 5: Generate 3 distractors
        distractors_few_shot = ChatPromptTemplate.from_messages(
            [
                ("human", "# Тема вопроса: {theme}\n# Вопрос по теме: {generated_question}\n# Правильный ответ: {correct_answer}\n\nСоздай 3 правдоподобных, но неправильных ответа (дистракторов) для данного вопроса. Для начала проанализируй вопрос и используй цепочку размышлений. После этого, сгенерируй 3 неправильных ответа (дистрактора). В итоге, после анализа и рассуждений, ты должен сгенерировать 3 неправильных ответа в формате:\n# Дистракторы:\n - <неправильный ответ 1>\n - <неправильный ответ 2>\n - <неправильный ответ 3>.\n\nНе добавляй номера или буквы к ответам.\n\n{format_instructions}"),
                ("ai", "{output}")
            ]
        )

        distractors_few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=distractors_few_shot,
            examples=few_shot_examples['distractors'][:num_examples],
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
    
    def generate_exam_question(theme, reference_question, existing_questions):
        # Шаг 2: Сгенерировать вопрос по теме
        generation_chain = get_generation_prompt(existing_questions)
        generation_args = {
            "theme": theme,
            "reference_question": reference_question,
            "existing_questions": existing_questions,
            "format_instructions": get_format_instructions(parser_generation)
        }
        while True:
            try:
                generated_question_result = generation_chain.invoke(generation_args)
                generated_question_json = str_to_json(generated_question_result)
                break
            except:
                pass

        generated_question = generated_question_json['question']
        correct_answer = generated_question_json['correct_answer']
        
        print(f"generated_question = {generated_question}")
        print(f"correct_answer = {correct_answer}")
        
        # Шаг 3: Генерация дистракторов
        distractors_chain = get_distractors_prompt()
        distractors_args = {
            "correct_answer": correct_answer,
            "theme": theme,
            "generated_question": generated_question,
            "format_instructions": get_format_instructions(parser_distractors)
        }
        print(f"distractors_args = {distractors_args}")
        while True:
            try:
                distractors_result = distractors_chain.invoke(distractors_args)
                distractors_json = str_to_json(distractors_result)
                break
            except:
                pass
        
        print(f"distractors_json = {distractors_json}")
        
        return {
            "theme": theme,
            "generated_question": generated_question,
            "correct_answer": correct_answer,
            "distractors": distractors_json
        }

    return generate_theme, generate_exam_question

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