from io import BytesIO
import random
import time
import streamlit as st
import pandas as pd
from typing import Any, Dict, List, Tuple, Union
import re
import json

from question_generation_ui import QuestionGenerationUI

from demo_type4.baseline_api import get_model
from utils import get_model_list
import streamlit.components.v1 as components

class Type4UI(QuestionGenerationUI):
    @classmethod
    def get_question_type_name(cls) -> str:
        return "Open Questions"
    
    @classmethod
    def get_params(cls) -> Dict[str, Any]:
        return {
            'has_examples': True,
            'has_theme_generation': False,
            'num_few_shot': 2
        }
    
    @classmethod
    def get_labels(cls) -> Dict[str, str]:
        return {
            'examples': 'Выберите пример компетенций из датасета',
            'example_button': 'Использовать выбранные компетенции',
            'generate_button': 'Сгенерировать вопросы',
            'theme': 'Сгенерированный кейс',
            'theme_analyze': 'Генерация кейса...'
        }
    
    # return list of (input_key, input_label, input_args)
    # input_args - additional args, i.e. height: 200
    def get_inputs(this) -> List[Tuple[str, str, Union[Dict, None]]]:
        return [('competence_list', 'Введите список компетенций. Кажджая компетенция должна быть на отдельной строке.', {'height': 200})]
    
    def get_model_list(this) -> List[str]:
        return get_model_list()
    
    def load_model(this, primary_model, num_examples):
        print(f'Loaded {primary_model} model with few-shot={num_examples}')
        return get_model(primary_model, num_examples)

    def load_examples(this) -> Tuple[List, List]:
        with open('./demo_type4/case_questions4.json', 'r') as f:
            data = json.load(f)

        res = []
        for case in data:
            competence_list = []
            for competence in case['competences']:
                competence_n = competence['competence']
                num = re.findall('№ \d+', competence_n)[0]
                name = re.findall('“(.+)”', competence_n)[0]
                competence_list.append((num, name))
            res.append(
                {
                    'competence_list_nums': ', '.join([num for num, name in competence_list]),
                    'competence_list': [name for num, name in competence_list]
                }
            )

        examples = res
        option_names = [f"Компетенции: {q['competence_list_nums']}" for i, q in enumerate(examples)]
        return option_names, examples

    def select_example(this, selected_option_name, selected_index, examples) -> str:
        selected_question = examples[selected_index]
        competence_list = '\n'.join(selected_question['competence_list'])
        return {'competence_list': competence_list}

    def generate(this, loaded_model, inputs, theme):
        generate_case, generate_questions = loaded_model
        competence_list = inputs['competence_list'].split('\n')
        case = generate_case(competence_list)
        questions = generate_questions(case, competence_list)
        return {
            'case': case,
            'questions': questions
        }
    
    def render_result(this, generation_result):
        generated_questions = generation_result['questions']
        case = generation_result['case']

        st.write(f"**Кейс:**")
        st.markdown(f"<div style='padding: 8px; background-color: rgba(255, 255, 255, 0.01); color: white; border: 1px solid rgba(255, 255, 255, 0.3); border-radius: 5px; margin-bottom: 16px;'>{case}</div>", unsafe_allow_html=True)

        st.write("**Неправильные ответы (дистракторы):**")
        num_q = len(generated_questions)
        for j, gen_q in enumerate(generated_questions):
            competence = gen_q['competence']
            question = gen_q['question']
            st.markdown(f"<div style='padding: 8px; background-color: rgba(255, 255, 255, 0.01); color: white; border: 1px solid rgba(255, 255, 255, 0.3); border-radius: 5px;{' margin-top: 8px;' if j > 0 else ''}{' margin-bottom: 16px;' if j == num_q - 1 else ''}'>\n    <div>\n        <b>Компетенция:</b>\n        <span>{competence}</span>\n    </div>\n    <div style='margin-top: 4px;'>\n        <b>Вопрос:</b>\n        <span>{question}</span>\n    </div>\n</div>", unsafe_allow_html=True)
