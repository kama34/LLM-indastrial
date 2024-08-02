from io import BytesIO
import random
import time
import streamlit as st
import pandas as pd
from typing import Any, Dict, List, Tuple, Union
import json

from question_generation_ui import QuestionGenerationUI

from demo_type2.api_type_2 import get_model, ExamQuestionType2
from utils import get_model_list
import streamlit.components.v1 as components

class Type2UI(QuestionGenerationUI):
    @classmethod
    def get_question_type_name(cls) -> str:
        return "Correct Sequence"
    
    @classmethod
    def get_params(cls) -> Dict[str, Any]:
        return {
            'has_examples': True,
            'has_theme_generation': True,
            'num_few_shot': 1
        }
    
    @classmethod
    def get_labels(cls) -> Dict[str, str]:
        return {
            'examples': 'Выберите пример компетенции из датасета',
            'example_button': 'Использовать выбранную компетенцию',
            'generate_button': 'Генерация вопроса',
            'theme': 'Сгенерированное описание кейса',
            'theme_analyze': 'Генерация кейса...'
        }
    
    # return list of (input_key, input_label, input_args)
    # input_args - additional args, i.e. height: 200
    def get_inputs(this) -> List[Tuple[str, str, Union[Dict, None]]]:
        return [('case_name', 'Введите описание кейса', {'height': 100}),
                ('competence', 'Введите название компетенции', {'height': 100}),]
    
    def get_model_list(this) -> List[str]:
        return get_model_list()
    
    def load_model(this, primary_model, num_examples):
        print(f'Loaded {primary_model} model with few-shot={num_examples}')
        return get_model(primary_model, num_examples)

    def load_examples(this) -> Tuple[List, List]:
        with open('./demo_type2/case_questions1.json', 'r') as f:
            data = json.load(f)

        res = []
        for q in data:
            res.append(
                ExamQuestionType2(
                    case_name=str(q['case_name']),
                    competence=str(q['competence']),
                )
            )

        examples = res[1:]
        option_names = [f"{i + 1}: {q.competence}" for i, q in enumerate(examples)]
        return option_names, examples

    def select_example(this, selected_option_name, selected_index, examples) -> str:
        selected_question = examples[selected_index]
        case_name = selected_question.to_string_case_name()
        competence = selected_question.to_string_competence()
        return {'case_name': case_name, 'competence': competence}

    def generate_theme(this, loaded_model, inputs):
        generate_desc, _ = loaded_model
        case_name = inputs['case_name']
        competence = inputs['competence']
        return generate_desc(case_name, competence)

    def generate(this, loaded_model, inputs, desc):
        _, generate_steps = loaded_model
        competence = inputs['competence']
        return generate_steps(competence, desc)
    
    def render_result(this, generation_result):
        generated_steps_right = generation_result['generated_steps_right']
        distractors = generation_result['distractors']

        st.write("**Вопрос:**")
        st.write('Какие из предложенных действий вы выполните? Расположите их в правильной последовательности')
        
        st.write("**Правильные шаги:**")
        steps_formatted = '\n'.join([f' {i+1}. {step}' for i, step in enumerate(generated_steps_right)])
        st.markdown(f"{steps_formatted}", unsafe_allow_html=True)
        
        num_q = len(distractors['negative_steps'])
        st.write("**Неправильные шаги (дистракторы):**")
        for j, distractor in enumerate(distractors['negative_steps']):
            st.markdown(f"<div style='padding: 5px; background-color: #f4cccc; color: #721c24; border: 1px solid #f5c6cb; border-radius: 5px;{' margin-top: 6px;' if j > 0 else ''}{' margin-bottom: 12px;' if j == num_q - 1 else ''}'>{distractor}</div>", unsafe_allow_html=True)