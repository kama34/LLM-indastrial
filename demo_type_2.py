from io import BytesIO
import random
import time
import streamlit as st
import pandas as pd
from typing import Any, Dict, List, Tuple, Union
import json

from question_generation_ui import QuestionGenerationUI

# Import the generate_theme and generate_exam_question functions from the other file
from baseline.api_type_2 import get_model, get_model_list, ExamQuestionType2
from baseline.baseline_vllm import get_model as get_model_saiga
import streamlit.components.v1 as components

class Type1UI(QuestionGenerationUI):
    @classmethod
    def get_question_type_name(cls) -> str:
        return "Multichoice Question"
    
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
            'examples': 'Выберите пример вопроса из датасета',
            'example_button': 'Использовать выбранный вопрос',
            'generate_button': 'Генерация',
            'theme': 'Сгенерированное описание кейса',
            'theme_analyze': 'Генерация описания кейса...'
        }
    
    # return list of (input_key, input_label, input_args)
    # input_args - additional args, i.e. height: 200
    def get_inputs(this) -> List[Tuple[str, str, Union[Dict, None]]]:
        return [('case_name', 'Введите название кейса', {'height': 100}),
                ('competence', 'Введите описание компетенция', {'height': 100}),]
    
    def get_model_list(this) -> List[str]:
        return get_model_list()
    
    def load_model(this, primary_model, num_examples):
        print(f'Loaded {primary_model} model with few-shot={num_examples}')
        if primary_model == 'saiga-8b':
            return get_model_saiga(num_examples)
        return get_model(primary_model, num_examples)

    def load_examples(this) -> Tuple[List, List]:
        with open('./case_questions1.json', 'r') as f:
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
        option_names = [f"Вопрос {i + 1}: {q.case_name}" for i, q in enumerate(examples)]
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
        
        st.write("**Правильные шаги:**")
        steps_formatted = '\n'.join([' - ' + step for step in generated_steps_right])
        st.markdown(f"{steps_formatted}", unsafe_allow_html=True)
        
        st.write("**Неправильные шаги (дистракторы):**")
        for j, distractor in enumerate(distractors['negative_steps']):
            st.markdown(f"<div style='padding: 5px; background-color: #f4cccc; color: #721c24; border: 1px solid #f5c6cb; border-radius: 5px;{' margin-top: 6px;' if j > 0 else ''}{' margin-bottom: 12px;' if j == 2 else ''}'>{distractor}</div>", unsafe_allow_html=True)