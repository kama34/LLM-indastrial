from io import BytesIO
import random
import time
import streamlit as st
import pandas as pd
from typing import Any, Dict, List, Tuple, Union
import json

from question_generation_ui import QuestionGenerationUI

from demo_type1.baseline_api import get_model, ExamQuestion
from demo_type1.baseline_vllm import get_model as get_model_saiga
from utils import get_model_list
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
            'num_few_shot': 3
        }
    
    @classmethod
    def get_labels(cls) -> Dict[str, str]:
        return {
            'examples': 'Выберите пример вопроса из датасета',
            'example_button': 'Использовать выбранный вопрос',
            'generate_button': 'Сгенерировать вопросы',
            'theme': 'Определенная тема',
            'theme_analyze': 'Анализ темы...'
        }
    
    # return list of (input_key, input_label, input_args)
    # input_args - additional args, i.e. height: 200
    def get_inputs(this) -> List[Tuple[str, str, Union[Dict, None]]]:
        return [('reference_question', 'Введите вопрос с 4 вариантами ответа на русском языке', {'height': 200})]
    
    def get_model_list(this) -> List[str]:
        return get_model_list() + ['saiga-8b']
    
    def load_model(this, primary_model, num_examples):
        print(f'Loaded {primary_model} model with few-shot={num_examples}')
        if primary_model == 'saiga-8b':
            return get_model_saiga(num_examples)
        return get_model(primary_model, num_examples)

    def load_examples(this) -> Tuple[List, List]:
        with open('./demo_type1/dataset.json', 'r') as f:
            data = json.load(f)

        res = []
        for q in data:
            res.append(
                ExamQuestion(
                    question=str(q['question']),
                    correct_answer=str(q['correct_answer']),
                    distractors=[str(d) for d in q['distractors']]
                )
            )

        examples = res[3:]
        option_names = [f"Вопрос {i + 1}: {q.question}" for i, q in enumerate(examples)]
        return option_names, examples

    def select_example(this, selected_option_name, selected_index, examples) -> str:
        selected_question = examples[selected_index]
        formatted_question = selected_question.to_string_with_distractors()
        return {'reference_question': formatted_question}

    def generate_theme(this, loaded_model, inputs):
        generate_theme, _ = loaded_model
        reference_question = inputs['reference_question']
        return generate_theme(reference_question)

    def generate(this, loaded_model, inputs, theme):
        _, generate_exam_question = loaded_model
        reference_question = inputs['reference_question']
        return generate_exam_question(
                        theme, reference_question)
    
    def render_result(this, generation_result):
        generated_question = generation_result['generated_question']
        correct_answer = generation_result['correct_answer']
        distractors = generation_result['distractors']

        st.write(f"**Вопрос:** {generated_question}")

        st.write("**Правильный ответ:**")
        st.markdown(f"{correct_answer}", unsafe_allow_html=True)

        num_q = len(distractors.values())
        st.write("**Неправильные ответы (дистракторы):**")
        for j, distractor in enumerate(distractors.values()):
            st.markdown(f"<div style='padding: 5px; background-color: #f4cccc; color: #721c24; border: 1px solid #f5c6cb; border-radius: 5px;{' margin-top: 6px;' if j > 0 else ''}{' margin-bottom: 12px;' if j == num_q - 1 else ''}'>{distractor}</div>", unsafe_allow_html=True)
