from io import BytesIO
import random
import time
import streamlit as st
import pandas as pd
from typing import Dict, List, Tuple
import json

# Import the generate_theme and generate_exam_question functions from the other file
import streamlit.components.v1 as components
from demo_type1.demo_type1 import Type1UI
from demo_type2.demo_type2 import Type2UI
from demo_type4.demo_type4 import Type4UI
from question_generation_ui import QuestionGenerationUI

@st.cache_resource
def load_question_types() -> Dict[str, QuestionGenerationUI]:
    question_types = [
        Type1UI,
        Type2UI,
        Type4UI
    ]

    question_types_map = {}
    question_types_map_index = {}
    for i, question_type in enumerate(question_types):
        question_type_name = question_type.get_question_type_name()
        question_types_map[question_type_name] = question_type()
        question_types_map_index[question_type_name] = i

    return question_types_map, question_types_map_index

question_types_map, question_types_map_index = load_question_types()

@st.cache_data
def load_examples(question_type):
    return question_types_map[question_type].load_examples()

@st.cache_resource
def load_model(question_type_name, primary_model, num_examples):
    question_class = question_types_map[question_type_name]
    return question_class.load_model(primary_model, num_examples)

def has_state(key):
    return key in st.session_state and st.session_state[key] is not None

def ChangeButtonSize(widget_label, size):
    htmlstr = f"""
<script>
 var elements = window.parent.document.querySelectorAll('button');
 for (var i = 0; i < elements.length; ++i) {{ 
    if (elements[i].innerText == '{widget_label}') {{ 
        elements[i].style.width = '{size}'
 }}
 }}
</script>
 """
    components.html(f"{htmlstr}", height=0, width=0)


def ChangeButtonColor(widget_label, font_color, background_color='transparent'):
    htmlstr = f"""
<script>
 var elements = window.parent.document.querySelectorAll('button');
 for (var i = 0; i < elements.length; ++i) {{ 
    if (elements[i].innerText == '{widget_label}') {{ 
        elements[i].style.color ='{font_color}';
        elements[i].style.background = '{background_color}'
 }}
 }}
</script>
 """
    components.html(f"{htmlstr}", height=0, width=0)

def validate_input(inputs):
    for input_key, input_value in inputs.items():
        if input_value is None or input_value.strip() == '':
            print(f'Validation failed for {input_key}: {input_value}')
            return False
        
    return True

st.markdown("""
    <style>
    .stCheckbox {
        display: flex;
        align-items: center;
        height: 100%;
        padding-top: 34px;
    }
    .stCheckbox {
        justify-content: center;
    }
    .block-container {
        flex-direction: row;
    }
    div:has(> iframe) {
        display: None;
        width: 0%;
    }
    </style>
 """, unsafe_allow_html=True)

st.title("Exam Question Generator")

current_index = 0
if has_state('question_type_name'):
    current_index = question_types_map_index[st.session_state.question_type_name]

question_type_name = st.selectbox(
            "Выберите тип вопроса",
            list(question_types_map.keys()),
            index=current_index
        )

st.session_state.question_type_name = question_type_name

question_type: QuestionGenerationUI = question_types_map[question_type_name]
question_params = question_type.get_params()
question_labels = question_type.get_labels()

# === Select Model and num examples ===

model_list = question_type.get_model_list()
col1, col2 = st.columns([3, 1])

with col1:
    model_name = st.selectbox(
        "Выберите модель",
        model_list,
        index=0
    )

with col2:
    if question_params['num_few_shot'] > 0:
        num_examples = st.number_input(
            "Количество примеров", min_value=0, 
            max_value=question_params['num_few_shot'], 
            value=question_params['num_few_shot'])
    else:
        num_examples = 0

if model_name is None or model_name not in model_list:
    model_name = model_list[0]

if num_examples < 0 or num_examples > question_params['num_few_shot']:
    num_examples = question_params['num_few_shot']

loaded_model = load_model(
    question_type_name,
    primary_model=model_name,
    num_examples=num_examples,
)

for input_key, _, _ in question_type.get_inputs():
    if not has_state(input_key):
        st.session_state[input_key] = ""

# === Select Example ===

if question_params['has_examples']:
    option_names, examples = question_type.load_examples()
    current_index = None
    if has_state('example_index'):
        current_index = st.session_state.example_index

    if current_index is not None and (current_index < 0 or current_index > len(option_names)):
        current_index = None
        st.session_state.example_index = None

    template_question = st.selectbox(
        question_labels['examples'],
        option_names,
        placeholder="Выберите пример",
        index=current_index
    )

    if template_question is not None:
        if st.button(question_labels['example_button']):
            index = option_names.index(template_question)
            st.session_state.example_index = index
            example_inputs = question_type.select_example(template_question, index, examples)

            for input_key, example_input in example_inputs.items():
                st.session_state[input_key] = example_input

# === Create Inputs ===

inputs = {}
for input_key, input_label, input_args in question_type.get_inputs():
    input_result = st.text_area(
        input_label,
        value=st.session_state[input_key],
        key=f'{input_key}_key',
        **input_args
    )
    inputs[input_key] = input_result

num_questions = st.number_input(
    "Количество вопросов для генерации", min_value=1, max_value=10, value=1)

# === Generate Button ===

if st.button(question_labels['generate_button']):
    if not validate_input(inputs):
        st.error("Пожалуйста, заполните все поля.")
    else:
        st.session_state.generate_button = True

if has_state('generate_button') and st.session_state.generate_button:

    # === Theme ===

    if question_params['has_theme_generation']:
        if not has_state('theme'):
            with st.spinner(question_labels['theme_analyze']):
                theme = question_type.generate_theme(loaded_model, inputs)
                st.session_state.theme = theme

        theme = st.session_state.theme
        st.write(f"**{question_labels['theme']}:**")
        st.markdown(f"<div style='padding: 8px; background-color: rgba(255, 255, 255, 0.01); color: white; border: 1px solid rgba(255, 255, 255, 0.3); border-radius: 5px; margin-bottom: 16px;'>{theme}</div>", unsafe_allow_html=True)

        col1_i, _ = st.columns(2)
        with col1_i:
            col1, col2 = st.columns([0.8, 0.2])
        with col1:
            if st.button("Подтвердить", key="confirm_theme"):
                st.session_state.theme_confirmed = True

        with col2:
            if st.button("X", key="reject_theme"):
                st.session_state.theme_confirmed = False

        ChangeButtonSize('Подтвердить', '100%')
        ChangeButtonColor('Подтвердить', 'white', '#4CAF50')
        ChangeButtonColor('X', 'white', '#f44336')
    else:
        theme = None

    # === Generate Questions ===

    if (not question_params['has_theme_generation']) or (has_state('theme_confirmed') and st.session_state.theme_confirmed):
        start_time = time.time()

        for i in range(num_questions):
            with st.spinner(f"Генерация вопроса {i + 1} из {num_questions}..."):
                try:
                    generation_result = question_type.generate(loaded_model, inputs, theme)

                    with st.expander(f"Сгенерированный вопрос {i + 1}", expanded=(i == 0)):
                        question_type.render_result(generation_result)
                except Exception as e:
                    st.error(f"Произошла ошибка при генерации вопроса {i + 1}: {str(e)}")

        end_time = time.time()
        elapsed_time = end_time - start_time
        st.success(f"Все вопросы сгенерированы за {elapsed_time:.2f} секунд")
