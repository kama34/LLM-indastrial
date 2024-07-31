from io import BytesIO
import random
import time
import streamlit as st
import pandas as pd
from typing import List, Tuple
import json

# Import the generate_theme and generate_exam_question functions from the other file
from baseline.api_type_1 import get_model, get_model_list, ExamQuestion
import streamlit.components.v1 as components

debug = False


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

# Function to export questions to an Excel file


def export_to_excel(questions: List[dict]) -> bytes:
    df = pd.DataFrame(questions)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name="Вопросы")
    return output.getvalue()


# Set up Streamlit app
st.title("Exam Question Generator")

# Load and cache the parsed questions


@st.cache_data
def load_parsed_questions():
    with open('dataset.json', 'r') as f:
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
    return res[2:]


@st.cache_resource
def load_model(primary_model, num_examples):
    print(f'Loaded {primary_model} model with few-shot={num_examples}')
    return get_model(primary_model, num_examples)


parsed_questions = load_parsed_questions()

model_list = get_model_list()

# Create two columns
col1, col2 = st.columns([3, 1])  # Adjust the width ratio as needed

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
    !div iframe {
        width: 0%;
    }
    </style>
 """, unsafe_allow_html=True)

allow_model_choice = True

if allow_model_choice:
    # Place the selectbox in the first column
    with col1:
        model_name = st.selectbox(
            "Выберите модель",
            model_list,
            index=0
        )

    # Place the checkbox in the second column
    with col2:
        num_examples = st.number_input(
    "Количество примеров", min_value=0, max_value=3, value=3)

    if model_name is None or model_name not in model_list:
        model_name = model_list[0]

    if num_examples is None or num_examples < 0 or num_examples > 3:
        num_examples = 3
    
    generate_theme, generate_exam_question = load_model(
        primary_model=model_name,
        num_examples=num_examples,
    )
    
else:
    generate_theme, generate_exam_question = load_model('qwen2-72b', 3)

template_question = st.selectbox(
    "Выберите пример вопроса из файла",
    [f"Вопрос {i + 1}: {q.question}" for i, q in enumerate(parsed_questions)],
    placeholder="Выберите пример",
)

if "reference_question" not in st.session_state:
    st.session_state.reference_question = ""

examples = random.choices(parsed_questions, k=5)  # depricated

if template_question is not None:
    selected_question_index = int(
        template_question.split(':')[0].split()[1]) - 1
    selected_question = parsed_questions[selected_question_index]
    formatted_question = selected_question.to_string_with_distractors()

    if st.button("Использовать выбранный вопрос"):
        st.session_state.reference_question = formatted_question

    examples = random.choices(
        parsed_questions[:selected_question_index] + parsed_questions[selected_question_index+1:], k=5)

# User input for the reference question
reference_question = st.text_area(
    "Введите вопрос с 4 вариантами ответа на русском языке",
    value=st.session_state.reference_question,
    height=200
)

num_questions = st.number_input(
    "Количество вопросов для генерации", min_value=1, max_value=10, value=1)
is_rendered = False

if st.button("Сгенерировать вопросы"):
    if reference_question.strip() == "":
        st.error("Пожалуйста, введите вопрос с вариантами ответа.")
    else:
        st.session_state.generate_button = True

if 'generate_button' in st.session_state and st.session_state.generate_button:
    # Analyze the theme with a spinner
    if 'theme' not in st.session_state:
        with st.spinner("Анализ темы..."):
            theme = generate_theme(reference_question)
            st.session_state.theme = theme

    theme = st.session_state.theme
    st.write("**Определенная тема:**")
    st.markdown(f"{theme}", unsafe_allow_html=True)

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

    if 'theme_confirmed' in st.session_state and st.session_state.theme_confirmed:
        questions = []
        questions2 = []
        start_time = time.time()

        for i in range(num_questions):
            with st.spinner(f"Генерация вопроса {i + 1} из {num_questions}..."):
                try:
                    exam_question = generate_exam_question(
                        theme, reference_question, questions2)
                    questions.append(exam_question)
                    questions2.append(exam_question['generated_question'])
                    
                    generated_question = exam_question['generated_question']
                    correct_answer = exam_question['correct_answer']
                    distractors = exam_question['distractors']

                    with st.expander(f"Сгенерированный вопрос {i + 1}", expanded=(i == 0)):
                        st.write(f"**Вопрос:** {generated_question}")

                        st.write("**Правильный ответ:**")
                        st.markdown(f"{correct_answer}", unsafe_allow_html=True)

                        st.write("**Неправильные ответы (дистракторы):**")
                        for j, distractor in enumerate(distractors.values()):
                            st.markdown(f"<div style='padding: 5px; background-color: #f4cccc; color: #721c24; border: 1px solid #f5c6cb; border-radius: 5px;{' margin-top: 6px;' if j > 0 else ''}{' margin-bottom: 12px;' if j == 2 else ''}'>{distractor}</div>", unsafe_allow_html=True)

                    if debug:
                        with st.expander("Debug Information", expanded=False):
                            st.json({
                                "Generated Question": generated_question,
                                "Correct Answer": correct_answer,
                                "Distractors": distractors,
                            })
                except Exception as e:
                    st.error(f"Произошла ошибка при генерации вопроса {i + 1}: {str(e)}")

        end_time = time.time()
        elapsed_time = end_time - start_time
        st.success(f"Все вопросы сгенерированы за {elapsed_time:.2f} секунд")
        is_rendered = True

        # Save the generated questions in the session state
        st.session_state.generated_questions = questions
        st.session_state.elapsed_time = elapsed_time

if "generated_questions" in st.session_state:
    questions = st.session_state.generated_questions
    elapsed_time = st.session_state.elapsed_time

    if not is_rendered:
        for i, exam_question in enumerate(questions.values()):
            with st.expander(f"Сгенерированный вопрос {i + 1}", expanded=(i == 0)):
                st.write(f"**Вопрос:** {exam_question['generated_question']}")

                st.write("**Правильный ответ:**")
                st.markdown(f"{exam_question['correct_answer']}",
                            unsafe_allow_html=True)

                st.write("**Неправильные ответы (дистракторы):**")
                for j, distractor in enumerate(exam_question['distractors'].values()):
                    st.markdown(f"<div style='padding: 5px; background-color: #f4cccc; color: #721c24; border: 1px solid #f5c6cb; border-radius: 5px;{' margin-top: 6px;' if j > 0 else ''}{' margin-bottom: 12px;' if j == 2 else ''}'>{distractor}</div>", unsafe_allow_html=True)

        st.success(f"Все вопросы сгенерированы за {elapsed_time:.2f} секунд")

    export_data = []
    for exam_question in questions:
        print(f"exam_question = {exam_question}")
        all_answers = [exam_question['correct_answer']] + \
            list(exam_question['distractors'].values())
        random.shuffle(all_answers)
        correct_index = all_answers.index(exam_question['correct_answer'])
        row = {
            "Вопрос по компетенции (Задание)": exam_question['generated_question'],
            "Правильный ответ": chr(ord('А') + correct_index),
            "Ответ А": all_answers[0],
            "Ответ Б": all_answers[1],
            "Ответ В": all_answers[2],
            "Ответ Г": all_answers[3],
        }
        export_data.append(row)

    excel_data = export_to_excel(export_data)
    st.download_button(
        label="Скачать сгенерированные вопросы в формате Excel",
        data=excel_data,
        file_name="generated_questions.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
