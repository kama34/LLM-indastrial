from io import BytesIO
import random
import time
import streamlit as st
import pandas as pd
from typing import List, Tuple
import json

# Import the generate_theme and generate_exam_question functions from the other file
from use_vllm import get_model, get_model_list, ExamQuestion
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
    return res


@st.cache_resource
def load_model(primary_model, few_shot=False):
    print(f'Loaded {primary_model} model with few-shot={few_shot}')
    return get_model(primary_model, few_shot)


parsed_questions = load_parsed_questions()

model_list = get_model_list()[::-1]

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

allow_model_choice = False

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
        few_shot_learning = st.checkbox("Few-shot learning")

    if model_name is None or model_name not in model_list:
        model_name = model_list[0]

    if few_shot_learning is None:
        few_shot_learning = False

    generate_theme, generate_exam_question = load_model(
        model_name, few_shot_learning)
else:
    generate_theme, generate_exam_question = load_model('saiga', True)

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
    st.markdown(f"{theme.question_theme}", unsafe_allow_html=True)

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
        start_time = time.time()

        for i in range(num_questions):
            with st.spinner(f"Генерация вопроса {i + 1} из {num_questions}..."):
                try:
                    exam_question = generate_exam_question(
                        theme, reference_question, questions)
                    questions.append(exam_question)

                    with st.expander(f"Сгенерированный вопрос {i + 1}", expanded=(i == 0)):
                        st.write(f"**Вопрос:** {exam_question.question}")

                        st.write("**Правильный ответ:**")
                        st.markdown(f"{exam_question.correct_answer}", unsafe_allow_html=True)

                        st.write("**Неправильные ответы (дистракторы):**")
                        for j, distractor in enumerate(exam_question.distractors):
                            st.markdown(f"<div style='padding: 5px; background-color: #f4cccc; color: #721c24; border: 1px solid #f5c6cb; border-radius: 5px;{' margin-top: 6px;' if j > 0 else ''}{' margin-bottom: 12px;' if j == 2 else ''}'>{distractor}</div>", unsafe_allow_html=True)

                    if debug:
                        with st.expander("Debug Information", expanded=False):
                            st.json({
                                "Generated Question": exam_question.question,
                                "Correct Answer": exam_question.correct_answer,
                                "Distractors": exam_question.distractors,
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
        for i, exam_question in enumerate(questions):
            with st.expander(f"Сгенерированный вопрос {i + 1}", expanded=(i == 0)):
                st.write(f"**Вопрос:** {exam_question.question}")

                st.write("**Правильный ответ:**")
                st.markdown(f"{exam_question.correct_answer}",
                            unsafe_allow_html=True)

                st.write("**Неправильные ответы (дистракторы):**")
                for j, distractor in enumerate(exam_question.distractors):
                    st.markdown(f"<div style='padding: 5px; background-color: #f4cccc; color: #721c24; border: 1px solid #f5c6cb; border-radius: 5px;{' margin-top: 6px;' if j > 0 else ''}{' margin-bottom: 12px;' if j == 2 else ''}'>{distractor}</div>", unsafe_allow_html=True)

        st.success(f"Все вопросы сгенерированы за {elapsed_time:.2f} секунд")

    export_data = []
    for exam_question in questions:
        all_answers = [exam_question.correct_answer] + \
            exam_question.distractors
        random.shuffle(all_answers)
        correct_index = all_answers.index(exam_question.correct_answer)
        row = {
            "Вопрос по компетенции (Задание)": exam_question.question,
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
