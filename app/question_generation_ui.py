from io import BytesIO
import random
import time
import streamlit as st
import pandas as pd
from typing import Any, Dict, List, Tuple, Union
import json
import streamlit.components.v1 as components

class QuestionGenerationUI:
    @classmethod
    def get_question_type_name(cls) -> str:
        raise NotImplementedError()
    
    @classmethod
    def get_params(cls) -> Dict[str, Any]:
        return {
            'has_examples': True,
            'has_theme_generation': True,
            'num_few_shot': 0
        }

    @classmethod
    def get_labels(cls) -> Dict[str, str]:
        return {
            'examples': 'Выберите пример из датасета',
            'example_button': 'Использовать выбранный пример',
            'generate_button': 'Сгенерировать вопросы',
            'theme': 'Определенная тема',
            'theme_analyze': 'Анализ темы...'
        }

    def get_model_list(this) -> List[str]:
        raise NotImplementedError()

    def load_model(this, primary_model, num_examples):
        raise NotImplementedError()

    def load_examples(this) -> Tuple[List, List]:
        pass
    
    # example for each input
    def select_example(this, selected_option_name, selected_index, examples) -> Dict[str, str]:
        pass

    # return list of (input_key, input_label, input_args)
    # input_args - additional args, i.e. height: 200
    def get_inputs(this) -> List[Tuple[str, str, Union[Dict, None]]]:
        return [('input', 'Введите вопрос на русском языке', None)]

    def generate_theme(this, loaded_model, inputs):
        pass

    def generate(this, loaded_model, inputs, theme):
        raise NotImplementedError()
    
    def render_result(this, generation_result):
        raise NotImplementedError()
