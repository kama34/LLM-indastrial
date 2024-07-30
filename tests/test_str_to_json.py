import json
import re

def str_to_json(output):
    if not isinstance(output, str):
        raise ValueError("output should be a string")

    # Попытка преобразовать строку в JSON
    try:
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
    except json.JSONDecodeError as e:
        print(f"Ошибка декодирования output в output_json: {e}")
        raise e
    
string1 = '```json{"question": "Какой тип анализа используется при критическом чтении и为何分析 of a literary text?", "correct_answer": "Филологический анализ"}```'
string2 = 'asdasdasd```json{"question": "Какой тип анализа используется при критическом чтении и为何分析 of a literary text?", "correct_answer": "Филологический анализ"}```'
string3 = '```json{"question": "Какой тип анализа используется при критическом чтении и为何分析 of a literary text?", "correct_answer": "Филологический анализ"}```asdasdasd'
string4 = 'asdasdasd```json{"question": "Какой тип анализа используется при критическом чтении и为何分析 of a literary text?", "correct_answer": "Филологический анализ"}```asdasdasd'
string5 = '{"question": "Какой тип анализа используется при критическом чтении и为何分析 of a literary text?", "correct_answer": "Филологический анализ"}```'
string6 = '```json{"question": "Какой тип анализа используется при критическом чтении и为何分析 of a literary text?", "correct_answer": "Филологический анализ"}'
string7 = '{"question": "Какой тип анализа используется при критическом чтении и为何分析 of a literary text?", "correct_answer": "Филологический анализ"}'

print(str_to_json(string1))
print(str_to_json(string2))
print(str_to_json(string3))
print(str_to_json(string4))
print(str_to_json(string5))
print(str_to_json(string6))
print(str_to_json(string7))