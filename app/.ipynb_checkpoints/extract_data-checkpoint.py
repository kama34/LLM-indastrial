import pandas as pd
from typing import List, Tuple
import json

def parse_excel_file(file) -> List[Tuple[str, List[str]]]:
    questions = []
    xl = pd.ExcelFile(file)
    for sheet in xl.sheet_names[:7]:  # Only first 7 sheets
        df = xl.parse(sheet)
        for index, row in df.iterrows():
            question = row.iloc[6]
            answers = [row.iloc[8], row.iloc[9], row.iloc[10], row.iloc[11]]
            correct_answer = ord(row.iloc[7]) - ord('А')
            if correct_answer < 0 or correct_answer >= 4:
                print('[ERROR] Sheet', sheet, '|', question, f'({row.iloc[7]} -> {correct_answer})')
            if question == "":
                break
            questions.append((question, answers, correct_answer))
    return questions

def export_questions_json(questions):
    data = []
    for question, answers, correct in questions:
        data.append({
            'question': question,
            'correct_answer': answers[correct],
            'distractors': answers[:correct] + answers[correct+1:]
        })
    return data

if __name__ == '__main__':
    questions = parse_excel_file('./question_references.xlsx')
    data = export_questions_json(questions)
    data_json = json.dumps(data, indent=4, ensure_ascii=False)
    with open("dataset.json", "w") as dataset_file:
        dataset_file.write(data_json)
    print(data_json)
    print('Number of questions:', len(data))