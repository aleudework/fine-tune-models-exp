from datasets import load_dataset
import pandas as pd

squad = load_dataset('squad_v2')
df = pd.DataFrame(squad['train'])


def format_instruction(row):

    # If the field is empty, it returns a default string
    context_ = row.get("context", 'No context provided')
    question_ = row.get("question", 'No question provided')

    answer_text = row.get("answers", {}).get("text", [])
    if not answer_text:
        answer_ = "No answer provided"
    else:
        answer_ = answer_text[0]

    prompt = (
        f"### Context: {context_} "
        f"### Question: {question_} "
        f"### Answer: {answer_} "
    )

    return prompt

print(format_instruction(squad['train'][1]))

