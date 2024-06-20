import os
import dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

import pandas as pd
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate

import random
from sklearn import metrics

# TODO Model
dotenv.load_dotenv()
HF_SECRET = os.environ.get('HF_SECRET')
HF_MODEL = 'allenai/OLMo-7B-hf'
N_TOKENS = 3

model = AutoModelForCausalLM.from_pretrained(
    HF_MODEL,
    token=HF_SECRET,
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    HF_MODEL,
    token=HF_SECRET,
    trust_remote_code=True
)

pipe = pipeline(
    task='text-generation',
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=N_TOKENS
)

print('Model: Initialized')

# TODO Template
I_EXAMPLES = [0, 5]

context = pd.read_json('../data/outcome_similarity_detection/context.json')
context = context.loc[0]
context = context.to_dict()

examples = pd.read_json('../data/outcome_similarity_detection/few_shot.json')
examples = examples.loc[I_EXAMPLES]
examples = examples.to_dict(orient='records')

examples_template = PromptTemplate(
    input_variables=['primary_outcome', 'reported_outcome', 'label'],
    template=f"""Definition: {context['definition']}
Question: {context['question']}
Primary outcome: {{primary_outcome}}
Reported outcome: {{reported_outcome}}
Answer: {{label}}""",
)

input_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=examples_template,
    input_variables=['primary_outcome', 'reported_outcome'],
    suffix=f"""Definition: {context['definition']}
Question: {context['question']}
Primary outcome: {{primary_outcome}}
Reported outcome: {{reported_outcome}}
"""
)

print('Template: Initialized')

# TODO Evaluate
N_DATA = 50

DATA = pd.read_csv('../data/outcome_similarity_detection/train.tsv', sep='\t')
DATA = DATA.iloc[:N_DATA]

predictions = []
invalid_answers = 0
for i in DATA.iterrows():
    row = i[1]
    input = input_template.format(
        primary_outcome=row['out1'],
        reported_outcome=row['out2']
    )
    output = pipe(input)

    answer = output[0]['generated_text'][len(input):]
    print(answer)
    if '0' in answer:
        predictions.append(0)
    elif '1' in answer:
        predictions.append(1)
    else:
        predictions.append(random.choice([0, 1]))
        invalid_answers += 1

labels = DATA['label'].tolist()

print('Predictions:')
print(f"0: {predictions.count(0)}")
print(f"1: {predictions.count(1)}")
print(f"Invalid: {invalid_answers}")
print('------------')
print(f"Accuracy: {metrics.accuracy_score(labels, predictions)}")
print(f"Precision: {metrics.precision_score(labels, predictions)}")
print(f"Recall: {metrics.recall_score(labels, predictions)}")
print(f"F1: {metrics.f1_score(labels, predictions)}")
