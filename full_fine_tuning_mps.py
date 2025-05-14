# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import pandas as pd
import torch

# Brug MPS hvis tilgængeligt (Apple M1/M2 chips)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
model = model.to(device)  # Flyt modellen til MPS

prompt = 'Hej med dig, hvordan går det?'

inputs = tokenizer(prompt, return_tensors='pt')
inputs = {k: v.to(device) for k, v in inputs.items()}  # Flyt input til MPS

print('DEVICE!!')
print(device)

# Generér svar
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    pad_token_id=tokenizer.eos_token_id
)

# Dekodér og vis
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

####################

squad = load_dataset('squad_v2')
squad_train = squad['train']
#df = pd.DataFrame(squad['train'])


#####################

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


###################

from trl import SFTConfig, SFTTrainer

trainer_config = SFTConfig(
    output_dir='outputs',
    learning_rate=5e-7,
    lr_scheduler_type='constant',
    max_grad_norm=0.3,
    weight_decay=0.001,
    max_steps=100,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    max_seq_length=1024,
    packing=True,
    fp16=False,
    logging_steps=1,
    seed=42
)

trainer = SFTTrainer(
    model=model,
    train_dataset=squad['train'],
    formatting_func=format_instruction,
    args=trainer_config
)

trainer.train()