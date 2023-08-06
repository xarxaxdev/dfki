from datasets import load_dataset

squad = load_dataset("squad", split="train[:5000]")

squad = squad.train_test_split(test_size=0.2)
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


tokenized_squad = squad.map(preprocess_function, batched=True, remove_columns=squad["train"].column_names)


from transformers import DefaultDataCollator

data_collator = DefaultDataCollator()


from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer

model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased").to('cuda')


epochs = 1 
lr = 2e-5 
pretrained_model = 'distilbert-base-uncased' 
model_name = f'{pretrained_model}_lr{lr}_epochs{epochs}'
training_args = TrainingArguments(
    output_dir=f"~/generated_models/{model_name}",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=epochs,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    #save_strategy="epoch",
    #load_best_model_at_end=True#,
    #push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=squad["train"],
    eval_dataset=squad["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    #compute_metrics=compute_metrics,
)

trainer.train()
import os
def save_model(model, filename):
    cur_path = os.path.split(os.path.realpath(__file__))[0]
    project_path = cur_path#os.path.split(cur_path)[0]
    datafile = os.path.join(os.path.expanduser('~'),'generated_models', filename)
    #torch.save(model, datafile)
    trainer.save_model(datafile)
    return True



save_model(model, model_name)
