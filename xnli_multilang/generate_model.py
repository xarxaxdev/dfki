from datasets import load_dataset


languages = ['ar', 'bg', 'de', 'el', 'en', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh']
#languages.append['all_languages'] #I'll think what to do with this one

xnli = {}
for l in languages:
    xnli[l] = load_dataset("xnli",l)
#xnli = load_dataset("xnli",'ar')

base_model = "bert-base-multilingual-cased"
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(base_model)

def preprocess_function(examples):
    print(examples)
    return tokenizer(examples["premise"] +tokenizer.sep_token + examples["premise"], truncation=True)

def preprocess_function_p(examples):
    return tokenizer(examples["premise"], truncation=True)

def preprocess_function_h(examples):
    return tokenizer(examples["hypothesis"],truncation=True)

#tokenized_xnli_p = {}
#tokenized_xnli_h = {}
tokenized_xnli = {}
for l in languages:
    #tokenized_xnli_p[l] = {}
    #tokenized_xnli_h[l] = {}
    #for s in ['train','test']:
    #    tokenized_xnli_p[l][s] = xnli[l][s].map(preprocess_function_p, batched=True)
    #    tokenized_xnli_h[l][s] = xnli[l][s].map(preprocess_function_h, batched=True)
    tokenized_xnli[l] = xnli[l].map(preprocess_function, batched=True)
    
#print(tokenized_xnli_p['es']['train'][0])
#print(tokenized_xnli_h['es']['train'][0])
print(tokenized_xnli['es']['train'][0])
assert(False)
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
import evaluate

f1 = evaluate.load("f1")

import numpy as np


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return f1.compute(predictions=predictions, references=labels)


id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}


from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer


for l in languages:

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=2, id2label=id2label, label2id=label2id
    ).to('cuda')

    epochs = 1 
    lr = 2e-5 
    model_name = f'{base_model}_lr{lr}_epochs{epochs}_l='
    training_args = TrainingArguments(
        output_dir=f"~/generated_models/{model_name}",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True#,
        #push_to_hub=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_xnli["train"],
        eval_dataset=tokenized_xnli["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
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
