#DATASET LOADING
from datasets import load_dataset
languages = ['ar', 'bg', 'de', 'el', 'en', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh']
languages = ['es']
#languages.append['all_languages'] #I'll think what to do with this one
xnli = {}
for l in languages:
    xnli[l] = load_dataset("xnli",l)

#TOKENIZING
base_model = "bert-base-multilingual-cased"
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)
def preprocess_function(examples):
    return tokenizer(examples["premise"], examples["hypothesis"], truncation=True)
tokenized_xnli = {}
for l in languages:
    tokenized_xnli[l] = xnli[l].map(preprocess_function, batched=True)
#COLLATION    
from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
#EVALUATION
import evaluate
f1 = evaluate.load("f1")
import numpy as np
def compute_metrics(eval_pred):
    print(eval_pred)
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return f1.compute(predictions=predictions, references=labels)


id2label = {0: "Entailment", 1: "Neutral",2:'Contradiction'}
label2id = {"Entailment": 0, "Neutral": 1, 'Contradiction':2}


from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

#labels = [i[f"label"] for i in tokenized_xnli[l]["train"]]
#print(labels)
#print(tokenized_xnli[l]["train"][0])
#assert(False)

epochs = 1 
lr = 2e-5 
import torch,os

for l in languages:
    print(f'training model for {l}')
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=len(id2label), id2label=id2label, label2id=label2id
    ).to('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = f'{base_model}_lr{lr}_epochs{epochs}_l={l}'

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
        train_dataset=tokenized_xnli[l]["train"],
        eval_dataset=tokenized_xnli[l]["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    def save_model(model, filename):
        cur_path = os.path.split(os.path.realpath(__file__))[0]
        project_path = cur_path
        datafile = os.path.join(os.path.expanduser('~'),'generated_models', filename)
        trainer.save_model(datafile)
        return True



    save_model(model, model_name)
