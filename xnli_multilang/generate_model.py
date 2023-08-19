#DATASET LOADING
from datasets import load_dataset
languages = ['ar', 'bg', 'de', 'el', 'en', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh']
languages = ['en']

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
    return f1.compute(predictions=predictions, references=labels,average='macro')


id2label = {0: "Entailment", 1: "Neutral",2:'Contradiction'}
label2id = {"Entailment": 0, "Neutral": 1, 'Contradiction':2}


from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer


epochs = 5
lrs = [1e-5,5e-6,2e-6]
import torch,os

for l in languages:
    for lr in lrs:
        print(f'training model for {l}')
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model, num_labels=len(id2label), id2label=id2label, label2id=label2id
        ).to('cuda' if torch.cuda.is_available() else 'cpu')
        model_name = f'{base_model}_lr{lr}_epochs{epochs}_l={l}'
        model_path = f"generated_models/{model_name}"

        training_args = TrainingArguments(
            output_dir=model_path,
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=epochs,
            weight_decay=0.01,
            metric_for_best_model='f1',
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True#,
            #push_to_hub=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_xnli[l]["train"],
            eval_dataset=tokenized_xnli[l]["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        print('------TRAINING FINISHED----------')
        cur_path = os.path.split(os.path.realpath(__file__))[0]
        datafile = os.path.join(cur_path, model_path)
        trainer.save_model(datafile)


        metrics_values = {'val_f1':[],'val_loss':[],'tra_loss':[]}
        for metrics in trainer.state.log_history:
            if 'eval_f1' in metrics:
                metrics_values['val_loss'].append(round(metrics['eval_loss'],3))
                metrics_values['val_f1'].append(round(metrics['eval_f1'],3))
            elif 'loss' in metrics :
                metrics_values['tra_loss'].append(round(metrics['loss'],3))

        def print_metrics():
            out = '\t'.join(['epoch'] + [str(i+1) for i in range(epochs)])
            for m in metrics_values:
                out += '\n' + '\t'.join([m]+[str(i) for i in metrics_values[m]])
            eval_res = trainer.evaluate(tokenized_xnli[l]["validation"])
            print(eval_res)
            out += f'\nBest F1 on evaluation is {round(eval_res["eval_f1"],3)}'
            test_res = trainer.evaluate(tokenized_xnli[l]["test"])
            print(test_res)
            out += f'\nBest F1 on testing is {round(test_res["eval_f1"],3)}'
            #out += trainer.evaluate(tokenized_xnli[l]["testing"])
            return out

        with open(datafile+'/metrics.csv','w') as f:
            f.write(print_metrics())

