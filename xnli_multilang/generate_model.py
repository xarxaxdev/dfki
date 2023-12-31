#DATASET LOADING
import evaluate,torch,os
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding,AutoTokenizer

languages = ['ar', 'bg', 'de', 'el', 'en', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh']
#languages = ['en']

xnli = {}
for l in languages:
    xnli[l] = load_dataset("xnli",l)

#TOKENIZING
base_model = "bert-base-multilingual-cased"


tokenizer = AutoTokenizer.from_pretrained(base_model)
def preprocess_function(examples):
    return tokenizer(examples["premise"], examples["hypothesis"], truncation=True)
tokenized_xnli = {}
for l in languages:
    tokenized_xnli[l] = xnli[l].map(preprocess_function, batched=True)

#COLLATION    

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
#EVALUATION
f1 = evaluate.load("f1")
def compute_metrics(eval_pred):
    print(eval_pred)
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return f1.compute(predictions=predictions, references=labels,average='macro')


id2label = {0: "Entailment", 1: "Neutral",2:'Contradiction'}
label2id = {"Entailment": 0, "Neutral": 1, 'Contradiction':2}




epochs = 10
lrs = [1e-6,2e-6,5e-6,1e-5,2e-5]
batch_sizes= [8,16,32 ]
lrs=[2e-5]#best params on English
batch_sizes=[32]#best params on English
skip_combinations = 0
#skip_combinations = 6

for l in languages:
    for lr in lrs:
        for bs in batch_sizes:

            if skip_combinations> 0:
                skip_combinations -= 1
                continue
                
            model = AutoModelForSequenceClassification.from_pretrained(
                base_model, num_labels=len(id2label), id2label=id2label, label2id=label2id
            ).to('cuda' if torch.cuda.is_available() else 'cpu')
            model_name = f'{base_model}_lr{lr}_bs{bs}_epochs{epochs}_l={l}'
            model_path = f"generated_models/{model_name}"
            print(f'--------Training model {model_name}--------')
            training_args = TrainingArguments(
                output_dir=model_path,
                learning_rate=lr,#2e-5,
                gradient_accumulation_steps=4 if bs== 32 else 1,
                per_device_train_batch_size=8 if bs== 32 else bs,
                per_device_eval_batch_size= 8 if bs== 32 else bs,
                num_train_epochs=epochs,
                weight_decay=0.01,
                evaluation_strategy="epoch",
                logging_strategy="epoch",
                #save_strategy="epoch",
                save_strategy="no",
                metric_for_best_model='f1',
                load_best_model_at_end=False#,
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
            if not os.path.exists(datafile):
                os.mkdir(datafile)

            #trainer.save_model(datafile)


            metrics_values = {'val_f1':[],'val_loss':[],'tra_loss':[]}
            for metrics in trainer.state.log_history:
                if 'eval_f1' in metrics:
                    metrics_values['val_loss'].append(round(metrics['eval_loss'],3))
                    metrics_values['val_f1'].append(round(metrics['eval_f1'],3))
                elif 'loss' in metrics :
                    metrics_values['tra_loss'].append(round(metrics['loss'],3))

            def print_metrics():
                out = model_name + '\n'
                out += '\t'.join(['epoch'] + [str(i+1) for i in range(epochs)])
                for m in metrics_values:
                    out += '\n' + '\t'.join([m]+[str(i) for i in metrics_values[m]])
                eval_res = max(metrics_values['val_f1'])
                print(eval_res)
                out += f'\nBest F1 on evaluation is {eval_res}'
                #test_res = trainer.evaluate(tokenized_xnli[l]["test"])
                #print(test_res)
                #out += f'\nBest F1 on testing is {round(test_res["eval_f1"],3)}'
                return out

            with open(datafile+'/metrics.csv','w') as f:
                f.write(print_metrics())

