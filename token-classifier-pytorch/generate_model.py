import argparse
import os,torch

def params():
    parser = argparse.ArgumentParser(
        description='Parameters for the model'
    )

    parser.add_argument(
        "--pretrained_model", dest="pretrained_model",
        help="pretrained_model to use as base"
    )

    parser.add_argument(
        "--epochs", dest="epochs",
        help="number of epochs"
    )
    
    parser.add_argument(
        "--lr", dest="lr",
        help='learning rate value'
    )


    return parser.parse_args()

args = params()
#default parameters for little work
epochs = int(args.epochs) if args.epochs else 3
lr = float(args.lr) if args.lr else 2e-5 
pretrained_model = args.pretrained_model if args.pretrained_model else 'distilbert-base-uncased' 
model_name = f'{pretrained_model}_lr{lr}_epochs{epochs}'

#load dataset
from datasets import load_dataset
wnut = load_dataset("wnut_17")
#look at BIO notation
label_list = wnut["train"].features[f"ner_tags"].feature.names

#need a tokenizer 
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

#and token alignment since BERT works with 
#characters not with words 
def tokenize_and_align_labels(examples):
    #tokenize
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


#apply to whole dataset
tokenized_wnut = wnut.map(tokenize_and_align_labels, batched=True)

#Pad data to BERT model input size
from transformers import DataCollatorForTokenClassification
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

### EVALUATION METRICS
import evaluate
seqeval = evaluate.load("seqeval")

import numpy as np

labels = [label_list[i] for i in wnut["train"][0][f"ner_tags"]]


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

#generate label2id and id2label for our trainer
id2label = {}
label2id = {}
i=0
for l in label_list:
    id2label[i] = l
    label2id[l] = i
    i+=1

#load pretrained model
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
model = AutoModelForTokenClassification.from_pretrained(
    pretrained_model, num_labels=len(id2label), id2label=id2label, label2id=label2id
).to('cuda' if torch.cuda.is_available() else 'cpu')


model_path = f"generated_models/{model_name}"

training_args = TrainingArguments(
    output_dir=model_path,
    learning_rate=lr,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=epochs,#2, # 2 was the original value but runnig it on non-GPU
    weight_decay=0.01,
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True#,
    #push_to_hub=True, #this is for sharing it in hugginface so commented it
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_wnut["train"],
    eval_dataset=tokenized_wnut["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

train_result = trainer.train()

metrics_values = {'val_f1':[],'val_loss':[],'tra_loss':[]}
for metrics in trainer.state.log_history:
    if 'eval_f1' in metrics:
        metrics_values['val_loss'].append(round(metrics['eval_loss'],3))
        metrics_values['val_f1'].append(round(metrics['eval_f1'],3))
    elif 'loss' in metrics :
        metrics_values['tra_loss'].append(round(metrics['loss'],3))

def print_metrics():
    out = '\t'.join(['epoch'] + [str(i) for i in range(epochs)])
    for m in metrics_values:
        out += '\n' + '\t'.join([m]+[str(i) for i in metrics_values[m]])
    return out


cur_path = os.path.split(os.path.realpath(__file__))[0]
datafile = os.path.join(cur_path, model_path)
trainer.save_model(datafile)

with open(datafile+'/metrics.csv','w') as f:
    f.write(print_metrics())