from datasets import load_dataset
wnut = load_dataset("wnut_17")
label_list = wnut["train"].features[f"ner_tags"].feature.names

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")



from preprocessing import tokenize_and_align_labels,id2label,label2id
from evaluation import compute_metrics,initialize_labeling

initialize_labeling(wnut["train"])
tokenized_wnut = wnut.map(tokenize_and_align_labels, batched=True)


from transformers import DataCollatorForTokenClassification

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)


import evaluate

seqeval = evaluate.load("seqeval")







from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

model = AutoModelForTokenClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=13, id2label=id2label, label2id=label2id
)

training_args = TrainingArguments(
    output_dir="my_awesome_wnut_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch"#,
    #load_best_model_at_end=True,
    #push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_wnut["train"],
    eval_dataset=tokenized_wnut["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

