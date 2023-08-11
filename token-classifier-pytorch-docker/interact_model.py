import os,torch
import argparse

def params():
    parser = argparse.ArgumentParser(
        description='Parameters for the model'
    )

    parser.add_argument(
        "--modelname", dest="modelname",
        help="modelname to load"
    )

    return parser.parse_args()

args = params()
modelname = args.modelname if args.modelname else 'distilbert-base-uncased_lr2e-05_epochs1' 
datafile = os.path.join(os.path.expanduser('~'),'generated_models', modelname)
model_path = f"generated_models/{modelname}"


#need a tokenizer 
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')


from transformers import pipeline
classifier = pipeline("ner", model=model_path)

print('=======BEGIN TYPING=======')
while True:
    #read input
    s = input()
    #print output after applying to model
    print(f"you said:{s}" )
    #token_s = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    print(classifier(s))
