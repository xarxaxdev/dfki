import os,torch
import argparse

from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer


def load_model(filename):
    cur_path = os.path.split(os.path.realpath(__file__))[0]
    project_path = cur_path#os.path.split(cur_path)[0]
    datafile = os.path.join(project_path, '~/generated_models', filename)
    #return torch.load(datafile,map_location=torch.device('cpu'))
    return AutoModelForTokenClassification.from_pretrained(datafile)






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
classifier = pipeline("sentiment-analysis", model=datafile)

print('=======BEGIN TYPING=======')
while True:
    #read input
    s = input()
    #print output after applying to model
    print(f"you said:{s}" )
    #token_s = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    print(classifier(s))
