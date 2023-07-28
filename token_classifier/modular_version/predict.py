### PREDICTION ###
text = "The Golden State Warriors are an American professional basketball team based in San Francisco."

from transformers import pipeline

def load_model(filename):
    cur_path = os.path.split(os.path.realpath(__file__))[0]
    project_path = os.path.split(cur_path)[0]
    datafile = os.path.join(project_path,'generated_models', filename)
    return torch.load(datafile,map_location=torch.device('cpu'))

model = load_model('allcode')
classifier = pipeline("ner", model="my_awesome_wnut_model")
classifier(text)

