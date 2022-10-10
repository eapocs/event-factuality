from transformers import AutoModel, AutoTokenizer
import os
import json
import tqdm
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def embed(tokens, i):
    assert isinstance(tokens[0], str) # batch size 1
    tokens = tokenizer(tokens, is_split_into_words=True, return_tensors="pt")
    output = model(**tokens).last_hidden_state[0]
    idx = [idx for idx, el in enumerate(tokens.word_ids()) if el == i]
    vector = output[idx].mean(dim=0)
    return vector

data = {
    file.split(".jsonl")[0]: [json.loads(l) for l in open(f"factuality_bert/glue_data/UDS_IH2/{file}")] for file in os.listdir("factuality_bert/glue_data/UDS_IH2/") if file.endswith(".jsonl")
}

for kind in data:
    for i, el in enumerate(data[kind]):
        data[kind][i]["tokens"] = el["text"].split()

model = AutoModel.from_pretrained("bert-base-cased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

with open("factuality_embed.jsonl", "w") as f:
    for kind in data:
        for i, elem in tqdm.tqdm(enumerate(data[kind])):
            for t in elem["targets"]:
                a = embed(elem["tokens"], t["span1"][0])

                f.write(json.dumps(
                    {   
                        "vector": [float(el) for el in a],
                        "word": t["span_text"],
                        "label": t["label"],
                        "split": kind,
                        "idx": i,
                    }
                )+"\n")


df = pd.DataFrame([json.loads(el) for el in open("factuality_embed.jsonl")])

train = df[df.split.isin(["train", "dev"])]
test = df[df.split.isin(["test"])]

X_train = np.vstack(train.vector)
y_train = train.label

X_test = np.vstack(test.vector)
y_test = test.label

reg = LinearRegression().fit(X_train, y_train)
preds = pd.DataFrame(data={"pred":reg.predict(X_test), "test": y_test})


## Пример

text = ["He", "spoke", "about", "this", "yesterday"]
index = 1 # spoke

print(reg.predict([embed(text, index).detach().numpy()])[0])

