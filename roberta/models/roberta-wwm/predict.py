import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification

model_path = "./models/roberta-sentiment"

tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

df = pd.read_csv("./data/predic.csv")


def predict(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        return torch.argmax(logits, dim=1).item()

df["pred"] = df["review"].apply(predict)

print(df)
df.to_csv("predict_result.csv", index=False)

from sklearn.metrics import classification_report

print(classification_report(df["label"], df["pred"], digits=4))