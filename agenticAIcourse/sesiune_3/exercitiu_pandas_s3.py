import pandas as pd
import json 

df = pd.read_json(
    "https://raw.githubusercontent.com/rishabhmisra/News-Headlines-Dataset-For-Sarcasm-Detection/master/Sarcasm_Headlines_Dataset.json",
    lines=True
)
print(df.head())
print("df shape", df.shape)

def filtrare(f):
    return f[f['headline'].str.contains("scientist", case=False)]

df2 = filtrare(df)
print("Subsetul care contine scientist:", df2)
print("df2 shape", df2.shape)

df2["numar_cuv"] = df2["headline"].str.split().str.len()
print("df2 dimensiune", df.ndim)
print("df2", df2["numar_cuv"])
print("Noul data frame :")
print(df2)

with open("corpus.jsonl", "w", encoding="utf-8") as f:
    for record in df2.to_dict(orient="records"):
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        