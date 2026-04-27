import json


def transforma(intrare: str, min_cuvinte: int) -> dict:
    total = 0
    pastrate = 0
    dic = {"total":0, "pastrate":0, "filtrate":0}

    with open(intrare, "r", encoding="utf-8") as f:
        for linie in f:
            linie = linie.strip()
            if not linie:
                continue

            doc = json.loads(linie)
            total += 1
            dic["total"] = total

            headline = doc.get("headline", "")
            if len(headline.split()) >= min_cuvinte:
                pastrate += 1
                dic["pastrate"] = pastrate
                ##print("headline pastrat",headline)
    dic["filtrate"] = total - pastrate
    return dic

stats = transforma("corpus.jsonl", min_cuvinte=19)
print(stats)
