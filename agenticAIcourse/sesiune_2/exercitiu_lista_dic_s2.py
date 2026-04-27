from functools import reduce

loguri = [
    {"nivel": "INFO",  "durata_ms": 230, "mesaj": "Succes"},
    {"nivel": "ERROR", "durata_ms": 150, "mesaj": "N/A"},
    {"nivel": "ERROR", "durata_ms": 150, "mesaj": "Eroare"},
    {"nivel": "INFO",  "durata_ms": 300, "mesaj": "E bine"},
    {"nivel": "ERROR", "durata_ms": 700, "mesaj": "Dureaza mult"},
    {"nivel": "ERROR", "durata_ms": 120, "mesaj": "Nu e bine"},
]

# erorile cu durata > 200
erori_lente = list(filter(lambda log: log["nivel"] == "ERROR" and log["durata_ms"] > 200, loguri))

# durata medie
durate = list(map(lambda log: log["durata_ms"], erori_lente))
durata_medie = reduce(lambda acc, d: acc + d, durate) / len(durate) if durate else 0

#mesajele concatenate, separate prin " | "
mesaje_concatenate = reduce(lambda acc, msg: acc + " | " + msg, map(lambda log: log["mesaj"], erori_lente))

print("Erori cu durata_ms > 200:")
for e in erori_lente:
    print(f"  {e}")

print(f"\ndurata medie: {durata_medie:.1f} ms")
print(f"\nmesaje: {mesaje_concatenate}")
