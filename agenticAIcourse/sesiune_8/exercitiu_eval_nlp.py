import nltk
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
 
# Descarcă resursele necesare NLTK
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
 
smoother = SmoothingFunction().method1
 
 
def tokenizeaza_simplu(text):
    """Tokenizare minimală la nivel de cuvânt."""
    return text.lower().split()
 
 
def calculeaza_bleu(referinta, generat):
    ref_tok = [tokenizeaza_simplu(referinta)]
    gen_tok = tokenizeaza_simplu(generat)
    return sentence_bleu(ref_tok, gen_tok, smoothing_function=smoother)
 
 
def calculeaza_rouge(referinta, generat):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    scoruri = scorer.score(referinta, generat)
    return {k: round(v.fmeasure, 3) for k, v in scoruri.items()}
 
 
# --- Experiment 1: Cazuri ilustrative ---
print("=" * 60)
print("Experiment 1: Cazuri Ilustrative")
print("=" * 60 + "\n")
 
referinta = "The cat sat on the mat near the window."
 
cazuri = [
    ("Identic cu referința",
     "The cat sat on the mat near the window."),
 
    ("Sinonim — același sens, cuvinte diferite",
     "A feline rested on the rug beside the glass."),
 
    ("Parțial corect — lipsesc informații",
     "The cat sat on the mat."),
 
    ("Ordine diferită — același conținut",
     "Near the window, on the mat, sat the cat."),
 
    ("Repetare de n-grame — fără sens",
     "The cat the cat the cat sat sat sat."),
 
    ("Complet irelevant",
     "The stock market fell sharply yesterday afternoon."),
]
 
for titlu, generat in cazuri:
    bleu = calculeaza_bleu(referinta, generat)
    rouge = calculeaza_rouge(referinta, generat)
    print(f"--- {titlu} ---")
    print(f"  Generat: \"{generat}\"")
    print(f"  BLEU:    {bleu:.3f}")
    print(f"  ROUGE-1: {rouge['rouge1']}  ROUGE-2: {rouge['rouge2']}  ROUGE-L: {rouge['rougeL']}")
    print()
 
 
# --- Experiment 2: Rezumare automată ---
print("=" * 60)
print("Experiment 2: Evaluare Rezumare")
print("=" * 60 + "\n")
 
doc_original = """
Cercetătorii de la Universitatea Stanford au publicat un studiu care arată că
exercițiul fizic regulat îmbunătățește semnificativ funcțiile cognitive.
Studiul a urmărit 500 de participanți timp de doi ani și a constatat că
persoanele care fac cel puțin 30 de minute de exerciții pe zi au o memorie
cu 20% mai bună față de persoanele sedentare. Autorii recomandă activitate
aerobică de intensitate moderată pentru beneficii optime.
"""
 
referinta_rez = "Exercițiul fizic de 30 de minute pe zi îmbunătățește memoria cu 20%, conform unui studiu Stanford pe 500 de participanți."
 
rezumate_candidate = [
    ("Rezumat bun",
     "Un studiu Stanford arată că 30 de minute de exerciții zilnice cresc memoria cu 20%."),
 
    ("Rezumat parțial",
     "Exercițiul fizic este benefic pentru sănătate."),
 
    ("Rezumat cu detalii greșite",
     "Studiul arată că o oră de exerciții pe zi îmbunătățește memoria cu 50%."),
 
    ("Copie directă din document",
     "Cercetătorii de la Universitatea Stanford au publicat un studiu care arată că exercițiul fizic regulat îmbunătățește semnificativ funcțiile cognitive."),

    ("Output care bifeaza metricile, dar nu e rezumat",
     "Studiul arată că o oră de exerciții pe zi îmbunătățește publica sanatatea cu 20%"),
     #"Cercetătorii Stanford fac exercițiu fizic regulat si publica studiu care îmbunătățește semnificativ funcțiile cognitive."),
]
 
scorer_rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
 
for titlu, rezumat in rezumate_candidate:
    bleu = calculeaza_bleu(referinta_rez, rezumat)
    r = scorer_rouge.score(referinta_rez, rezumat)
    print(f"--- {titlu} ---")
    print(f"  Rezumat: \"{rezumat}\"")
    print(f"  BLEU: {bleu:.3f} | R-1: {r['rouge1'].fmeasure:.3f} | R-2: {r['rouge2'].fmeasure:.3f} | R-L: {r['rougeL'].fmeasure:.3f}")
    print()
 
print("Observație: Copia directă din document poate obține scoruri ridicate")
print("chiar dacă nu este un rezumat util. Metricile nu pot evalua compresia.")
 
# Întrebare de reflecție:
# De ce textul sinonim obține BLEU scăzut deși sensul este identic?
# Ce tip de evaluare ar fi mai potrivit pentru a surprinde calitatea reală?