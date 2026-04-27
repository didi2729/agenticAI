import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import defaultdict
import random
 
random.seed(42)
np.random.seed(42)
 
 
# --- Corpus sintetic cu structură semantică ---
# Grupăm cuvintele în contexte similare pentru ca Word2Vec să le asocieze
corpus_raw = [
    # Animale de companie
    "pisica mananca hrana pisica doarme pisica joaca",
    "cainele mananca hrana cainele doarme cainele alearga",
    "iepurele mananca morcovi iepurele doarme iepurele sare",
    "pisica iepurele cainele sunt animale de companie",
    "veterinarul ingrijeste pisica cainele iepurele",
 
    # Orașe și țări
    "parisul este capitala frantei franta este tara",
    "roma este capitala italiei italia este tara",
    "berlinul este capitala germaniei germania este tara",
    "bucurestiul este capitala romaniei romania este tara",
    "londra este capitala angliei anglia este tara",
 
    # Regalitate
    "regele conduce tara regina conduce tara",
    "printul este fiul regelui printesa este fiica reginei",
    "regele este barbat regina este femeie",
    "printul este barbat printesa este femeie",
 
    # Profesii
    "doctorul vindeca pacientii doctorul lucreaza in spital",
    "profesorul preda lectii profesorul lucreaza in scoala",
    "inginerul construieste poduri inginerul lucreaza in firma",
    "doctorul profesorul inginerul sunt profesionisti",
]
 
# Tokenizare simplă
propozitii = [p.split() for p in corpus_raw]
 
 
# --- Implementare simplificată Skip-gram ---
# (în practică se folosește gensim sau PyTorch)
class Word2VecSimplu:
    def __init__(self, dim=50, fereastra=2, lr=0.025, epoci=900):
        self.dim = dim
        self.fereastra = fereastra
        self.lr = lr
        self.epoci = epoci
 
    def antreneaza(self, propozitii):
        # Construiește vocabularul
        cuvinte = [c for p in propozitii for c in p]
        self.vocab = sorted(set(cuvinte))
        self.cuv2id = {c: i for i, c in enumerate(self.vocab)}
        V = len(self.vocab)
 
        # Inițializare aleatoare a embeddings-urilor
        self.W_in  = np.random.randn(V, self.dim) * 0.01   # embeddings intrare
        self.W_out = np.random.randn(self.dim, V) * 0.01   # embeddings ieșire
 
        for epoca in range(self.epoci):
            lr = self.lr * (1 - epoca / self.epoci) + 0.001  # lr descrescător
            pierdere_totala = 0
 
            for prop in propozitii:
                ids = [self.cuv2id[c] for c in prop if c in self.cuv2id]
                for i, id_central in enumerate(ids):
                    # Context: cuvintele din fereastra din jur
                    start = max(0, i - self.fereastra)
                    stop  = min(len(ids), i + self.fereastra + 1)
                    context = [ids[j] for j in range(start, stop) if j != i]
 
                    # Forward: embedding cuvânt central
                    h = self.W_in[id_central]   # (dim,)
 
                    for id_ctx in context:
                        # Scor pentru fiecare cuvânt din vocabular
                        scoruri = h @ self.W_out   # (V,)
                        scoruri -= scoruri.max()   # stabilitate numerică
                        exp_s = np.exp(scoruri)
                        probs = exp_s / exp_s.sum()
 
                        pierdere_totala -= np.log(probs[id_ctx] + 1e-10)
 
                        # Backward: gradient simplu
                        dL = probs.copy()
                        dL[id_ctx] -= 1.0
 
                        # Actualizare ponderi
                        self.W_out -= lr * np.outer(h, dL)
                        self.W_in[id_central] -= lr * (self.W_out @ dL)
 
        # Embeddings finale = media W_in și W_out (practică comună)
        self.embeddings = (self.W_in + self.W_out.T) / 2
 
    def vector(self, cuv):
        if cuv not in self.cuv2id:
            return None
        return self.embeddings[self.cuv2id[cuv]]
 
    def similare(self, cuv, top_n=5):
        v = self.vector(cuv)
        if v is None:
            return []
        scoruri = {}
        for c in self.vocab:
            if c == cuv:
                continue
            u = self.vector(c)
            cos = np.dot(v, u) / (np.linalg.norm(v) * np.linalg.norm(u) + 1e-10)
            scoruri[c] = cos
        return sorted(scoruri.items(), key=lambda x: -x[1])[:top_n]
 
    def analogie(self, a, b, c):
        """a este față de b ca c față de ___?  ex: rege-bărbat+femeie≈regină"""
        va, vb, vc = self.vector(a), self.vector(b), self.vector(c)
        if any(v is None for v in [va, vb, vc]):
            return None
        tinta = vb - va + vc
        scoruri = {}
        pentru_exclus = {a, b, c}
        for cuv in self.vocab:
            if cuv in pentru_exclus:
                continue
            u = self.vector(cuv)
            cos = np.dot(tinta, u) / (np.linalg.norm(tinta) * np.linalg.norm(u) + 1e-10)
            scoruri[cuv] = cos
        return max(scoruri.items(), key=lambda x: x[1])
 
 
# Antrenare
print("Antrenare Word2Vec...")
model = Word2VecSimplu(dim=30, fereastra=10, epoci=800)
model.antreneaza(propozitii)
print(f"Vocabular: {len(model.vocab)} cuvinte\n")
 
# Cuvinte similare
for cuv in ['pisica', 'regele', 'franta']:
    print(f"Similare cu '{cuv}':")
    for c, s in model.similare(cuv, top_n=4):
        print(f"  {c:20s} cosine={s:.3f}")
    print()
 
# Analogii
print("Analogii vectoriale:")
analogii = [
    ('regele', 'barbat', 'femeie'),     # regele - bărbat + femeie ≈ ?
    ('parisul', 'franta', 'italia'),    # paris - franța + italia ≈ ?
    ('regele', 'franta', 'romania'),    # rege - franța + românia ≈ ?
]
for a, b, c in analogii:
    rez = model.analogie(a, b, c)
    if rez:
        print(f"  {a} - {b} + {c} ≈ {rez[0]} (score={rez[1]:.3f})")
 
 
# --- Vizualizare t-SNE ---
cuvinte_viz = ['pisica', 'cainele', 'iepurele',
               'regele', 'regina', 'printul', 'printesa',
               'parisul', 'franta', 'roma', 'italia', 'berlinul', 'germania',
               'doctorul', 'profesorul', 'inginerul']
cuvinte_viz = [c for c in cuvinte_viz if c in model.cuv2id]
 
vectori = np.array([model.vector(c) for c in cuvinte_viz])
coords = TSNE(n_components=2, perplexity=5, random_state=42).fit_transform(vectori)
 
culori = {'animal': 'tab:blue', 'regalitate': 'tab:orange',
          'oras': 'tab:green', 'profesie': 'tab:red'}
grupuri = {
    'pisica': 'animal', 'cainele': 'animal', 'iepurele': 'animal',
    'regele': 'regalitate', 'regina': 'regalitate',
    'printul': 'regalitate', 'printesa': 'regalitate',
    'parisul': 'oras', 'roma': 'oras', 'berlinul': 'oras',
    'franta': 'oras', 'italia': 'oras', 'germania': 'oras',
    'doctorul': 'profesie', 'profesorul': 'profesie', 'inginerul': 'profesie',
}
 
fig, ax = plt.subplots(figsize=(10, 7))
for i, cuv in enumerate(cuvinte_viz):
    grup = grupuri.get(cuv, 'altul')
    culoare = culori.get(grup, 'gray')
    ax.scatter(coords[i, 0], coords[i, 1], c=culoare, s=100, zorder=3)
    ax.annotate(cuv, (coords[i, 0] + 0.5, coords[i, 1] + 0.5), fontsize=9)
 
from matplotlib.patches import Patch
legenda = [Patch(color=c, label=g) for g, c in culori.items()]
ax.legend(handles=legenda)
ax.set_title('Vizualizare t-SNE a embeddings-urilor Word2Vec')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
 
# Întrebare de reflecție:
# De ce cuvintele din același grup semantic se grupează în spațiu?
# Ce s-ar întâmpla cu mai mult text de antrenare?