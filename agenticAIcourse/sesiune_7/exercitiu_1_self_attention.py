import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
 
torch.manual_seed(42)
 
 
class SelfAttention(nn.Module):
    """
    Un singur cap de self-attention.
 
    Intrare: secvență de vectori X ∈ R^(T × d_model)
    Ieșire:  secvență de vectori Y ∈ R^(T × d_v)
    """
    def __init__(self, d_model, d_k, d_v):
        super().__init__()
        self.d_k = d_k
 
        # Proiecțiile Q, K, V sunt matrice de parametri învățați
        self.W_Q = nn.Linear(d_model, d_k, bias=False)
        self.W_K = nn.Linear(d_model, d_k, bias=False)
        self.W_V = nn.Linear(d_model, d_v, bias=False)
 
    def forward(self, x, mask=None):
        # x: (batch, T, d_model)
        Q = self.W_Q(x)   # (batch, T, d_k)
        K = self.W_K(x)   # (batch, T, d_k)
        V = self.W_V(x)   # (batch, T, d_v)
 
        # Scoruri de compatibilitate: cât de mult se „potrivesc" Q și K
        scores = torch.bmm(Q, K.transpose(1, 2)) / (self.d_k ** 0.5)
        # shape: (batch, T, T) — pentru fiecare pereche (i, j)
 
        if mask is not None:
            # Setăm pozițiile mascate la -inf → softmax le dă 0
            scores = scores.masked_fill(mask == 0, float('-inf'))
 
        # Normalizăm scorurile în distribuție de probabilitate
        weights = F.softmax(scores, dim=-1)   # (batch, T, T)
 
        # Suma ponderată a valorilor
        output = torch.bmm(weights, V)   # (batch, T, d_v)
 
        return output, weights
 
 
class MultiHeadAttention(nn.Module):
    """
    Multi-head attention: h capete de atenție rulate în paralel.
    Fiecare cap poate captura tipuri diferite de relații.
    """
    def __init__(self, d_model, h):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
 
        # h capete, fiecare cu dimensiune d_k = d_model / h
        self.heads = nn.ModuleList([
            SelfAttention(d_model, self.d_k, self.d_k)
            for _ in range(h)
        ])
        # Proiecție finală care combină outputurile tuturor capetelor
        self.W_O = nn.Linear(d_model, d_model)
 
    def forward(self, x, mask=None):
        # Rulăm fiecare cap independent
        outputs = []
        all_weights = []
        for head in self.heads:
            out, w = head(x, mask)
            outputs.append(out)
            all_weights.append(w)
 
        # Concatenăm outputurile și proiectăm înapoi la d_model
        concat = torch.cat(outputs, dim=-1)   # (batch, T, d_model)
        return self.W_O(concat), all_weights
 
 
# --- Demonstrație ---
d_model = 64   # dimensiunea embedding-urilor
h = 4          # numărul de capete de atenție
T = 8          # lungimea secvenței
 
# Simulăm o secvență de token-uri random (în practică ar fi embeddings)
x = torch.randn(1, T, d_model)
 
mha = MultiHeadAttention(d_model, h)
output, weights = mha(x)
 
print(f"Input shape:  {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Număr de matrice de atenție: {len(weights)}")
print(f"Shape matrice atenție (per cap): {weights[0].shape}")
 
 
# --- Vizualizare matrice de atenție ---
cuvinte = ['The', 'cat', 'sat', 'on', 'the', 'mat', 'today', '.']
 
fig, axe = plt.subplots(1, h, figsize=(14, 3))
for i, ax in enumerate(axe):
    w = weights[i][0].detach().numpy()   # (T, T)
    im = ax.imshow(w, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(T))
    ax.set_yticks(range(T))
    ax.set_xticklabels(cuvinte, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(cuvinte, fontsize=8)
    ax.set_title(f'Cap {i+1}', fontsize=10)
 
plt.suptitle('Matricele de atenție pentru fiecare cap (valori random — fără antrenare)')
plt.tight_layout()
plt.show()
 
print("\nObservație: Capetele vor dezvolta tipare diferite după antrenare.")
print("Unele vor urmări sintaxa, altele semantica, altele referințele.")