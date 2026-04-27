import numpy as np
import matplotlib.pyplot as plt
 
# Cele 4 combinații posibile de intrări binare
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
 
# Etichetele corecte pentru fiecare operație logică
y_and = np.array([0, 0, 0, 1])   # AND: 1 doar când ambele sunt 1
y_or  = np.array([0, 1, 1, 1])   # OR:  1 când cel puțin unul e 1
y_xor = np.array([0, 1, 1, 0])   # XOR: 1 când exact unul e 1
 
 
class Perceptron:
    def __init__(self, rata_invatare=0.1, epoci=20):
        self.rata_invatare = rata_invatare
        self.epoci = epoci
 
    def antreneaza(self, X, y):
        self.w = np.zeros(X.shape[1])   # o pondere per feature, inițial zero
        self.b = 0.0
        self.erori_per_epoca = []
 
        for _ in range(self.epoci):
            erori = 0
            for xi, yi in zip(X, y):
                # Predicție: 1 dacă suma ponderată depășește pragul 0, altfel 0
                y_prezis = 1 if np.dot(self.w, xi) + self.b > 0 else 0
 
                # Regula de actualizare a perceptronului:
                # dacă predicția e corectă, eroare = 0 → ponderile nu se schimbă
                # dacă greșește, ponderile se mișcă în direcția corectă
                eroare = yi - y_prezis
                self.w += self.rata_invatare * eroare * xi
                self.b += self.rata_invatare * eroare
 
                erori += int(y_prezis != yi)
            self.erori_per_epoca.append(erori)
    def prezice(self, X):
        return np.array([1 if np.dot(self.w, xi) + self.b > 0 else 0 for xi in X])
 
 
def vizualizeaza(model, X, y, titlu):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
 
    # --- Stânga: frontiera de decizie ---
    ax1.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', s=300, zorder=3,
                edgecolors='black', linewidths=1.5)
 
    # Adaugă eticheta fiecărui punct
    for (x0, x1), eticheta in zip(X, y):
        ax1.text(x0 + 0.05, x1 + 0.05, str(eticheta), fontsize=12)
 
    # Desenează frontiera de decizie: w[0]*x + w[1]*y + b = 0
    # => y = -(w[0]*x + b) / w[1]
    if model.w[1] != 0:
        x_linie = np.linspace(-0.5, 1.5, 100)
        y_linie = -(model.w[0] * x_linie + model.b) / model.w[1]
        ax1.plot(x_linie, y_linie, 'k--', linewidth=2, label='frontieră de decizie')
        ax1.legend()
    else:
        ax1.text(0.5, 0.5, 'nicio frontieră găsită', transform=ax1.transAxes,
                 ha='center', color='gray')
 
    ax1.set_xlim(-0.5, 1.5)
    ax1.set_ylim(-0.5, 1.5)
    ax1.set_title(titlu)
    ax1.grid(True)

# --- Dreapta: numărul de erori per epocă ---
    ax2.plot(model.erori_per_epoca, marker='o', color='steelblue')
    ax2.set_xlabel('Epocă')
    ax2.set_ylabel('Erori de clasificare')
    ax2.set_title('Convergență')
    ax2.set_ylim(-0.2, 4.2)
    ax2.grid(True)
 
    plt.tight_layout()
    plt.show()
 
    acuratete = np.mean(model.prezice(X) == y) * 100
    print(f"{titlu}: acuratețe finală = {acuratete:.0f}%\n")
 
# --- Rulăm experimentele ---
for titlu, y in [('AND', y_and), ('OR', y_or), ('XOR', y_xor)]:
    p = Perceptron(rata_invatare=0.9, epoci=5)
    p.antreneaza(X, y)
    vizualizeaza(p, X, y, titlu)