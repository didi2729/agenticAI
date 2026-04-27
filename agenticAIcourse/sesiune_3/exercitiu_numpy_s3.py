import numpy as np

matrice = np.random.rand(100, 384).astype(np.float32)
first_rows = matrice[:5]
first_columns = matrice[5:]

print("primele 5 randuri:", first_rows)
print("primele 5 coloane:", first_columns)


