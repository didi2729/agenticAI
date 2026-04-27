# List comprehension
patrate = [x ** 2 for x in range(10)]
print (f"patrate", patrate)

# Cu filtrare
pare = [x for x in range(20) if x % 2 == 0]
print("pare", pare)

# Dict comprehension
lungimi = {cuvant: len(cuvant) for cuvant in ["python", "AI", "agent"]}
print("lungimi", lungimi)
 
# Set comprehension
unice = {x % 5 for x in range(20)}
print ("unice", unice)

#generator
def numere_pare(n):
    for i in range(n):
        if i % 2 == 0:
            yield i
 
for numar in numere_pare(10):
    print(numar)  # 0, 2, 4, 6, 8

# Generator expression — calculează câte unul, la cerere
patrate_gen = (x ** 2 for x in range(1_000_000))   # aproape zero memorie
for i in patrate_gen:
    print ("*", i)

