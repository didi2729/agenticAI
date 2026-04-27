NOTA_MIN = 1.0
NOTA_MAX = 10

studenti = {}
 
def adaugare_nota(nume, nota):
    """Ädauga o nota unui student"""
    
    if (NOTA_MIN <= nota <= NOTA_MAX):
        studenti.setdefault(nume, []).append(nota)
    else:
        raise ValueError("Nota trebuie sa fie intre", NOTA_MIN , "si ", NOTA_MAX);
 
def medie(nume):
    """Returneaza media studentului"""
    if nume not in studenti or len(studenti[nume]) == 0:
        return 0
    return sum(studenti[nume]) / len(studenti[nume])
 
def listare():
    """Afiseaza toti studentii cu media si notele lor"""
    if not studenti:
        print("Nu exista studenti inregistrati")
    for k in studenti:
        print(k, round(medie(k), 2), studenti[k])

def stergereStudent(n):
    """Sterge student"""
    if n in studenti:
        del studenti[n]
    else:
        print("Student inexistent")
 
def peste_prag(prag:float):
    """Lista studenti cu media mai mare sau egala cu prag dat"""
    lista = []
    for k in studenti:
        if medie(k) >= prag:
            lista.append(k)
    return lista

def topStudent():
    """Top student"""
    if not studenti:
        raise ValueError("Nu există studenți înregistrați.")
    return max(studenti, key=medie)

   
def stats():
    """Statistici"""
    listaMedieStudenti = []
    for k in studenti:
        listaMedieStudenti.append(round(medie(k), 2))

    print("media generala este :", sum(listaMedieStudenti)/len(listaMedieStudenti))
    print("nr studenti", len(studenti))
    print("top student", topStudent())
    

 
def run():
    print("comenzi: add <nume> <nota>, list, top <prag>, remove <nume>, stats, quit")
    while True:
        i = input("> ").strip()
        if i == "quit":
            break
        else:
            p = i.split()
            if p[0] == "add":
                adaugare_nota(p[1], float(p[2]))
            elif p[0] == "list":
                listare()
            elif p[0] == "top":
                print(peste_prag(float(p[1])))
            elif p[0] == "remove":
                stergereStudent(p[1])
            elif p[0] == "stats":
                stats()
 
run()
