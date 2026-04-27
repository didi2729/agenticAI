import chromadb
from sentence_transformers import SentenceTransformer
 
print("Se încarcă modelul de embeddings...")
encoder = SentenceTransformer('all-MiniLM-L6-v2')
print("Model încărcat.\n")
 
# --- Corpus de documente cu metadata ---
documente = [
    {"id": "doc_001", "text": "Transformers use self-attention to process sequences in parallel.", "categorie": "nlp", "an": 2017},
    {"id": "doc_002", "text": "BERT is pre-trained on masked language modeling.", "categorie": "nlp", "an": 2018},
    {"id": "doc_003", "text": "GPT generates text autoregressively, predicting the next token.", "categorie": "nlp", "an": 2018},
    {"id": "doc_004", "text": "Word2Vec learns word embeddings from co-occurrence statistics.", "categorie": "embeddings", "an": 2013},
    {"id": "doc_005", "text": "GloVe combines global co-occurrence counts with local context.", "categorie": "embeddings", "an": 2014},
    {"id": "doc_006", "text": "SBERT produces sentence embeddings suitable for semantic search.", "categorie": "embeddings", "an": 2019},
    {"id": "doc_007", "text": "Convolutional networks detect local patterns in images.", "categorie": "vision", "an": 2012},
    {"id": "doc_008", "text": "ResNet introduced residual connections to train very deep networks.", "categorie": "vision", "an": 2016},
    {"id": "doc_009", "text": "GANs train a generator and discriminator in opposition.", "categorie": "generative", "an": 2014},
    {"id": "doc_010", "text": "VAEs encode inputs into a probabilistic latent space.", "categorie": "generative", "an": 2013},
    {"id": "doc_011", "text": "FAISS enables efficient similarity search over millions of vectors.", "categorie": "retrieval", "an": 2017},
    {"id": "doc_012", "text": "Vector databases store embeddings and support approximate nearest neighbor search.", "categorie": "retrieval", "an": 2020},
    {"id": "doc_013", "text": "RAG combines retrieval with language model generation.", "categorie": "retrieval", "an": 2020},
    {"id": "doc_014", "text": "Reinforcement learning from human feedback aligns LLMs with preferences.", "categorie": "nlp", "an": 2022},
    {"id": "doc_015", "text": "Scaling laws show that model performance improves predictably with compute.", "categorie": "nlp", "an": 2020},
]
 
texte = [d["text"] for d in documente]
 
# --- Construim colecția Chroma ---
# EphemeralClient = stocare in-memory, fără persistență pe disc
client = chromadb.EphemeralClient()
colectie = client.create_collection(
    name="documente_ml",
    metadata={"hnsw:space": "cosine"}   # metrica de distanță folosită intern
)
 
# Calculăm embeddings și le adăugăm împreună cu documentele și metadata
print(f"Calculez embeddings pentru {len(documente)} documente...")
embeddings = encoder.encode(texte, normalize_embeddings=True)
 
colectie.add(
    ids=[d["id"] for d in documente],
    embeddings=embeddings.tolist(),
    documents=texte,
    metadatas=[{"categorie": d["categorie"], "an": d["an"]} for d in documente],
)
print(f"Colecție creată: {colectie.count()} documente indexate.\n")
 
 
# --- Funcție helper pentru căutare ---
def cauta(query, k=3, where=None, eticheta=""):
    """
    Caută semantic în colecție.
    `where` permite filtrare după metadata: {"categorie": "nlp"} sau {"an": {"$gte": 2019}}
    """
    emb_q = encoder.encode([query], normalize_embeddings=True).tolist()
    rez = colectie.query(
        query_embeddings=emb_q,
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"]
    )
    docs      = rez["documents"][0]
    metadatas = rez["metadatas"][0]
    distante  = rez["distances"][0]
 
    titlu = f"Interogare: '{query}'"
    if eticheta:
        titlu += f"  [{eticheta}]"
    print(titlu)
    for doc, meta, dist in zip(docs, metadatas, distante):
        sim = 1 - dist   # Chroma returnează distanță cosinus: 0=identic, 2=opus
        print(f"  [{sim:.3f}] [{meta['categorie']}/{meta['an']}] {doc}")
    print()
    return docs   # returnăm textele pentru a le folosi ca context RAG
 
 
# ==========================================
# 1. Căutare semantică simplă
# ==========================================
print("=" * 55)
print("1. Căutare semantică fără filtre")
print("=" * 55 + "\n")
 
cauta("How do attention mechanisms work?")
cauta("What are sentence embeddings used for?")
cauta("How to store and search vector representations efficiently?")
 
 
# ==========================================
# 2. Căutare cu filtrare după metadata
# ==========================================
print("=" * 55)
print("2. Căutare cu filtrare după metadata")
print("=" * 55 + "\n")
 
# Filtrare după categorie
cauta("vector representations of words",
      where={"categorie": "embeddings"},
      eticheta="filtre: categorie=embeddings")
 
# Filtrare după an (operatori: $eq, $ne, $gt, $gte, $lt, $lte, $in)
cauta("neural language models",
      where={"an": {"$gte": 2019}},
      eticheta="filtre: an >= 2019")
 
# Filtrare combinată cu $and
cauta("generative models",
      where={"$and": [{"categorie": "generative"}, {"an": {"$lte": 2014}}]},
      eticheta="filtre: categorie=generative AND an <= 2014")
 
 
# ==========================================
# 3. Actualizare și ștergere documente
# ==========================================
print("=" * 55)
print("3. Operații CRUD — actualizare și ștergere")
print("=" * 55 + "\n")
 
# Adăugăm un document nou
doc_nou = "Mixture of Experts scales LLMs by activating only a subset of parameters per token."
emb_nou = encoder.encode([doc_nou], normalize_embeddings=True).tolist()
colectie.add(
    ids=["doc_016"],
    embeddings=emb_nou,
    documents=[doc_nou],
    metadatas=[{"categorie": "nlp", "an": 2024}]
)
print(f"Document adăugat. Total: {colectie.count()} documente.")
 
# Actualizăm metadata unui document existent
colectie.update(
    ids=["doc_001"],
    metadatas=[{"categorie": "nlp", "an": 2017, "citari": 50000}]
)
print("Metadata doc_001 actualizată (am adăugat câmpul 'citari').")
 
# Verificăm că apare în căutare
cauta("sparse activation mixture of experts", k=2,
      eticheta="după adăugare doc_016")
 
# Ștergem documentul adăugat
colectie.delete(ids=["doc_016"])
print(f"Document șters. Total: {colectie.count()} documente.\n")