from contextlib import contextmanager
import time
 
@contextmanager
def timer(nume: str):
    """Măsoară timpul de execuție al unui bloc de cod"""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"[{nume}] Timp: {elapsed:.3f}s")
 
with timer("apel LLM"):
    time.sleep(0.1)  # simulăm un apel API
# [apel LLM] Timp: 0.100s

@contextmanager
def sesiune_llm(model: str):
    print(f"[{model}] sesiune deschisă")
    start = time.perf_counter()
    sesiune = {"model": model, "apeluri": 0}
    try:
        yield sesiune
    finally:
        elapsed = time.perf_counter() - start
        print(f"[{model}] închis după {sesiune['apeluri']} apeluri, {elapsed:.3f}s")


with sesiune_llm("claude") as s:
    time.sleep(0.05)
    s["apeluri"] += 1
    time.sleep(0.05)
    s["apeluri"] += 1
# [gpt-4o] sesiune deschisă
# [gpt-4o] închis după 2 apeluri, 0.100s