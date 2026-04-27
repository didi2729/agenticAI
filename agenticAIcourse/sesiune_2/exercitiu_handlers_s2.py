from typing import TypedDict, NotRequired, Protocol


class Mesaj(TypedDict):
    role: str
    content: str
    tokens: NotRequired[int]


class Handler(Protocol):
    def __call__(self, mesaj: Mesaj) -> Mesaj: ...


def ruleaza(handlers: list[Handler], mesaj: Mesaj) -> Mesaj:
    for handler in handlers:
        mesaj = handler(mesaj)
    return mesaj


# Handler_1 completează câmpul `tokens` cu lungimea conținutului
class TokenCounter:
    def __call__(self, mesaj: Mesaj) -> Mesaj:
        return Mesaj(
            role=mesaj["role"],
            content=mesaj["content"],
            tokens=len(mesaj["content"]),
        )


# Handler_2 'role' la lowercase
class NormalizeRole:
    def __call__(self, mesaj: Mesaj) -> Mesaj:
        mesaj_nou = Mesaj(
            role=mesaj["role"].lower(),
            content=mesaj["content"],
        )
        if "tokens" in mesaj:
            mesaj_nou["tokens"] = mesaj["tokens"]
        return mesaj_nou


if __name__ == "__main__":
    mesaj_initial: Mesaj = {"role": "User", "content": "Salut, lume!"}
    handlers: list[Handler] = [NormalizeRole(), TokenCounter()]
    rezultat = ruleaza(handlers, mesaj_initial)
    print(rezultat)
