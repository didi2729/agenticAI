class Agent:
    def __init__(self, nume: str, model :str, token_budget: int):
        self.nume = nume
        self.model = model
        self._token_budget = token_budget
        self._tokeni_folositi = 0
        self._tokeni_ramasi = 0
 
    @property
    def tokeni_ramasi(self) -> int:
        return self._token_budget - self._tokeni_folositi
 
    @property
    def epuizat(self) -> bool:
        return self._tokeni_folositi >= self._token_budget
 
    def consuma(self, n: int) -> None:
        if n > self.tokeni_ramasi:
            raise ValueError(f"Insuficienți tokeni: {self.tokeni_ramasi} disponibili")
        self._tokeni_folositi += n

    def trimite(self, prompt: str, tokens:int):
        if tokens > self.tokeni_ramasi:
            raise ValueError(f"Insuficienti tokeni")
        self._token_budget -= tokens

    def __repr__(self) -> str:
        return f"Agent(nume={self.nume!r}, model={self.model!r}, tokeni_ramasi={self.tokeni_ramasi})"

agent = Agent("Researcher","claude-sonnet-4", 1000)
agent.consuma(400)
agent.trimite("Trimite", 20)
print(agent.tokeni_ramasi)  # 580
print(agent.epuizat) 
print(repr(agent))
agent.trimite("", 10)