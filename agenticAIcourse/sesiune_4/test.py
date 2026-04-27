from pydantic import BaseModel
from typing import Optional
 
class Alegere(BaseModel):
    index: int
    mesaj: dict
    motiv_oprire: Optional[str] = None
 
class RaspunsChat(BaseModel):
    id: str
    model: str
    alegeri: list[Alegere]
    tokens_folositi: Optional[int] = None
 
# Deserializare din dict (răspuns JSON de la API)
date_json = {
    "id": "chatcmpl-abc123",
    "model": "gpt-4o-mini",
    "alegeri": [
        {"index": 0, "mesaj": {"role": "assistant", "content": "Bună ziua!"}, "motiv_oprire": "stop"}
    ],
    "tokens_folositi": 42,
}
 
raspuns = RaspunsChat.model_validate(date_json)
print(raspuns.id)                          # chatcmpl-abc123
print(raspuns.alegeri[0].mesaj["content"]) # Bună ziua!
 
# Serializare înapoi la dict/JSON
print("dict python", raspuns.model_dump())                # dict Python
print("JSON", raspuns.model_dump_json())    