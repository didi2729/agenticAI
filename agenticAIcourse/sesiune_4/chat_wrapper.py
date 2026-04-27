import requests
import sys
sys.path.insert(0, "/Users/didi/AgenticAI")
import time
from pydantic import BaseModel, Field, ValidationError as PydanticValidationError
from typing import Optional
import os
from dotenv import load_dotenv
import logging
import agenticAI.api_error as api_error


logger = logging.getLogger(__name__)

class Message(BaseModel):
    role: str
    content: Optional[str] = None
 
class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)  # ge=min, le=max
    max_tokens: Optional[int] = None

class Choice(BaseModel):
    message: Message
    finishReason: Optional[str] = Field(alias="finish_reason")
    index: int = 0

class ChatResponse(BaseModel):
    id: str
    choices: list[Choice]
    usage: Optional[dict]= None    
    max_tokens: Optional[int] = None

load_dotenv()

def creeaza_sesiune_autentificata(token: str) -> requests.Session:
    """Creează o sesiune requests cu autentificare Bearer aplicată global."""
    sesiune = requests.Session()
    sesiune.headers.update({
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    })
    return sesiune

class ChatWrapper():
    def __init__(self):
        self._api_key = os.environ.get("API_KEY")
        self._base_url = os.environ.get("BASE_URL")

        if not self._api_key:
            print("Eroare: API_KEY missing from .env file")
            sys.exit(1)
        if not self._base_url:
            print("Eroare: BASE_URL missing from .env file")
            sys.exit(1)
        
    @property
    def api_key(self) -> str:
        return self._api_key

    @property
    def base_url(self) -> str:
        return self._base_url    
    
    def test_connection(self, model: str = "gemini-3-flash-preview") -> None:
        print("Testing conection...")
        response = self.chat(
            messages=[{"role": "user", "content": "Hello, test connection!"}],
            model=model,
            max_tokens=700
        )
        print("Conexiune reusita! Raspuns:", response.choices[0].message.content)

    def chat(self, messages: list[dict], model:str="gemini-3-flash-preview", temperature: float=0.7, max_tokens: int = 700, system: Optional[str] = None) -> ChatResponse:
        if system:
            messages = [{"role": "system", "content": system}] + list(messages)
        request = ChatRequest(messages=messages, model=model, temperature=temperature, max_tokens=max_tokens)
        sesiune = creeaza_sesiune_autentificata(self.api_key)

        try:
            raspuns = sesiune.post(self.base_url, json=request.model_dump(exclude_none=True),timeout=30)
        except requests.exceptions.Timeout:
            raise api_error.APIConnectionError("Expired request - server did not reply in 30s")
        except requests.exceptions.ConnectionError:
            raise api_error.APIConnectionError("Network error - server connection error")
        

        treat_http_error(raspuns)

        try:
            return ChatResponse.model_validate(raspuns.json())
        except PydanticValidationError as e:
            logger.error(f"Rapsuns neasteptat: {e}")
            raise api_error.APIError(f"Structura raspunsului API s-a schimbat: {e}") from e

    def call_with_retries(self, *args, retries=4, delay=2, **kwargs) -> ChatResponse:
        """Apelează metoda chat cu retry la erori de rețea sau rate limit."""
        for attempt in range(1, retries + 1):
            try:
                return self.chat(*args, **kwargs)
            except (api_error.APIConnectionError, api_error.RateLimitError) as e:
                if attempt == retries:
                    raise
                logger.warning(f"Attempt {attempt} failed: {e}. Retrying in {delay} seconds...")
                time.sleep(delay)

def treat_http_error(response: requests.Response) -> None:
    """Converteste erorile in exceptii"""
    if response.ok:
        return

    cod = response.status_code
    try:
        details = response.json().get("error",{}).get("message", response.text)
    except Exception:
        details = response.text
    if cod == 401:
        raise api_error.AuthenticationError(f"Authentification failuire: {details}", cod)
    if cod == 429:
        raise api_error.RateLimitError(f"Rate limit reached: {details}", cod)
    if cod in (400, 422):
        raise api_error.ValidationError(f"Invalid request: {details}", cod)
    if cod in (500, 502, 503):
        raise api_error.APIError(f"Server error: {details}", cod)
    else:
        raise api_error.APIError(f"API error({cod}):{details}", cod)
 
wrapper = ChatWrapper()
wrapper.test_connection("gemini-3-flash-preview")
response = wrapper.call_with_retries(
    messages=[{"role": "user", "content": "How did the number of deaths per year from natural disasters change over the last 100 years?"}],
    model="gemini-3-flash-preview",   # sau alt model compatibil
    system="Fii concis, maximum 100 de cuvinte.",
    temperature=0.5,
    retries=4,
    delay=2,
)

print(response.choices[0].message.content)