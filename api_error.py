class APIError(Exception):
    """Eroare de bază pentru toate problemele cu API-ul."""
    def __init__(self, mesaj: str, status_code: int | None = None):
        super().__init__(mesaj)
        self.status_code = status_code
 
class AuthenticationError(APIError):
    """Cheie API invalidă sau expirată (401)."""
    pass
 
class RateLimitError(APIError):
    """Limita de rată depășită (429)."""
    pass
 
class APIConnectionError(APIError):
    """Eroare de rețea — serverul nu a răspuns."""
    pass
 
class ValidationError(APIError):
    """Request invalid — parametri greșiți (400, 422)."""
    pass