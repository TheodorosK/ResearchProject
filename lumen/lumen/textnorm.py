"""How text is compared in searches: case, accents and Greek final sigma don't matter."""
import unicodedata


def searchable(text: str) -> str:
    """'Καλημέρα ΑΘΗΝΑ café' -> 'καλημερα αθηνα cafe'."""
    decomposed = unicodedata.normalize("NFD", text or "")
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).casefold()
