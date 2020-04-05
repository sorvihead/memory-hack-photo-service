from dataclasses import dataclass


@dataclass
class FacesRequest:
    chat_id: str
    base64_string: str
    type: str
