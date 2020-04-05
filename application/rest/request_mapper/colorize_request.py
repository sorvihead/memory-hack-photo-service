from dataclasses import dataclass


@dataclass
class ColorizeRequest:
    base64_string: str
    chat_id: str
