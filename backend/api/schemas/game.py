"""schemas/game.py — Modelos Pydantic dos endpoints do jogo de soletração."""

from typing import List, Optional

from pydantic import BaseModel, Field


class NewGameRequest(BaseModel):
    texto: str = Field(
        ...,
        description="Palavra ou frase a ser soletrada. Acentos são removidos e o "
        "texto é convertido para maiúsculas no back-end (mesma normalização de normalizar_texto).",
        examples=["Oi tudo bem"],
    )


class NewGameResponse(BaseModel):
    session_id: str
    texto_original: str
    total_letras: int
    letra_esperada: Optional[str]


class ObserveRequest(BaseModel):
    image_base64: str = Field(..., description="Frame atual da webcam, em base64.")


class ObserveResponse(BaseModel):
    hand_detected: bool
    letter: Optional[str] = None
    confidence: float = 0.0
    error: Optional[str] = None


class GameStateResponse(BaseModel):
    status: str = Field(..., description="AGUARDANDO | CORRETO | INCORRETO")
    indice_atual: int
    total_letras: int
    letra_esperada: Optional[str]
    acertos: int
    erros: int
    puladas: int
    finalizado: bool
    mensagem: Optional[str] = None


class GameResultResponse(BaseModel):
    texto_original: str
    letras_esperadas: List[str]
    letras_reconhecidas: List[str]
    acertos: int
    erros: int
    puladas: int
    pontuacao: float
