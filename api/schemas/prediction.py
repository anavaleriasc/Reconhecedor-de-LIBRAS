"""schemas/prediction.py — Modelos Pydantic do endpoint de predição avulsa
(fora do contexto de uma partida do jogo)."""

from typing import Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    image_base64: str = Field(
        ...,
        description="Imagem em base64 (string pura ou data URL) capturada pelo front.",
    )
    include_debug_image: bool = Field(
        default=False,
        description="Se True, devolve também a imagem com os landmarks desenhados, em base64.",
    )


class PredictResponse(BaseModel):
    hand_detected: bool
    letter: Optional[str] = None
    confidence: float = 0.0
    debug_image_base64: Optional[str] = None
    error: Optional[str] = None
