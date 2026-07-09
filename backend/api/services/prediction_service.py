"""
services/prediction_service.py — Regras de aplicação para predição avulsa
(uma imagem por vez, fora do contexto de uma partida).

Usado pelo endpoint POST /predict — útil para telas de calibração/teste no
front, ou qualquer fluxo que só precise "qual letra é essa imagem?", sem
o controle de progresso/pontuação do jogo.

Não conhece FastAPI/Pydantic — recebe e devolve tipos simples ou objetos
de src.prediction, para poder ser testado sem subir um servidor.
"""

from typing import Optional

from backend.src.prediction import (
    PredictionResult,
    carregar_modelo,
    predict_from_base64,
    predict_from_bytes,
)

# Referências ao modelo/label encoder, preenchidas em preload_model() no
# startup da API. Se ficarem None, predict_from_base64/bytes caem no
# carregamento sob demanda de src.prediction (mesmo resultado, só que sem
# o ganho de já estar pronto na primeira requisição).
_modelo = None
_label_encoder = None


def preload_model(model_path: Optional[str] = None, encoder_path: Optional[str] = None) -> None:
    """Chamado no startup da API (ver api/main.py) para carregar o modelo
    antes da primeira requisição, evitando que o primeiro usuário pague o
    custo de leitura do disco."""
    global _modelo, _label_encoder
    _modelo, _label_encoder = carregar_modelo(model_path, encoder_path)


def predict_image_base64(image_base64: str, include_debug_image: bool = False) -> PredictionResult:
    return predict_from_base64(
        image_base64,
        modelo=_modelo,
        label_encoder=_label_encoder,
        include_debug_image=include_debug_image,
    )


def predict_image_bytes(image_bytes: bytes, include_debug_image: bool = False) -> PredictionResult:
    return predict_from_bytes(
        image_bytes,
        modelo=_modelo,
        label_encoder=_label_encoder,
        include_debug_image=include_debug_image,
    )
