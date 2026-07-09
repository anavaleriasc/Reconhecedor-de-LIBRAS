"""routes/prediction.py — Endpoints de predição avulsa (uma imagem por vez,
sem estado de jogo). Útil para telas de calibração/teste no front."""

from fastapi import APIRouter, File, HTTPException, UploadFile

from api.schemas.prediction import PredictRequest, PredictResponse
from api.services import prediction_service

router = APIRouter(prefix="/predict", tags=["predict"])


@router.post("/", response_model=PredictResponse)
def predict_base64(payload: PredictRequest) -> PredictResponse:
    """Recebe uma imagem em base64 (ex.: capturada de um <canvas> no React)
    e devolve a letra reconhecida — equivalente a executar_predicao() de
    predict_image.py, sem salvar nada em disco."""
    try:
        resultado = prediction_service.predict_image_base64(
            payload.image_base64,
            include_debug_image=payload.include_debug_image,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return PredictResponse(**resultado.to_dict())


@router.post("/upload", response_model=PredictResponse)
async def predict_upload(
    file: UploadFile = File(..., description="Arquivo de imagem (jpg/png)."),
    include_debug_image: bool = False,
) -> PredictResponse:
    """Recebe a imagem como multipart/form-data — útil para testar direto
    pelo Swagger UI (/docs) ou upload de arquivo a partir do front."""
    conteudo = await file.read()

    try:
        resultado = prediction_service.predict_image_bytes(conteudo, include_debug_image=include_debug_image)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return PredictResponse(**resultado.to_dict())
