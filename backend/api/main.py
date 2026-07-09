"""
main.py — Ponto de entrada da API FastAPI.

Uso (a partir da raiz do projeto, com src/ e api/ como pacotes irmãos):

    uvicorn api.main:app --reload

Documentação interativa disponível em /docs (Swagger) e /redoc.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import game as game_routes
from api.routes import prediction as prediction_routes
from api.services import prediction_service

app = FastAPI(
    title="Libras Game API",
    description="API para o jogo de reconhecimento do alfabeto manual da Libras via MediaPipe.",
    version="1.0.0",
)

# Ajuste as origens conforme o ambiente (dev do Vite/CRA, produção etc.).
# Em produção, troque "*" ou localhost pelos domínios reais do front.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "https://reconhecedor-de-libras.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    """Carrega o modelo e o label encoder uma vez, antes da primeira
    requisição — evita que o primeiro usuário da API pague o custo de
    leitura do disco em pleno request."""
    prediction_service.preload_model()


@app.get("/health", tags=["health"])
def health() -> dict:
    """Healthcheck simples, útil para orquestradores/monitoramento."""
    return {"status": "ok"}


app.include_router(prediction_routes.router)
app.include_router(game_routes.router)
