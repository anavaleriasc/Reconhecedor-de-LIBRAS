"""routes/game.py — Endpoints REST + WebSocket do jogo de soletração.

Rotas REST: uma chamada por ação do jogador (observe/confirm/skip/finish),
o front chama /observe a cada frame (ex.: um a cada N ms) e as demais sob
demanda (quando o usuário aperta o botão equivalente a SPACE/N/Q).

WebSocket: canal opcional para quem preferir manter um único socket aberto
durante a partida, mais parecido com o loop local do realtime_game.py
original.
"""

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from backend.api.schemas.game import (
    GameResultResponse,
    GameStateResponse,
    NewGameRequest,
    NewGameResponse,
    ObserveRequest,
    ObserveResponse,
)
from backend.api.services import game_service
from backend.api.services.game_service import GameSessionNotFoundError

router = APIRouter(prefix="/game", tags=["game"])


@router.post("/", response_model=NewGameResponse)
def new_game(payload: NewGameRequest) -> NewGameResponse:
    """Equivalente a solicitar_texto_usuario(): inicia uma nova partida com
    a palavra/frase enviada pelo front."""
    try:
        return NewGameResponse(**game_service.start_game(payload.texto))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/{session_id}/observe", response_model=ObserveResponse)
def observe(session_id: str, payload: ObserveRequest) -> ObserveResponse:
    """Processa um frame da webcam e atualiza a última letra reconhecida
    da sessão, sem avançar a palavra."""
    try:
        resultado = game_service.observe_frame(session_id, payload.image_base64)
    except GameSessionNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    return ObserveResponse(**resultado.to_dict())


@router.post("/{session_id}/confirm", response_model=GameStateResponse)
def confirm(session_id: str) -> GameStateResponse:
    """Equivalente à tecla SPACE."""
    try:
        return GameStateResponse(**game_service.confirm_attempt(session_id))
    except GameSessionNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.post("/{session_id}/skip", response_model=GameStateResponse)
def skip(session_id: str) -> GameStateResponse:
    """Equivalente à tecla N."""
    try:
        return GameStateResponse(**game_service.skip_letter(session_id))
    except GameSessionNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.post("/{session_id}/finish", response_model=GameStateResponse)
def finish(session_id: str) -> GameStateResponse:
    """Equivalente à tecla Q — encerra a partida antes do fim."""
    try:
        return GameStateResponse(**game_service.finish_game(session_id))
    except GameSessionNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.get("/{session_id}", response_model=GameStateResponse)
def get_state(session_id: str) -> GameStateResponse:
    """Estado atual da partida — útil para o front sincronizar a UI
    (ex.: ao reconectar ou dar refresh na página)."""
    try:
        return GameStateResponse(**game_service.get_state(session_id))
    except GameSessionNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.get("/{session_id}/result", response_model=GameResultResponse)
def get_result(session_id: str) -> GameResultResponse:
    """Equivalente à tela final de exibir_resultado_final(), como JSON."""
    try:
        return GameResultResponse(**game_service.get_result(session_id))
    except GameSessionNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.delete("/{session_id}", status_code=204)
def delete_game(session_id: str) -> None:
    """Remove a sessão (ex.: usuário saiu da tela do jogo sem terminar)."""
    try:
        game_service.delete_game(session_id)
    except GameSessionNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.websocket("/{session_id}/ws")
async def game_websocket(websocket: WebSocket, session_id: str) -> None:
    """
    Canal em tempo real, opcional, para quem preferir manter um único
    socket aberto durante a partida em vez de um POST por frame — mais
    parecido com o loop local do realtime_game.py original.

    Protocolo (mensagens JSON):

        Cliente -> servidor:
            {"type": "frame", "image_base64": "..."}
            {"type": "confirm"}
            {"type": "skip"}
            {"type": "finish"}

        Servidor -> cliente:
            {"type": "observation", "hand_detected": ..., "letter": ..., "confidence": ..., "error": ...}
            {"type": "state", "status": ..., "indice_atual": ..., ...}
            {"type": "error", "detail": "..."}

    A sessão precisa já existir (criada via POST /game/) antes de abrir o
    WebSocket; se o session_id for inválido, o socket recebe um erro e é
    encerrado.
    """
    await websocket.accept()

    try:
        while True:
            mensagem = await websocket.receive_json()
            tipo = mensagem.get("type")

            try:
                if tipo == "frame":
                    resultado = game_service.observe_frame(session_id, mensagem.get("image_base64", ""))
                    await websocket.send_json({"type": "observation", **resultado.to_dict()})

                elif tipo == "confirm":
                    estado = game_service.confirm_attempt(session_id)
                    await websocket.send_json({"type": "state", **estado})

                elif tipo == "skip":
                    estado = game_service.skip_letter(session_id)
                    await websocket.send_json({"type": "state", **estado})

                elif tipo == "finish":
                    estado = game_service.finish_game(session_id)
                    await websocket.send_json({"type": "state", **estado})

                else:
                    await websocket.send_json(
                        {"type": "error", "detail": f"Tipo de mensagem desconhecido: {tipo}"}
                    )

            except GameSessionNotFoundError as e:
                await websocket.send_json({"type": "error", "detail": str(e)})
                break

    except WebSocketDisconnect:
        pass
