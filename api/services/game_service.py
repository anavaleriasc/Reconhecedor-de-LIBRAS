"""
services/game_service.py — Regras de aplicação do jogo de soletração.

Faz a ponte entre as rotas (camada HTTP/WebSocket) e o núcleo de domínio
(GameSession, em src/prediction.py) + o registro de sessões (api/game.py).

Não conhece FastAPI/Pydantic — recebe e devolve tipos simples/dicts (ou
PredictionResult), para poder ser testado sem subir um servidor. Erros de
domínio (sessão inexistente) são sinalizados com GameSessionNotFoundError;
quem decide o código HTTP é a camada de rotas.
"""

from src.prediction import GameSession, PredictionResult, decode_base64_image
from api.game import game_session_manager


class GameSessionNotFoundError(Exception):
    """Levantada quando um session_id não corresponde a nenhuma sessão ativa."""


def _get_session(session_id: str) -> GameSession:
    try:
        return game_session_manager.get(session_id)
    except KeyError as e:
        raise GameSessionNotFoundError(str(e)) from e


def start_game(texto: str) -> dict:
    """Cria uma nova sessão a partir da palavra/frase informada.
    Equivalente a solicitar_texto_usuario() + montar letras_esperadas no
    realtime_game.py original — só que o texto vem do front, não do input()."""
    session_id, session = game_session_manager.create(texto)
    return {
        "session_id": session_id,
        "texto_original": session.texto_original,
        "total_letras": session.total_letras,
        "letra_esperada": session.letra_esperada,
    }


def observe_frame(session_id: str, image_base64: str) -> PredictionResult:
    """Processa um frame da webcam e atualiza a última letra reconhecida da
    sessão, sem avançar a palavra — equivalente a cada iteração do loop
    local, antes de checar a tecla pressionada."""
    session = _get_session(session_id)

    frame = decode_base64_image(image_base64)
    if frame is None:
        return PredictionResult(hand_detected=False, error="Não foi possível decodificar a imagem base64.")

    return session.observe(frame)


def confirm_attempt(session_id: str) -> dict:
    """Equivalente à tecla SPACE."""
    return _get_session(session_id).confirm()


def skip_letter(session_id: str) -> dict:
    """Equivalente à tecla N."""
    return _get_session(session_id).skip()


def finish_game(session_id: str) -> dict:
    """Equivalente à tecla Q — encerra a partida antes do fim."""
    return _get_session(session_id).finalizar_forcado()


def get_state(session_id: str) -> dict:
    return _get_session(session_id).snapshot()


def get_result(session_id: str) -> dict:
    """Equivalente a exibir_resultado_final(), como dict pronto para JSON."""
    return _get_session(session_id).resultado_final()


def delete_game(session_id: str) -> None:
    if not game_session_manager.exists(session_id):
        raise GameSessionNotFoundError(f"Sessão de jogo '{session_id}' não encontrada.")
    game_session_manager.remove(session_id)
