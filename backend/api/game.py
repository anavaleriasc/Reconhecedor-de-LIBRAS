"""
game.py — Gerenciamento de sessões de jogo em memória.

Cada sessão é uma instância de GameSession (definida em src/prediction.py),
mantida viva enquanto a partida está em andamento. Como o jogo agora é
dirigido pelo front (React) via chamadas HTTP/WebSocket em vez de um loop
local com webcam e teclado, alguém precisa guardar o estado entre uma
chamada e outra — é isso que este módulo faz.

Importante: este armazenamento é em memória do processo. Se a API rodar
com múltiplos workers/processos, cada um terá seu próprio dicionário de
sessões — nesse cenário, considerar trocar por um backend compartilhado
(Redis, por exemplo) antes de escalar horizontalmente.
"""

import threading
import uuid
from typing import Dict, Tuple

from backend.src.prediction import GameSession


class GameSessionManager:
    """Registro thread-safe de sessões de jogo ativas, indexadas por
    session_id (UUID gerado na criação da sessão)."""

    def __init__(self) -> None:
        self._sessions: Dict[str, GameSession] = {}
        self._lock = threading.Lock()

    def create(self, texto: str) -> Tuple[str, GameSession]:
        """Cria uma nova sessão a partir do texto/frase a ser soletrada.
        Pode levantar ValueError se o texto for inválido/vazio após
        normalização (propagado por GameSession.new)."""
        session = GameSession.new(texto)
        session_id = uuid.uuid4().hex

        with self._lock:
            self._sessions[session_id] = session

        return session_id, session

    def get(self, session_id: str) -> GameSession:
        """Retorna a sessão correspondente, ou lança KeyError se não
        existir (a camada de services converte isso em um erro de domínio,
        e a camada de rotas em HTTP 404)."""
        with self._lock:
            session = self._sessions.get(session_id)

        if session is None:
            raise KeyError(f"Sessão de jogo '{session_id}' não encontrada.")

        return session

    def remove(self, session_id: str) -> None:
        """Remove a sessão (ex.: ao final do jogo ou se o usuário abandonar)."""
        with self._lock:
            self._sessions.pop(session_id, None)

    def exists(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._sessions

    def count(self) -> int:
        """Número de sessões ativas — útil para um endpoint de métricas/debug."""
        with self._lock:
            return len(self._sessions)


# Instância única compartilhada por toda a API (singleton em nível de módulo).
# Os services importam esta instância em vez de criar a própria, garantindo
# que todas as rotas/websockets enxerguem as mesmas sessões.
game_session_manager = GameSessionManager()
