"""
prediction.py — Módulo core de predição para uso pela API (FastAPI).

Encapsula a MESMA lógica de interpretação usada em predict_image.py e
webcam_app.py:

    frame (BGR) -> landmarks via MediaPipe -> normalização -> features
    -> predição do classificador (+ confiança)

... porém totalmente desacoplada de:
    - leitura/escrita em disco (cv2.imread / cv2.imwrite)
    - janelas do OpenCV (cv2.imshow / cv2.waitKey)
    - entrada via terminal (input())
    - sys.exit() em caso de erro

A camada de API decide o que fazer com os erros (ex.: retornar HTTP 400/500)
e como os frames chegam (upload multipart, base64 via JSON, WebSocket etc.).

Este módulo também replica a MÁQUINA DE ESTADOS do jogo de soletração de
webcam_app.py através da classe GameSession — só que em vez de reagir a
teclas (SPACE/N/Q) num loop local com webcam, ela é dirigida por chamadas
explícitas (observe/confirm/skip/finalizar_forcado), que a API expõe como
endpoints REST ou mensagens de WebSocket para o front em React.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np
import joblib

from src import config
from src.utils import normalizar_texto
from src.landmarks import extract_hand_landmarks
from src.features import normalize_landmarks, extract_features_from_landmarks


# =============================================================================
# Carregamento do modelo (singleton em memória do processo da API)
# =============================================================================

class _ModelRegistry:
    """Mantém modelo e label encoder carregados em memória, evitando
    releitura do disco a cada requisição da API."""

    _modelo = None
    _label_encoder = None
    _model_path: Optional[str] = None
    _encoder_path: Optional[str] = None

    @classmethod
    def get(cls, model_path: Optional[str] = None, encoder_path: Optional[str] = None):
        model_path = model_path or config.MODEL_PATH
        encoder_path = encoder_path or config.LABEL_ENCODER_PATH

        precisa_recarregar = (
            cls._modelo is None
            or cls._label_encoder is None
            or cls._model_path != model_path
            or cls._encoder_path != encoder_path
        )

        if precisa_recarregar:
            cls._modelo = joblib.load(model_path)
            cls._label_encoder = joblib.load(encoder_path)
            cls._model_path = model_path
            cls._encoder_path = encoder_path

        return cls._modelo, cls._label_encoder

    @classmethod
    def reset(cls) -> None:
        """Útil em testes, ou se a API expuser um endpoint de reload de modelo."""
        cls._modelo = None
        cls._label_encoder = None
        cls._model_path = None
        cls._encoder_path = None


def carregar_modelo(model_path: Optional[str] = None, encoder_path: Optional[str] = None):
    """
    Carrega (ou reaproveita, se já carregado) o modelo e o label encoder.

    Ao contrário das versões em predict_image.py/webcam_app.py, NÃO chama
    sys.exit() em caso de erro — lança RuntimeError, para a API decidir como
    responder (ex.: HTTP 500 com detail=str(e)).
    """
    try:
        return _ModelRegistry.get(model_path, encoder_path)
    except FileNotFoundError as e:
        raise RuntimeError(f"Modelo ou label encoder não encontrado: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Erro ao carregar modelo/encoder: {e}") from e


# =============================================================================
# Utilidades de imagem (decodificação vinda da API)
# =============================================================================

def decode_image_bytes(image_bytes: bytes) -> Optional[np.ndarray]:
    """Decodifica bytes crus (ex.: corpo de um upload multipart) em uma
    imagem BGR do OpenCV. Retorna None se a decodificação falhar."""
    if not image_bytes:
        return None
    np_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    return frame


def decode_base64_image(base64_str: str) -> Optional[np.ndarray]:
    """Decodifica uma imagem em base64 (formato comum quando o front em
    React captura frames de <video>/<canvas> e envia via JSON/WebSocket).
    Aceita tanto uma string base64 pura quanto uma data URL
    ('data:image/jpeg;base64,...')."""
    if not base64_str:
        return None

    if base64_str.strip().startswith("data:") and "," in base64_str:
        base64_str = base64_str.split(",", 1)[1]

    try:
        raw_bytes = base64.b64decode(base64_str)
    except Exception:
        return None

    return decode_image_bytes(raw_bytes)


def encode_image_base64(frame_bgr: np.ndarray, ext: str = ".jpg") -> str:
    """Codifica um frame BGR em base64, útil para devolver ao front a
    imagem com os landmarks desenhados (debug_image do MediaPipe), da
    mesma forma que a versão local desenhava o esqueleto da mão na tela."""
    ok, buffer = cv2.imencode(ext, frame_bgr)
    if not ok:
        return ""
    return base64.b64encode(buffer).decode("utf-8")


# =============================================================================
# Resultado de uma predição
# =============================================================================

@dataclass
class PredictionResult:
    """Resultado de uma predição de uma letra a partir de um frame/imagem."""
    hand_detected: bool
    letter: Optional[str] = None
    confidence: float = 0.0
    debug_image_base64: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "hand_detected": self.hand_detected,
            "letter": self.letter,
            "confidence": round(self.confidence, 4),
            "debug_image_base64": self.debug_image_base64,
            "error": self.error,
        }


# =============================================================================
# Predição stateless a partir de um frame já decodificado
# =============================================================================

def predict_frame(
    frame_bgr: np.ndarray,
    modelo=None,
    label_encoder=None,
    model_path: Optional[str] = None,
    encoder_path: Optional[str] = None,
    include_debug_image: bool = False,
) -> PredictionResult:
    """
    Executa o mesmo pipeline usado em predict_image.py / webcam_app.py:
    frame -> landmarks (MediaPipe) -> normalização -> features -> predição.

    Pode receber modelo/label_encoder já carregados (recomendado dentro da
    API, para reaproveitar entre requisições/frames) ou carregá-los sob
    demanda via model_path/encoder_path.

    Se include_debug_image=True, devolve também a imagem com os landmarks
    desenhados (em base64), para o front exibir o overlay da mão como a
    versão local fazia com cv2.imshow.
    """
    if modelo is None or label_encoder is None:
        modelo, label_encoder = carregar_modelo(model_path, encoder_path)

    try:
        landmarks_array, debug_image, success, handedness = extract_hand_landmarks(frame_bgr)
    except Exception as e:
        return PredictionResult(hand_detected=False, error=f"Erro no MediaPipe: {e}")

    if not success or landmarks_array is None:
        debug_b64 = encode_image_base64(debug_image) if include_debug_image and debug_image is not None else None
        return PredictionResult(hand_detected=False, debug_image_base64=debug_b64)

    try:
        normalized = normalize_landmarks(landmarks_array, handedness)
        feature_vector = extract_features_from_landmarks(normalized)
        feature_vector_2d = feature_vector.reshape(1, -1)

        predicao_codificada = modelo.predict(feature_vector_2d)
        letra_prevista = label_encoder.inverse_transform(predicao_codificada)[0]

        confianca = 0.0
        if hasattr(modelo, "predict_proba"):
            try:
                probabilidades = modelo.predict_proba(feature_vector_2d)
                confianca = float(np.max(probabilidades))
            except Exception:
                confianca = 0.0

        debug_b64 = encode_image_base64(debug_image) if include_debug_image else None

        return PredictionResult(
            hand_detected=True,
            letter=str(letra_prevista),
            confidence=confianca,
            debug_image_base64=debug_b64,
        )
    except Exception as e:
        return PredictionResult(hand_detected=True, error=f"Erro na predição: {e}")


def predict_from_bytes(
    image_bytes: bytes,
    modelo=None,
    label_encoder=None,
    model_path: Optional[str] = None,
    encoder_path: Optional[str] = None,
    include_debug_image: bool = False,
) -> PredictionResult:
    """Ponto de entrada típico para um endpoint POST /predict que recebe
    a imagem como multipart/form-data (UploadFile no FastAPI)."""
    frame = decode_image_bytes(image_bytes)
    if frame is None:
        return PredictionResult(hand_detected=False, error="Não foi possível decodificar a imagem.")
    return predict_frame(frame, modelo, label_encoder, model_path, encoder_path, include_debug_image)


def predict_from_base64(
    base64_str: str,
    modelo=None,
    label_encoder=None,
    model_path: Optional[str] = None,
    encoder_path: Optional[str] = None,
    include_debug_image: bool = False,
) -> PredictionResult:
    """Ponto de entrada típico para um endpoint/WebSocket que recebe frames
    em base64 — fluxo comum quando o React captura a webcam em um <canvas>
    e envia o frame como string em um payload JSON."""
    frame = decode_base64_image(base64_str)
    if frame is None:
        return PredictionResult(hand_detected=False, error="Não foi possível decodificar a imagem base64.")
    return predict_frame(frame, modelo, label_encoder, model_path, encoder_path, include_debug_image)


# =============================================================================
# Estado de uma partida do jogo de soletração
# (mesma jogabilidade de webcam_app.py, sem OpenCV/webcam/teclado)
# =============================================================================

STATUS_AGUARDANDO = "AGUARDANDO"
STATUS_CORRETO = "CORRETO"
STATUS_INCORRETO = "INCORRETO"


@dataclass
class GameSession:
    """
    Réplica stateful (em memória) da lógica de jogo de webcam_app.py,
    adaptada para ser dirigida por chamadas de API/WebSocket em vez de
    teclas (SPACE/N/Q) e uma janela OpenCV local.

    A API deve manter uma instância de GameSession por partida em andamento
    (ex.: num dicionário em memória indexado por session_id, ou em Redis
    serializando resultado_final()/_snapshot() se precisar persistir).

    Fluxo esperado:
        session = GameSession.new(texto="OI TUDO BEM")

        # a cada frame recebido do front (ex.: via WebSocket):
        resultado = session.observe(frame_bgr, modelo, label_encoder)
        # -> devolve letra reconhecida no frame atual, sem avançar a palavra

        # quando o usuário confirma a tentativa (equivalente à tecla SPACE):
        estado = session.confirm()

        # quando o usuário pula a letra atual (equivalente à tecla N):
        estado = session.skip()

        # se o usuário encerrar antes do fim (equivalente à tecla Q):
        estado = session.finalizar_forcado()

        # ao final:
        resumo = session.resultado_final()
    """
    texto_original: str
    letras_esperadas: List[str]
    letras_reconhecidas: List[str] = field(default_factory=list)
    indice_atual: int = 0
    acertos: int = 0
    erros: int = 0
    puladas: int = 0
    status_atual: str = STATUS_AGUARDANDO
    ultima_letra_reconhecida: Optional[str] = None
    ultima_confianca: float = 0.0

    @classmethod
    def new(cls, texto: str) -> "GameSession":
        """Equivalente a solicitar_texto_usuario(), mas recebendo o texto
        já digitado no front em vez de via input() no terminal."""
        texto_normalizado = normalizar_texto(texto)
        if not texto_normalizado:
            raise ValueError(
                "Texto inválido ou vazio após normalização (apenas letras A-Z são aceitas)."
            )
        return cls(texto_original=texto_normalizado, letras_esperadas=list(texto_normalizado))

    @property
    def total_letras(self) -> int:
        return len(self.letras_esperadas)

    @property
    def finalizado(self) -> bool:
        return self.indice_atual >= self.total_letras

    @property
    def letra_esperada(self) -> Optional[str]:
        return None if self.finalizado else self.letras_esperadas[self.indice_atual]

    def observe(self, frame_bgr: np.ndarray, modelo=None, label_encoder=None) -> PredictionResult:
        """
        Processa um frame vindo da webcam do front e atualiza a última
        letra reconhecida — sem avançar o índice da palavra. No jogo
        local isso acontecia a cada iteração do loop, antes de checar a
        tecla pressionada; aqui vira uma chamada explícita por frame.
        """
        resultado = predict_frame(frame_bgr, modelo, label_encoder)
        self.ultima_letra_reconhecida = resultado.letter if resultado.hand_detected else None
        self.ultima_confianca = resultado.confidence
        return resultado

    def confirm(self) -> dict:
        """Equivalente à tecla SPACE: confirma a tentativa usando a última
        letra reconhecida por observe()."""
        if self.finalizado:
            return self._snapshot(mensagem="Jogo já finalizado.")

        if self.ultima_letra_reconhecida is None:
            return self._snapshot(mensagem="Mão não detectada. Posicione a mão e tente novamente.")

        letra_esperada = self.letra_esperada
        letra_reconhecida = self.ultima_letra_reconhecida

        if letra_reconhecida == letra_esperada:
            self.acertos += 1
            self.status_atual = STATUS_CORRETO
        else:
            self.erros += 1
            self.status_atual = STATUS_INCORRETO

        self.letras_reconhecidas.append(letra_reconhecida)
        self.indice_atual += 1
        self.ultima_letra_reconhecida = None

        return self._snapshot()

    def skip(self) -> dict:
        """Equivalente à tecla N: pula a letra atual (conta como erro)."""
        if self.finalizado:
            return self._snapshot(mensagem="Jogo já finalizado.")

        self.erros += 1
        self.puladas += 1
        self.letras_reconhecidas.append("-")
        self.indice_atual += 1
        self.status_atual = STATUS_INCORRETO
        self.ultima_letra_reconhecida = None

        return self._snapshot()

    def finalizar_forcado(self) -> dict:
        """Equivalente à tecla Q: encerra a partida antes do fim,
        preenchendo as letras restantes como não reconhecidas ('-')."""
        while len(self.letras_reconhecidas) < self.total_letras:
            self.letras_reconhecidas.append("-")
        self.indice_atual = self.total_letras
        return self._snapshot()

    def resultado_final(self) -> dict:
        """Equivalente a exibir_resultado_final(), mas retornando um dict
        em vez de imprimir no terminal — pronto para virar JSON na API."""
        total_tentativas = self.acertos + self.erros
        pontuacao = (self.acertos / total_tentativas * 100) if total_tentativas > 0 else 0.0
        return {
            "texto_original": self.texto_original,
            "letras_esperadas": self.letras_esperadas,
            "letras_reconhecidas": self.letras_reconhecidas,
            "acertos": self.acertos,
            "erros": self.erros,
            "puladas": self.puladas,
            "pontuacao": round(pontuacao, 2),
        }

    def snapshot(self, mensagem: Optional[str] = None) -> dict:
        """Versão pública de _snapshot(), para a camada de services da API
        consultar o estado atual sem depender de um método "privado"."""
        return self._snapshot(mensagem)

    def _snapshot(self, mensagem: Optional[str] = None) -> dict:
        return {
            "status": self.status_atual,
            "indice_atual": self.indice_atual,
            "total_letras": self.total_letras,
            "letra_esperada": self.letra_esperada,
            "acertos": self.acertos,
            "erros": self.erros,
            "puladas": self.puladas,
            "finalizado": self.finalizado,
            "mensagem": mensagem,
        }
