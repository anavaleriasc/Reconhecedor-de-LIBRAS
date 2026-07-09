"""
landmarks.py — Extração de landmarks de mãos usando MediaPipe Tasks API.
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "hand_landmarker.task"

# Conexões dos dedos para desenho manual (já que drawing_utils não existe no Python 3.12)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Polegar
    (0, 5), (5, 6), (6, 7), (7, 8),        # Indicador
    (5, 9), (9, 10), (10, 11), (11, 12),   # Medio
    (9, 13), (13, 14), (14, 15), (15, 16), # Anelar
    (13, 17), (17, 18), (18, 19), (19, 20),# Minimo
    (0, 17)                                # Base da palma
]

# Inicialização global do detector para otimização
_detector = None

def get_detector():
    global _detector
    if _detector is None:
        try:
            base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=1,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            _detector = vision.HandLandmarker.create_from_options(options)
        except Exception as e:
            print(f"Erro ao inicializar o modelo do MediaPipe: {e}")
            return None
    return _detector

def extract_hand_landmarks(image_bgr):
    """
    Extrai landmarks da mão usando MediaPipe Hands Tasks API.
    
    Parâmetros
    ----------
    image_bgr : np.ndarray
        Imagem de entrada no formato BGR.
        
    Retorna
    -------
    tuple[np.ndarray ou None, np.ndarray, bool]
        - landmarks_array: array numpy (21, 3) com as coordenadas (x, y, z) ou None.
        - debug_image: cópia da imagem com os landmarks desenhados.
        - success: booleano indicando se uma mão foi detectada.
    """
    debug_image = image_bgr.copy()
    detector = get_detector()
    
    if detector is None:
        return None, debug_image, False

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    detection_result = detector.detect(mp_image)

    if not detection_result.hand_landmarks:
        return None, debug_image, False, None

    hand_landmarks = detection_result.hand_landmarks[0]
    handedness = detection_result.handedness[0][0].category_name
    landmarks_array = np.zeros((21, 3), dtype=np.float32)
    
    h, w, _ = debug_image.shape

    # Definir paleta de cores para os dedos (BGR)
    CORES_DEDOS = [
        (0, 200, 255),    # Polegar (Amarelo/Laranja Neon)
        (255, 100, 0),    # Indicador (Azul Escuro Neon)
        (255, 255, 0),    # Médio (Ciano)
        (100, 255, 100),  # Anelar (Verde Claro)
        (200, 100, 255),  # Mínimo (Rosa/Roxo Neon)
        (255, 255, 255)   # Base da palma (Branco)
    ]
    
    # Mapear índices de conexões para cores
    def obter_cor_conexao(idx):
        if idx <= 4: return CORES_DEDOS[0]
        if idx <= 8: return CORES_DEDOS[1]
        if idx <= 12: return CORES_DEDOS[2]
        if idx <= 16: return CORES_DEDOS[3]
        if idx <= 20: return CORES_DEDOS[4]
        return CORES_DEDOS[5]

    # Desenhar conexões com brilho (linhas mais grossas e suaves)
    for connection in HAND_CONNECTIONS:
        start_idx = connection[0]
        end_idx = connection[1]
        
        pt1 = hand_landmarks[start_idx]
        pt2 = hand_landmarks[end_idx]
        
        x1, y1 = int(pt1.x * w), int(pt1.y * h)
        x2, y2 = int(pt2.x * w), int(pt2.y * h)
        
        cor = obter_cor_conexao(end_idx)
        
        # Sombra/Brilho (linha grossa e semi-transparente seria ideal, mas via cv2 puro fazemos sobreposição)
        cv2.line(debug_image, (x1, y1), (x2, y2), cor, 4, cv2.LINE_AA)
        cv2.line(debug_image, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA) # Núcleo branco

    # Extrair coordenadas e desenhar pontos (bolinhas)
    for i, landmark in enumerate(hand_landmarks):
        landmarks_array[i] = [landmark.x, landmark.y, landmark.z]
        cx, cy = int(landmark.x * w), int(landmark.y * h)
        
        cor = obter_cor_conexao(i)
        cv2.circle(debug_image, (cx, cy), 6, cor, -1, cv2.LINE_AA)
        cv2.circle(debug_image, (cx, cy), 3, (255, 255, 255), -1, cv2.LINE_AA)

    return landmarks_array, debug_image, True, handedness
