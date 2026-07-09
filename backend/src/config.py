"""
config.py — Configurações globais do sistema de reconhecimento do alfabeto manual da Libras.

Todas as constantes e parâmetros do projeto são centralizados aqui para facilitar
ajustes e garantir consistência entre os módulos.
"""

import os

# ==============================================================================
# Caminhos do Projeto
# ==============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
SAMPLES_DIR = os.path.join(PROJECT_ROOT, "data", "samples")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "results", "figures")

MODEL_PATH = os.path.join(MODELS_DIR, "classifier.joblib")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.joblib")

# ==============================================================================
# Parâmetros de Imagem
# ==============================================================================
IMAGE_SIZE = (224, 224)                # Tamanho padrão para redimensionamento
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")

# ==============================================================================
# Parâmetros do MediaPipe Hands
# ==============================================================================
MP_MIN_DETECTION_CONFIDENCE = 0.5
MP_MIN_TRACKING_CONFIDENCE = 0.5

# ==============================================================================
# Parâmetros de Treinamento
# ==============================================================================
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# Parâmetros dos classificadores
KNN_N_NEIGHBORS = 5
SVM_KERNEL = "rbf"
SVM_C = 10.0
SVM_GAMMA = "scale"
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = None

# ==============================================================================
# Letras Dinâmicas (J e Z)
# ==============================================================================
# Se True, exclui letras que exigem movimento (J e Z) do treinamento
EXCLUDE_DYNAMIC_LETTERS = False
DYNAMIC_LETTERS = ["J", "Z"]

# ==============================================================================
# Parâmetros do Modo Interativo (Webcam)
# ==============================================================================
WEBCAM_INDEX = 0
WEBCAM_WINDOW_NAME = "Libras Alphabet Game"
MASK_WINDOW_NAME = "Hand Mask"
WEBCAM_FPS_DELAY = 1                   # Delay em ms para cv2.waitKey

# Cores para desenho na janela (BGR)
COR_VERDE = (0, 255, 0)
COR_VERMELHO = (0, 0, 255)
COR_AZUL = (255, 0, 0)
COR_AMARELO = (0, 255, 255)
COR_BRANCO = (255, 255, 255)
COR_PRETO = (0, 0, 0)
COR_CIANO = (255, 255, 0)
COR_LARANJA = (0, 165, 255)

# Fonte para texto sobre o frame
FONT = 0                              # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 2
