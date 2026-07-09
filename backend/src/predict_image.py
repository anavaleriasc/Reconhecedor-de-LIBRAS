"""
predict_image.py — Módulo de predição para imagens isoladas do alfabeto manual da Libras.

Carrega um modelo treinado e um codificador de rótulos, processa uma imagem
de entrada, extrai landmarks da mão via MediaPipe e exibe a letra
prevista junto com a confiança da classificação.

Uso:
    python -m src.predict_image --image data/samples/teste.jpg --model models/classifier.joblib
"""

import argparse
import os
import sys
import cv2
import numpy as np
import joblib

from src import config
from src.utils import garantir_diretorio, desenhar_texto


def criar_parser() -> argparse.ArgumentParser:
    """Cria e retorna o parser de argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description="Predição de uma imagem isolada do alfabeto manual da Libras via Landmarks."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Caminho para a imagem de entrada (obrigatório).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=config.MODEL_PATH,
        help=f"Caminho para o modelo treinado (padrão: {config.MODEL_PATH}).",
    )
    parser.add_argument(
        "--label-encoder",
        type=str,
        default=config.LABEL_ENCODER_PATH,
        help=f"Caminho para o codificador de rótulos (padrão: {config.LABEL_ENCODER_PATH}).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(config.FIGURES_DIR, "predict_image_result.png"),
        help="Caminho para salvar a imagem com a visualização do resultado.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Se presente, exibe a imagem em uma janela do OpenCV.",
    )
    return parser


def carregar_modelo(caminho_modelo: str, caminho_encoder: str):
    """Carrega o modelo treinado e o codificador de rótulos."""
    if not os.path.isfile(caminho_modelo):
        print(f"Erro: arquivo do modelo não encontrado em '{caminho_modelo}'.")
        sys.exit(1)

    if not os.path.isfile(caminho_encoder):
        print(f"Erro: arquivo do codificador de rótulos não encontrado em '{caminho_encoder}'.")
        sys.exit(1)

    try:
        modelo = joblib.load(caminho_modelo)
        label_encoder = joblib.load(caminho_encoder)
    except Exception as e:
        print(f"Erro ao carregar o modelo ou codificador: {e}")
        sys.exit(1)

    return modelo, label_encoder


def criar_visualizacao(
    debug_image: np.ndarray,
    letra_prevista: str,
    confianca: float,
) -> np.ndarray:
    """
    Cria uma imagem de visualização com o resultado da predição.
    Usa a imagem já com os landmarks desenhados pelo MediaPipe.
    """
    vis = debug_image.copy()
    altura, largura = vis.shape[:2]

    # Monta o texto com a predição
    texto_letra = f"Letra: {letra_prevista}"
    texto_confianca = f"Confianca: {confianca * 100:.1f}%"

    # Fundo semi-transparente para melhor legibilidade do texto
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (largura, 80), config.COR_PRETO, cv2.FILLED)
    cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)

    # Desenha os textos
    desenhar_texto(vis, texto_letra, (10, 30), escala=1.0, cor=config.COR_VERDE, espessura=2)
    desenhar_texto(vis, texto_confianca, (10, 65), escala=0.8, cor=config.COR_BRANCO, espessura=2)

    return vis


def executar_predicao(args: argparse.Namespace) -> None:
    """Executa o pipeline completo de predição para uma imagem isolada."""
    # Importação local para garantir que os subagentes já tenham criado o módulo
    try:
        from src.landmarks import extract_hand_landmarks
        from src.features import normalize_landmarks, extract_features_from_landmarks
    except ImportError as e:
        print(f"Erro ao importar módulos core. Verifique se foram implementados: {e}")
        sys.exit(1)

    print(f"Carregando modelo de '{args.model}'...")
    print(f"Carregando codificador de '{args.label_encoder}'...")
    modelo, label_encoder = carregar_modelo(args.model, args.label_encoder)

    if not os.path.isfile(args.image):
        print(f"Erro: imagem não encontrada em '{args.image}'.")
        sys.exit(1)

    imagem_bgr = cv2.imread(args.image)
    if imagem_bgr is None:
        print(f"Erro: não foi possível ler a imagem '{args.image}'.")
        sys.exit(1)

    print(f"Imagem carregada: {args.image} ({imagem_bgr.shape[1]}x{imagem_bgr.shape[0]})")
    print(f"Extraindo landmarks da mão via MediaPipe...")
    
    landmarks_array, debug_image, success, handedness = extract_hand_landmarks(imagem_bgr)

    if not success or landmarks_array is None:
        print("Erro: nenhuma mão detectada na imagem pelo MediaPipe.")
        sys.exit(1)

    normalized = normalize_landmarks(landmarks_array, handedness)
    feature_vector = extract_features_from_landmarks(normalized)

    print(f"Vetor geométrico extraído: {feature_vector.shape[0]} dimensões.")

    # Realizar predição
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

    print("\n" + "=" * 40)
    print("       RESULTADO DA PREDIÇÃO")
    print("=" * 40)
    print(f"  Letra prevista : {letra_prevista}")
    print(f"  Confiança      : {confianca * 100:.2f}%")
    print("=" * 40 + "\n")

    visualizacao = criar_visualizacao(debug_image, letra_prevista, confianca)

    diretorio_saida = os.path.dirname(args.output)
    if diretorio_saida:
        garantir_diretorio(diretorio_saida)
    cv2.imwrite(args.output, visualizacao)
    print(f"Visualização salva em: {args.output}")

    if args.show:
        nome_janela = "Predicao - Libras (MediaPipe)"
        cv2.imshow(nome_janela, visualizacao)
        print("Pressione qualquer tecla para fechar a janela...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    parser = criar_parser()
    args = parser.parse_args()
    executar_predicao(args)


if __name__ == "__main__":
    main()
