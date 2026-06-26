"""
dataset.py — Carregamento e preparação do dataset do alfabeto manual da Libras.

Percorre diretórios organizados por classe (uma pasta por letra),
extrai features de cada imagem e retorna os dados prontos para treinamento.
"""

import os
import cv2
import numpy as np
from tqdm import tqdm

from src import config
from src.landmarks import extract_hand_landmarks
from src.features import normalize_landmarks, extract_features_from_landmarks


def load_dataset(data_dir=None):
    """
    Carrega dataset organizado por pastas.

    Cada subpasta de data_dir representa uma classe (letra).
    Percorre todas as imagens, extrai features, retorna X, y.

    Args:
        data_dir: caminho para o diretório com pastas por classe.
                  Padrão: config.DATA_DIR

    Retorna:
        X: np.ndarray com matriz de features (n_amostras x n_features)
        y: np.ndarray com rótulos (strings das letras)
        image_paths: lista com caminhos das imagens usadas
    """
    # Definir diretório de dados
    if data_dir is None:
        data_dir = config.DATA_DIR

    # Verificar se o diretório existe
    if not os.path.isdir(data_dir):
        print(f"[ERRO] Diretório de dados não encontrado: {data_dir}")
        print("Verifique se o caminho está correto e se o dataset foi extraído.")
        return np.array([]), np.array([]), []

    # Listar e ordenar subpastas (cada uma representa uma classe/letra)
    todas_pastas = sorted([
        pasta for pasta in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, pasta))
    ])

    if len(todas_pastas) == 0:
        print(f"[ERRO] Nenhuma subpasta de classe encontrada em: {data_dir}")
        return np.array([]), np.array([]), []

    # Filtrar letras dinâmicas (J e Z) se configurado
    if config.EXCLUDE_DYNAMIC_LETTERS:
        pastas_excluidas = [p for p in todas_pastas if p.upper() in config.DYNAMIC_LETTERS]
        pastas = [p for p in todas_pastas if p.upper() not in config.DYNAMIC_LETTERS]
        if pastas_excluidas:
            print(f"[INFO] Excluindo letras dinâmicas: {', '.join(pastas_excluidas)}")
    else:
        pastas = todas_pastas

    print(f"[INFO] Diretório de dados: {data_dir}")
    print(f"[INFO] Classes encontradas ({len(pastas)}): {', '.join(pastas)}")

    # Listas para armazenar resultados
    features_list = []
    labels_list = []
    image_paths = []

    # Contadores para estatísticas
    total_imagens = 0
    total_sucessos = 0
    total_falhas = 0
    stats_por_classe = {}

    # Contar total de imagens para a barra de progresso
    total_arquivos = 0
    for pasta in pastas:
        caminho_pasta = os.path.join(data_dir, pasta)
        arquivos = [
            arq for arq in os.listdir(caminho_pasta)
            if arq.lower().endswith(config.IMAGE_EXTENSIONS)
        ]
        total_arquivos += len(arquivos)

    # Processar cada classe
    barra_progresso = tqdm(total=total_arquivos, desc="Extraindo features", unit="img")

    for pasta in pastas:
        caminho_pasta = os.path.join(data_dir, pasta)
        letra = pasta.upper()

        # Listar arquivos de imagem válidos
        arquivos = sorted([
            arq for arq in os.listdir(caminho_pasta)
            if arq.lower().endswith(config.IMAGE_EXTENSIONS)
        ])

        sucessos_classe = 0
        falhas_classe = 0

        for arquivo in arquivos:
            caminho_imagem = os.path.join(caminho_pasta, arquivo)
            total_imagens += 1

            # Carregar imagem em BGR
            imagem_bgr = cv2.imread(caminho_imagem)

            if imagem_bgr is None:
                print(f"\n[AVISO] Não foi possível carregar a imagem: {caminho_imagem}")
                falhas_classe += 1
                total_falhas += 1
                barra_progresso.update(1)
                continue

            # Extrair features com MediaPipe
            landmarks, _, success, handedness = extract_hand_landmarks(imagem_bgr)

            if not success:
                falhas_classe += 1
                total_falhas += 1
                barra_progresso.update(1)
                continue

            landmarks_norm = normalize_landmarks(landmarks, handedness)
            feat = extract_features_from_landmarks(landmarks_norm)


            # Armazenar resultado
            features_list.append(feat)
            labels_list.append(letra)
            image_paths.append(caminho_imagem)
            sucessos_classe += 1
            total_sucessos += 1

            barra_progresso.update(1)

        stats_por_classe[letra] = {
            "total": len(arquivos),
            "sucessos": sucessos_classe,
            "falhas": falhas_classe,
        }

    barra_progresso.close()

    # Imprimir estatísticas
    print("\n" + "=" * 60)
    print("ESTATÍSTICAS DO CARREGAMENTO DO DATASET")
    print("=" * 60)
    print(f"  Total de imagens encontradas: {total_imagens}")
    print(f"  Features extraídas com sucesso: {total_sucessos}")
    print(f"  Falhas na extração: {total_falhas}")
    if total_imagens > 0:
        taxa_sucesso = (total_sucessos / total_imagens) * 100
        print(f"  Taxa de sucesso: {taxa_sucesso:.1f}%")
    print("-" * 60)
    print(f"  {'Classe':<10} {'Total':<10} {'Sucessos':<10} {'Falhas':<10}")
    print("-" * 60)
    for letra in sorted(stats_por_classe.keys()):
        s = stats_por_classe[letra]
        print(f"  {letra:<10} {s['total']:<10} {s['sucessos']:<10} {s['falhas']:<10}")
    print("=" * 60)

    # Converter para arrays numpy
    if len(features_list) > 0:
        X = np.array(features_list)
        y = np.array(labels_list)
    else:
        X = np.array([])
        y = np.array([])
        print("[AVISO] Nenhuma feature foi extraída com sucesso.")

    print(f"\n[INFO] Formato final de X: {X.shape if X.size > 0 else '(vazio)'}")
    print(f"[INFO] Formato final de y: {y.shape if y.size > 0 else '(vazio)'}")

    return X, y, image_paths
