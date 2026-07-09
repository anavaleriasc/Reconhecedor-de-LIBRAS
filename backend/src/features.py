"""
features.py — Extração de descritores a partir de landmarks do MediaPipe.
"""

import numpy as np

def normalize_landmarks(landmarks_array, handedness="Right"):
    """
    Normaliza os landmarks usando o punho (0) como origem e a distância
    entre o punho (0) e a base do dedo médio (9) como fator de escala.
    Também espelha o eixo X se a mão for esquerda, para que fique
    numericamente idêntica a uma mão direita.
    """
    if landmarks_array is None:
        return None

    # Fazer cópia para não alterar o array original
    landmarks = landmarks_array.copy()

    # Origem no punho (índice 0)
    wrist = landmarks[0].copy()
    landmarks -= wrist

    # Espelhar eixo X se for mão esquerda (Left)
    if handedness == "Left":
        landmarks[:, 0] = -landmarks[:, 0]

    # Escala pela distância do punho (0) até a base do dedo médio (9)
    distance = np.linalg.norm(landmarks[9] - landmarks[0])

    epsilon = 1e-6
    scale = distance if distance > epsilon else epsilon

    landmarks /= scale

    return landmarks

def extract_features_from_landmarks(landmarks):
    """
    Extrai as features a partir de landmarks JÁ NORMALIZADOS.
    Retorna array 1D numpy contendo:
    a) Coordenadas (x,y,z) achatadas (63 features)
    b) Distâncias de cada ponta de dedo (4, 8, 12, 16, 20) até o punho (0)
    c) Distâncias entre pontas adjacentes (4-8, 8-12, 12-16, 16-20)
    d) Ângulo entre os vetores de cada dedo adjacente (crucial para diferenciar U e V)
    """
    if landmarks is None:
        return None

    # a) Todas as coordenadas (x,y,z) achatadas (63 features)
    coords_flattened = landmarks.flatten()

    # b) Distâncias de cada ponta de dedo até o punho
    fingertip_indices = [4, 8, 12, 16, 20]
    wrist = landmarks[0]
    
    dist_to_wrist = [np.linalg.norm(landmarks[idx] - wrist) for idx in fingertip_indices]

    # c) Distâncias entre pontas adjacentes (4-8, 8-12, 12-16, 16-20)
    dist_adjacent = []
    for i in range(len(fingertip_indices) - 1):
        idx1 = fingertip_indices[i]
        idx2 = fingertip_indices[i+1]
        dist_adjacent.append(np.linalg.norm(landmarks[idx1] - landmarks[idx2]))

    # d) Ângulos entre dedos adjacentes
    # Usaremos o vetor que vai da base (MCP) até a ponta (TIP) para cada dedo
    # Polegar: 1 -> 4
    # Indicador: 5 -> 8
    # Médio: 9 -> 12
    # Anelar: 13 -> 16
    # Mínimo: 17 -> 20
    bases = [1, 5, 9, 13, 17]
    tips = [4, 8, 12, 16, 20]
    
    vetores_dedos = []
    for base_idx, tip_idx in zip(bases, tips):
        vec = landmarks[tip_idx] - landmarks[base_idx]
        vetores_dedos.append(vec)
        
    angulos = []
    for i in range(len(vetores_dedos) - 1):
        v1 = vetores_dedos[i]
        v2 = vetores_dedos[i+1]
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            angulos.append(0.0)
        else:
            cos_theta = np.dot(v1, v2) / (norm1 * norm2)
            # Clip para evitar erros numéricos (ex: 1.0000000002)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            angulos.append(np.arccos(cos_theta))

    # e) Distâncias do polegar (4) para as outras pontas (8, 12, 16, 20)
    # Fundamental para letras onde o polegar interage com os outros dedos (ex: A, S, E, T, F)
    thumb_tip = landmarks[4]
    other_tips = [8, 12, 16, 20]
    dist_thumb_others = [np.linalg.norm(landmarks[idx] - thumb_tip) for idx in other_tips]

    # f) "Dobramento" dos 4 dedos principais (indicador, médio, anelar, mínimo)
    # Razão entre a distância da ponta (TIP) ao punho e a base (MCP) ao punho.
    # Dedo esticado: razão > 1. Dedo dobrado: razão < 1.
    dobramento = []
    for tip_idx, mcp_idx in zip([8, 12, 16, 20], [5, 9, 13, 17]):
        dist_tip = np.linalg.norm(landmarks[tip_idx] - wrist)
        dist_mcp = np.linalg.norm(landmarks[mcp_idx] - wrist)
        dobramento.append(dist_tip / (dist_mcp + 1e-6))

    # g) Projeção Lateral (Identificação de Cruzamento de Dedos, vital para U vs R)
    # Vetor que aponta da esquerda para a direita na mão (do MCP do indicador para o MCP do mínimo)
    v_right = landmarks[17] - landmarks[5]
    norm_r = np.linalg.norm(v_right)
    
    cruzamento = []
    for i in range(len(fingertip_indices) - 1):
        idx1 = fingertip_indices[i]
        idx2 = fingertip_indices[i+1]
        v_tips = landmarks[idx2] - landmarks[idx1]
        norm_t = np.linalg.norm(v_tips)
        
        if norm_r > 0 and norm_t > 0:
            # Produto escalar normalizado (cosseno do ângulo)
            # Se for > 0, o dedo 2 está à direita do dedo 1 (ex: U).
            # Se for < 0, o dedo 2 está cruzado à esquerda do dedo 1 (ex: R).
            proj = np.dot(v_right, v_tips) / (norm_r * norm_t)
        else:
            proj = 0.0
        cruzamento.append(proj)

    # h) Diferença Explícita Polegar vs Indicador (F vs T)
    # T e F diferem muito em profundidade (Z) e sobreposição.
    diff_thumb_index = landmarks[8] - landmarks[4]

    # i) Vetores Apontados para Baixo (M vs N vs Q)
    # Avalia o componente Y (que aponta para baixo na câmera) do vetor do dedo.
    # Isso diferencia 3 dedos caídos (M) de 2 dedos caídos (N).
    down_vectors = []
    for tip_idx, mcp_idx in zip([8, 12, 16], [5, 9, 13]):
        vec = landmarks[tip_idx] - landmarks[mcp_idx]
        down_vectors.append(vec[1])  # Y componente

    # j) Métrica de Gancho do Indicador (X)
    # Mede o grau de curvatura do indicador. Se reto = ~1.0, se gancho (X) < 0.8
    hook_index_dist_straight = np.linalg.norm(landmarks[8] - landmarks[6]) + np.linalg.norm(landmarks[6] - landmarks[5])
    hook_index = [np.linalg.norm(landmarks[8] - landmarks[5]) / (hook_index_dist_straight + 1e-6)]

    # k) Bússola Interna da Mão (Independência de Rotação da Câmera)
    # Em vez de confiar no eixo Y da câmera, criamos um vetor "Bússola"
    # que aponta do punho (0) para a base do dedo médio (9).
    v_compass = landmarks[9] - landmarks[0]
    norm_compass = np.linalg.norm(v_compass)
    if norm_compass > 0:
        v_compass_norm = v_compass / norm_compass
    else:
        v_compass_norm = np.array([0, 1, 0])
        
    compass_fingers = []
    # Para cada dedo principal (indicador, médio, anelar, mínimo), 
    # medimos a similaridade do vetor (MCP -> TIP) com a Bússola.
    for tip_idx, mcp_idx in zip([8, 12, 16, 20], [5, 9, 13, 17]):
        vec = landmarks[tip_idx] - landmarks[mcp_idx]
        norm_vec = np.linalg.norm(vec)
        if norm_vec > 0:
            vec_norm = vec / norm_vec
            compass_fingers.append(np.dot(v_compass_norm, vec_norm))
        else:
            compass_fingers.append(0.0)
            
    # Bússola cruzada com o Polegar (Crucial para separar Q de N)
    v_thumb = landmarks[4] - landmarks[2]
    norm_thumb = np.linalg.norm(v_thumb)
    if norm_thumb > 0:
        v_thumb_norm = v_thumb / norm_thumb
        compass_thumb = [np.dot(v_compass_norm, v_thumb_norm)]
    else:
        compass_thumb = [0.0]

    features = np.concatenate([
        coords_flattened,      # 63
        dist_to_wrist,         # 5
        dist_adjacent,         # 4
        angulos,               # 4
        dist_thumb_others,     # 4
        dobramento,            # 4
        cruzamento,            # 4
        diff_thumb_index,      # 3
        down_vectors,          # 3
        hook_index,            # 1
        compass_fingers,       # 4
        compass_thumb          # 1
    ])  # Total = 100 features

    return features.astype(np.float32)
