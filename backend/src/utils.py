"""
utils.py — Utilitários gerais do projeto de reconhecimento do alfabeto manual da Libras.

Contém funções auxiliares para manipulação de diretórios, normalização de texto
e desenho de informações sobre frames OpenCV.
"""

import os
import unicodedata

import cv2

from backend.src import config


def garantir_diretorio(caminho):
    """
    Cria o diretório especificado caso ele ainda não exista.

    Parâmetros
    ----------
    caminho : str
        Caminho absoluto ou relativo do diretório a ser criado.

    Retorna
    -------
    str
        O próprio caminho recebido, para facilitar encadeamento.
    """
    os.makedirs(caminho, exist_ok=True)
    return caminho


def normalizar_texto(texto):
    """
    Normaliza texto para o modo interativo.

    Converte para maiúsculas, remove acentos e diacríticos e mantém
    apenas letras ASCII de A a Z.

    Parâmetros
    ----------
    texto : str
        Texto de entrada (pode conter acentos, espaços, pontuação, etc.).

    Retorna
    -------
    str
        Texto normalizado contendo apenas letras maiúsculas A-Z.
    """
    if not texto:
        return ""

    # Converter para maiúsculas
    texto = texto.upper()

    # Decompor caracteres Unicode (NFD separa letra base dos diacríticos)
    texto_nfd = unicodedata.normalize("NFD", texto)

    # Remover marcas diacríticas (categoria Unicode "Mn" = Mark, Nonspacing)
    texto_sem_acentos = "".join(
        char for char in texto_nfd if unicodedata.category(char) != "Mn"
    )

    # Manter apenas letras A-Z
    texto_limpo = "".join(char for char in texto_sem_acentos if char.isalpha() and char.isascii())

    return texto_limpo


def desenhar_texto(frame, texto, posicao, cor, escala=0.7, espessura=2):
    """
    Desenha texto com sombra sobre um frame OpenCV.

    A sombra preta é desenhada com um deslocamento de (2, 2) pixels
    para dar contraste e legibilidade em qualquer fundo.

    Parâmetros
    ----------
    frame : np.ndarray
        Frame OpenCV (BGR) onde o texto será desenhado (modificado in-place).
    texto : str
        Texto a ser exibido.
    posicao : tuple[int, int]
        Coordenadas (x, y) do canto inferior-esquerdo do texto.
    cor : tuple[int, int, int]
        Cor do texto em BGR.
    escala : float, opcional
        Escala da fonte (padrão: 0.7).
    espessura : int, opcional
        Espessura dos traços da fonte (padrão: 2).
    """
    if frame is None or texto is None:
        return

    x, y = posicao

    # Sombra preta com deslocamento de (2, 2) pixels
    cv2.putText(
        frame, texto, (x + 2, y + 2),
        config.FONT, escala, config.COR_PRETO, espessura + 1, cv2.LINE_AA
    )

    # Texto colorido por cima
    cv2.putText(
        frame, texto, (x, y),
        config.FONT, escala, cor, espessura, cv2.LINE_AA
    )


def desenhar_info_frame(frame, info_dict):
    """
    Desenha múltiplas linhas de informação no canto superior-esquerdo do frame.

    Cada entrada do dicionário é exibida como "chave: valor" em uma
    linha separada, com espaçamento vertical automático.

    Parâmetros
    ----------
    frame : np.ndarray
        Frame OpenCV (BGR) onde as informações serão desenhadas (modificado in-place).
    info_dict : dict
        Dicionário com pares chave-valor a serem exibidos. As chaves e
        valores são convertidos para string automaticamente.
    """
    if frame is None or not info_dict:
        return

    x_inicial = 10
    y_inicial = 30
    espacamento = 30

    for i, (chave, valor) in enumerate(info_dict.items()):
        linha = f"{chave}: {valor}"
        posicao = (x_inicial, y_inicial + i * espacamento)
        desenhar_texto(frame, linha, posicao, config.COR_VERDE)


def redimensionar_para_exibicao(imagem, largura_max=800):
    """
    Redimensiona imagem mantendo a proporção para exibição em tela.

    Se a largura da imagem já for menor ou igual a ``largura_max``,
    a imagem é retornada sem modificações.

    Parâmetros
    ----------
    imagem : np.ndarray
        Imagem OpenCV a ser redimensionada.
    largura_max : int, opcional
        Largura máxima desejada em pixels (padrão: 800).

    Retorna
    -------
    np.ndarray
        Imagem redimensionada (ou a original, se já estiver dentro do limite).
    """
    if imagem is None:
        return None

    altura, largura = imagem.shape[:2]

    if largura <= largura_max:
        return imagem

    # Calcular nova altura mantendo a proporção
    proporcao = largura_max / largura
    nova_altura = int(altura * proporcao)

    return cv2.resize(imagem, (largura_max, nova_altura), interpolation=cv2.INTER_AREA)
