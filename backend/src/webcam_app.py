"""
webcam_app.py — Modo interativo com webcam para reconhecimento do alfabeto manual da Libras.

O usuário digita uma palavra ou frase e o sistema solicita que ele reproduza cada
letra com a mão na frente da webcam. O reconhecimento é feito em tempo real e o
sistema exibe um HUD com a letra esperada, a letra reconhecida, o status da tentativa
e a pontuação acumulada.

Controles:
    SPACE  → Confirmar tentativa
    N      → Pular letra
    Q      → Sair do jogo

Uso:
    python -m src.webcam_app --model models/classifier.joblib
"""

import argparse
import os
import sys
import warnings

# Suprimir os alertas chatos de depreciação do Protobuf causados pelo MediaPipe
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")
import time
import cv2
import numpy as np
import joblib

from backend.src import config
from backend.src.utils import normalizar_texto, desenhar_texto


# =============================================================================
# Constantes internas do módulo
# =============================================================================
# Altere para "game" se desejar jogar, ou "analysis" para inspecionar predições
DEFAULT_MODE = "analysis"

_STATUS_AGUARDANDO = "AGUARDANDO"
_STATUS_CORRETO = "CORRETO"
_STATUS_INCORRETO = "INCORRETO"

# Quantidade de frames que o status CORRETO/INCORRETO permanece visível
_FRAMES_STATUS_VISIVEL = 10

# Margem e dimensões do HUD
_HUD_MARGEM_TOPO = 10
_HUD_ALTURA = 200
_HUD_PADDING_ESQUERDA = 15
_HUD_ESPACO_LINHA = 30


# =============================================================================
# Funções auxiliares
# =============================================================================

def criar_parser() -> argparse.ArgumentParser:
    """Cria e retorna o parser de argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description="Modo interativo com webcam para reconhecimento do alfabeto manual da Libras."
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
        "--camera",
        type=int,
        default=config.WEBCAM_INDEX,
        help=f"Índice da câmera a utilizar (padrão: {config.WEBCAM_INDEX}).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["analysis", "game"],
        default=DEFAULT_MODE,
        help="Modo de operação: 'analysis' para depurar probabilidades ou 'game' para jogar.",
    )
    return parser


def carregar_modelo(caminho_modelo: str, caminho_encoder: str):
    """
    Carrega o modelo treinado e o codificador de rótulos.

    Parâmetros:
        caminho_modelo: Caminho do arquivo do modelo (.joblib).
        caminho_encoder: Caminho do arquivo do codificador de rótulos (.joblib).

    Retorna:
        Tupla (modelo, label_encoder).

    Levanta:
        SystemExit se os arquivos não forem encontrados ou não puderem ser carregados.
    """
    if not os.path.isfile(caminho_modelo):
        print(f"Erro: arquivo do modelo não encontrado em '{caminho_modelo}'.")
        sys.exit(1)

    if not os.path.isfile(caminho_encoder):
        print(f"Erro: arquivo do codificador de rótulos não encontrado em '{caminho_encoder}'.")
        sys.exit(1)

    try:
        modelo = joblib.load(caminho_modelo)
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        sys.exit(1)

    try:
        label_encoder = joblib.load(caminho_encoder)
    except Exception as e:
        print(f"Erro ao carregar o codificador de rótulos: {e}")
        sys.exit(1)

    return modelo, label_encoder


def solicitar_texto_usuario() -> str:
    """
    Solicita ao usuário que digite uma palavra ou frase no terminal.

    Continua solicitando até que o texto normalizado não esteja vazio.

    Retorna:
        Texto normalizado (maiúsculas, sem acentos, somente A-Z).
    """
    while True:
        entrada = input("\nDigite uma palavra ou frase para soletrar: ").strip()
        texto_normalizado = normalizar_texto(entrada)

        if texto_normalizado:
            return texto_normalizado

        print("Texto inválido ou vazio após normalização. Tente novamente.")
        print("(Apenas letras A-Z são aceitas. Acentos são removidos automaticamente.)")


def desenhar_hud(
    frame: np.ndarray,
    letra_esperada: str,
    letra_reconhecida: str,
    status: str,
    acertos: int,
    erros: int,
    puladas: int,
    indice_atual: int,
    total_letras: int,
    texto_original: str,
) -> np.ndarray:
    """
    Desenha o Heads-Up Display (HUD) moderno com Pillow.
    """
    from backend.src.ui import UIRenderer
    
    ui = UIRenderer(frame)
    altura_frame, largura_frame = frame.shape[:2]
    
    # ---------------------------------------------------------
    # 1. Painel Superior (Fundo do HUD)
    # ---------------------------------------------------------
    hud_bg = [15, 15, largura_frame - 15, 185]
    ui.draw_rounded_rect(hud_bg, (25, 25, 25), alpha=210, radius=12)
    ui.draw_rounded_rect_outline(hud_bg, (80, 80, 80), alpha=255, radius=12, width=1)
    
    # --- Barra de Progresso ---
    progresso = (indice_atual) / max(total_letras, 1) if status == _STATUS_AGUARDANDO else (indice_atual + 1) / max(total_letras, 1)
    bar_rect = [30, 30, largura_frame - 30, 45]
    ui.draw_progress_bar(bar_rect, progresso, color_bgr=(0, 255, 150), bg_color_bgr=(60, 60, 60), alpha=255)
    
    # Texto de Progresso
    ui.draw_text(f"Progresso: {indice_atual + 1}/{total_letras}", (30, 50), (200, 200, 200), size=14)
    
    # Lógica de renderização da palavra com destaque na letra atual
    janela_texto = texto_original[max(0, indice_atual-10) : min(len(texto_original), indice_atual+10)]
    ui.draw_text(f"Frase: {janela_texto}", (largura_frame - 250, 50), (200, 200, 200), size=14)
    
    # ---------------------------------------------------------
    # 2. Caixas de Status e Letras
    # ---------------------------------------------------------
    
    # Letra Esperada Box
    box_esperada = [30, 75, 200, 165]
    ui.draw_rounded_rect(box_esperada, (40, 40, 40), alpha=180, radius=8)
    ui.draw_text("ESPERADA", (115, 95), (150, 150, 150), size=12, bold=True, anchor="mm")
    ui.draw_text(letra_esperada, (115, 130), (0, 220, 255), size=45, bold=True, anchor="mm")
    
    # Letra Reconhecida Box
    box_rec = [220, 75, 450, 165]
    cor_borda_rec = (60, 60, 60)
    cor_texto_rec = (255, 255, 255)
    
    if letra_reconhecida != "---":
        if status == _STATUS_CORRETO:
            cor_borda_rec = (0, 255, 0)
            cor_texto_rec = (0, 255, 0)
        elif status == _STATUS_INCORRETO:
            cor_borda_rec = (0, 0, 255)
            cor_texto_rec = (0, 0, 255)
            
    ui.draw_rounded_rect(box_rec, (40, 40, 40), alpha=180, radius=8)
    ui.draw_rounded_rect_outline(box_rec, cor_borda_rec, alpha=255, radius=8, width=2)
    ui.draw_text("RECONHECIDA", (335, 95), (150, 150, 150), size=12, bold=True, anchor="mm")
    
    # Ajustar tamanho se for mensagem longa ("Mao nao detectada")
    tamanho_fonte = 45 if len(letra_reconhecida) == 1 else 16
    ui.draw_text(letra_reconhecida, (335, 130), cor_texto_rec, size=tamanho_fonte, bold=True, anchor="mm")
    
    # Status da Pontuação Box
    box_status = [470, 75, largura_frame - 30, 165]
    ui.draw_rounded_rect(box_status, (40, 40, 40), alpha=180, radius=8)
    
    # Infos de Score
    ui.draw_text("✅", (485, 90), (255, 255, 255), size=16, is_emoji=True)
    ui.draw_text(f"Acertos: {acertos}", (510, 90), (0, 255, 0), size=16, bold=True)
    
    ui.draw_text("❌", (485, 115), (255, 255, 255), size=16, is_emoji=True)
    ui.draw_text(f"Erros: {erros}", (510, 115), (0, 0, 255), size=16, bold=True)
    
    ui.draw_text("⏭️", (485, 140), (255, 255, 255), size=16, is_emoji=True)
    ui.draw_text(f"Puladas: {puladas}", (510, 140), (255, 180, 0), size=16, bold=True)
    
    # ---------------------------------------------------------
    # 3. Controles (Painel Inferior)
    # ---------------------------------------------------------
    panel_bottom = [15, altura_frame - 45, largura_frame - 15, altura_frame - 15]
    ui.draw_rounded_rect(panel_bottom, (25, 25, 25), alpha=210, radius=8)
    msg_controles = "SPACE = Confirmar   |   N = Pular   |   Q = Sair"
    ui.draw_text(msg_controles, (largura_frame//2, altura_frame - 30), (255, 255, 255), size=14, bold=True, anchor="mm")
    
    # ---------------------------------------------------------
    # 4. Feedback Visual Gigante no Centro da Tela
    # ---------------------------------------------------------
    if status == _STATUS_CORRETO:
        ui.draw_text("✅", (largura_frame//2, altura_frame//2 + 50), (255, 255, 255), size=180, anchor="mm", is_emoji=True)
    elif status == _STATUS_INCORRETO:
        ui.draw_text("❌", (largura_frame//2, altura_frame//2 + 50), (255, 255, 255), size=180, anchor="mm", is_emoji=True)

    return ui.render()





def exibir_resultado_final(
    texto_original: str,
    letras_esperadas: list,
    letras_reconhecidas: list,
    acertos: int,
    erros: int,
    puladas: int,
    tempo_total: float,
) -> None:
    """
    Exibe o resultado final do jogo no terminal.

    Parâmetros:
        texto_original: Texto normalizado digitado pelo usuário.
        letras_esperadas: Lista de letras esperadas.
        letras_reconhecidas: Lista de letras reconhecidas (ou '-' para puladas).
        acertos: Número de acertos.
        erros: Número de erros.
        puladas: Número de letras puladas.
        tempo_total: Tempo total de jogo em segundos.
    """
    total_tentativas = acertos + erros
    porcentagem = (acertos / total_tentativas * 100) if total_tentativas > 0 else 0.0

    seq_esperada = " ".join(letras_esperadas)
    seq_reconhecida = " ".join(letras_reconhecidas)

    print("\n")
    print("=" * 40)
    print("         RESULTADO FINAL")
    print("=" * 40)
    print(f"Palavra/frase: {texto_original}")
    print(f"Sequência esperada:    {seq_esperada}")
    print(f"Sequência reconhecida: {seq_reconhecida}")
    print(f"Acertos: {acertos}")
    print(f"Erros: {erros}")
    print(f"Puladas: {puladas}")
    print(f"Tempo total: {tempo_total:.1f}s")
    print(f"Pontuação final: {porcentagem:.2f}%")
    print("=" * 40)
    print()


def loop_principal(
    modelo,
    label_encoder,
    camera_index: int,
    texto_normalizado: str,
) -> None:
    """
    Executa o loop principal do jogo interativo com a webcam.

    Parâmetros:
        modelo: Modelo de classificação carregado.
        label_encoder: Codificador de rótulos carregado.
        camera_index: Índice da câmera a utilizar.
        texto_normalizado: Texto normalizado (somente A-Z).
    """
    try:
        from backend.src.landmarks import extract_hand_landmarks
        from backend.src.features import normalize_landmarks, extract_features_from_landmarks
    except ImportError as e:
        print(f"Erro ao carregar os módulos core: {e}")
        sys.exit(1)
    # Gerar lista de letras a partir do texto
    letras_esperadas = list(texto_normalizado)
    total_letras = len(letras_esperadas)
    letras_reconhecidas = []

    # Contadores de pontuação
    acertos = 0
    erros = 0
    puladas = 0

    # Controle do status visual
    status_atual = _STATUS_AGUARDANDO
    frames_restantes_status = 0

    # Letra reconhecida pelo modelo no frame atual
    letra_reconhecida_atual = "---"

    # Índice da letra atual na sequência
    indice_atual = 0

    # Abrir webcam
    print(f"\nAbrindo webcam (índice {camera_index})...")
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(
            "Erro: não foi possível acessar a webcam. "
            "Verifique se ela está conectada ou se está sendo usada por outro aplicativo."
        )
        sys.exit(1)

    print("Webcam aberta com sucesso!")
    print(f"\nPalavra/frase: {texto_normalizado}")
    print(f"Total de letras: {total_letras}")
    print("Controles: SPACE = confirmar | N = pular | Q = sair\n")

    tempo_inicio = time.time()

    try:
        while indice_atual < total_letras:
            # Capturar frame
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            # Flip horizontal (espelho)
            frame = cv2.flip(frame, 1)

            # Letra esperada atual
            letra_esperada = letras_esperadas[indice_atual]

            # Extrair características via MediaPipe
            try:
                landmarks_array, debug_image, success, handedness = extract_hand_landmarks(frame)
            except Exception:
                landmarks_array, debug_image, success, handedness = None, frame, False, None

            if success and landmarks_array is not None:
                try:
                    normalized = normalize_landmarks(landmarks_array, handedness)
                    feature_vector = extract_features_from_landmarks(normalized)
                    
                    # Fazer predição
                    feature_vector_2d = feature_vector.reshape(1, -1)
                    predicao_codificada = modelo.predict(feature_vector_2d)
                    letra_reconhecida_atual = label_encoder.inverse_transform(predicao_codificada)[0]
                except Exception:
                    letra_reconhecida_atual = "---"
            else:
                letra_reconhecida_atual = "Mao nao detectada"

            # Gerenciar contagem regressiva do status visual
            if frames_restantes_status > 0:
                frames_restantes_status -= 1
                if frames_restantes_status == 0:
                    status_atual = _STATUS_AGUARDANDO

            # Desenhar HUD
            frame_final = desenhar_hud(
                frame=debug_image,
                letra_esperada=letra_esperada,
                letra_reconhecida=letra_reconhecida_atual,
                status=status_atual,
                acertos=acertos,
                erros=erros,
                puladas=puladas,
                indice_atual=indice_atual,
                total_letras=total_letras,
                texto_original=texto_normalizado,
            )

            # Exibir frame principal
            cv2.imshow(config.WEBCAM_WINDOW_NAME, frame_final)

            # Ler tecla pressionada
            tecla = cv2.waitKey(config.WEBCAM_FPS_DELAY) & 0xFF

            # --- SPACE (32): confirmar tentativa ---
            if tecla == 32:
                if letra_reconhecida_atual not in ("---", "Mao nao detectada"):
                    if letra_reconhecida_atual == letra_esperada:
                        acertos += 1
                        status_atual = _STATUS_CORRETO
                        letras_reconhecidas.append(letra_reconhecida_atual)
                        print(f"  ✓ Letra '{letra_esperada}' — CORRETO!")
                    else:
                        erros += 1
                        status_atual = _STATUS_INCORRETO
                        letras_reconhecidas.append(letra_reconhecida_atual)
                        print(
                            f"  ✗ Esperada '{letra_esperada}', "
                            f"reconhecida '{letra_reconhecida_atual}' — INCORRETO"
                        )

                    frames_restantes_status = _FRAMES_STATUS_VISIVEL
                    indice_atual += 1
                else:
                    # Mão não detectada — não confirma
                    print("  ⚠ Mão não detectada. Posicione a mão e tente novamente.")

            # --- N: pular letra ---
            elif tecla == ord("n") or tecla == ord("N"):
                erros += 1
                puladas += 1
                letras_reconhecidas.append("-")
                print(f"  → Letra '{letra_esperada}' pulada (contada como erro).")
                indice_atual += 1
                status_atual = _STATUS_INCORRETO
                frames_restantes_status = _FRAMES_STATUS_VISIVEL

            # --- Q: sair do jogo ---
            elif tecla == ord("q") or tecla == ord("Q"):
                print("\nJogo encerrado pelo usuário.")
                # Preencher letras restantes como não reconhecidas
                while len(letras_reconhecidas) < total_letras:
                    letras_reconhecidas.append("-")
                break

    except KeyboardInterrupt:
        print("\nJogo interrompido pelo teclado.")
        while len(letras_reconhecidas) < total_letras:
            letras_reconhecidas.append("-")

    # Mostrar Tela Final Visual se o frame existir
    if 'frame_final' in locals():
        from backend.src.ui import UIRenderer
        ui = UIRenderer(frame_final)
        altura, largura = frame_final.shape[:2]
        cx, cy = largura // 2, altura // 2
        
        # Fundo escuro cobrindo a tela toda
        ui.draw_rounded_rect([0, 0, largura, altura], (0, 0, 0), alpha=230, radius=0)
        
        total_tentativas = acertos + erros
        pontuacao = int((acertos / total_tentativas * 100) if total_tentativas > 0 else 0)
        
        # Caixa de resultados central
        box = [cx - 200, cy - 150, cx + 200, cy + 150]
        ui.draw_rounded_rect(box, (30, 30, 30), alpha=255, radius=15)
        ui.draw_rounded_rect_outline(box, (100, 100, 100), alpha=255, radius=15, width=2)
        
        ui.draw_text("FIM DE JOGO", (cx, cy - 100), (0, 255, 255), size=30, bold=True, anchor="mm")
        ui.draw_text("🏆", (cx, cy - 40), (255, 255, 255), size=50, anchor="mm", is_emoji=True)
        
        cor_pontuacao = (0, 255, 0) if pontuacao >= 70 else (0, 200, 255) if pontuacao >= 50 else (0, 0, 255)
        ui.draw_text(f"Pontuação: {pontuacao} / 100", (cx, cy + 30), cor_pontuacao, size=28, bold=True, anchor="mm")
        ui.draw_text(f"Acertos: {acertos}   Erros: {erros}", (cx, cy + 70), (200, 200, 200), size=16, anchor="mm")
        ui.draw_text("Pressione qualquer tecla para fechar", (cx, cy + 120), (120, 120, 120), size=14, anchor="mm")
        
        cv2.imshow(config.WEBCAM_WINDOW_NAME, ui.render())
        # Aguardar tecla do usuário indefinidamente para não fechar a tela sozinho
        cv2.waitKey(0)

    # O Finally fecha as janelas com release()
    tempo_total = time.time() - tempo_inicio
    cap.release()
    cv2.destroyAllWindows()

    # Exibir resultado final
    exibir_resultado_final(
        texto_original=texto_normalizado,
        letras_esperadas=letras_esperadas,
        letras_reconhecidas=letras_reconhecidas,
        acertos=acertos,
        erros=erros,
        puladas=puladas,
        tempo_total=tempo_total,
    )


# (O loop_analise foi movido para src/analysis_mode.py para melhor organização)


def main():
    """Ponto de entrada principal do modo interativo com webcam."""
    parser = criar_parser()
    args = parser.parse_args()

    print("=" * 50)
    print("  RECONHECIMENTO DO ALFABETO MANUAL DA LIBRAS")
    print(f"          Modo: {args.mode.upper()}")
    print("=" * 50)

    # Carregar modelo e codificador
    print(f"\nCarregando modelo de '{args.model}'...")
    print(f"Carregando codificador de rótulos de '{args.label_encoder}'...")
    modelo, label_encoder = carregar_modelo(args.model, args.label_encoder)
    print("Modelo e codificador carregados com sucesso!\n")

    if args.mode == "game":
        # Solicitar texto ao usuário
        texto_normalizado = solicitar_texto_usuario()
        print(f"\nTexto normalizado: {texto_normalizado}")
        print(f"Letras a soletrar: {' '.join(list(texto_normalizado))}")

        # Executar loop principal
        loop_principal(
            modelo=modelo,
            label_encoder=label_encoder,
            camera_index=args.camera,
            texto_normalizado=texto_normalizado,
        )
    else:
        # Executar modo de análise avançado
        try:
            from backend.src.analysis_mode import run_analysis_mode
            from datetime import datetime
        except ImportError as e:
            print(f"Erro ao carregar o módulo de análise: {e}")
            sys.exit(1)
            
        print("\n" + "-"*50)
        print("  METADADOS DA SESSÃO DE ANÁLISE")
        print("-" * 50)
        
        default_name = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session_name = input(f"Nome da sessão (padrão: {default_name}): ").strip()
        if not session_name:
            session_name = default_name
            
        condition = input("Condição (ex: boa_iluminacao) (padrão: unspecified): ").strip()
        if not condition:
            condition = "unspecified"
            
        notes = input("Notas: ").strip()
        
        metadata = {
            "session_name": session_name,
            "condition": condition,
            "notes": notes
        }
            
        run_analysis_mode(
            modelo=modelo,
            label_encoder=label_encoder,
            camera_index=args.camera,
            metadata=metadata
        )


if __name__ == "__main__":
    main()
