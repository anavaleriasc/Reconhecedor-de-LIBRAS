"""
evaluate.py — Avaliação do modelo treinado para reconhecimento do alfabeto manual da Libras.

Gera relatório de classificação, matriz de confusão, métricas em JSON
e identifica as letras mais confundidas pelo classificador.
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Backend não-interativo
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split

from src import config
from src.dataset import load_dataset
from src.utils import garantir_diretorio


def main():
    """Função principal de avaliação do modelo treinado."""

    # -------------------------------------------------------------------------
    # Parsing de argumentos da linha de comando
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Avaliação do modelo treinado para reconhecimento do alfabeto Libras."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=config.DATA_DIR,
        help="Caminho para o diretório do dataset (padrão: config.DATA_DIR)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=config.MODEL_PATH,
        help="Caminho para o modelo treinado (padrão: config.MODEL_PATH)"
    )
    parser.add_argument(
        "--label-encoder",
        type=str,
        default=config.LABEL_ENCODER_PATH,
        help="Caminho para o LabelEncoder (padrão: config.LABEL_ENCODER_PATH)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=config.RESULTS_DIR,
        help="Diretório para salvar os resultados (padrão: config.RESULTS_DIR)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("AVALIAÇÃO DO MODELO — LIBRAS ALPHABET")
    print("=" * 60)
    print(f"  Diretório de dados: {args.data}")
    print(f"  Modelo: {args.model}")
    print(f"  LabelEncoder: {args.label_encoder}")
    print(f"  Diretório de resultados: {args.output_dir}")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Carregamento do modelo e do LabelEncoder
    # -------------------------------------------------------------------------
    print("\n[1/6] Carregando modelo e LabelEncoder...")

    if not os.path.isfile(args.model):
        print(f"[ERRO] Arquivo de modelo não encontrado: {args.model}")
        print("Execute o treinamento primeiro: python -m src.train")
        return

    if not os.path.isfile(args.label_encoder):
        print(f"[ERRO] Arquivo do LabelEncoder não encontrado: {args.label_encoder}")
        print("Execute o treinamento primeiro: python -m src.train")
        return

    modelo = joblib.load(args.model)
    label_encoder = joblib.load(args.label_encoder)
    print(f"[INFO] Modelo carregado: {type(modelo).__name__}")
    print(f"[INFO] Classes do LabelEncoder: {list(label_encoder.classes_)}")

    # -------------------------------------------------------------------------
    # Carregamento do dataset
    # -------------------------------------------------------------------------
    print("\n[2/6] Carregando dataset e extraindo features...")
    X, y, image_paths = load_dataset(data_dir=args.data)

    if X.size == 0:
        print("\n[ERRO] Nenhuma feature foi extraída. Verifique o dataset e tente novamente.")
        print("Abortando avaliação.")
        return

    # Codificar rótulos com o mesmo LabelEncoder do treinamento
    y_encoded = label_encoder.transform(y)

    # -------------------------------------------------------------------------
    # Divisão treino/teste (mesma divisão do treinamento)
    # -------------------------------------------------------------------------
    print("\n[3/6] Dividindo dados em treino e teste (mesma divisão do treinamento)...")
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y_encoded,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y_encoded
    )
    print(f"[INFO] Amostras de teste: {X_teste.shape[0]}")

    # -------------------------------------------------------------------------
    # Predição no conjunto de teste
    # -------------------------------------------------------------------------
    print("\n[4/6] Realizando predições no conjunto de teste...")
    y_pred = modelo.predict(X_teste)

    # Converter índices para nomes das classes
    nomes_classes = label_encoder.classes_
    y_teste_letras = label_encoder.inverse_transform(y_teste)
    y_pred_letras = label_encoder.inverse_transform(y_pred)

    # -------------------------------------------------------------------------
    # Cálculo das métricas
    # -------------------------------------------------------------------------
    print("\n[5/6] Calculando métricas de avaliação...")

    acuracia = accuracy_score(y_teste, y_pred)
    precisao_macro = precision_score(y_teste, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_teste, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_teste, y_pred, average="macro", zero_division=0)

    print(f"\n  Acurácia:         {acuracia:.4f}")
    print(f"  Precisão (macro): {precisao_macro:.4f}")
    print(f"  Recall (macro):   {recall_macro:.4f}")
    print(f"  F1-Score (macro): {f1_macro:.4f}")

    # Relatório de classificação completo
    relatorio_texto = classification_report(
        y_teste_letras, y_pred_letras,
        target_names=nomes_classes,
        zero_division=0
    )
    print(f"\n{'=' * 60}")
    print("RELATÓRIO DE CLASSIFICAÇÃO")
    print("=" * 60)
    print(relatorio_texto)

    # -------------------------------------------------------------------------
    # Salvamento dos resultados
    # -------------------------------------------------------------------------
    print("\n[6/6] Salvando resultados...")
    garantir_diretorio(args.output_dir)

    # --- Matriz de confusão (PNG) ---
    caminho_matriz = os.path.join(args.output_dir, "confusion_matrix.png")

    cm = confusion_matrix(y_teste_letras, y_pred_letras, labels=nomes_classes)
    num_classes = len(nomes_classes)

    # Ajustar tamanho da figura com base no número de classes
    tamanho_fig = max(10, num_classes * 0.6)
    fig, ax = plt.subplots(figsize=(tamanho_fig, tamanho_fig))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=nomes_classes)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")

    ax.set_title("Matriz de Confusão — Alfabeto Libras", fontsize=14, fontweight="bold")
    ax.set_xlabel("Classe Predita", fontsize=12)
    ax.set_ylabel("Classe Verdadeira", fontsize=12)

    # Rotacionar labels para melhor legibilidade
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()

    fig.savefig(caminho_matriz, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Matriz de confusão salva em: {caminho_matriz}")

    # --- Relatório de classificação (CSV) ---
    caminho_relatorio = os.path.join(args.output_dir, "classification_report.csv")

    relatorio_dict = classification_report(
        y_teste_letras, y_pred_letras,
        target_names=nomes_classes,
        output_dict=True,
        zero_division=0
    )
    df_relatorio = pd.DataFrame(relatorio_dict).transpose()
    df_relatorio.to_csv(caminho_relatorio, float_format="%.4f")
    print(f"[INFO] Relatório de classificação salvo em: {caminho_relatorio}")

    # --- Métricas principais (JSON) ---
    caminho_metricas = os.path.join(args.output_dir, "metrics.json")

    metricas = {
        "accuracy": round(float(acuracia), 4),
        "precision_macro": round(float(precisao_macro), 4),
        "recall_macro": round(float(recall_macro), 4),
        "f1_macro": round(float(f1_macro), 4),
        "num_classes": int(num_classes),
        "num_test_samples": int(len(y_teste)),
    }

    with open(caminho_metricas, "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Métricas salvas em: {caminho_metricas}")

    # -------------------------------------------------------------------------
    # Identificação das 5 letras mais confundidas
    # -------------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("TOP 5 — LETRAS MAIS CONFUNDIDAS")
    print("=" * 60)

    # Zerar a diagonal (acertos) para encontrar confusões
    cm_confusoes = cm.copy().astype(float)
    np.fill_diagonal(cm_confusoes, 0)

    # Encontrar os pares com mais confusões
    confusoes = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm_confusoes[i, j] > 0:
                confusoes.append({
                    "verdadeira": nomes_classes[i],
                    "predita": nomes_classes[j],
                    "quantidade": int(cm_confusoes[i, j]),
                })

    # Ordenar por quantidade de confusões (decrescente)
    confusoes.sort(key=lambda x: x["quantidade"], reverse=True)

    if confusoes:
        top_5 = confusoes[:5]
        for idx, conf in enumerate(top_5, 1):
            print(
                f"  {idx}. '{conf['verdadeira']}' confundida com "
                f"'{conf['predita']}': {conf['quantidade']} vezes"
            )
    else:
        print("  Nenhuma confusão encontrada! Classificação perfeita.")

    # -------------------------------------------------------------------------
    # Resumo final
    # -------------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("RESUMO DA AVALIAÇÃO")
    print("=" * 60)
    print(f"  Acurácia:         {acuracia:.4f}")
    print(f"  Precisão (macro): {precisao_macro:.4f}")
    print(f"  Recall (macro):   {recall_macro:.4f}")
    print(f"  F1-Score (macro): {f1_macro:.4f}")
    print(f"  Número de classes: {num_classes}")
    print(f"  Amostras de teste: {len(y_teste)}")
    print(f"\n  Arquivos gerados:")
    print(f"    - {caminho_matriz}")
    print(f"    - {caminho_relatorio}")
    print(f"    - {caminho_metricas}")
    print("=" * 60)
    print("\nAvaliação finalizada com sucesso!")


if __name__ == "__main__":
    main()
