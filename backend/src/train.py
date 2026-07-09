"""
train.py — Treinamento de classificadores para reconhecimento do alfabeto manual da Libras.

Treina três classificadores (KNN, SVM, Random Forest) com validação cruzada,
seleciona automaticamente o melhor e salva o modelo treinado.
"""

import argparse
import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from src import config
from src.dataset import load_dataset
from src.utils import garantir_diretorio


def main():
    """Função principal de treinamento dos classificadores."""

    # -------------------------------------------------------------------------
    # Parsing de argumentos da linha de comando
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Treinamento de classificadores para reconhecimento do alfabeto Libras."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=config.DATA_DIR,
        help="Caminho para o diretório do dataset (padrão: config.DATA_DIR)"
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default=config.MODEL_PATH,
        help="Caminho para salvar o modelo treinado (padrão: config.MODEL_PATH)"
    )
    parser.add_argument(
        "--output-label-encoder",
        type=str,
        default=config.LABEL_ENCODER_PATH,
        help="Caminho para salvar o LabelEncoder (padrão: config.LABEL_ENCODER_PATH)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("TREINAMENTO DO CLASSIFICADOR — LIBRAS ALPHABET")
    print("=" * 60)
    print(f"  Diretório de dados: {args.data}")
    print(f"  Caminho do modelo: {args.output_model}")
    print(f"  Caminho do LabelEncoder: {args.output_label_encoder}")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Carregamento do dataset
    # -------------------------------------------------------------------------
    print("\n[1/6] Carregando dataset e extraindo features...")
    X, y, image_paths = load_dataset(data_dir=args.data)

    # Verificar se há dados suficientes
    if X.size == 0:
        print("\n[ERRO] Nenhuma feature foi extraída. Verifique o dataset e tente novamente.")
        print("Abortando treinamento.")
        return

    print(f"\n[INFO] Dataset carregado: {X.shape[0]} amostras, {X.shape[1]} features")
    print(f"[INFO] Classes únicas: {len(np.unique(y))}")

    # -------------------------------------------------------------------------
    # Codificação dos rótulos
    # -------------------------------------------------------------------------
    print("\n[2/6] Codificando rótulos com LabelEncoder...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(f"[INFO] Classes mapeadas: {list(label_encoder.classes_)}")

    # -------------------------------------------------------------------------
    # Divisão treino/teste
    # -------------------------------------------------------------------------
    print("\n[3/6] Dividindo dados em treino e teste...")
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y_encoded,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y_encoded
    )
    print(f"[INFO] Treino: {X_treino.shape[0]} amostras")
    print(f"[INFO] Teste:  {X_teste.shape[0]} amostras")

    # -------------------------------------------------------------------------
    # Definição das pipelines de classificação
    # -------------------------------------------------------------------------
    print("\n[4/6] Criando pipelines de classificação...")

    pipelines = {
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(
                n_neighbors=config.KNN_N_NEIGHBORS
            ))
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel=config.SVM_KERNEL,
                C=config.SVM_C,
                gamma=config.SVM_GAMMA,
                probability=True,
                random_state=config.RANDOM_STATE
            ))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=config.RF_N_ESTIMATORS,
                max_depth=config.RF_MAX_DEPTH,
                random_state=config.RANDOM_STATE
            ))
        ]),
    }

    # -------------------------------------------------------------------------
    # Validação cruzada para seleção do melhor classificador
    # -------------------------------------------------------------------------
    print("\n[5/6] Executando validação cruzada (cv=5) nos dados de treino...")
    print("-" * 60)

    resultados = {}

    for nome, pipeline in pipelines.items():
        print(f"\n  Avaliando {nome}...")
        scores = cross_val_score(pipeline, X_treino, y_treino, cv=5, scoring="accuracy")
        media = scores.mean()
        desvio = scores.std()
        resultados[nome] = {
            "pipeline": pipeline,
            "media": media,
            "desvio": desvio,
            "scores": scores,
        }
        print(f"  {nome}: Acurácia = {media:.4f} (+/- {desvio:.4f})")
        print(f"    Scores individuais: {[f'{s:.4f}' for s in scores]}")

    # Selecionar o melhor classificador
    melhor_nome = max(resultados, key=lambda k: resultados[k]["media"])
    melhor_info = resultados[melhor_nome]
    melhor_pipeline = melhor_info["pipeline"]

    print("\n" + "-" * 60)
    print(f"[RESULTADO] Melhor classificador: {melhor_nome}")
    print(f"  Acurácia média (CV): {melhor_info['media']:.4f} (+/- {melhor_info['desvio']:.4f})")

    # -------------------------------------------------------------------------
    # Treinamento final do melhor classificador
    # -------------------------------------------------------------------------
    print(f"\n[6/6] Treinando {melhor_nome} no conjunto de treino completo...")
    melhor_pipeline.fit(X_treino, y_treino)
    print("[INFO] Treinamento concluído.")

    # Avaliar no conjunto de teste
    acuracia_teste = melhor_pipeline.score(X_teste, y_teste)
    print(f"\n[RESULTADO] Acurácia no conjunto de teste: {acuracia_teste:.4f}")

    # -------------------------------------------------------------------------
    # Salvamento do modelo e do LabelEncoder
    # -------------------------------------------------------------------------
    print("\n[INFO] Salvando modelo e LabelEncoder...")

    # Garantir que os diretórios existam
    garantir_diretorio(os.path.dirname(args.output_model))
    garantir_diretorio(os.path.dirname(args.output_label_encoder))

    joblib.dump(melhor_pipeline, args.output_model)
    print(f"[INFO] Modelo salvo em: {args.output_model}")

    joblib.dump(label_encoder, args.output_label_encoder)
    print(f"[INFO] LabelEncoder salvo em: {args.output_label_encoder}")

    # -------------------------------------------------------------------------
    # Resumo final
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RESUMO DO TREINAMENTO")
    print("=" * 60)
    print(f"  Melhor classificador: {melhor_nome}")
    print(f"  Acurácia (validação cruzada): {melhor_info['media']:.4f} (+/- {melhor_info['desvio']:.4f})")
    print(f"  Acurácia (conjunto de teste): {acuracia_teste:.4f}")
    print(f"  Total de amostras: {X.shape[0]}")
    print(f"  Features por amostra: {X.shape[1]}")
    print(f"  Número de classes: {len(label_encoder.classes_)}")
    print(f"  Modelo salvo: {args.output_model}")
    print(f"  LabelEncoder salvo: {args.output_label_encoder}")
    print("=" * 60)
    print("\nTreinamento finalizado com sucesso!")


if __name__ == "__main__":
    main()
