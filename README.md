# Sistema Interativo de Reconhecimento do Alfabeto Manual da Libras

## Descrição

Este projeto implementa um sistema de **visão computacional híbrida** para reconhecimento
das letras do alfabeto manual da **Língua Brasileira de Sinais (Libras)**. O sistema utiliza
o **MediaPipe Hands** para detecção precisa e extração de landmarks (21 pontos 3D da mão),
e em seguida calcula descritores geométricos que são alimentados em classificadores
clássicos de aprendizado de máquina.

## Justificativa Técnica

Inicialmente, este projeto testou abordagens de segmentação baseadas em cor de pele e
subtração de fundo. Contudo, essas abordagens mostraram-se instáveis na webcam devido à
semelhança de cor entre a mão, o braço e o rosto, além de grande sensibilidade à iluminação.
Dessa forma, o pipeline foi migrado para utilizar landmarks (MediaPipe) como base primária,
isolando a estrutura e a pose da mão do fundo e de outros ruídos, oferecendo uma
experiência interativa mais robusta.

## Objetivo

- **Reconhecer** automaticamente as 26 letras do alfabeto manual da Libras a partir de imagens estáticas ou vídeo em tempo real.
- **Oferecer um modo interativo (jogo)** onde o usuário digita uma palavra ou frase e tenta reproduzir cada letra com a mão na frente da webcam.
- **Avaliar e pontuar** o desempenho do usuário, exibindo estatísticas ao final de cada sessão.

## Abordagem

O sistema segue o seguinte pipeline:

```text
Imagem
   │
   ▼
MediaPipe Hands (Extração de 21 landmarks da mão)
   │
   ▼
Normalização dos landmarks (Translação e Escala)
   │
   ▼
Extração de descritores geométricos (Distâncias e relações)
   │
   ▼
Classificador clássico (SVM / KNN / Random Forest)
   │
   ▼
Letra prevista
```

## Estrutura do Projeto

```
libras-alphabet-cv/
├── README.md                         # Este arquivo
├── requirements.txt                  # Dependências do projeto
├── data/
│   ├── raw/                          # Dataset organizado por letra
│   │   ├── A/                        # Imagens da letra A
│   │   ├── B/                        # Imagens da letra B
│   │   ├── ...
│   │   └── Z/                        # Imagens da letra Z
│   ├── processed/                    # Dados processados (features extraídas)
│   └── samples/                      # Imagens de exemplo para teste
├── models/
│   ├── classifier.joblib             # Modelo treinado
│   └── label_encoder.joblib          # Codificador de rótulos
├── results/
│   └── figures/                      # Gráficos, matrizes de confusão, etc.
└── src/
    ├── __init__.py                   # Inicialização do pacote
    ├── config.py                     # Configurações globais do projeto
    ├── landmarks.py                  # Detecção e extração de landmarks via MediaPipe
    ├── features.py                   # Normalização e descritores geométricos
    ├── dataset.py                    # Carregamento do dataset
    ├── utils.py                      # Funções utilitárias
    ├── train.py                      # Treinamento do classificador
    ├── evaluate.py                   # Avaliação do modelo
    ├── predict_image.py              # Predição de imagem isolada
    └── webcam_app.py              # Modo interativo com webcam (jogo)
```

## Dependências

| Biblioteca      | Versão Mínima | Finalidade                                     |
|-----------------|---------------|-------------------------------------------------|
| opencv-python   | 4.5.0         | Processamento de imagem e captura de webcam     |
| numpy           | 1.21.0        | Operações numéricas e manipulação de arrays     |
| scikit-learn    | 1.0.0         | Classificadores e métricas de avaliação         |
| matplotlib      | 3.4.0         | Geração de gráficos e visualizações             |
| pandas          | 1.3.0         | Manipulação de dados tabulares                  |
| joblib          | 1.1.0         | Serialização de modelos e codificadores         |
| mediapipe       | 0.10.0        | Detecção e extração de landmarks da mão         |
| tqdm            | 4.62.0        | Barras de progresso no terminal                 |

## Instalação

```bash
# 1. Clonar o repositório
git clone https://github.com/seu-usuario/libras-alphabet-cv.git
cd libras-alphabet-cv

# 2. (Recomendado) Criar um ambiente virtual
python -m venv .venv
# Ativar no Windows (PowerShell):
.venv\Scripts\Activate.ps1

# 3. Instalar dependências
pip install -r requirements.txt
```

## Organização do Dataset

O dataset deve ser organizado na pasta `data/` com uma subpasta para cada letra do
alfabeto. Cada subpasta deve conter imagens da respectiva letra em formato JPG, PNG ou
similar:

```
data/
├── A/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
├── B/
│   ├── img_001.jpg
│   └── ...
├── ...
└── Z/
    ├── img_001.jpg
    └── ...
```

### Dataset Utilizado

Para o treinamento deste modelo, foi utilizado o seguinte dataset público do Kaggle:
- **[Libras Brazilian Sign Language Dataset](https://www.kaggle.com/datasets/gabrielbraga2536/libras-brazilian-sign-language?select=dataset)**

Ele deve ser baixado, extraído e organizado dentro da pasta `data/` conforme a estrutura descrita acima.

### Dicas para um bom dataset

- O foco principal é que a **mão esteja visível e nítida**.
- Fundos complexos não afetam mais a segmentação de forma drástica, mas iluminações muito extremas podem dificultar a leitura do MediaPipe.
- Capture imagens de **diferentes pessoas e tamanhos de mão**.

## Treinamento

Para treinar o classificador com o dataset organizado em `data/`:

```bash
.venv\Scripts\Activate.ps1
python -m src.train --data data
```

### Opções disponíveis

| Argumento         | Descrição                                      | Padrão      |
|-------------------|-------------------------------------------------|-------------|
| `--data`          | Caminho para o diretório do dataset             | `data`  |

O modelo treinado será salvo em `models/classifier.joblib` e o codificador de rótulos em
`models/label_encoder.joblib`.

## Avaliação

Para avaliar o modelo treinado utilizando o dataset:

```bash
.venv\Scripts\Activate.ps1
python -m src.evaluate --data data --model models/classifier.joblib
```

O módulo de avaliação gera:
- **Relatório de classificação** com precisão, recall e F1-score por letra.
- **Matriz de confusão** salva em `results/figures/`.
- **Acurácia geral** do modelo.

## Predição de Imagem Isolada

Para classificar uma imagem individual:

```bash
.venv\Scripts\Activate.ps1
python -m src.predict_image --image data/samples/teste.jpg --model models/classifier.joblib
```

### Opções disponíveis

| Argumento         | Descrição                                      | Padrão                               |
|-------------------|-------------------------------------------------|---------------------------------------|
| `--image`         | Caminho para a imagem de entrada (obrigatório)  | —                                     |
| `--model`         | Caminho para o modelo treinado                  | `models/classifier.joblib`            |
| `--label-encoder` | Caminho para o codificador de rótulos           | `models/label_encoder.joblib`         |
| `--output`        | Caminho para salvar a visualização              | `results/figures/predict_image_result.png` |
| `--show`          | Exibe a imagem em janela do OpenCV              | desativado                            |

A visualização gerada inclui os **landmarks** e conexões da mão e a **letra prevista**
com a respectiva **confiança**.

## Modo Interativo com Webcam

Este é o modo principal do sistema. O usuário digita uma palavra ou frase e o sistema
solicita que reproduza cada letra com a mão na frente da webcam:

```bash
.venv\Scripts\Activate.ps1
python -m src.webcam_app --model models/classifier.joblib
```

### Como funciona

1. O sistema pede para o usuário digitar uma palavra ou frase no terminal.
2. O texto é normalizado (maiúsculas, sem acentos, somente letras A-Z).
3. A webcam é aberta e o sistema exibe um **HUD** (Heads-Up Display) com:
   - **Letra esperada** — a próxima letra a ser reproduzida.
   - **Letra reconhecida** — o que o modelo está detectando em tempo real.
   - **Status** — AGUARDANDO, CORRETO ou INCORRETO.
   - **Pontuação** — acertos, erros e letras puladas.
   - **Progresso** — posição atual na sequência de letras.

### Controles

| Tecla   | Ação                                      |
|---------|-------------------------------------------|
| `SPACE` | Confirmar a tentativa atual               |
| `N`     | Pular a letra atual                       |
| `Q`     | Sair do jogo                              |

### Opções disponíveis

| Argumento         | Descrição                                      | Padrão                               |
|-------------------|-------------------------------------------------|---------------------------------------|
| `--model`         | Caminho para o modelo treinado                  | `models/classifier.joblib`            |
| `--label-encoder` | Caminho para o codificador de rótulos           | `models/label_encoder.joblib`         |
| `--mode`          | Define o funcionamento principal do app         | `analysis`                            |
| `--camera`        | Índice da câmera a utilizar                     | `0`                                   |

## Modo Análise (Diagnóstico e Logs)

O sistema conta com um **Modo Análise Avançado** construído especificamente para depuração científica do classificador (foco em coletar métricas e diagnosticar problemas). Ele é executado por padrão ao chamar:

```bash
python -m src.webcam_app
```
*(Para acessar o Modo Jogo, utilize `--mode game`)*

O Modo Análise não solicita texto. Ele abre um HUD (Heads-Up Display) de depuração exibindo:
- Predição Suavizada (Votação temporal dos últimos 15 frames para evitar oscilações).
- Top 3 Probabilidades matemáticas do classificador em tempo real.
- Margem de Incerteza (Top1 - Top2).

### Ferramentas de Diagnóstico

Você pode utilizar os seguintes controles no teclado para gerar dados reais de teste:

| Tecla   | Ação                                      |
|---------|-------------------------------------------|
| `S`     | **Iniciar/Parar Gravação da Sessão**. Gera um arquivo `.csv` frame a frame e um `.json` de resumo matemático na pasta `results/analysis/`. |
| `F`     | **Salvar Frame Problemático**. Captura a imagem exata do erro e um `.json` contendo as predições daquele momento. |
| `Q`     | Sair da análise                           |

### Exemplo de Resumo de Sessão (JSON)
```json
{
    "duration_seconds": 12.45,
    "valid_frames": 280,
    "most_frequent_top1": "U",
    "average_top1_confidence": 0.8923,
    "class_switches": 12,
    "temporal_stability": 0.75
}
```

### Exemplo de resultado final (Modo Jogo)

```
========================================
         RESULTADO FINAL
========================================
Palavra/frase: CASA
Sequência esperada:    C A S A
Sequência reconhecida: C A B A
Acertos: 3
Erros: 1
Puladas: 0
Tempo total: 45.2s
Pontuação final: 75 / 100
========================================
```

## Pipeline de Processamento (Landmarks)

```text
1. ENTRADA
   └─ Imagem BGR (webcam ou arquivo)

2. DETECÇÃO (MediaPipe)
   └─ Localização da mão e extração de 21 coordenadas (x,y,z)

3. NORMALIZAÇÃO
   ├─ Origem transferida para o punho (Landmark 0)
   └─ Escala normalizada pela distância do punho ao dedo médio

4. EXTRAÇÃO DE DESCRITORES GEOMÉTRICOS
   ├─ Coordenadas normalizadas dos 21 pontos
   ├─ Distâncias de cada ponta (4, 8, 12, 16, 20) até o punho
   └─ Distâncias entre pontas de dedos vizinhos

5. CLASSIFICAÇÃO
   └─ SVM / KNN / Random Forest
```

## Limitações Conhecidas

- **Dependência de landmarks**: O sistema é refém da qualidade da detecção do MediaPipe. Se a mão estiver muito fora do enquadramento ou sobreposta de forma complexa, o rastreamento falha, e o frame é ignorado.
- **Diferença entre sinais completos vs soletração**: O alfabeto Libras foca na imagem estática, mas algumas letras, especialmente em discursos fluídos, envolvem nuances gestuais que a soletração pura não capta.
- **Letras dinâmicas (J e Z)**: As letras J e Z envolvem **movimento** e não podem ser reconhecidas por uma análise de frame único. O sistema pode opcionalmente excluí-las do treinamento.
- **Variação entre usuários**: Estruturas ósseas muito discrepantes e flexibilidade dos dedos podem afetar marginalmente a performance de alguns gestos finos (como M e N).

## Melhorias Futuras

- 🎬 **Reconhecimento de letras dinâmicas**: Implementar análise temporal de landmarks (ex: LSTM ou GRU em cima dos vetores do MediaPipe) para reconhecer as letras J e Z.
- 📊 **Interface gráfica**: Criar uma interface gráfica completa (Tkinter, PyQt ou web) para facilitar o uso por pessoas não técnicas.
- 🗃️ **Banco de dados de progresso**: Armazenar o histórico de sessões do usuário para acompanhar a evolução do aprendizado ao longo do tempo.
- 🔤 **Palavras e frases**: Expandir o modo interativo para incluir um dicionário de palavras em Libras com significados e exemplos.

## Licença

Este projeto é distribuído sob a licença **MIT**. Consulte o arquivo [LICENSE](LICENSE) para mais detalhes.

```
MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
