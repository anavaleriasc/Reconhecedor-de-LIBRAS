# Sistema Interativo de Reconhecimento do Alfabeto Manual de Libras

## Descrição

Este projeto implementa um sistema de **visão computacional híbrida** para reconhecimento
das letras do alfabeto manual da **Língua Brasileira de Sinais (Libras)**. O sistema utiliza
o **MediaPipe Hands** para detecção precisa e extração de landmarks (21 pontos 3D da mão),
e em seguida calcula descritores geométricos que são alimentados em classificadores
clássicos de aprendizado de máquina.

**[Assista a demonstração de uso](https://youtu.be/tmEsHRk0ohA)**

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

## Estrutura do projeto
```
reconhecedor-de-LIBRAS/
├── README.md
├── .gitignore
│
├── frontend/                         # Cliente React que consome a API
│   ├── public/
│   ├── src/
│   │   ├── main.jsx                  # Ponto de entrada do React
│   │   ├── App.jsx                   # Rotas (Home / Game / Results)
│   │   ├── global.css                # Tokens de design e estilos base
│   │   ├── pages/
│   │   │   ├── Home.jsx              # Digitação da palavra/frase e criação da sessão
│   │   │   ├── Game.jsx              # Captura da webcam e jogo em si
│   │   │   └── Results.jsx           # Resultado final da partida
│   │   └── services/
│   │       └── api.js                # Camada única de chamadas HTTP/WebSocket à API
│   │       # HUD, Header e Card ainda serão extraídos como componentes próprios
│   ├── package.json
│   └── vite.config.js
│
└── backend/
    ├── requirements.txt              # Dependências do backend
    │
    ├── api/
    │   ├── game.py                   # Gerenciamento de sessões
    │   ├── main.py                   # Ponto de entrada da API
    │   │
    │   ├── routes/
    │   │   ├── game.py
    │   │   └── prediction.py
    │   │
    │   ├── schemas/
    │   │   ├── game.py
    │   │   └── prediction.py
    │   │
    │   └── services/
    │       ├── game_service.py
    │       └── prediction_service.py
    │
    ├── src/
    │   ├── __init__.py                   # Inicialização do pacote
    │   ├── aggregate_sessions.py         # Consolida os JSONs de resumo do Modo Análise em um CSV único
    │   ├── analysis_mode.py              # Lógica do Modo Análise (gravação, captura automática, HUD de diagnóstico)
    │   ├── config.py                     # Configurações globais do projeto
    │   ├── dataset.py                    # Carregamento do dataset
    │   ├── evaluate.py                   # Avaliação do modelo
    │   ├── features.py                   # Normalização e descritores geométricos
    │   ├── landmarks.py                  # Detecção e extração de landmarks via MediaPipe
    │   ├── plot_logs.py                  # Gráficos de linha do tempo e frequência a partir dos logs CSV
    │   ├── plot_worst_classes.py         # Ranking das 10 piores classes por F1-score
    │   ├── predict_image.py              # Predição de uma imagem isolada (CLI)
    │   ├── prediction.py                 # Módulo stateless de predição usado exclusivamente pela API
    │   ├── webcam_app.py                 # Modo interativo com webcam (jogo e análise), antigo realtime_game.py
    │   ├── train.py                      # Treinamento do classificador
    │   ├── ui.py                         # Renderização do OpenCV
    │   └── utils.py                      # Funções utilitárias
    │
    ├── models/
    │   ├── classifier.joblib             # Modelo treinado
    │   ├── label_encoder.joblib          # Codificador de rótulos
    │   └── hand_landmarker.task          # Modelo do MediaPipe Hands
    │
    ├── data/
    │   ├── raw/                          # Dataset organizado por letra
    │   │   ├── A/                        # Imagens da letra A
    │   │   ├── B/
    │   │   ├── ...
    │   │   └── Z/                        # Imagens da letra Z
    │   ├── processed/                    # Dados processados (features extraídas)
    │   └── samples/                      # Imagens de exemplo para teste
    │
    └── results/
        ├── figures/                      # Gráficos, matrizes de confusão, etc. (gerados pelo evaluate.py)
        ├── classification_report.csv     # Métricas por classe, geradas pelo evaluate.py
        ├── worst_f1_classes.png          # Top 10 piores classes por F1-score (plot_worst_classes.py)
        └── analysis/
            ├── logs/                     # Um .csv por sessão gravada, frame a frame
            ├── summaries/                # Um .json de resumo por sessão + os .png gerados por plot_logs.py
            ├── frames/
            │   └── <sessão>/             # Capturas automáticas (a cada 2s) e manuais (tecla F)
            └── session_summary.csv       # Consolidado de todas as sessões (aggregate_sessions.py)
```

> **Nota sobre `prediction.py`:** ele existe à parte de `webcam_app.py` e `analysis_mode.py` porque estes dois dependem de janela do OpenCV, teclado e loop bloqueante — nada disso faz sentido dentro de uma requisição HTTP. `prediction.py` reaproveita os mesmos módulos de baixo nível (`src.landmarks`, `src.features`) e replica a mesma máquina de estados do jogo (`GameSession`), mas devolve dicionários/objetos Python simples em vez de desenhar na tela. É o único módulo de `src/` importado pela `api/`.

> **Nota Importante:** As pastas `models/` e `results/` (dentro de `backend/`) deste repositório já vêm acompanhadas de um **modelo previamente treinado** (junto com os relatórios e gráficos técnicos gerados durante o seu treinamento) e também contêm **resultados e evidências de experimentos** reais feitos através do Modo de Análise. Dessa forma, você pode iniciar testes com a câmera imediatamente, sem a necessidade de baixar o dataset para treinar o classificador do zero.

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
git clone https://github.com/seu-usuario/reconhecedor-de-LIBRAS.git
cd reconhecedor-de-LIBRAS

# 2. (Recomendado) Criar um ambiente virtual
python -m venv .venv
# Ativar no Windows (PowerShell):
.venv\Scripts\Activate.ps1

# 3. Entrar na pasta do backend e instalar as dependências
cd backend
pip install -r requirements.txt
```

> Todo o código Python (treinamento, avaliação, modo webcam, API) vive em `backend/`. Antes de rodar qualquer comando das seções abaixo, entre nessa pasta (`cd backend`) — a partir daí, todos os comandos e caminhos usados neste README (`python -m src.xxx`, `data/`, `models/`, `results/`) são relativos a `backend/`.

## Organização do Dataset

O dataset deve ser organizado na pasta `data/` (dentro de `backend/`) com uma subpasta para cada letra do
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

Ele deve ser baixado, extraído e organizado dentro da pasta `data/` (em `backend/`) conforme a estrutura descrita acima.

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
| `--mode`          | Define o funcionamento principal do app: `game` ou `analysis` | `analysis`             |
| `--camera`        | Índice da câmera a utilizar                     | `0`                                   |

> Não existem flags de linha de comando para o texto do jogo nem para os metadados do Modo Análise — ambos são pedidos interativamente no terminal depois que o modelo é carregado (`input()`), antes da webcam abrir.

## Modo Análise (Diagnóstico e Logs)

O sistema conta com um **Modo Análise Avançado** construído especificamente para depuração científica do classificador (foco em coletar métricas e diagnosticar problemas). Ele é executado por padrão ao chamar:

```bash
python -m src.webcam_app
```
*(Para acessar o Modo Jogo, utilize `--mode game`)*

O Modo Análise não pede um texto para soletrar. Antes de abrir a webcam, ele pede três metadados no terminal (todos opcionais — se você só der Enter, os padrões são usados):

```
Nome da sessão (padrão: analysis_20260708_143000):
Condição (ex: boa_iluminacao) (padrão: unspecified):
Notas:
```

Esses três valores (`session_name`, `condition`, `notes`) vão junto no `.json` de resumo de cada sessão, para ajudar a identificar as condições de cada teste depois.

Uma vez com a webcam aberta, o HUD (Heads-Up Display) de depuração exibe:
- Predição Suavizada (Votação temporal dos últimos 15 frames para evitar oscilações).
- Top 3 Probabilidades matemáticas do classificador em tempo real.
- Margem de Incerteza (Top1 - Top2).

### Ferramentas de Diagnóstico

Você pode utilizar os seguintes controles no teclado para gerar dados reais de teste:

| Tecla   | Ação                                      |
|---------|-------------------------------------------|
| `S`     | **Iniciar/Parar Gravação da Sessão**. Cada gravação tem um alvo fixo de **10 segundos** e para sozinha ao atingir esse tempo (dá para parar antes apertando `S` de novo). Gera um `.csv` frame a frame em `results/analysis/logs/` e um `.json` de resumo em `results/analysis/summaries/`. |
| `N`     | **Nova Sessão**. Encerra a gravação atual (se houver) e abre uma janela para digitar o nome da próxima sessão, reiniciando todos os contadores. |
| `F`     | **Salvar Captura Manual**. A qualquer momento (gravando ou não), salva a imagem original, a imagem com landmarks e um `.json` com a predição daquele instante em `results/analysis/frames/<sessão>/`. |
| `Q`     | Sair da análise                           |

Além da captura manual (`F`), **enquanto a gravação está ativa o sistema também salva sozinho** uma captura automática nos instantes 2s, 4s, 6s, 8s e 10s da sessão — sem precisar apertar nada. Isso garante uma amostra mínima de imagens mesmo que você esqueça de usar `F`.

### Exemplo de Resumo de Sessão (JSON)

O `.json` salvo em `results/analysis/summaries/` tem mais campos do que aparecem no HUD — por exemplo:

```json
{
    "session_name": "boa_iluminacao_01",
    "condition": "boa_iluminacao",
    "notes": "Teste com luz natural, letra U",
    "duration_seconds": 10.02,
    "session_duration_target": 10.0,
    "completed_full_duration": true,
    "total_frames": 302,
    "valid_frames": 280,
    "hand_detection_rate": 0.9272,
    "most_frequent_top1": "U",
    "top3_frequent_classes": [["U", 210], ["V", 45], ["R", 25]],
    "average_top1_confidence": 0.8923,
    "average_margin": 0.412,
    "class_switches": 12,
    "temporal_stability": 0.75,
    "auto_captures_target": 5,
    "auto_capture_times_seconds": [2.0, 4.0, 6.0, 8.0, 10.0],
    "auto_captures_saved": 5
}
```

> Se a gravação for interrompida antes de completar os 10s (`completed_full_duration: false`), o resumo também ganha o campo `interruption_reason`.

> **Atenção ao consolidar:** o `aggregate_sessions.py` (próxima seção) só copia um subconjunto fixo dessas colunas para o CSV final — `top3_frequent_classes`, `auto_captures_saved` e `interruption_reason`, por exemplo, ficam de fora do consolidado e só existem no `.json` de cada sessão individual.

### Comandos de Relatório e Gráficos

Para facilitar a documentação científica do seu TCC, criamos 4 scripts que processam os dados gerados pelo Modo Análise:

1. **Gráficos da Sessão (Linhas e Barras)**
   Para cada CSV em `results/analysis/logs/` (ou um específico via `--log caminho.csv`), gera dois PNGs dentro de `results/analysis/summaries/`: `<sessão>_plot.png` (probabilidades Top-3, margem de incerteza e estabilidade de classe ao longo do tempo) e `<sessão>_class_frequency.png` (barras com a frequência de cada letra detectada).
   ```bash
   python -m src.plot_logs
   ```
2. **Consolidação de Sessões**
   Varre todos os JSONs de `results/analysis/summaries/` e agrupa um subconjunto das métricas (duração, detecção de mão, trocas de classe, estabilidade etc.) em `results/analysis/session_summary.csv`. Campos mais detalhados de cada sessão (como `top3_frequent_classes` ou `auto_captures_saved`) permanecem só nos JSONs individuais.
   ```bash
   python -m src.aggregate_sessions
   ```
3. **Calcanhar de Aquiles (Piores Classes)**
   Lê `results/classification_report.csv` (gerado pelo `evaluate.py`) e desenha, em `results/worst_f1_classes.png`, um gráfico ranqueando as 10 letras com menor F1-Score.
   ```bash
   python -m src.plot_worst_classes
   ```

### Exemplo de resultado final (Modo Jogo)

Ao terminar (ou sair de) uma partida, o terminal imprime:

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
Pontuação final: 75.00%
========================================
```

Ao mesmo tempo, a janela da webcam mostra uma tela de "FIM DE JOGO" com a pontuação (0 a 100), acertos/erros e um troféu, que fica visível até que qualquer tecla seja pressionada.

## API (FastAPI)

A API expõe o reconhecimento e a jogabilidade para o `frontend/` via HTTP/WebSocket, sem depender de OpenCV/janelas. Ela é construída em cima de `src/prediction.py` (veja a nota na Estrutura do Projeto acima).

```bash
# a partir de backend/ (cd backend), com src/ e api/ como pacotes irmãos
uvicorn api.main:app --reload
```

Documentação interativa em `http://localhost:8000/docs`. Principais endpoints:

| Rota | Descrição |
|------|-----------|
| `POST /predict/` e `/predict/upload` | Predição avulsa de uma imagem (base64 ou upload), fora do contexto de uma partida |
| `POST /game/` | Cria uma partida a partir de uma palavra/frase, devolve `session_id` |
| `POST /game/{id}/observe` | Processa um frame da webcam sem avançar a palavra |
| `POST /game/{id}/confirm` `/skip` `/finish` | Equivalentes às teclas `SPACE`/`N`/`Q` do jogo local |
| `GET /game/{id}` e `/game/{id}/result` | Estado atual e resultado final da partida |
| `WS /game/{id}/ws` | Canal único para o mesmo fluxo acima via mensagens JSON |

### `prediction.py` ainda é necessário?

**Sim, e não precisou de nenhuma mudança de lógica** depois da renomeação de `realtime_game.py` para `webcam_app.py` — só corrigi os comentários que ainda citavam o nome antigo. A razão de ele continuar existindo separado de `webcam_app.py`/`analysis_mode.py`:

- Os três módulos fazem a **mesma sequência de inferência** (`extract_hand_landmarks` → `normalize_landmarks` → `extract_features_from_landmarks` → `modelo.predict`), importando das mesmas fontes (`src.landmarks`, `src.features`). Se você alterar essa lógica em `src/features.py` ou `src/landmarks.py`, os três se beneficiam automaticamente, sem precisar tocar em `prediction.py`.
- O que **não é compartilhado** é a "cola" em volta disso — como `webcam_app.py` lê teclado e desenha na tela, e `prediction.py` só devolve dicionários — por isso essa parte está duplicada de propósito entre os três arquivos. A fórmula de pontuação (`acertos / (acertos + erros) * 100`) é idêntica nos dois lugares, então o resultado de uma partida via API bate com o resultado de uma partida local.
- Se no futuro vocês mudarem a fórmula de pontuação, o formato do HUD ou as regras do jogo (o que conta como erro, por exemplo) diretamente em `webcam_app.py`, será preciso replicar manualmente a mudança na classe `GameSession` de `prediction.py` — não há um único lugar fazendo isso hoje.

## Frontend

```bash
cd frontend
npm install
npm run dev
```

O front (Vite + React) espera a API rodando em `http://localhost:8000` por padrão — configurável via `frontend/.env` (`VITE_API_URL`). As telas `Home`, `Game` e `Results` já consomem `src/services/api.js` de ponta a ponta; os componentes `HUD`, `Header` e `Card` (pontuação, entrada etc.) ainda serão extraídos numa próxima etapa.

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

- **Reconhecimento de letras dinâmicas**: Implementar análise temporal de landmarks (ex: LSTM ou GRU em cima dos vetores do MediaPipe) para reconhecer as letras J e Z.
- **Interface gráfica**: Criar uma interface gráfica completa (Tkinter, PyQt ou web) para facilitar o uso por pessoas não técnicas.
- **Banco de dados de progresso**: Armazenar o histórico de sessões do usuário para acompanhar a evolução do aprendizado ao longo do tempo.
- **Palavras e frases**: Expandir o modo interativo para incluir um dicionário de palavras em Libras com significados e exemplos.
- **Componentização do frontend**: Extrair `HUD`, `Header` e `Card` (pontuação, entrada de texto) das telas `Home`/`Game`/`Results` para componentes reutilizáveis.

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
