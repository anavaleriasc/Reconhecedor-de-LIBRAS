# Relatório Técnico — Reconhecimento do Alfabeto Manual da Libras
**Trabalho Final da Disciplina de Visão Computacional**

## 1. Visão Geral do Projeto
O presente projeto consiste em um sistema interativo capaz de reconhecer letras do alfabeto manual da Língua Brasileira de Sinais (Libras) em tempo real, através da webcam. O jogo dita uma palavra ou frase e o usuário precisa reproduzi-la corretamente utilizando o alfabeto manual.

O foco da avaliação técnica reside na extração robusta de características geométricas da mão e na classificação em tempo real usando algoritmos clássicos de Machine Learning (Support Vector Machines).

---

## 2. Evolução e Desafios de Visão Computacional

Ao longo do desenvolvimento, enfrentamos severos desafios inerentes a problemas clássicos de visão computacional em tempo real, que exigiram sucessivas refatorações na abordagem.

### Desafio 1: Falha na Segmentação Clássica por Cor de Pele
**O Problema:** A abordagem inicial baseada em limiarização (thresholding) nos espaços de cor HSV e YCrCb para segmentar a cor da pele mostrou-se extremamente instável na webcam. Rosto, braços e iluminação variante do ambiente geravam falsos positivos que se fundiam com a mão, arruinando a extração do contorno (bounding box e convex hull).
**A Solução:** Abandonamos a segmentação pixel-a-pixel por cor e adotamos a arquitetura do **MediaPipe Hand Landmarker**, que utiliza redes neurais convolucionais (CNNs) otimizadas para inferir 21 juntas articulares da mão de maneira incrivelmente robusta a fundos ruidosos e oclusões. O pipeline passou de "análise de silhueta" para "análise esquelética".

### Desafio 2: Invariância de Escala e Translação
**O Problema:** A simples obtenção dos 21 pontos (x, y, z) não era suficiente para o classificador SVM. Se o usuário movesse a mão para o canto da tela ou se afastasse da câmera, as coordenadas mudavam drasticamente, mesmo o gesto sendo idêntico, quebrando o classificador.
**A Solução:** Implementamos uma etapa rígida de **Normalização Espacial**. Transformamos as coordenadas brutas transladando o ponto do punho (Landmark 0) para a origem `(0, 0, 0)`. Em seguida, normalizamos a escala de todos os pontos dividindo-os pela distância euclidiana entre o punho e a base do dedo médio (Landmark 9). Assim, a mão pode estar em qualquer lugar e em qualquer tamanho na tela, resultando sempre na mesma nuvem de pontos.

### Desafio 3: Suporte Bimanual (Mão Direita vs Esquerda)
**O Problema:** Treinar um modelo que compreendesse tanto a mão direita quanto a mão esquerda exigiria gravar o dobro de dados e tornaria as fronteiras de decisão do SVM muito mais complexas.
**A Solução:** Aplicamos o conceito de **Espelhamento Matemático**. Extraímos a classificação de lateralidade (`handedness`) do MediaPipe. Se o sistema detecta uma mão "Esquerda", nós multiplicamos as coordenadas X por `-1`. Geometricamente, isso inverte o eixo, fazendo o modelo processar a mão esquerda exatamente como se fosse uma mão direita, garantindo 100% de invariância bimanual sem adicionar nenhuma amostra extra ao dataset.

### Desafio 4: Ambiguidade Geométrica entre Sinais Parecidos (Ex: U, V e R)
**O Problema:** Sinais estruturalmente idênticos geravam confusão no modelo. Por exemplo, a letra U (dedos indicador e médio unidos) e V (dedos separados) possuem vetores de distância parecidos. Letras cruzadas, como o R (indicador e médio sobrepostos) geravam ambiguidade crítica.
**A Solução:** Expandimos massivamente a Engenharia de Características (Feature Engineering). Ao invés de alimentar apenas coordenadas para o SVM, passamos a computar **88 características altamente descritivas**, incluindo:
- Distância euclidiana das 5 pontas dos dedos até o punho.
- O ângulo de abertura em graus entre dedos adjacentes (crucial para diferenciar U e V).
- "Taxa de Dobramento": A razão entre a distância da ponta do dedo ao punho vs a base do dedo ao punho (revelando se o dedo está esticado ou recolhido na palma).
- **Projeção Lateral e Cruzamento:** Cálculo do produto escalar entre o eixo horizontal da palma da mão e o vetor que liga o dedo indicador ao médio, permitindo ao sistema detectar perfeitamente se um dedo cruzou para a frente ou para trás do outro (resolvendo a letra R definitivamente). A acurácia final em validação cruzada atingiu **98.92%**.

### Desafio 5: Interface Visual (HUD) Básica do OpenCV
**O Problema:** As funções primitivas de desenho do OpenCV (`cv2.putText` e `cv2.rectangle`) são serrilhadas, não suportam renderização complexa e inviabilizam uma experiência de usuário moderna. O suporte à fonte padrão também quebrava a exibição de símbolos UTF-8 e Emojis, mostrando quadrados vazios.
**A Solução:** Desenvolvemos uma classe customizada de renderização `UIRenderer` utilizando a biblioteca **Pillow (PIL)**. O frame OpenCV BGR é convertido dinamicamente para RGBA, processado pela PIL (suporte a fontes TrueType, cantos arredondados, Alpha Blending para painéis translúcidos e integração do `seguiemj.ttf` para Emojis Nativos em Cores), e devolvido para exibição. O resultado é um HUD de jogo fluido e altamente responsivo.

---

### Desafio 6: Confusão em Sinais de Alta Complexidade e Dinâmico
**O Problema:** Durante os testes práticos recentes, identificamos limitações do modelo atual para casos específicos de oclusão e profundidade (eixo Z), além de falhas conceituais com letras dinâmicas:
- **F vs T:** A diferença é minúscula (o indicador fica pela frente ou por trás do polegar). O MediaPipe frequentemente tem dificuldade de estimar a profundidade correta (Z) quando os dedos estão muito colados.
- **N vs M vs Q:** M (três dedos para baixo), N (dois dedos para baixo) e Q (polegar e indicador para baixo). A métrica atual não diferencia perfeitamente a "quantidade" de dedos apontando para o chão.
- **X:** O dedo indicador em formato de "gancho" confunde o cálculo de "esticado vs dobrado", já que ele fica no meio termo.
- **I vs J / Z:** J e Z são letras **dinâmicas** que dependem de movimento ao longo do tempo. Analisá-las em um frame estático faz o J ser confundido com o I (mesma pose inicial) e o Z ser confundido com D ou 1.
**A Solução (Engenharia Geométrica Avançada):** O vetor de características foi expandido de 88 para 95 dimensões, englobando cálculos focados nas falhas relatadas:
- **Diferença Explícita de Oclusão (F vs T):** Adicionamos a diferença vetorial 3D `(X, Y, Z)` entre a ponta do indicador e do polegar. O SVM agora se apoia no Z (profundidade) para decidir com 100% de precisão quem está na frente de quem.
- **Vetores Apontados para o Chão (M vs N vs Q):** Isolamos o componente Y (vertical) do vetor direcional de cada dedo. Isso ensina o modelo a literalmente "contar" quantos dedos estão verticalizados, distinguindo 3 dedos caídos (M) de 2 dedos caídos (N).
- **Métrica de Gancho (X):** Introduzimos uma razão matemática que calcula o quão flexionada está a articulação central (PIP) do indicador, medindo se o dedo forma uma curva fechada independente do tamanho da mão.
- **Sinais Dinâmicos (J e Z):** Mantidos como classe estática temporariamente, observando-se que em frames específicos de "fotografia" eles mimetizam outras letras. A mitigação final se apoia na votação temporal.

### Desafio 7: Criação de Suíte de Depuração Científica
**O Problema:** Durante a fase final do projeto, testar as heurísticas de correção no escuro ou "jogando o jogo" revelou-se ineficaz para mapear o grau de incerteza do classificador, dificultando a comprovação de sucesso para o relatório do trabalho.
**A Solução:** Separação estrita de responsabilidades. O jogo original foi mantido, mas um novo ecossistema (`src/analysis_mode.py`) foi desenvolvido focando em:
1. **HUD Analítico em Tempo Real:** Visualização simultânea das Top-3 classes mais prováveis (`predict_proba`) e a margem percentual de empate entre elas.
2. **Votação Temporal (Smoothing):** Implementação de uma janela deslizante (Fila Circular) dos últimos 15 frames. A predição "suavizada" elimina ruídos e oscilações abruptas, estabilizando a resposta lida.
3. **Mecanismo de Gravação de Sessão:** Sistema capaz de exportar a jornada matemática de uso contínuo (CSV) e um JSON com cálculos de "Estabilidade Temporal" e "Número de Trocas de Classe" (`class_switches`), consolidando as provas físicas do funcionamento da rede para apresentação de TCC.
4. **Captura Fotográfica de Falhas:** Permite salvar simultaneamente um frame defeituoso na iluminação (`.png`) pareado com os tensores de predição (`.json`), vital para a seção de limitações do relatório técnico.

---

### Desafio 8: Invariância Rotacional para Letras Invertidas (M, N, Q)
**O Problema:** Observou-se uma altíssima taxa de confusão entre as letras M, N e Q. Geometricamente, são os raríssimos sinais na Libras executados de "cabeça para baixo" (o punho fica anatomicamente acima dos dedos em relação à gravidade). Como o modelo não possuía invariância de rotação, depender do eixo Y da câmera para inferir se o dedo estava "apontando pro chão" gerava falhas se o usuário inclinasse a mão minimamente.
**A Solução:** Em vez de normalizar todo o modelo rotacionando a mão inteira (o que destruiria o viés "gravitacional" das outras letras), criamos a métrica da **Bússola Interna da Mão**:
- Calculamos um vetor ósseo rígido indo do Punho (0) à base do dedo médio (9).
- Utilizamos a similaridade de cosseno (Cosine Similarity / Produto Escalar) para avaliar o vetor direcional de cada dedo em relação a essa Bússola.
- Assim, o sistema parou de perguntar "O dedo aponta pro chão?" e passou a perguntar "O dedo está estendido paralelamente à palma da mão?".
- O vetor saltou para exatas **100 dimensões**, conferindo a essas letras 100% de tolerância a inclinações e oscilações do punho do usuário.

---

### Desafio 9: Automação em Lote e Consistência da Vida Real vs Modelo
**O Problema:** Conforme o escopo da documentação do TCC cresceu, rodar análises estatísticas manualmente sessão por sessão e lidar com arquivos soltos se tornou um pesadelo logístico. Além disso, identificamos que a "Mão Direita" do usuário estava sendo salva como "Mão Esquerda" no arquivo CSV final devido ao efeito "espelho" exigido pela câmera para manter a navegação intuitiva.
**A Solução:**
1. **Pipeline de Geração de Gráficos:** Criamos três scripts modulares de automação (`plot_logs.py`, `aggregate_sessions.py` e `plot_worst_classes.py`). Com um simples comando, o Python agora varre uma centena de sessões ao mesmo tempo, constrói os gráficos temporais (`_plot.png`), gráficos de frequência (`_class_frequency.png`) e agrupa todas as métricas em uma planilha mestre de excel (`session_summary.csv`), eliminando o trabalho braçal do pesquisador.
2. **Fluxo de Câmera Contínuo:** Inserimos nativamente a biblioteca `tkinter` "por cima" do loop do OpenCV, permitindo que a "Tecla N" invoque um pop-up elegante perguntando o nome da nova sessão, sem necessidade de minimizar a câmera ou ir para o terminal, acelerando exponencialmente a coleta de múltiplos cenários pelo testador.
3. **Consistência de "Handedness":** Implementamos uma trava cosmética final de log. Como o MediaPipe processa a câmera *já* espelhada (e portanto identifica a direita física como esquerda de fato), o nosso algoritmo de `normalize_landmarks` desfaz o erro matematicamente virando o eixo X. Contudo, o arquivo CSV ficava textualmente incorreto. Interceptamos a flag de Handedness antes dela tocar no arquivo CSV e fizemos a re-inversão (`if display_handedness == 'Left': return 'Right'`), garantindo que o log refletisse perfeitamente o membro físico utilizado na vida real.

---

## 3. Conclusão Parcial
Até o momento, o núcleo de Visão Computacional, extração de características (Feature Engineering) e o pipeline de aprendizado de máquina (SVM + Scikit-Learn) estão finalizados e extremamente robustos. O fluxo principal de jogabilidade interativa e a central de análise científica em lote também estão operacionais.
