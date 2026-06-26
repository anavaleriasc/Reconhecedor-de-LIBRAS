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

## 3. Conclusão Parcial
Até o momento, o núcleo de Visão Computacional, extração de características (Feature Engineering) e o pipeline de aprendizado de máquina (SVM + Scikit-Learn) estão finalizados e extremamente robustos. O fluxo principal de jogabilidade interativa também está operacional. O projeto está preparado para a apresentação final da disciplina.
