/**
 * services/api.js — Camada única de comunicação com a API FastAPI.
 *
 * Centraliza toda chamada HTTP/WebSocket para o back-end, espelhando os
 * endpoints de api/routes/prediction.py e api/routes/game.py. Os
 * componentes React não devem chamar fetch() diretamente — sempre passar
 * por aqui, para manter a URL base, os headers e o tratamento de erro
 * em um único lugar.
 *
 * Configuração:
 *   Crie um arquivo .env na raiz do frontend com:
 *     VITE_API_URL=http://localhost:8000
 *   (se omitido, cai no default abaixo)
 */

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000" || "https://reconhecedor-de-libras.onrender.com";

/**
 * Wrapper de fetch que já monta a URL, define JSON por padrão e lança um
 * erro com a mensagem vinda de `detail` (padrão do FastAPI/HTTPException)
 * quando a resposta não é 2xx.
 */
async function request(path, { method = "GET", body, headers, isFormData = false } = {}) {
  const response = await fetch(`${API_URL}${path}`, {
    method,
    headers: isFormData
      ? headers
      : { "Content-Type": "application/json", ...headers },
    body: isFormData ? body : body ? JSON.stringify(body) : undefined,
  });

  // 204 No Content (ex.: DELETE /game/{id}) não tem corpo para parsear
  if (response.status === 204) {
    return null;
  }

  const data = await response.json().catch(() => null);

  if (!response.ok) {
    const detail = data?.detail || `Erro ${response.status} ao chamar ${path}`;
    throw new Error(detail);
  }

  return data;
}

/**
 * Converte um frame capturado de um <video>/<canvas> em base64 (data URL),
 * pronto para ser enviado em ObserveRequest/PredictRequest.
 *
 * Uso típico:
 *   const canvas = document.createElement("canvas");
 *   canvas.width = video.videoWidth;
 *   canvas.height = video.videoHeight;
 *   canvas.getContext("2d").drawImage(video, 0, 0);
 *   const imageBase64 = canvasToBase64(canvas);
 */
export function canvasToBase64(canvas, quality = 0.8) {
  return canvas.toDataURL("image/jpeg", quality);
}

// =============================================================================
// Predição avulsa — POST /predict/*
// =============================================================================

/**
 * Envia uma imagem em base64 (ex.: gerada por canvasToBase64) para
 * predição avulsa, fora do contexto de uma partida.
 */
export function predictImageBase64(imageBase64, includeDebugImage = false) {
  return request("/predict/", {
    method: "POST",
    body: { image_base64: imageBase64, include_debug_image: includeDebugImage },
  });
}

/**
 * Envia um arquivo de imagem (File/Blob) via multipart/form-data para
 * predição avulsa — útil para telas de upload direto de arquivo.
 */
export function predictImageUpload(file, includeDebugImage = false) {
  const formData = new FormData();
  formData.append("file", file);

  const query = includeDebugImage ? "?include_debug_image=true" : "";

  return request(`/predict/upload${query}`, {
    method: "POST",
    body: formData,
    isFormData: true,
  });
}

// =============================================================================
// Jogo de soletração — REST /game/*
// =============================================================================

/** Cria uma nova partida a partir da palavra/frase digitada pelo usuário. */
export function createGame(texto) {
  return request("/game/", {
    method: "POST",
    body: { texto },
  });
}

/** Envia um frame da webcam para a sessão, sem avançar a palavra. */
export function observeFrame(sessionId, imageBase64) {
  return request(`/game/${sessionId}/observe`, {
    method: "POST",
    body: { image_base64: imageBase64 },
  });
}

/** Equivalente à tecla SPACE: confirma a tentativa com a última letra observada. */
export function confirmAttempt(sessionId) {
  return request(`/game/${sessionId}/confirm`, { method: "POST" });
}

/** Equivalente à tecla N: pula a letra atual (conta como erro). */
export function skipLetter(sessionId) {
  return request(`/game/${sessionId}/skip`, { method: "POST" });
}

/** Equivalente à tecla Q: encerra a partida antes do fim. */
export function finishGame(sessionId) {
  return request(`/game/${sessionId}/finish`, { method: "POST" });
}

/** Estado atual da partida — útil para sincronizar a UI ao montar/reconectar. */
export function getGameState(sessionId) {
  return request(`/game/${sessionId}`, { method: "GET" });
}

/** Resultado final (acertos, erros, pontuação) ao terminar a partida. */
export function getGameResult(sessionId) {
  return request(`/game/${sessionId}/result`, { method: "GET" });
}

/** Remove a sessão (ex.: usuário saiu da tela sem terminar o jogo). */
export function deleteGame(sessionId) {
  return request(`/game/${sessionId}`, { method: "DELETE" });
}

// =============================================================================
// Jogo de soletração — WebSocket /game/{id}/ws (canal em tempo real, opcional)
// =============================================================================

/**
 * Abre o WebSocket de uma partida e devolve um pequeno wrapper com métodos
 * para enviar frame/confirm/skip/finish, além de registrar os callbacks de
 * evento. Usar como alternativa às chamadas REST por frame quando quiser
 * manter um único socket aberto durante o jogo todo.
 *
 * Uso típico:
 *   const socket = connectGameSocket(sessionId, {
 *     onObservation: (obs) => setLetraReconhecida(obs.letter),
 *     onState: (state) => setGameState(state),
 *     onError: (msg) => console.error(msg),
 *   });
 *   socket.sendFrame(imageBase64);
 *   socket.confirm();
 *   socket.close();
 */
export function connectGameSocket(sessionId, { onObservation, onState, onError, onOpen, onClose } = {}) {
  const wsUrl = `${API_URL.replace(/^http/, "ws")}/game/${sessionId}/ws`;
  const socket = new WebSocket(wsUrl);

  socket.onopen = () => onOpen?.();
  socket.onclose = () => onClose?.();

  socket.onmessage = (event) => {
    let mensagem;
    try {
      mensagem = JSON.parse(event.data);
    } catch {
      onError?.("Mensagem inválida recebida do servidor.");
      return;
    }

    if (mensagem.type === "observation") {
      onObservation?.(mensagem);
    } else if (mensagem.type === "state") {
      onState?.(mensagem);
    } else if (mensagem.type === "error") {
      onError?.(mensagem.detail);
    }
  };

  const sendJson = (payload) => {
    if (socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(payload));
    }
  };

  return {
    sendFrame: (imageBase64) => sendJson({ type: "frame", image_base64: imageBase64 }),
    confirm: () => sendJson({ type: "confirm" }),
    skip: () => sendJson({ type: "skip" }),
    finish: () => sendJson({ type: "finish" }),
    close: () => socket.close(),
    raw: socket,
  };
}