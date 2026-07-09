import { createContext, useContext, useMemo, useState } from "react";

/**
 * GameContext — estado compartilhado entre as três telas do fluxo.
 *
 * Home chama `iniciarPartida(response)` com o NewGameResponse inteiro
 * ({ session_id, texto_original, total_letras, letra_esperada }) assim
 * que POST /game/ responde — isso já dá pra Game.jsx começar a renderizar
 * a letra-alvo e o total de letras sem esperar a primeira mensagem "state"
 * do WebSocket (que só chega após o primeiro confirm/skip/finish).
 * Ao terminar, Game grava `result` (o GameResultResponse). Results lê `result`.
 */
const GameContext = createContext(null);

const ESTADO_INICIAL = {
  texto: "",
  sessionId: null,
  totalLetras: 0,
  letraEsperada: null,
  result: null,
};

export function GameProvider({ children }) {
  const [estado, setEstado] = useState(ESTADO_INICIAL);

  const value = useMemo(
    () => ({
      ...estado,
      setResult: (result) => setEstado((atual) => ({ ...atual, result })),
      /** Recebe o NewGameResponse de POST /game/ e popula tudo de uma vez. */
      iniciarPartida: (response) =>
        setEstado((atual) => ({
          ...atual,
          texto: response.texto_original,
          sessionId: response.session_id,
          totalLetras: response.total_letras,
          letraEsperada: response.letra_esperada,
          result: null,
        })),
      /** Limpa o estado ao voltar para a Home / iniciar novo jogo. */
      resetGame: () => setEstado(ESTADO_INICIAL),
    }),
    [estado]
  );

  return <GameContext.Provider value={value}>{children}</GameContext.Provider>;
}

export function useGame() {
  const ctx = useContext(GameContext);
  if (!ctx) {
    throw new Error("useGame precisa ser usado dentro de <GameProvider>.");
  }
  return ctx;
}
