import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { createGame } from "../services/api";
import { useGame } from "../context/GameContext";
import Button from "../components/Button";

export default function Home() {
  const [valor, setValor] = useState("");
  const [modo, setModo] = useState(null); // "normal" | "didatico"
  const [carregando, setCarregando] = useState(false);
  const [erro, setErro] = useState(null);

  const navigate = useNavigate();
  const { iniciarPartida, resetGame } = useGame();

  async function handleSubmit(event) {
    event.preventDefault();

    const texto = valor.trim();
    if (!texto) return;

    setCarregando(true);
    setErro(null);

    try {
      resetGame();

      const partida = await createGame(texto);

      iniciarPartida(partida);

      navigate(modo === "didatico" ? "/gameDidatico" : "/game");
    } catch (err) {
      setErro(
        err.message || "Putz não deu, tenta ai de novo."
      );
    } finally {
      setCarregando(false);
    }
  }

  return (
    <main className="home-page">
      <div className="container">
        <div className="home-panel">
          <p className="home-eyebrow">Reconhecimento em Libras</p>

          <h1 className="home-title">
            Escolha o modo de jogo
          </h1>

          <p className="home-subtitle">
            Selecione uma modalidade para iniciar o reconhecimento da
            soletração manual.
          </p>

          {!modo ? (
            <div className="game-mode-grid">
              <button
                type="button"
                className="game-mode-card"
                onClick={() => setModo("normal")}
              >
                <h2>Modo Básico</h2>
                <p>
                  Mostra cada letra que deve ser realizada em Libras.
                </p>
              </button>

              <button
                type="button"
                className="game-mode-card"
                onClick={() => setModo("didatico")}
              >
                <h2>Modo Didático</h2>
                <p>
                  Mostra a imagem do sinal correspondente a cada letra.
                </p>
              </button>
            </div>
          ) : (
            <>
              <form className="home-form" onSubmit={handleSubmit}>
                <div>
                  <input
                    className="text-input"
                    type="text"
                    autoComplete="off"
                    autoFocus
                    placeholder="Ex.: ESCOLA"
                    value={valor}
                    onChange={(e) => setValor(e.target.value)}
                    maxLength={40}
                    disabled={carregando}
                  />

                  <p className="field-hint">
                    Acentos são removidos e o texto será convertido para
                    maiúsculas automaticamente.
                  </p>
                </div>

                {erro && (
                  <p className="error-banner" role="alert">
                    {erro}
                  </p>
                )}

                <div className="home-actions">
                  <Button
                    variant="ghost"
                    type="button"
                    onClick={() => {
                      setModo(null);
                      setValor("");
                    }}
                  >
                    Voltar
                  </Button>

                  <Button
                    type="submit"
                    disabled={carregando || !valor.trim()}
                  >
                    {carregando ? "Iniciando..." : "Começar"}
                  </Button>
                </div>
              </form>
            </>
          )}
        </div>
      </div>
    </main>
  );
}