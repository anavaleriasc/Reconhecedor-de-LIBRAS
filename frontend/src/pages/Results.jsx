import { useNavigate } from "react-router-dom";
import { useGame } from "../context/GameContext";
import Card from "../components/Card";
import WordProgress from "../components/WordProgress";
import Button from "../components/Button";

/**
 * Formato real de GameSession.resultado_final() (src/prediction.py):
 *   { texto_original, letras_esperadas, letras_reconhecidas,
 *     acertos, erros, puladas, pontuacao }
 * `pontuacao` já vem em 0–100 (acertos / total_tentativas * 100).
 * `letras_reconhecidas[i]` é "-" quando a letra foi pulada ou o jogo
 * encerrado antes de chegar nela.
 */
export default function Results() {
  const { result, resetGame } = useGame();
  const navigate = useNavigate();

  const texto = result?.texto_original ?? "";
  const esperadas = result?.letras_esperadas ?? [];
  const reconhecidas = result?.letras_reconhecidas ?? [];
  const acertos = result?.acertos ?? 0;
  const erros = result?.erros ?? 0;
  const puladas = result?.puladas ?? 0;
  const pontuacao = result?.pontuacao ?? 0;

  const resultadosPorLetra = esperadas.map((letra, i) =>
    reconhecidas[i] === undefined ? undefined : reconhecidas[i] === letra ? "correta" : "erro"
  );

  function jogarNovamente() {
    resetGame();
    navigate("/");
  }

  return (
    <main className="results-page">
      <div className="container">
        <div className="results-panel">
          <div>
            <p className="home-eyebrow" style={{ textAlign: "center" }}>
              Partida concluída
            </p>
            <p className="results-word">{texto}</p>
          </div>

          <WordProgress texto={texto} indiceAtual={-1} resultados={resultadosPorLetra} />

          <div className="results-stats">
            <Card stat label="Acertos" value={acertos} tone="signal" />
            <Card stat label="Erros" value={erros} tone="alert" />
            <Card stat label="Pontuação" value={`${pontuacao}%`} />
          </div>

          {puladas > 0 && (
            <p className="field-hint" style={{ textAlign: "center" }}>
              {puladas} letra{puladas > 1 ? "s" : ""} pulada{puladas > 1 ? "s" : ""}.
            </p>
          )}

          <div className="results-actions">
            <Button variant="primary" onClick={jogarNovamente}>
              Jogar novamente
            </Button>
          </div>
        </div>
      </div>
    </main>
  );
}
