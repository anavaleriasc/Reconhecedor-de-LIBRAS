/**
 * HUD — barra de status do jogo. Recebe dados já prontos (não conhece a
 * API); Game.jsx é responsável por calcular/formatar e repassar via props.
 */
export default function HUD({ acertos = 0, erros = 0, indiceAtual = 0, total = 0, tempo }) {
  return (
    <div className="hud">

      <div className="hud-divider" />

      <div className="hud-stat">
        <span className="hud-stat-label">Progresso</span>
        <span className="hud-stat-value mono-data">
          {Math.min(indiceAtual + 1, total)}/{total}
        </span>
      </div>

      <div className="hud-divider" />

      <div className="hud-stat">
        <span className="hud-stat-label">Acertos</span>
        <span className="hud-stat-value signal mono-data">{acertos}</span>
      </div>

      <div className="hud-stat">
        <span className="hud-stat-label">Erros</span>
        <span className="hud-stat-value alert mono-data">{erros}</span>
      </div>

      <div className="hud-spacer" />

      {tempo != null && (
        <div className="hud-stat">
          <span className="hud-stat-label">Tempo</span>
          <span className="hud-stat-value mono-data">{tempo}</span>
        </div>
      )}
    </div>
  );
}
