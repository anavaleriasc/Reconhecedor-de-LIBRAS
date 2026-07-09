/**
 * Card — painel de assinatura do produto. Por padrão exibe os cantos de
 * mira (.viewfinder) definidos em global.css; `active` acende os cantos
 * em --signal (usado quando uma letra é reconhecida com boa confiança).
 *
 * Duas formas de uso:
 *   <Card label="Câmera">{...}</Card>                      → painel livre
 *   <Card stat label="Acertos" value={7} tone="signal" />   → tile de estatística
 */
export default function Card({
  children,
  label,
  value,
  tone,
  stat = false,
  active = false,
  viewfinder = true,
  className = "",
}) {
  const classes = ["card", viewfinder ? "viewfinder" : "", active ? "is-active" : "", className]
    .filter(Boolean)
    .join(" ");

  if (stat) {
    return (
      <div className={`${classes} card-stat`}>
        {label && <span className="card-label">{label}</span>}
        <span className={`card-value ${tone ?? ""}`}>{value}</span>
      </div>
    );
  }

  return (
    <div className={classes}>
      {label && <span className="card-label">{label}</span>}
      {children}
    </div>
  );
}
