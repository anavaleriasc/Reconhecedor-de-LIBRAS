/** ConfidenceMeter — barra 0–1 de confiança do reconhecimento do sinal atual. */
export default function ConfidenceMeter({ confidence = 0 }) {
  const pct = Math.round(Math.max(0, Math.min(1, confidence)) * 100);

  return (
    <div className="confidence">
      <div className="confidence-track">
        <div className="confidence-fill" style={{ width: `${pct}%` }} />
      </div>
      <span className="confidence-value mono-data">{pct}%</span>
    </div>
  );
}
