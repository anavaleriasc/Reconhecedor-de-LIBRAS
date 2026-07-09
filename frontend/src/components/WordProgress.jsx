/**
 * WordProgress — a palavra/frase inteira, letra a letra, com o status de
 * cada uma. `resultados` é um array paralelo a `texto` com "correta" |
 * "erro" | undefined (ainda não tentada) para as letras já passadas.
 */
export default function WordProgress({ texto = "", indiceAtual = 0, resultados = [] }) {
  return (
    <div className="word-progress" role="list" aria-label="Progresso da palavra">
      {texto.split("").map((letra, i) => {
        let status = "";
        if (i === indiceAtual) status = "is-current";
        else if (resultados[i] === "correta") status = "is-correct";
        else if (resultados[i] === "erro") status = "is-error";

        return (
          <span key={`${letra}-${i}`} className={`letter-tile ${status}`} role="listitem">
            {letra === " " ? "␣" : letra}
          </span>
        );
      })}
    </div>
  );
}
