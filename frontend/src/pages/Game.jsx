import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { connectGameSocket, getGameResult, deleteGame } from "../services/api";
import { useGame } from "../context/GameContext";
import Card from "../components/Card";
import HUD from "../components/HUD";
import CameraFeed from "../components/CameraFeed";
import ConfidenceMeter from "../components/ConfidenceMeter";
import WordProgress from "../components/WordProgress";
import Button from "../components/Button";

const CONFIDENCE_LIMIAR = 0.6;

export default function Game() {
  const { texto, sessionId, totalLetras: totalInicial, letraEsperada: letraInicial, setResult, resetGame } = useGame();
  const navigate = useNavigate();
  const socketRef = useRef(null);
  const prevRef = useRef({ acertos: 0, erros: 0, indice: 0 });
  const inicioRef = useRef(Date.now());

  const [conectado, setConectado] = useState(false);
  const [indiceAtual, setIndiceAtual] = useState(0);
  const [totalLetras, setTotalLetras] = useState(totalInicial);
  const [letraEsperada, setLetraEsperada] = useState(letraInicial);
  const [acertos, setAcertos] = useState(0);
  const [erros, setErros] = useState(0);
  const [resultados, setResultados] = useState([]);
  const [finalizado, setFinalizado] = useState(false);
  const [letraReconhecida, setLetraReconhecida] = useState(null);
  const [confidence, setConfidence] = useState(0);
  const [erro, setErro] = useState(null);
  const [aviso, setAviso] = useState(null); 
  const [tempo, setTempo] = useState("00:00");

  const handleState = useCallback((state) => {
    const novoIndice = state.indice_atual ?? 0;
    const novosAcertos = state.acertos ?? 0;
    const novosErros = state.erros ?? 0;

    setResultados((atual) => {
      if (novoIndice > prevRef.current.indice) {
        const copia = [...atual];
        const idxAnterior = prevRef.current.indice;
        if (novosAcertos > prevRef.current.acertos) copia[idxAnterior] = "correta";
        else if (novosErros > prevRef.current.erros) copia[idxAnterior] = "erro";
        return copia;
      }
      return atual;
    });

    prevRef.current = { acertos: novosAcertos, erros: novosErros, indice: novoIndice };
    setIndiceAtual(novoIndice);
    setTotalLetras(state.total_letras ?? 0);
    setLetraEsperada(state.letra_esperada ?? null);
    setAcertos(novosAcertos);
    setErros(novosErros);
    setAviso(state.mensagem || null);
    if (state.finalizado) setFinalizado(true);
  }, []);

  useEffect(() => {
    const socket = connectGameSocket(sessionId, {
      onOpen: () => setConectado(true),
      onClose: () => setConectado(false),
      onState: handleState,
      onObservation: (obs) => {
        setLetraReconhecida(obs.hand_detected ? obs.letter ?? null : null);
        setConfidence(obs.confidence ?? 0);
        if (obs.error) setAviso(obs.error);
      },
      onError: (msg) => setErro(msg || "Erro de comunicação com o servidor."),
    });
    socketRef.current = socket;
    return () => socket.close();
  }, [sessionId, handleState]);

  useEffect(() => {
    const id = setInterval(() => {
      const segundos = Math.floor((Date.now() - inicioRef.current) / 1000);
      const m = String(Math.floor(segundos / 60)).padStart(2, "0");
      const s = String(segundos % 60).padStart(2, "0");
      setTempo(`${m}:${s}`);
    }, 1000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    if (!finalizado) return;
    let ativo = true;

    getGameResult(sessionId)
      .then((res) => {
        if (!ativo) return;
        setResult(res);
        navigate("/results");
      })
      .catch((err) => setErro(err.message || "Vish kk"));

    return () => {
      ativo = false;
    };
  }, [finalizado, sessionId, setResult, navigate]);

  const confirmar = useCallback(() => socketRef.current?.confirm(), []);
  const pular = useCallback(() => socketRef.current?.skip(), []);
  const encerrar = useCallback(() => socketRef.current?.finish(), []);

  useEffect(() => {
    function onKeyDown(event) {
      if (event.code === "Space") {
        event.preventDefault();
        confirmar();
      } else if (event.key.toLowerCase() === "n") {
        pular();
      } else if (event.key.toLowerCase() === "q") {
        encerrar();
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [confirmar, pular, encerrar]);

  function sairDoJogo() {
    socketRef.current?.close();
    if (!finalizado) {
      deleteGame(sessionId).catch(() => {});
    }
    resetGame();
    navigate("/");
  }

  const letraAlvo = letraEsperada ?? texto?.[indiceAtual] ?? "";
  const reconhecimentoBate =
    letraReconhecida && letraReconhecida.toUpperCase() === letraAlvo?.toUpperCase() && confidence >= CONFIDENCE_LIMIAR;

  return (
    <main className="game-page">
      {erro && (<div className="container"> <p className="error-banner" role="alert">{erro}</p> </div>)}
      {aviso && !erro && (<div className="container"> <p className="error-banner" role="status">{aviso}</p> </div> )}

      <div className="game-layout">
        <div className="container">
          <Card label="Câmera" active={reconhecimentoBate}>
            <CameraFeed
              active={!finalizado}
              onFrame={(frame) => socketRef.current?.sendFrame(frame)}
            />
            <div className="detected-letter">
              <span className="detected-letter-label">Detectando</span>
              <span
                className={`detected-letter-value ${
                  reconhecimentoBate ? "signal" : letraReconhecida ? "alert" : ""
                }`}
              >
                {letraReconhecida ?? "—"}
              </span>
            </div>
            <ConfidenceMeter confidence={confidence} />
          </Card>
        </div>

        <div className="game-layout">
          <div className="container">
            <HUD
              indiceAtual={indiceAtual}
              acertos={acertos}
              erros={erros}          
              tempo={tempo}
              total={totalLetras || texto.length}
            />

            <WordProgress texto={texto} indiceAtual={indiceAtual} resultados={resultados} />

            <Card label="Letra atual" className="game-target-card" active={reconhecimentoBate}>
              <span className="letter-target">{letraAlvo}</span>

              <div className="game-actions">
                <Button variant="primary" onClick={confirmar} disabled={!conectado}> Confirmar
                </Button>
                <Button variant="secondary" onClick={pular} disabled={!conectado}>   Pular
                </Button>
                <Button variant="danger" onClick={encerrar} disabled={!conectado}>   Encerrar
                </Button>
              </div>

              <p className="game-shortcut-hint">
                <kbd>Espaço</kbd> confirma · <kbd>N</kbd> pula · <kbd>Q</kbd> encerra
              </p>

              <Button variant="ghost" size="sm" onClick={sairDoJogo}>
                Sair sem terminar
              </Button>
            </Card>
          </div>
        </div>
      </div>
    </main>
  );
}
