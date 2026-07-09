import { useEffect, useRef, useState } from "react";
import { canvasToBase64 } from "../services/api";

/**
 * CameraFeed — pede acesso à webcam, exibe a prévia e, enquanto `active`
 * for true, chama `onFrame(imageBase64)` a cada `intervalMs`. O componente
 * não sabe nada sobre WebSocket/REST — quem decide o que fazer com o frame
 * é o chamador (Game.jsx), o que mantém esta peça reutilizável.
 */
export default function CameraFeed({ active = true, intervalMs = 400, onFrame }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(document.createElement("canvas"));
  const [status, setStatus] = useState("solicitando"); // solicitando | pronta | negada

  useEffect(() => {
    let stream;

    async function start() {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        }
        setStatus("pronta");
      } catch {
        setStatus("negada");
      }
    }

    start();

    return () => {
      stream?.getTracks().forEach((track) => track.stop());
    };
  }, []);

  useEffect(() => {
    if (status !== "pronta" || !active || !onFrame) return;

    const captura = setInterval(() => {
      const video = videoRef.current;
      if (!video || video.readyState < 2) return;

      const canvas = canvasRef.current;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext("2d").drawImage(video, 0, 0);
      onFrame(canvasToBase64(canvas));
    }, intervalMs);

    return () => clearInterval(captura);
  }, [status, active, intervalMs, onFrame]);

  return (
    <div className="camera-frame">
      <video ref={videoRef} muted playsInline aria-label="Prévia da webcam" />
      {status !== "pronta" && (
        <div className="camera-placeholder">
          {status === "solicitando" && <p>Aguardando permissão da câmera…</p>}
          {status === "negada" && (
            <p>Não foi possível acessar a câmera. Verifique as permissões do navegador e recarregue a página.</p>
          )}
        </div>
      )}
    </div>
  );
}
