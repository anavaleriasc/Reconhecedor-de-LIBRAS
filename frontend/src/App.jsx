import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { GameProvider, useGame } from "./context/GameContext";
import Header from "./components/Header";
import Footer from "./components/Footer";
import Home from "./pages/Home";
import Game from "./pages/Game";
import GameDidatico from "./pages/GameDidatico";
import Results from "./pages/Results";

/**
 * Impede acesso direto a /game ou /results sem uma partida em andamento —
 * ex.: usuário atualiza a página ou cola a URL manualmente.
 */
function RequireSession({ children }) {
  const { sessionId } = useGame();
  if (!sessionId) {
    return <Navigate to="/" replace />;
  }
  return children;
}

function RequireResult({ children }) {
  const { result } = useGame();
  if (!result) {
    return <Navigate to="/" replace />;
  }
  return children;
}

function AppRoutes() {
  return (
    <div className="page">
      <Header />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route
          path="/game"
          element={
            <RequireSession>
              <Game />
            </RequireSession>
          }
        />
        <Route
          path="/gameDidatico"
          element={
            <RequireSession>
              <GameDidatico />
            </RequireSession>
          }
        />
        <Route
          path="/results"
          element={
            <RequireResult>
              <Results />
            </RequireResult>
          }
        />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
      <Footer />
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <GameProvider>
        <AppRoutes />
      </GameProvider>
    </BrowserRouter>
  );
}
