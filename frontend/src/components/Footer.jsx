import { Link } from "react-router-dom";

export default function Footer() {
  return (
    <footer className="site-footer">
      <div className="container footer-content">
        {/* Projeto */}
        <div className="footer-section">
            <Link to="/" className="brand">
            <img
                src="/logo.png"
                alt="Reconhecedor de Libras"
                className="brand-logo"
            />
            </Link>
            <p>Reconhecedor de LIBRAS</p>
            <p>
                Projeto de visão computacional para reconhecimento de letras do
                alfabeto em LIBRAS.
            </p>
        </div>

        {/* Instituição */}
        <div className="footer-section">
            <p className="home-eyebrow">Instituição</p>
            <p>Universidade Federal do Piauí</p>
            <p>Centro de Ciências da Natureza</p>
            <p>Visão Computacional</p>
        </div>

        {/* Equipe */}
        <div className="footer-section">
            <p className="home-eyebrow">Equipe</p>
            <p>Ana Valéria Silva Coelho</p>
            <p>João Victor Cruz Silva</p>
            <p>Teresina | PI</p>
        </div>
      </div>
    </footer>
  );
}