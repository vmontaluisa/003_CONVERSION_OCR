import React from "react";

const Footer = () => {
  return (
    <footer className="footer">
      <p>© {new Date().getFullYear()} - Buscador de Documentos | Desarrollado por <strong>Estrategos & Software</strong></p>
    </footer>
  );
};

export default Footer;