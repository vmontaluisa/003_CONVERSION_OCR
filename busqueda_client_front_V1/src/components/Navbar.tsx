import { Link } from "react-router-dom";
import logo from "../assets/logo2024.png"; // Asegúrate de tener el logo en /src/assets/



const Navbar = () => {
  return (
    <nav className="navbar">
        <div className="logo">
            <img src={logo} alt="Logo" className="logo-img"  />
      </div>
      <ul>
      <li><Link to="/">🏠 Inicio </Link></li>
      <li><Link to="/buscar">🔎 Búsqueda </Link></li>
        <li><Link to="/chat">💬 Chat Legal</Link></li>
      </ul>
    </nav>
  );
};

export default Navbar;