import React, { useState } from "react";
import { searchLLMS } from "../api/Api";

const ChatPage: React.FC = () => {
  const [mensaje, setMensaje] = useState("");
  const [respuesta, setRespuesta] = useState("");

  const handleSendMessage = async () => {
    try {
      const data = await searchLLMS(mensaje);
      setRespuesta(data.respuesta);
    } catch (error) {
      console.error("Error en la API de chat:", error);
    }
  };

  return (
    <div>

      <div className="header-container">
        <h1 className="title">💬  Chat Legal</h1>
      </div>
      <br />

      <div className="search-container">
          <label>Pregunta:</label>

      <input
        type="text"
        value={mensaje}
        onChange={(e) => setMensaje(e.target.value)}
        placeholder="Escribe tu mensaje"
          className="search-input"
      />
      <button onClick={handleSendMessage}   className="search-button"   >Enviar</button>

</div>

      <p>Respuesta: {respuesta}</p>
    </div>
  );
};

export default ChatPage;