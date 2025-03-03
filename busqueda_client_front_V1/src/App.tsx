import React, { useState, useEffect } from "react";
import { getDocuments } from "./api/Api"; // Importa la función desde Api.ts

const App: React.FC = () => {
  const [documents, setDocuments] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadDocuments = async () => {
      try {
        const data = await getDocuments();
        setDocuments(data.documentos || []);  // ✅ Si `documentos` no existe, usa un array vacío
      } catch (error) {
        console.error("Error al cargar documentos:", error);
      } finally {
        setLoading(false);
      }
    };
    loadDocuments();
  }, []);

  return (
    <div className="container">
      <div className="header-container">
        <h1 className="title">📂 LEYES DISPONIBLES</h1>
      </div>      <br></br>
      {loading ? (
        <p>Cargando documentos...</p>
      ) : documents.length > 0 ? (
        <table className="document-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Nombre del Archivo</th>
              {/*  
              <th>Título</th>
              */}
              <th>Páginas</th>
              <th>Tamaño (KB)</th>
              <th>Archivo PDF</th>
              
            </tr>
          </thead>
          <tbody>
            {documents.map((doc, index) => (
              <tr key={index}>
                <td>{index + 1}</td>
                <td className="tabla-nombre-archivo"  >{doc.archivo_nombre}</td>
 {/*                <td>{doc.archivo_titulo || "Sin título"}</td>  */}
                <td>{doc.archivo_paginas || "N/A"}</td>
                <td>{doc.archivo_size_kb ? `${doc.archivo_size_kb} KB` : "N/A"}</td>
                <td>
                <a href={`./02_PDF_PROCESADOS/${doc.archivo_nombre}`} download>
                      <img src="/imagenes/pdf.png" alt="Descargar PDF" width="20" height="20" />
                    </a>

                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p>No hay documentos disponibles</p>
      )}
    </div>
  );
};

export default App;