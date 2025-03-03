import React, { useState, useEffect } from "react";
import { getDocuments } from "./api/Api"; // Importa la función desde Api.ts

const App: React.FC = () => {
  const [documents, setDocuments] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [currentPage, setCurrentPage] = useState(1);
  const documentsPerPage = 9; // 📌 Cantidad de documentos por página

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

  // 📌 Cálculo de paginación
  const indexOfLastDocument = currentPage * documentsPerPage;
  const indexOfFirstDocument = indexOfLastDocument - documentsPerPage;
  const currentDocuments = documents.slice(indexOfFirstDocument, indexOfLastDocument);
  const totalPages = Math.ceil(documents.length / documentsPerPage);

  // 📌 Funciones para cambiar de página
  const handleNextPage = () => {
    if (currentPage < totalPages) setCurrentPage(currentPage + 1);
  };

  const handlePrevPage = () => {
    if (currentPage > 1) setCurrentPage(currentPage - 1);
  };

  return (
    <div className="container">
      <div className="header-container">
        <h1 className="title">📂 LEYES DISPONIBLES</h1>
      </div>      
      <br />

      {loading ? (
        <p>Cargando documentos...</p>
      ) : documents.length > 0 ? (
        <div>
          <table className="document-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Nombre del Archivo</th>
                <th>Páginas</th>
                <th>Tamaño (KB)</th>
                <th>Archivo PDF</th>
              </tr>
            </thead>
            <tbody>
              {currentDocuments.map((doc, index) => (
                <tr key={index}>
                  <td>{indexOfFirstDocument + index + 1}</td>
                  <td className="tabla-nombre-archivo">{doc.archivo_nombre}</td>
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

          {/* 📌 Paginador */}
          <div className="pagination">
            <button disabled={currentPage === 1} onClick={handlePrevPage}>
              ⬅ Anterior
            </button>
            <span>
              Página {currentPage} de {totalPages}
            </span>
            <button disabled={currentPage === totalPages} onClick={handleNextPage}>
              Siguiente ➡
            </button>
          </div>
        </div>
      ) : (
        <p>No hay documentos disponibles</p>
      )}
    </div>
  );
};

export default App;