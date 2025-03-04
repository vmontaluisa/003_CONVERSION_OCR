import React, { useState } from "react";
import { searchChromaDB, searchFAISS } from "../api/Api";
import ImageViewer from "../components/ImageViewer";
const SearchPage: React.FC = () => {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<any[]>([]);
  const [searchEngine, setSearchEngine] = useState<"faiss" | "chromadb">("chromadb");
  const [topK, setTopK] = useState(20); // Cantidad de resultados por búsqueda
  const [currentPage, setCurrentPage] = useState(1); // Página actual
  const resultsPerPage = 7; // Cantidad de resultados por página
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [selectedCoords, setSelectedCoords] = useState<any>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);


  // Función para manejar la búsqueda
  const handleSearch = async () => {
    try {
      let data;
      if (searchEngine === "faiss") {
        data = await searchFAISS(query, topK);
      } else {
        data = await searchChromaDB(query, topK);
      }
      setResults(data.resultados);
      setCurrentPage(1); // Reiniciar a la primera página
    } catch (error) {
      console.error("Error en la búsqueda:", error);
    }
  };



  const openImageViewer = (imageUrl: string, coords: any) => {
    setSelectedImage(imageUrl);
    setSelectedCoords(coords);
    setIsModalOpen(true);
  };

  const closeModal = () => {
    setIsModalOpen(false);
  };

  // Calcular índices de paginación
  const indexOfLastResult = currentPage * resultsPerPage;
  const indexOfFirstResult = indexOfLastResult - resultsPerPage;
  const currentResults = results.slice(indexOfFirstResult, indexOfLastResult);
  const totalPages = Math.ceil(results.length / resultsPerPage);

  return (
    <div className="container">
      <div className="header-container">
        <h1 className="title">🔎 Buscar en {searchEngine === "faiss" ? "FAISS" : "ChromaDB"}</h1>
      </div>
      <br />

      {/* Selección de motor de búsqueda */}
      <div className="search-options">
        
      <div className="radio-buttons">
        <label>
          <input
            type="radio"
            value="faiss"
            checked={searchEngine === "faiss"}
            onChange={() => setSearchEngine("faiss")}
            className="radio-input"
          />
          🔍 FAISS
        </label>
        <label>
          <input
            type="radio"
            value="chromadb"
            checked={searchEngine === "chromadb"}
            onChange={() => setSearchEngine("chromadb")}
            className="radio-input"
          />
          📚 ChromaDB
        </label>
      </div>
      
      {/* Selección de cantidad de resultados */}
      <div className="input-container">
          <label>Resultados:</label>
        <input
          type="number"
          value={topK}
          onChange={(e) => setTopK(Number(e.target.value))}
          min={1}
          max={50}
           className="input-large"
        />
      </div>


      </div>


      {/* Input de búsqueda */}
      <div className="search-container">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ingrese su búsqueda"
             className="search-input"
        />
        <button className="search-button"  onClick={handleSearch}>Buscar</button>
      </div>

      {/* Tabla de resultados */}
      {results.length > 0 ? (
        <div>
          <table className="results-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Texto</th>
                <th>Archivo</th>
                <th>Página</th>
                <th>Artículo</th>
                <th>Distancia</th>
                <th>Imagen</th>
                <th>Archivo PDF</th>
              </tr>
            </thead>
            <tbody>
              {currentResults.map((result, index) => {
                const imageUrl = result.metadata?.pagina_imagen
                  ? `./06_IMAGENES/${result.metadata.pagina_imagen}`
                  : null;
                const pdfUrl = result.metadata?.archivo_nombre_archivo_pdf
                  ? `./02_PDF_PROCESADOS/${result.metadata.archivo_nombre_archivo_pdf}`
                  : null;

                  const coords = {
                    x0: result.metadata?.parrafo_coordenadas_x0 || 0,
                    y0: result.metadata?.parrafo_coordenadas_y0 || 0,
                    x1: result.metadata?.parrafo_coordenadas_x1 || 0,
                    y1: result.metadata?.parrafo_coordenadas_y1 || 0,
                  };

                return (
                  <tr key={index}>
                    <td>{indexOfFirstResult + index + 1}</td>
                    <td>{result.metadata?.parrafo_texto_original || "Sin texto"}</td>
                    <td>{result.metadata?.archivo_nombre_archivo_pdf || "Desconocido"}</td>
                    <td>{result.metadata?.pagina_numero || "-"}</td>
                    <td>{result.metadata?.articulo_numero || "-"}</td>
                    <td>{result.distancia.toFixed(4)}</td>
                    <td>
               
                    {imageUrl ? (
                        <button onClick={() => openImageViewer(imageUrl, coords)}>
                          <img src="/imagenes/parrafo.png" alt="Ver Imagen" width="20" height="20" />
                        </button>
                      ) : (
                        "No disponible"
                      )}
                      
     
                    </td>
                    <td>
                      {pdfUrl ? (
                        <a href={pdfUrl} download>
                          <img src="/imagenes/pdf.png" alt="Descargar PDF" width="20" height="20" />
                        </a>
                      ) : (
                        "No disponible"
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>

          {/* Paginación */}
          <div className="pagination">
            <button disabled={currentPage === 1} onClick={() => setCurrentPage(currentPage - 1)}>
              ⬅ Anterior
            </button>
            <span>
              Página {currentPage} de {totalPages}
            </span>
            <button disabled={currentPage === totalPages} onClick={() => setCurrentPage(currentPage + 1)}>
              Siguiente ➡
            </button>
          </div>
        </div>
      ) : (
        <p>No hay resultados</p>
      )}

      {/* Modal Popup para mostrar ImageViewer */}
      {isModalOpen && selectedImage && (
        <div className="modal-overlay" onClick={closeModal}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <button className="close-button" onClick={closeModal}>✖</button>
            <ImageViewer imageUrl={selectedImage} x0={selectedCoords.x0} y0={selectedCoords.y0} x1={selectedCoords.x1} y1={selectedCoords.y1} dpiResolution={300} />
          </div>
        </div>
      )}
      


    </div>
  );
};

export default SearchPage;