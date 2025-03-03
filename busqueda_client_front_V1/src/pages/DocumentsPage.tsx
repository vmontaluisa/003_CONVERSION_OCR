import { useState, useEffect } from "react";
import { getDocuments } from "../api/Api";

const DocumentsPage: React.FC = () => {
    const [documents, setDocuments] = useState<any[]>([]); // ✅ Asegurar que es un array

    useEffect(() => {
        const fetchDocuments = async () => {
            try {
                const data = await getDocuments();
                console.log("Datos recibidos de la API:", data);  // 🔍 Depuración
                setDocuments(data.documentos || []);  // ✅ Si `documentos` no existe, usa un array vacío
            } catch (error) {
                console.error("Error al obtener documentos:", error);
            }
        };
        fetchDocuments();
    }, []);

    return (
        <div>
            <h1>Documentos</h1>
            <ul>
                {Array.isArray(documents) && documents.map((doc, index) => (
                    <li key={index}>{doc.archivo_nombre}</li>
                ))}
            </ul>
        </div>
    );
};

export default DocumentsPage;