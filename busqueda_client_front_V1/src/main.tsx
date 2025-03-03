import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import App from "./App";
import ChatPage from "./pages/ChatPage";
import SearchPage from "./pages/SearchPage";
import DocumentsPage from "./pages/DocumentsPage";
import Navbar from "./components/Navbar";  //
import Footer from "./components/Footer"

import "./index.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>


    <Router>
        <div className="app-container">
                <Navbar />  
                <main className="content">
                      <Routes>
                        <Route path="/" element={<App />} />
                        <Route path="/buscar" element={<SearchPage />} />
                        <Route path="/chat" element={<ChatPage />} />
                        <Route path="/documentos" element={<DocumentsPage />} />

                      </Routes>
                    
                  </main>
                  
        </div>
        <Footer />


    </Router>

  
  </React.StrictMode>
);