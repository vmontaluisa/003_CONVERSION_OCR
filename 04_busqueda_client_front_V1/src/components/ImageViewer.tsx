import React from "react";

interface ImageViewerProps {
  imageUrl: string;
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  dpiResolution: number;
}

const ImageViewer: React.FC<ImageViewerProps> = ({ imageUrl, x0, y0, x1, y1, dpiResolution }) => {
  // Convertir coordenadas de DPI a píxeles
  const convertToPixels = (value: number) => (value * dpiResolution) / 72;
  const convertToPixels_X = (value: number) => (value) ;
  const convertToPixels_Y = (value: number) => (value * dpiResolution) / 150;

  const boxStyle: React.CSSProperties = {
    position: "absolute",
    top: `${convertToPixels_Y(y0)}px`,
    left: `${convertToPixels_X(x0)}px`,
    width: `${convertToPixels(x1 - x0)}px`,
    height: `${convertToPixels(y1 - y0)}px`,
//    border: "2px solid red",
 //   backgroundColor: "rgba(255, 0, 0, 0.2)",
  };

  return (
    <div style={{ position: "relative", display: "inline-block" }}>
      <img src={imageUrl} alt="Documento" style={{ width: "100%", height: "auto" }} />
      <div style={boxStyle}></div>
    </div>
  );
};

export default ImageViewer;