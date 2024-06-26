// components/ImageUploadForm.tsx
"use client";
import React, { useState, ChangeEvent, FormEvent } from "react";
import axios from "axios";

const ImageUploadForm: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [fileUrls, setFileUrls] = useState<string[]>([]);

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0] || null;
    setFile(selectedFile);
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();

    if (!file) {
      setStatus("Selecione um arquivo ZIP.");
      return;
    }

    if (!file.name.endsWith(".zip")) {
      setStatus("Por favor, selecione um arquivo ZIP válido.");
      return;
    }

    const formData = new FormData();
    formData.append("image", file);

    setIsLoading(true);
    setStatus("");

    try {
      const response = await axios.post(
        "http://localhost:8000/api/upload/",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      if (response.status === 200) {
        setStatus("Arquivos extraídos com sucesso.");
        setFileUrls(response.data.file_urls);
      } else {
        setStatus(`Erro ao enviar arquivo ZIP: ${response.data.error}`);
      }
    } catch (error: any) {
      setStatus(`Erro ao enviar arquivo ZIP: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div>
      <h1>Upload de Arquivo ZIP</h1>
      <form onSubmit={handleSubmit}>
        <div>
          <label htmlFor="image">Escolha um arquivo ZIP:</label>
          <input
            type="file"
            id="image"
            accept=".zip"
            onChange={handleFileChange}
          />
        </div>
        <br />
        <button type="submit" disabled={isLoading}>
          Enviar
        </button>
      </form>
      {isLoading && <p>Enviando arquivo...</p>}
      {status && <p>{status}</p>}
      <div>
        {fileUrls.length > 0 && <h2>Arquivos Extraídos:</h2>}
        <div style={{ display: "flex", flexWrap: "wrap" }}>
          {fileUrls.map((url, index) => (
            <div key={index} style={{ margin: "10px" }}>
              <a
                href={url}
                target="_blank"
                rel="noopener noreferrer"
              >{`Arquivo ${index + 1}`}</a>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default ImageUploadForm;
