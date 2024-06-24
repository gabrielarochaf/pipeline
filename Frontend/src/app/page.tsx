// pages/upload.js
"use client";
import { useState } from "react";
import Image from "next/image";

//teste

export default function Upload() {
  const [selectedFile, setSelectedFile] = useState(null);

  const handleFileChange = (e: any) => {
    setSelectedFile(e.target.files[0]);
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      console.error("Nenhum arquivo selecionado");
      return;
    }

    const formData = new FormData();
    formData.append("image", selectedFile);

    try {
      const response = await fetch("http://localhost:8000/api/upload/", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Falha ao enviar arquivo");
      }

      console.log("Arquivo enviado com sucesso");
      // Aqui você pode redirecionar ou exibir uma mensagem de sucesso
    } catch (error) {
      console.error("Erro ao enviar arquivo:", error);
      // Aqui você pode exibir uma mensagem de erro para o usuário
    }
  };

  return (
    <div>
      <h1>Upload de Imagem</h1>
      <input type="file" onChange={handleFileChange} />
      <button onClick={handleUpload}>Enviar</button>

      <Image
        src="http://localhost:8000/media/images/image.png"
        alt=""
        width={30}
        height={30}
      />
      <Image
        src="http://localhost:8000/media/images/photo_4969979131383098687_y.jpg"
        alt=""
        width={30}
        height={30}
      />
    </div>
  );
}
