// pages/upload.js
"use client";
import { useState } from "react";

export default function Upload() {
  // const [title, setTitle] = useState("");
  const [image, setImage] = useState(null);
  const [imageUrl, setImageUrl] = useState("");

  const handleSubmit = async (e: any) => {
    e.preventDefault();

    if (!image) {
      console.error("No file selected");
      return;
    }

    const formData = new FormData();
    // formData.append("title", title);
    formData.append("image", image);

    const response = await fetch("http://127.0.0.1:8000/api/upload/", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    if (response.ok) {
      setImageUrl(data.image_url);
    } else {
      console.error(data.error);
    }
  };

  const handleFileChange = (e: any) => {
    if (e.target.files && e.target.files.length > 0) {
      setImage(e.target.files[0]);
    }
  };

  console.log(imageUrl);

  return (
    <div>
      <h1>Upload Image</h1>
      <form onSubmit={handleSubmit}>
        {/* <input
          type="text"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          placeholder="Title"
          required
        /> */}
        <input type="file" onChange={handleFileChange} required />
        <button type="submit">Upload</button>
      </form>
      {imageUrl && (
        <div>
          <h2>Uploaded Image</h2>
          <img src={imageUrl} alt="Uploaded" />
        </div>
      )}
    </div>
  );
}
