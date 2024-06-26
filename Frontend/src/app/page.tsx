"use client";
import axios from "axios";
import ImageUploadForm from "../components/FormImage";
import { useEffect, useState } from "react";

export default function Home() {
  const [data, setData] = useState();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const getDictionaries = async () => {
    try {
      const response = await axios.get("http://localhost:8000/api/images/", {
        headers: {
          "Content-Type": "application/json",
        },
      });
      return response.data;
    } catch (error) {
      console.error("Error fetching dictionaries:", error);
      throw error;
    }
  };

  useEffect(() => {
    const fetchData = async () => {
      try {
        const result = await getDictionaries();
        console.log(result);
        setData(result);
      } catch (err: any) {
        setError(err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) return <p>Loading...</p>;
  if (error) return <p>Error: {error.message}</p>;

  return (
    <div>
      <ImageUploadForm />
      <pre>{JSON.stringify(data, null, 2)}</pre>
    </div>
  );
}
