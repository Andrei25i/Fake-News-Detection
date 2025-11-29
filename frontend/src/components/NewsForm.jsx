import { useState } from "react";
import api from "../services/api";
import Results from "./Results";
import "./NewsForm.css";

const NewsForm = () => {
  const [title, setTitle] = useState("");
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await api.post("/predict", {
        title,
        text,
      });

      setResult(response.data);
    } catch (err) {
      console.error("Request error:", err);
      setError(err.response?.data?.detail || "Error connecting to server");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="form-wrapper">
      <form onSubmit={handleSubmit} className="news-form">
        <div className="form-group">
          <label htmlFor="title">Title</label>
          <input
            type="text"
            id="title"
            name="title"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="Enter news title here..."
            required
          />
        </div>

        <div className="form-group">
          <label htmlFor="title">Description</label>
          <textarea
            type="textarea"
            id="text"
            name="text"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Enter news description here..."
            rows={5}
            required
          />
        </div>

        <button type="submit" disabled={loading} className="submit-btn">
          {loading ? "Analyzing..." : "Analyze"}
        </button>

        {error && <div className="error-msg">{error}</div>}
      </form>

      <Results results={result} />
    </div>
  );
};

export default NewsForm;
