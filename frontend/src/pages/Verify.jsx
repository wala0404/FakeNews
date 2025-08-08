import React, { useState } from "react";
import { classifyNews, ocrImage } from "../api/news";
import UploadBox from "../components/UploadBox";

const Verify = () => {
  const [text, setText] = useState("");
  const [ocrResult, setOcrResult] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleClassify = async () => {
    setLoading(true);
    const res = await classifyNews(text);
    setResult(res);
    setLoading(false);
  };

  const handleFile = async (file) => {
    setLoading(true);
    const ocr = await ocrImage(file);
    setOcrResult(ocr.text);
    setText(ocr.text);
    setLoading(false);
  };

  return (
    <div className="max-w-xl mx-auto mt-8">
      <h1 className="text-2xl font-bold mb-4">Verify News</h1>
      <textarea
        className="w-full border rounded p-2 mb-2"
        rows={5}
        placeholder="Paste news text here or use OCR below..."
        value={text}
        onChange={e => setText(e.target.value)}
      />
      <UploadBox onFileChange={handleFile} />
      <button
        className="bg-blue-600 text-white px-4 py-2 rounded"
        onClick={handleClassify}
        disabled={loading || !text}
      >
        {loading ? "Processing..." : "Classify"}
      </button>
      {result && (
        <div className="mt-4 p-4 bg-gray-100 rounded">
          <div>Label: <b>{result.label}</b></div>
          <div>Score: {result.score.toFixed(3)}</div>
        </div>
      )}
      {ocrResult && (
        <div className="mt-2 text-sm text-gray-500">OCR Text: {ocrResult}</div>
      )}
    </div>
  );
};

export default Verify;

