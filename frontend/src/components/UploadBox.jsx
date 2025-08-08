import React from "react";

const UploadBox = ({ onFileChange }) => (
  <div className="flex flex-col items-center border-2 border-dashed border-gray-400 p-4 rounded mb-4">
    <input type="file" accept="image/*" onChange={e => onFileChange(e.target.files[0])} />
    <p className="text-gray-500 mt-2">Upload an image to extract text (OCR)</p>
  </div>
);

export default UploadBox;

