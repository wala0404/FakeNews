import React from "react";

const ArticleCard = ({ article }) => (
  <div className="bg-white rounded shadow p-4 mb-4">
    <h2 className="text-lg font-bold">{article.title}</h2>
    <p className="text-gray-700">{article.content}</p>
    {article.url && (
      <a href={article.url} className="text-blue-500 underline" target="_blank" rel="noopener noreferrer">
        Read more
      </a>
    )}
  </div>
);

export default ArticleCard;

