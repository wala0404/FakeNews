import React, { useEffect, useState } from "react";
import { getRecommendations } from "../api/news";
import ArticleCard from "../components/ArticleCard";

const Feed = () => {
  const [articles, setArticles] = useState([]);
  useEffect(() => {
    getRecommendations().then(setArticles);
  }, []);
  return (
    <div className="max-w-2xl mx-auto mt-8">
      <h1 className="text-2xl font-bold mb-4">Recommended News</h1>
      {articles.map((a, i) => (
        <ArticleCard key={i} article={a} />
      ))}
    </div>
  );
};

export default Feed;

