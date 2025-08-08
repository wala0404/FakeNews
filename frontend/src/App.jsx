import React from "react";
import { BrowserRouter, Routes, Route, Link } from "react-router-dom";
import Feed from "./pages/Feed";
import Verify from "./pages/Verify";

function App() {
  return (
    <BrowserRouter>
      <nav className="bg-gray-800 p-4 text-white flex justify-between">
        <div className="font-bold">FakeNews Platform</div>
        <div>
          <Link className="mr-4" to="/">Feed</Link>
          <Link to="/verify">Verify</Link>
        </div>
      </nav>
      <Routes>
        <Route path="/" element={<Feed />} />
        <Route path="/verify" element={<Verify />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;

