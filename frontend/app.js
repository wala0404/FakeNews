// ---------------------------
// app.js
// ---------------------------
const mainContent = document.getElementById("main-content");

// ---------------------------
// NEWS API (Live News)
// ---------------------------

const API_KEY = "a2098a3a19f1474387f0ba2d622092ad"
const url = "https://newsapi.org/v2/everything?q=";
async function fetchNews(query) {
  const res = await fetch(`${url}${query}&apiKey=${API_KEY}`);
  const data = await res.json();
  return data.articles || [];
}

function bindData(articles) {
  const cardsContainer = document.getElementById("cardscontainer");
  const newsCardTemplate = document.getElementById("template-news-card");

  cardsContainer.innerHTML = "";

  articles.forEach((article) => {
    if (!article.urlToImage) return;

    const cardClone = newsCardTemplate.content.cloneNode(true);
    fillDataInCard(cardClone, article);
    cardsContainer.appendChild(cardClone);
  });
}

function fillDataInCard(cardClone, article) {
  const newsImg = cardClone.querySelector("#news-img");
  const newsTitle = cardClone.querySelector("#news-title");
  const newsSource = cardClone.querySelector("#news-source");
  const newsDesc = cardClone.querySelector("#news-desc");

  newsImg.src = article.urlToImage;
  newsTitle.innerHTML = article.title?.slice(0, 60) + "...";
  newsDesc.innerHTML = article.description?.slice(0, 150) + "...";

  const date = new Date(article.publishedAt).toLocaleString("en-US", {
    timeZone: "Asia/Jakarta",
  });

  newsSource.innerHTML = `${article.source.name} · ${date}`;

  cardClone.firstElementChild.addEventListener("click", () => {
    window.open(article.url, "_blank");
  });
}

// ---------------------------
// Navigation
// ---------------------------
let curSelectedNav = null;

function onNavItemClick(id) {
  fetchNews(id).then((articles) => bindData(articles));

  const navItem = document.getElementById(id);
  if (curSelectedNav) curSelectedNav.classList.remove("active");
  curSelectedNav = navItem;
  curSelectedNav.classList.add("active");
}

// ---------------------------
// Search
// ---------------------------
const searchButton = document.getElementById("search-button");
const searchText = document.getElementById("search-text");

if (searchButton) {
  searchButton.addEventListener("click", () => {
    const query = searchText.value.trim();
    if (!query) return;
    fetchNews(query).then((articles) => bindData(articles));
    if (curSelectedNav) curSelectedNav.classList.remove("active");
    curSelectedNav = null;
  });
}

// ---------------------------
// Feed Page
// ---------------------------
function renderFeed() {
  mainContent.innerHTML = `
    <div id="articles"></div>
    <h1 style="margin-top:2rem;">Live News</h1>
    <div class="cards-container container flex" id="cardscontainer"></div>
  `;

  // Fetch recommended news from backend
  fetch("http://localhost:8000/api/recommend")
    .then((res) => res.json())
    .then((articles) => {
      const articlesDiv = document.getElementById("articles");
      articlesDiv.innerHTML = articles
        .map(
          (article) => `
        <div class="card">
          <h2>${article.title}</h2>
          <p>${article.content}</p>
          ${
            article.url
              ? `<a href="${article.url}" target="_blank">Read more</a>`
              : ""
          }
        </div>
      `
        )
        .join("");
    });

  // Default live news
  fetchNews("Technology").then((articles) => bindData(articles));
}

// ---------------------------
// Verify Page
// ---------------------------
function renderVerify() {
  mainContent.innerHTML = `
    <h1>Verify News</h1>
    <textarea id="news-text" rows="5" style="width:100%;margin-bottom:1rem;" placeholder="Paste news text here..."></textarea>
    <div class="upload-box">
      <input type="file" id="ocr-file" accept="image/*">
      <p>Upload an image to extract text (OCR)</p>
      <div class="language-options">
        <label><input type="radio" name="lang" value="ara" checked> Arabic</label>
        <label><input type="radio" name="lang" value="eng"> English</label>
      </div>
      <button class="button" id="ocr-only-btn" style="margin-top:0.5rem;">Read Text from Image</button>
    </div>
    <button class="button" id="classify-btn" style="margin-top:0.5rem;">Classify</button>
    <div id="verify-result" style="margin-top:1rem;"></div>
    <div id="ocr-result" style="margin-top:0.5rem;color:#6b7280;"></div>
  `;

  const newsTextEl = document.getElementById("news-text");
  const ocrResultEl = document.getElementById("ocr-result");

  // OCR Button
  document.getElementById("ocr-only-btn").addEventListener("click", async () => {
    const fileInput = document.getElementById("ocr-file");
    const langInput = document.querySelector("input[name='lang']:checked");
    const button = document.getElementById("ocr-only-btn");

    if (!fileInput.files.length) {
      ocrResultEl.textContent = "Please select an image first.";
      return;
    }

    const langParam = langInput.value;
    const isArabic = langParam === "ara";

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);
    formData.append("lang", langParam);

    const originalButtonText = button.textContent;
    button.disabled = true;
    button.textContent = "Processing...";
    ocrResultEl.textContent = "Processing image...";
    ocrResultEl.classList.remove("error");

    try {
      const response = await fetch("http://localhost:8000/api/ocr", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      if (!response.ok) throw new Error(data?.error || `Server error: ${response.status}`);

      newsTextEl.value = data.text || "";
      newsTextEl.dir = isArabic ? "rtl" : "ltr";
      newsTextEl.style.textAlign = isArabic ? "right" : "left";
      ocrResultEl.textContent = "Text extracted successfully!";
    } catch (err) {
      console.error("OCR Error:", err);
      ocrResultEl.textContent = `Error: ${err.message}`;
      ocrResultEl.classList.add("error");
    } finally {
      button.disabled = false;
      button.textContent = originalButtonText;
    }
  });

  // Classify Button
  document.getElementById("classify-btn").addEventListener("click", async () => {
    const text = newsTextEl.value.trim();
    if (!text) return;

    const resultEl = document.getElementById("verify-result");
    resultEl.textContent = "Processing...";

    try {
      const response = await fetch("http://localhost:8000/api/classify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (!response.ok) throw new Error("Classification failed");

      const result = await response.json();
      resultEl.innerHTML = `
        <div>Label: <b>${result.label}</b></div>
        <div>Score: ${result.score.toFixed(3)}</div>
      `;
    } catch (error) {
      resultEl.textContent = `Error: ${error.message}`;
      console.error("Classification error:", error);
    }
  });
}

// ---------------------------
// Routing
// ---------------------------
function handleHashChange() {
  if (location.hash === "#verify") renderVerify();
  else renderFeed();
}

// Attach link events
document.getElementById("verify-link").addEventListener("click", () => {
  location.hash = "#verify";
});

// Navigation links
["breaking", "politics", "sports", "gaza", "frontpage"].forEach((id) => {
  const el = document.getElementById(id);
  if (el) el.addEventListener("click", () => onNavItemClick(id));
});

// Listen to hash changes
window.addEventListener("hashchange", handleHashChange);

// Initial render
handleHashChange();
