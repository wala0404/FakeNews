const mainContent = document.getElementById('main-content');

// ----------- Feed -----------
function renderFeed() {
  mainContent.innerHTML = `
    <h1>Recommended News</h1>
    <div id="articles"></div>
  `;

  fetch('http://localhost:8000/api/recommend')
    .then(res => res.json())
    .then(articles => {
      const articlesDiv = document.getElementById('articles');
      articlesDiv.innerHTML = articles.map(article => `
        <div class="card">
          <h2>${article.title}</h2>
          <p>${article.content}</p>
          ${article.url ? `<a href="${article.url}" target="_blank">Read more</a>` : ''}
        </div>
      `).join('');
    });
}

// ----------- Verify -----------
function renderVerify() {
  mainContent.innerHTML = `
    <h1>Verify News</h1>
    <textarea id="news-text" rows="5" style="width:100%;margin-bottom:1rem;"></textarea>
    <div class="upload-box">
      <input type="file" id="ocr-file" accept="image/*">
      <p>Upload an image to extract text (OCR)</p>
      <div class="language-options">
        <label><input type="radio" name="lang" value="ara" checked> Arabic</label>
        <label><input type="radio" name="lang" value="eng"> English</label>
      </div>
      <button class="button" id="ocr-only-btn" style="margin-top:0.5rem;">Read Text from Image</button>
    </div>
    <button class="button" id="classify-btn">Classify</button>
    <div id="verify-result" style="margin-top:1rem;"></div>
    <div id="ocr-result" style="margin-top:0.5rem;color:#6b7280;"></div>
  `;

  const newsTextEl = document.getElementById('news-text');
  const ocrResultEl = document.getElementById('ocr-result');

  // ----------- OCR Button Handler -----------
  document.getElementById('ocr-only-btn').addEventListener('click', async () => {
    const fileInput = document.getElementById('ocr-file');
    const langInput = document.querySelector('input[name="lang"]:checked');
    const button = document.getElementById('ocr-only-btn');

    if (!fileInput.files.length) {
      ocrResultEl.textContent = 'Please select an image first.';
      return;
    }
    if (!langInput) {
      ocrResultEl.textContent = 'Please select a language.';
      return;
    }

    const langParam = langInput.value;
    const isArabic = langParam === 'ara';

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('lang', langParam);

    // Update UI
    const originalButtonText = button.textContent;
    button.disabled = true;
    button.textContent = 'Processing...';
    ocrResultEl.textContent = 'Processing image...';
    ocrResultEl.classList.remove('error');

    try {
      const response = await fetch('http://localhost:8000/api/ocr', {
        method: 'POST',
        body: formData
      });

      const data = await response.json();
      if (!response.ok) throw new Error(data?.error || `Server error: ${response.status}`);

      console.log('OCR Response:', data);

      // Set text direction
      newsTextEl.value = data.text || '';
      newsTextEl.dir = isArabic ? 'rtl' : 'ltr';
      newsTextEl.style.textAlign = isArabic ? 'right' : 'left';
      ocrResultEl.textContent = 'Text extracted successfully!';

    } catch (err) {
      console.error('OCR Error:', err);
      ocrResultEl.textContent = `Error: ${err.message}`;
      ocrResultEl.classList.add('error');
    } finally {
      button.disabled = false;
      button.textContent = originalButtonText;
    }
  });

  // ----------- Classify Button Handler -----------
  document.getElementById('classify-btn').onclick = async () => {
    const text = newsTextEl.value.trim();
    if (!text) return;

    const resultEl = document.getElementById('verify-result');
    resultEl.textContent = 'Processing...';

    try {
      const response = await fetch('http://localhost:8000/api/classify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });
      if (!response.ok) throw new Error('Classification failed');

      const result = await response.json();
      resultEl.innerHTML = `
        <div>Label: <b>${result.label}</b></div>
        <div>Score: ${result.score.toFixed(3)}</div>
      `;
    } catch (error) {
      resultEl.textContent = `Error: ${error.message}`;
      console.error('Classification error:', error);
    }
  };
}

// ----------- Routing -----------
function handleHashChange() {
  if (location.hash === '#verify') renderVerify();
  else renderFeed();
}

document.getElementById('feed-link').onclick = () => { location.hash = '#feed'; };
document.getElementById('verify-link').onclick = () => { location.hash = '#verify'; };
window.addEventListener('hashchange', handleHashChange);

// Initial render
handleHashChange();
