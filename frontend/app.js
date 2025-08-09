const mainContent = document.getElementById('main-content');

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

  // Classify Button Handler
  document.getElementById('classify-btn').onclick = async () => {
    const text = document.getElementById('news-text').value.trim();
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

  // OCR Button Handler
  document.getElementById('ocr-only-btn').addEventListener('click', async () => {
    const fileInput = document.getElementById('ocr-file');
    const ocrResultEl = document.getElementById('ocr-result');
    const newsTextEl = document.getElementById('news-text');
    const button = document.getElementById('ocr-only-btn');

    // Validate inputs
    if (!fileInput.files.length) {
      ocrResultEl.textContent = 'Please select an image first.';
      return;
    }

    const langInput = document.querySelector('input[name="lang"]:checked');
    if (!langInput) {
      ocrResultEl.textContent = 'Please select a language.';
      return;
    }

    // Prepare request
    const file = fileInput.files[0];
    const langParam = langInput.value;
    const isArabic = langParam === 'ara';

    // Validate file type
    if (!file.type.match('image.*')) {
      ocrResultEl.textContent = 'Please upload an image file (JPEG/PNG).';
      return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('lang', langParam);

    // UI state
    const originalButtonText = button.textContent;
    button.disabled = true;
    button.textContent = 'Processing...';
    ocrResultEl.textContent = 'Processing image...';
    ocrResultEl.classList.remove('error');

    try {
      // API call
      const response = await fetch('http://localhost:8000/api/ocr', {
        method: 'POST',
        body: formData
      });

      // Handle response
      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        throw new Error(errorData?.error || `Server error: ${response.status}`);
      }

      const data = await response.json();

      // Apply RTL styling if Arabic
      if (isArabic) {
        ocrResultEl.classList.add('rtl');
        newsTextEl.classList.add('rtl');
        newsTextEl.style.textAlign = 'right';
        newsTextEl.dir = 'rtl';
      } else {
        ocrResultEl.classList.remove('rtl');
        newsTextEl.classList.remove('rtl');
        newsTextEl.style.textAlign = 'left';
        newsTextEl.dir = 'ltr';
      }

      // Update UI with results
      ocrResultEl.textContent = 'Extracted Text:';
      newsTextEl.value = data.text;
      console.log('OCR Success:', data);

    } catch (error) {
      console.error('OCR Error:', error);
      ocrResultEl.textContent = `Error: ${error.message}`;
      ocrResultEl.classList.add('error');
    } finally {
      button.disabled = false;
      button.textContent = originalButtonText;
    }
  });
}


function handleHashChange() {
  if (location.hash === '#verify') {
    renderVerify();
  } else {
    renderFeed();
  }
}

document.getElementById('feed-link').onclick = () => { location.hash = '#feed'; };
document.getElementById('verify-link').onclick = () => { location.hash = '#verify'; };
window.addEventListener('hashchange', handleHashChange);

// Initial render
handleHashChange();
