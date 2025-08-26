Fake News Detection & Bilingual News Recommendation (Phase 3–4)

What’s included (code only)
- FastAPI service: /api/recommend (retrieval → optional re‑ranking)
- Transient FAISS index loader + filters (lang/recency/per‑source)
- Worker/scripts: build/verify FAISS, fetch_and_index, train/eval ranker
- Ranker: Logistic Regression (default) and optional XGBoost

Quick start (serve recommendations)
1) Prepare an index directory with FAISS files (faiss.index, item_vectors.npy, article_ids.npy, meta.json). You can create a small sample by running the worker one‑shot with a local JSON.
2) Run API:
	- PowerShell: set INDEX_DIR then start uvicorn.
3) Test: POST /api/recommend returns an array of items; GET /api/recommend returns cold‑start.

Important
- Embeddings and FAISS files are NOT stored in git (too large). Generate locally via:
	- scripts/build_embeddings.py (if available) or scripts/build_index.py from your precomputed embeddings
	- scripts/fetch_and_index.py (fetch → embed → index) for a quick demo

Train ranker (Phase 4)
1) Datasets
	- Train: data/MINDsub_train_100k or data/MINDsub_train_200k (news.tsv + behaviors.tsv)
	- Dev: data/MINDlarge_dev (news.tsv + behaviors.tsv)
2) Install training deps (inside your venv)
	- pip install -r requirements-train.txt
3) Train (two methods)
	 - Method A — Logistic Regression (faster, stable):
		 - scripts/train_ranker.py --data_dir data/MINDsub_train_200k --out_dir models/ranker --model logreg
	 - Method B — XGBoost (potentially more accurate, slower):
		 - scripts/train_ranker.py --data_dir data/MINDsub_train_200k --out_dir models/ranker --model xgb
4) Evaluate (dev)
	- You must choose a model explicitly (no auto):
	  - scripts/eval_ranker.py --data_dir data/MINDlarge_dev --model_dir models/ranker --k 10 --prefer logreg
	  - scripts/eval_ranker.py --data_dir data/MINDlarge_dev --model_dir models/ranker --k 10 --prefer xgb
5) Serve with re‑ranking
	- API picks models/ranker/ranker.pkl (logreg) if available; otherwise ranker.xgb; falls back to retrieval if no model.


Method comparison (observed on our runs)
- Speed: Logistic Regression is faster to train and evaluate; XGBoost is slower.
- Accuracy: On our dev split, Logistic Regression gave better AUC/NDCG in this setup; XGBoost can improve with more tuning/rounds.

Commands we used (examples)
- Train (200k):
	- scripts/train_ranker.py --data_dir data/MINDsub_train_200k --out_dir models/ranker --model logreg
	- scripts/train_ranker.py --data_dir data/MINDsub_train_200k --out_dir models/ranker --model xgb
- Evaluate (dev):
	- scripts/eval_ranker.py --data_dir data/MINDlarge_dev --model_dir models/ranker --k 10 --prefer logreg
	- scripts/eval_ranker.py --data_dir data/MINDlarge_dev --model_dir models/ranker --k 10 --prefer xgb
- Chunked training (XGBoost):
	- First 50k:  --max_samples 50000
	- Second 50k: --skip_samples 50000 --resume_from models/ranker/ranker.xgb --num_boost_round 200

Troubleshooting
- If you see MemoryError during training, use caps: --limit_sessions or --max_samples.
- If xgboost isn’t installed, use --model logreg or install xgboost in your venv.

Runtime/training dependencies
Run commands (PowerShell examples)
- Serve API
	- $env:INDEX_DIR=".\models\index"; uvicorn app.main:app --host 127.0.0.1 --port 8011 --reload
- Worker one‑shot to create a tiny index
	- python scripts/fetch_and_index.py --one_shot --json_file .\models\index\sample_articles.json --out_dir .\models\index
- Train (200k)
	- python scripts/train_ranker.py --data_dir data/MINDsub_train_200k --out_dir models/ranker --model logreg
	- python scripts/train_ranker.py --data_dir data/MINDsub_train_200k --out_dir models/ranker --model xgb
- Evaluate (dev)
	- python scripts/eval_ranker.py --data_dir data/MINDlarge_dev --model_dir models/ranker --k 10 --prefer logreg
	- python scripts/eval_ranker.py --data_dir data/MINDlarge_dev --model_dir models/ranker --k 10 --prefer xgb
- Runtime: fastapi, uvicorn, numpy, faiss-cpu, pydantic
- Training: scikit-learn, joblib, numpy, scipy, optional xgboost

