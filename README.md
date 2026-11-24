# StockSense AI — Stock Price Predictor

Educational project combining Machine Learning, optional Generative AI explanations, and cloud deployment.

## Run locally
Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501. Example tickers: AAPL, MSFT, RELIANCE.NS, TCS.NS, INFY.NS.

## Enable Gemini
Get an API key from Google AI Studio and set `GEMINI_API_KEY` as an environment variable. For Streamlit Cloud, add this in App settings → Secrets:

```toml
GEMINI_API_KEY = "your_api_key"
```
Never commit API keys to GitHub. The AI explanation is optional; prediction works without it.

## Deploy to Streamlit Community Cloud
1. Upload app.py, requirements.txt and README.md to a GitHub repository.
2. Sign in at https://share.streamlit.io/.
3. Select the repository and app.py entry point, then deploy.
4. Add Gemini key in app secrets if needed.

## Docker / AWS
```bash
docker build -t stocksense-ai .
docker run -p 8501:8501 -e GEMINI_API_KEY="your_api_key" stocksense-ai
```
Publish the image to Amazon ECR and deploy using AWS App Runner or ECS. Set port 8501 and provide secrets securely.

## Workflow
Yahoo Finance historical data → feature engineering (returns, moving averages, volatility, volume change) → Random Forest regression → chronological holdout MAE/RMSE evaluation → multi-day experimental estimates → optional Gemini explanation → Streamlit dashboard.

## Limitations
This is not financial advice or a trading system. Forecasts are uncertain and recursively generated; holdout error metrics do not guarantee future accuracy. Yahoo Finance data can be delayed, incomplete, rate-limited or unavailable.
