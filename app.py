import os
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title='StockSense AI', page_icon='📈', layout='wide')
st.markdown('''<style>.stApp{background:linear-gradient(135deg,#07111f,#0b1830 60%,#101d36);color:#eef4ff} [data-testid="stSidebar"]{background:#0c1930} h1,h2,h3{color:#f3f7ff!important} div[data-testid="stMetric"]{background:#132541;border:1px solid #254365;padding:15px;border-radius:14px} .stButton>button{background:linear-gradient(90deg,#2563eb,#06b6d4);color:white;border:0;border-radius:9px;font-weight:700}</style>''', unsafe_allow_html=True)
st.title('📈 StockSense AI')
st.caption('Machine-learning stock estimates • Generative AI explanations • Cloud-ready dashboard')
with st.sidebar:
    st.header('⚙️ Dashboard settings')
    ticker=st.text_input('Stock ticker','AAPL').strip().upper()
    period=st.selectbox('Historical data',['1y','2y','5y','10y'],index=1)
    days=st.slider('Forecast horizon (trading days)',1,30,5)
    st.caption('Examples: AAPL · MSFT · RELIANCE.NS · TCS.NS · INFY.NS')
    st.button('Analyze stock',use_container_width=True)

@st.cache_data(ttl=900)
def load_data(symbol,hist_period):
    df=yf.download(symbol,period=hist_period,interval='1d',auto_adjust=True,progress=False)
    if df is None or df.empty: return pd.DataFrame()
    if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
    df=df.rename_axis('Date').reset_index()
    if not all(c in df.columns for c in ['Date','Close','Volume']): return pd.DataFrame()
    df['Date']=pd.to_datetime(df['Date'])
    return df[['Date','Close','Volume']].dropna()

def features(df):
    x=df.copy(); x['ret']=x.Close.pct_change(); x['ma5']=x.Close.rolling(5).mean(); x['ma10']=x.Close.rolling(10).mean(); x['ma20']=x.Close.rolling(20).mean(); x['volatility']=x.ret.rolling(10).std(); x['volume_change']=x.Volume.pct_change(); x['target']=x.Close.shift(-1)
    return x.replace([np.inf,-np.inf],np.nan).dropna()

def predict(df,n):
    cols=['Close','Volume','ret','ma5','ma10','ma20','volatility','volume_change']; f=features(df)
    if len(f)<40: raise ValueError('Not enough data. Select a longer historical period.')
    cut=int(len(f)*.8); model=RandomForestRegressor(n_estimators=200,max_depth=8,random_state=42)
    model.fit(f[cols].iloc[:cut],f.target.iloc[:cut]); test=model.predict(f[cols].iloc[cut:]); mae=mean_absolute_error(f.target.iloc[cut:],test); rmse=np.sqrt(mean_squared_error(f.target.iloc[cut:],test))
    model.fit(f[cols],f.target); hist=df.copy(); rows=[]
    for _ in range(n):
        ff=features(hist); price=float(model.predict(ff[cols].tail(1))[0]); dt=hist.Date.iloc[-1]+pd.tseries.offsets.BDay(1); rows.append((dt,price)); hist=pd.concat([hist,pd.DataFrame([{'Date':dt,'Close':price,'Volume':float(df.Volume.iloc[-1])}])],ignore_index=True)
    return pd.DataFrame(rows,columns=['Date','Predicted Close']),mae,rmse

with st.spinner('Loading market data...'): data=load_data(ticker,period)
if data.empty:
    st.error('No data found. Check ticker/internet. Indian symbols often use .NS, e.g. RELIANCE.NS.'); st.stop()
try: forecast,mae,rmse=predict(data,days)
except Exception as e: st.error(str(e)); st.stop()
latest=float(data.Close.iloc[-1]); prev=float(data.Close.iloc[-2]); end=float(forecast['Predicted Close'].iloc[-1])
a,b,c,d=st.columns(4); a.metric('Latest close',f'{latest:,.2f}',f'{(latest/prev-1)*100:+.2f}%'); b.metric(f'Estimate in {days} days',f'{end:,.2f}',f'{(end/latest-1)*100:+.2f}%'); c.metric('Backtest MAE',f'{mae:,.3f}'); d.metric('Backtest RMSE',f'{rmse:,.3f}')
st.subheader('📊 Historical price and model estimate')
fig=go.Figure(); fig.add_trace(go.Scatter(x=data.Date,y=data.Close,name='Historical close',line=dict(color='#38bdf8',width=2))); fig.add_trace(go.Scatter(x=forecast.Date,y=forecast['Predicted Close'],name='Model estimate',line=dict(color='#a78bfa',width=3,dash='dash'))); fig.update_layout(template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(8,20,38,.7)',height=450,margin=dict(l=10,r=10,t=20,b=10)); st.plotly_chart(fig,use_container_width=True)
left,right=st.columns(2)
with left:
    st.subheader('🔮 Forecast table'); st.dataframe(forecast,use_container_width=True,hide_index=True); st.download_button('⬇️ Download forecast CSV',forecast.to_csv(index=False).encode(),file_name=f'{ticker}_forecast.csv',mime='text/csv')
with right:
    st.subheader('✨ Generative AI explanation'); st.caption('Optional Gemini API integration. Add GEMINI_API_KEY to environment or Streamlit secrets.')
    if st.button('Generate AI explanation'):
        key=os.getenv('GEMINI_API_KEY') or st.secrets.get('GEMINI_API_KEY','')
        if not key: st.warning('Gemini key not configured. Add GEMINI_API_KEY to enable this feature.')
        else:
            try:
                import google.generativeai as genai
                genai.configure(api_key=key); gm=genai.GenerativeModel('gemini-1.5-flash')
                prompt=f'Explain these educational model outputs simply. Do not give investment advice or claim certainty. Ticker {ticker}; latest close {latest:.2f}; model estimate after {days} trading days {end:.2f}; backtest MAE {mae:.3f}; RMSE {rmse:.3f}. Explain uncertainty and limitations.'
                st.write(gm.generate_content(prompt).text)
            except Exception as e: st.error(f'Gemini error: {e}')
st.markdown('---')
st.subheader('☁️ Cloud deployment'); st.write('Deploy on Streamlit Community Cloud from GitHub, or build the included Docker image and deploy it on AWS App Runner/ECS. Configure port 8501 and store API keys as cloud secrets.')
st.caption('Educational prototype only — not financial advice. Forecasts are experimental estimates, not guaranteed future prices. Market data availability depends on Yahoo Finance.')
