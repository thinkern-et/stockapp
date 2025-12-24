#streamlit 배포용 코드 정리
import streamlit as st
import yfinance as yf
import pandas as pd
import ta
from prophet import Prophet
import matplotlib.pyplot as plt
import datetime
import warnings

# 설정 및 경고 무시
warnings.filterwarnings('ignore')
st.set_page_config(page_title="나만의 AI 주식 예측 대시보드", layout="wide")

# --- 데이터 분석 함수 ---
def get_stock_data(ticker):
    df = yf.download(ticker, period='2y', interval='1d', progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def run_analysis(df, ticker):
    # 데이터가 비어있거나 너무 적은지 확인 (최소 20일치 권장)
    if df is None or len(df) < 20:
        raise ValueError(f"{ticker}: 분석을 위한 충분한 데이터가 없습니다. (현재 {len(df) if df is not None else 0}개)")

    close_series = df['Close'].squeeze()
    
    # RSI 계산
    rsi_series = ta.momentum.rsi(close_series, window=14)
    
    # 계산 결과가 유효한지 다시 확인
    if rsi_series.dropna().empty:
        rsi = 50.0 # 데이터 부족 시 중립 값 부여 혹은 에러 발생
    else:
        rsi = rsi_series.iloc[-1]
    vol_focus = df['Volume'].iloc[-1] / df['Volume'].rolling(window=20).mean().iloc[-1]
    
    # Prophet 예측
    p_df = df[['Close']].reset_index()
    p_df.columns = ['ds', 'y']
    p_df['ds'] = p_df['ds'].dt.tz_localize(None)
    
    model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    model.fit(p_df)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    
    curr_p = float(close_series.iloc[-1])
    pred_p = float(forecast['yhat'].iloc[-1])
    return_pct = (pred_p - curr_p) / curr_p * 100
    
    return {
        'model': model,
        'forecast': forecast,
        'current_p': curr_p,
        'pred_p': pred_p,
        'return_pct': return_pct,
        'rsi': rsi,
        'vol_focus': vol_focus
    }

# --- Markdown 보고서 생성 함수 ---
def generate_report(summary_df):
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    md = f"# 🤖 AI 투자 전략 보고서\n"
    md += f"> **작성일자:** {now} | **분석 모델:** Prophet & TA\n\n"
    md += "## 📊 분석 요약\n"
    md += "| 티커 | 현재가 | 예측가 | 수익률 | RSI | 화제성 |\n"
    md += "| :--- | :--- | :--- | :--- | :--- | :--- |\n"
    
    for _, row in summary_df.iterrows():
        md += f"| {row['Ticker']} | {row['Current']} | {row['Predicted']} | **{row['Return%']}%** | {row['RSI']} | {row['Vol_Focus']} |\n"
    
    md += "\n## 💡 상세 의견\n"
    for _, row in summary_df.iterrows():
        status = "관망"
        if row['Return%'] > 7 and row['RSI'] < 70: status = "✅ 적극 매수"
        elif row['Return%'] > 0: status = "🟡 보유/추적"
        elif row['RSI'] > 75: status = "⚠️ 과매수 주의"
        
        md += f"### 🔍 {row['Ticker']}: {status}\n"
        md += f"- 예상 수익률: {row['Return%']}% | RSI: {row['RSI']} | 화제성: {row['Vol_Focus']}\n\n"
    return md

# --- UI 레이아웃 ---
st.title("🚀 나만의 AI 주식 예측 대시보드")
st.sidebar.header("🛠️ 설정")

mode = st.sidebar.radio("분석 모드 선택", ["단일 종목 상세 분석", "주요 종목 일괄 분석"])

if mode == "단일 종목 상세 분석":
    ticker = st.sidebar.text_input("티커 입력 (예: AAPL, 005930.KS)", "AAPL")
    if st.sidebar.button("분석 시작"):
        with st.spinner(f'{ticker} 분석 중...'):
            df = get_stock_data(ticker)
            res = run_analysis(df, ticker)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("현재가", f"{res['current_p']:.2f}")
            col2.metric("30일 후 예측", f"{res['pred_p']:.2f}", f"{res['return_pct']:.2f}%")
            col3.metric("RSI (상대강도)", f"{res['rsi']:.2f}")

            st.subheader("📈 향후 30일 가격 예측 차트")
            fig = res['model'].plot(res['forecast'])
            plt.axvline(x=df.index[-1], color='red', linestyle='--')
            st.pyplot(fig)

elif mode == "주요 종목 일괄 분석":
    kr_tickers = ['005930.KS', '000660.KS', '005490.KS', '035420.KS', '035720.KS', '005380.KS', '051910.KS', '207940.KS', '006400.KS', '068270.KS']
    us_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'AMD', 'AVGO']
    
    if st.sidebar.button("20개 종목 일괄 분석 실행"):
        all_results = []
        progress_bar = st.progress(0)
        combined_tickers = kr_tickers + us_tickers
        
        for i, ticker in enumerate(combined_tickers):
            try:
                df = get_stock_data(ticker)
                res = run_analysis(df, ticker)
                all_results.append({
                    'Ticker': ticker, 'Current': round(res['current_p'], 2),
                    'Predicted': round(res['pred_p'], 2), 'Return%': round(res['return_pct'], 2),
                    'RSI': round(res['rsi'], 2), 'Vol_Focus': round(res['vol_focus'], 2)
                })
            except:
                st.error(f"{ticker} 데이터 분석 실패")
            progress_bar.progress((i + 1) / len(combined_tickers))
            
        summary_df = pd.DataFrame(all_results).sort_values('Return%', ascending=False)
        
        st.subheader("📊 종합 분석 요약 리스트")
        st.dataframe(summary_df, use_container_width=True)

        markdown_output = generate_report(summary_df)
        report_md = generate_report(summary_df)
        st.subheader("📝 자동 생성된 투자 전략 보고서")
        st.markdown(markdown_output)
        
        st.download_button("보고서 다운로드 (.md)", markdown_output, "investment_report.md")
