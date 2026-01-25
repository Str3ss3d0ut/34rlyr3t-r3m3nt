import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import io
from datetime import datetime, timedelta

# --- 1. App Title and Setup ---
st.set_page_config(page_title="My Alpha Screener", layout="wide")
st.title("🚀 Personal Stock Alpha Screener (Future Wealth Edition)")

# --- Sector Map ---
SECTOR_MAP = {
    "NVDA": "Applied AI", "TSLA": "EV/Robotics", "AAPL": "Consumer Tech",
    "MSFT": "Applied AI", "AMD": "Applied AI", "AMZN": "E-Commerce",
    "GOOGL": "AdTech", "META": "AdTech", "NFLX": "Streaming",
    "COIN": "Crypto", "PLTR": "Defense/AI", "SOFI": "Fintech",
    "ROKU": "Streaming", "SHOP": "E-Commerce", "SQ": "Fintech", 
    "MSTR": "Crypto", "SNDK": "Storage", "LRCX": "Semi Equip", "MU": "Memory",
    "SMCI": "AI Server", "ULTA": "Retail", "CMI": "Industrial", "WBD": "Media"
}

# Default Universe
default_tickers = (
    "SNDK, WDC, MU, TER, STX, LRCX, INTC, WBD, ALB, FIX, NEM, CHRW, AMAT, GLW, GOOGL, GOOG, KLAC, HII, GM, CMI, CNC, HAL, CAT, APH, MRNA, BG, MPWR, TMO, LMT, VTRS, SLB, APA, EXPD, ROST, IVZ, LLY, JBHT, CRL, MRK, WAT, GS, MNST, STLD, ULTA, DG, BKR, DD, TECH, CVS, FDX, RTX, CAH, EL, CVNA, FCX, CFG, ELV, LHX, MS, PH, JNJ, VLO, KEYS, FOXA, ADI, EA, RL, IQV, FOX, DHR, AIZ, PCAR, NUE, EPAM, F, FITB, NDSN, TPR, XOM, WMT, TJX, DAL, ADM, LOW, GD, AME, GE, HWM, USB, BK, PLD, IBKR, WSM, RVTY, PWR, ROK, LDOS, EIX, CBRE, GILD, NOC, KEY, TRGP, MTD, DAY, MAR, VTR, SWK, STE, HST, DVN, AKAM, PNC, LUV, BMY, FE, PFG, NTRS, CTRA, HAS, BDX, HOLX, JBL, ROL, HBAN, SPG, PSX, AMGN, CVX, GL, SNA, TFC, HUBB, CBOE, MTB, COO, MLM, MSCI, EVRG, TSN, VMC, WMB, TXT, TDY, NDAQ, EW, RF, TPL, UPS, DOV, CTSH, MDT, WAB, O, AEP, NEE"
)

# --- SIDEBAR INPUTS ---
st.sidebar.header("User Input")

with st.sidebar.expander("📝 Edit Watchlist", expanded=False):
    ticker_input = st.text_area("Enter Tickers:", default_tickers, height=150)

stop_loss_pct = st.sidebar.slider("Trailing Stop Loss %", 5, 40, 30, 1) / 100
max_positions = st.sidebar.slider("Number of Positions", 1, 5, 1, help="1 = Sniper Mode.")
allocation_pct = st.sidebar.slider("Max Capital Allocation %", 50, 100, 80, 5) / 100

# [CRITICAL] Volume Sensitivity Slider
vol_threshold = st.sidebar.slider(
    "Min Volume Ratio (1.0 = Average)", 
    0.1, 2.0, 1.0, 0.1, 
    help="1.0 is standard. 1.2 is aggressive breakouts. Lower to 0.8 if stocks say 'WAIT'."
)

st.sidebar.divider()
st.sidebar.header("🛡️ Future Preservation Mode")
use_vol_target = st.sidebar.checkbox("Enable Volatility Targeting?", value=False, help="Recommended when account > $100k. Reduces size on risky stocks.")
target_vol_ann = st.sidebar.slider("Target Annual Volatility %", 10, 50, 20, 5) / 100

st.sidebar.divider()
st.sidebar.header("💰 Wealth Accelerator")
account_balance_input = st.sidebar.number_input("Starting Account Balance ($)", 100, value=1400, step=100)
monthly_contribution = st.sidebar.slider("Monthly Savings Add ($)", 0, 2000, 500, 50)

tickers = [x.strip().upper() for x in ticker_input.split(',') if x.strip()]

# --- DAILY TRADING TOOLS ---
st.sidebar.markdown("---")
st.sidebar.subheader("🧮 Daily Tools")
with st.sidebar.expander("Stop Loss Calculator", expanded=True):
    st.caption("Enter the Highest Price the stock has hit:")
    curr_high = st.number_input("Stock High ($)", value=0.0)
    if curr_high > 0:
        st.error(f"Set Broker Stop Order to: **${curr_high * (1 - stop_loss_pct):.2f}**")

with st.sidebar.expander("🏠 House Money Tracker", expanded=True):
    entry_p = st.number_input("Your Entry Price ($)", value=0.0)
    if entry_p > 0 and curr_high > 0:
        current_stop = curr_high * (1 - stop_loss_pct)
        target_price = entry_p / (1 - stop_loss_pct)
        
        if current_stop >= entry_p:
            st.success("🎉 HOUSE MONEY! Stop is above Entry. Risk is $0.")
            st.progress(1.0)
        else:
            diff = target_price - curr_high
            progress = max(0.0, min(1.0, (curr_high - entry_p) / (target_price - entry_p))) if target_price > entry_p else 0.0
            st.info(f"Price needs to hit **${target_price:.2f}** to be safe (+${diff:.2f}).")
            st.progress(progress)

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Live Screener & Sizing", "📉 Backtest & Wealth Sim", "🔎 S&P 500/400 Scanner"])

# ==========================================
# TAB 1: LIVE SCREENER (TRANSPARENT SIGNALS)
# ==========================================
with tab1:
    with st.expander("👮 Pre-Flight Checklist"):
        st.markdown("""
        1. **Earnings Check:** Does the stock report earnings in the next 5 days? (If YES -> WAIT).
        2. **Time Check:** Is it the last trading day of the month?
        """)
    
    use_live_regime = st.checkbox("🛑 Enable Market Safety Lock (SPY > 200 SMA)?", value=True)

    if st.button("Find Alpha (Live Scan)"):
        market_healthy = True
        if use_live_regime:
            with st.spinner("Checking Market Health..."):
                try:
                    spy = yf.Ticker("SPY").history(period="1y")
                    if not spy.empty:
                        spy_curr = spy['Close'].iloc[-1]
                        spy_sma = spy['Close'].rolling(200).mean().iloc[-1]
                        if spy_curr < spy_sma:
                            market_healthy = False
                            st.error(f"⚠️ **BEAR MARKET DETECTED!** SPY (${spy_curr:.0f}) < 200-SMA (${spy_sma:.0f}). Logic says: **CASH**.")
                except: st.warning("⚠️ Could not check SPY. Proceeding carefully.")

        if market_healthy:
            status_text = st.empty()
            status_text.text(f"⏳ Downloading data for {len(tickers)} stocks (Batch Mode)...")
            
            try:
                data = yf.download(tickers, period="6mo", group_by='ticker', auto_adjust=True, threads=True)
                status_text.text("✅ Data Downloaded. Processing Signals...")
                
                alpha_data = []
                for symbol in tickers:
                    try:
                        if len(tickers) == 1:
                            hist = data
                        else:
                            if symbol not in data.columns.levels[0]:
                                continue
                            hist = data[symbol]

                        hist = hist.dropna(how='all')
                        if len(hist) < 30: continue

                        current_price = hist['Close'].iloc[-1]
                        current_vol_qty = hist['Volume'].iloc[-1]
                        daily_returns = hist['Close'].pct_change()
                        
                        # Volatility Calcs
                        daily_std = daily_returns.tail(20).std()
                        volatility_20 = daily_std * np.sqrt(20) # Monthly proxy for ranking
                        ann_volatility = daily_std * np.sqrt(252) # Annualized for Sizing
                        
                        if volatility_20 == 0: volatility_20 = 1
                        
                        avg_vol_20 = hist['Volume'].tail(20).mean()
                        vol_ratio = current_vol_qty / avg_vol_20 if avg_vol_20 > 0 else 0
                        
                        r1m = ((current_price / hist['Close'].iloc[-22]) - 1) * 100 if len(hist) >= 22 else 0
                        r3m = ((current_price / hist['Close'].iloc[-64]) - 1) * 100 if len(hist) >= 64 else 0
                        
                        sma_50 = hist['Close'].tail(50).mean()
                        trend_status = "UP" if current_price > sma_50 else "DOWN"
                        raw_momentum = (r1m * 2) + r3m
                        alpha_score = (raw_momentum / 100) / volatility_20
                        
                        signal = ""
                        if trend_status == "DOWN":
                            signal = "❄️ COLD"
                        elif alpha_score <= 0:
                            signal = "WAIT (Score)"
                        elif vol_ratio < vol_threshold:
                            signal = "WAIT (Low Vol)"
                        else:
                            signal = "🔥 HOT"

                        # --- SIZING LOGIC ---
                        # 1. Base Allocation (Sniper)
                        base_alloc = allocation_pct
                        
                        # 2. Volatility Target Adjustment (Preservation)
                        if use_vol_target and ann_volatility > 0:
                            vol_alloc = target_vol_ann / ann_volatility
                            final_alloc = min(base_alloc, vol_alloc) # Cap at max alloc
                        else:
                            final_alloc = base_alloc
                        
                        capital_per_trade = (account_balance_input * final_alloc) / max_positions
                        shares_to_buy = capital_per_trade / current_price
                        initial_stop = current_price * (1 - stop_loss_pct)

                        alpha_data.append({
                            "Symbol": symbol,
                            "Sector": SECTOR_MAP.get(symbol, "Growth"),
                            "Price": current_price,
                            "SHARES": shares_to_buy, 
                            "Stop Price": initial_stop,
                            "Signal": signal,
                            "Vol Ratio": vol_ratio,
                            "Score (Risk Adj)": alpha_score,
                            "Ann Vol": ann_volatility,
                            "Alloc %": final_alloc * 100,
                            "Link": f"https://finance.yahoo.com/quote/{symbol}"
                        })
                    except Exception as e:
                        continue

                status_text.empty()
                if alpha_data:
                    df = pd.DataFrame(alpha_data).sort_values(by="Score (Risk Adj)", ascending=False).reset_index(drop=True)
                    top_stock = df.iloc[0]
                    if top_stock['Signal'] == "🔥 HOT":
                        st.success(f"🏆 Top Alpha: **{top_stock['Symbol']}** | Buy **{top_stock['SHARES']:.4f}** shares")
                        if use_vol_target:
                            st.info(f"🛡️ Vol Target Active: Allocation reduced to **{top_stock['Alloc %']:.1f}%** due to volatility ({top_stock['Ann Vol']:.1%}).")
                        else:
                            st.info(f"🚀 Sniper Mode: Full **{top_stock['Alloc %']:.0f}%** Allocation.")
                        st.error(f"🛑 **Set Initial Stop:** ${top_stock['Stop Price']:.2f}")
                    else:
                        st.warning(f"🏆 Top stock ({top_stock['Symbol']}) is {top_stock['Signal']}. Reason: {top_stock['Signal']}")
                    
                    st.dataframe(
                        df,
                        column_config={
                            "Link": st.column_config.LinkColumn("News"),
                            "Score (Risk Adj)": st.column_config.ProgressColumn("Score", min_value=-5, max_value=5, format="%.2f"),
                            "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
                            "Vol Ratio": st.column_config.NumberColumn("Vol Ratio", format="%.2f"),
                            "Ann Vol": st.column_config.NumberColumn("Ann Vol", format="%.1%"),
                            "Alloc %": st.column_config.NumberColumn("Alloc %", format="%.1f%%"),
                        },
                        column_order=("Symbol", "Price", "Signal", "Alloc %", "SHARES", "Stop Price", "Vol Ratio", "Score (Risk Adj)", "Link")
                    )
                else:
                    st.error("❌ No data returned.")
            except Exception as e:
                st.error(f"Critical Download Error: {e}")

# ==========================================
# TAB 2: BACKTEST & WEALTH SIMULATION
# ==========================================
with tab2:
    st.header(f"Strategy Analysis & Wealth Simulation")
    st.caption(f"Starting with ${account_balance_input:,} and adding ${monthly_contribution:,}/month.")
    use_regime = st.checkbox("Enable SPY 200 SMA Regime Filter? (Optional)", value=False)

    if st.button("Run Full Wealth Analysis"):
        with st.spinner("Processing 5 years of data..."):
            all_tickers = tickers + ["SPY"]
            data = yf.download(all_tickers, period="5y", interval="1d", progress=False, threads=False, auto_adjust=True)
            if isinstance(data.columns, pd.MultiIndex):
                try: data = data['Close']
                except KeyError: pass
            
            daily_returns = data.pct_change()
            rolling_volatility = daily_returns.rolling(window=20).std() * np.sqrt(20) # Monthly Vol
            annualized_volatility = daily_returns.rolling(window=20).std() * np.sqrt(252) # Annual Vol for Sizing
            
            spy_daily = data["SPY"]
            spy_200_sma = spy_daily.rolling(window=200).mean()
            
            monthly_data = data.resample('ME').last()
            monthly_sma = spy_200_sma.resample('ME').last()
            monthly_vol = rolling_volatility.resample('ME').last()
            monthly_ann_vol = annualized_volatility.resample('ME').last()
            
            # --- START SIMULATION ---
            strat_bal, save_bal, bench_bal = account_balance_input, account_balance_input, account_balance_input
            results = []
            
            for i in range(4, len(monthly_data)-1):
                dt, nxt_dt = monthly_data.index[i], monthly_data.index[i+1]
                
                # Monthly Contribution
                strat_bal += monthly_contribution
                save_bal += monthly_contribution
                bench_bal += monthly_contribution
                
                scores = {}
                for tick in tickers:
                    if tick not in data.columns: continue
                    try:
                        ret_1m = (monthly_data.loc[dt, tick] - monthly_data.iloc[i-1][tick]) / monthly_data.iloc[i-1][tick]
                        ret_3m = (monthly_data.loc[dt, tick] - monthly_data.iloc[i-3][tick]) / monthly_data.iloc[i-3][tick]
                        vol = monthly_vol.loc[dt, tick]
                        if pd.isna(vol) or vol == 0: vol = 1.0
                        raw_momentum = (ret_1m * 2) + ret_3m
                        scores[tick] = (raw_momentum / vol) if raw_momentum > 0 else raw_momentum
                    except: scores[tick] = -999 
                
                top_picks = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:max_positions]
                
                global_bull = True
                if use_regime and monthly_data.loc[dt, "SPY"] < monthly_sma.loc[dt]: global_bull = False

                m_rets = []
                if global_bull:
                    for s_name, s_score in top_picks:
                        if s_score > 0:
                            # --- SIZING LOGIC FOR BACKTEST ---
                            curr_ann_vol = monthly_ann_vol.loc[dt, s_name]
                            if use_vol_target and not pd.isna(curr_ann_vol) and curr_ann_vol > 0:
                                vol_alloc = target_vol_ann / curr_ann_vol
                                dynamic_alloc = min(allocation_pct, vol_alloc)
                            else:
                                dynamic_alloc = allocation_pct
                            
                            buy_p = monthly_data.loc[dt, s_name]
                            d_slice = data.loc[dt:nxt_dt, s_name]
                            if not d_slice.empty:
                                peak, exit_p = buy_p, d_slice.iloc[-1]
                                for p in d_slice:
                                    if p > peak: peak = p
                                    if (p - peak) / peak <= -stop_loss_pct:
                                        exit_p = p; break
                                m_rets.append(((exit_p / buy_p) - 1) * dynamic_alloc)
                            else: m_rets.append(0.0)
                        else: m_rets.append(0.0)
                else: m_rets = [0.0] * max_positions

                avg_m_ret = sum(m_rets) / len(m_rets) if m_rets else 0.0
                spy_ret = (monthly_data.loc[nxt_dt, "SPY"] / monthly_data.loc[dt, "SPY"]) - 1
                
                strat_bal *= (1 + avg_m_ret)
                bench_bal *= (1 + spy_ret)
                
                results.append({"Date": nxt_dt.date(), "Monthly Return": avg_m_ret, "Strategy": strat_bal, "SPY": bench_bal, "Savings Only": save_bal})
            
            res_df = pd.DataFrame(results)
            total_invested = account_balance_input + (len(res_df) * monthly_contribution)
            
            # Metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Final Balance", f"${strat_bal:,.0f}")
            c2.metric("Total Invested", f"${total_invested:,.0f}", help="Total cash you added")
            c3.metric("Profit/Loss", f"${(strat_bal - total_invested):,.0f}", delta=f"{((strat_bal/total_invested)-1)*100:.1f}%")
            c4.metric("Max Drawdown", f"{( (res_df['Strategy'] / res_df['Strategy'].cummax()) - 1 ).min():.2%}")

            # Charts
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=res_df['Date'], y=res_df['Strategy'], name='Strategy', line=dict(width=3, color='blue')))
            fig.add_trace(go.Scatter(x=res_df['Date'], y=res_df['Savings Only'], name='Savings Only', line=dict(dash='dash', color='gray')))
            fig.add_trace(go.Scatter(x=res_df['Date'], y=res_df['SPY'], name='SPY', line=dict(dash='dot', color='green')))
            st.plotly_chart(fig, width="stretch")

            # Forecast
            st.subheader("📊 5-Year Wealth Forecast")
            avg_y_ret = (strat_bal / total_invested) ** (1/5) - 1
            forecast = []
            f_bal = account_balance_input
            for y in range(1, 6):
                for m in range(12): f_bal = (f_bal + monthly_contribution) * (1 + (avg_y_ret / 12))
                forecast.append({"Year": f"Year {y}", "Estimated Balance": f"${f_bal:,.0f}"})
            st.table(pd.DataFrame(forecast))

            # Monte Carlo
            st.write("### 🎲 Monte Carlo")
            sim_ends = []
            monthly_rets = res_df['Monthly Return'].values
            for _ in range(200):
                shuffled = np.random.choice(monthly_rets, size=len(res_df), replace=True)
                s_bal = account_balance_input
                for r in shuffled: s_bal = (s_bal + monthly_contribution) * (1 + r)
                sim_ends.append(s_bal)
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Median Future", f"${np.median(sim_ends):,.0f}")
            m2.metric("Worst Case", f"${np.min(sim_ends):,.0f}")
            m3.metric("Best Case", f"${np.max(sim_ends):,.0f}")

            # Walk Forward
            st.write("### 🚶 Walk-Forward Consistency")
            res_df['Rolling 12M'] = res_df['Monthly Return'].rolling(12).apply(lambda x: (np.prod(1+x)-1)*100)
            st.plotly_chart(go.Figure(go.Bar(x=res_df['Date'], y=res_df['Rolling 12M'])))

# ==========================================
# TAB 3: S&P 500/400 UNIVERSE SCANNER
# ==========================================
with tab3:
    st.header("🔎 Sniper Universe Scanner")
    index_choice = st.radio("Select Universe:", ["S&P 500 (Large Cap)", "S&P 400 (Mid Cap)"], index=0, horizontal=True)
    
    c1, c2 = st.columns(2)
    dist_threshold = c1.slider("Max % from 52W High", 1, 20, 10) / 100
    min_volume = c2.number_input("Min Daily Volume ($M)", value=50)

    if st.button("Scan Market (Batch Mode)"):
        status_text = st.empty()
        if "500" in index_choice:
            status_text.text("⏳ Step 1: Fetching S&P 500 list...")
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        else:
            status_text.text("⏳ Step 1: Fetching S&P 400 list...")
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_400_companies'
        
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers)
            sp_df = pd.read_html(io.StringIO(response.text))[0]
            tickers_list = [t.replace('.', '-') for t in sp_df['Symbol'].tolist()]
            
            status_text.text(f"✅ Found {len(tickers_list)} tickers. Step 2: Downloading data...")
            data = yf.download(tickers_list + ["SPY"], period="6mo", group_by='ticker', auto_adjust=True, threads=True)
            
            status_text.text("✅ Data Downloaded. Step 3: Filtering...")
            winners = []
            if "SPY" in data.columns.levels[0]:
                spy_ret_3m = (data["SPY"]["Close"].iloc[-1] / data["SPY"]["Close"].iloc[0] - 1)
            else: spy_ret_3m = 0.0
            
            for ticker in tickers_list:
                try:
                    if ticker not in data.columns.levels[0]: continue
                    stock_df = data[ticker]
                    closes = stock_df['Close']
                    if closes.isna().all(): continue
                    curr_price = closes.iloc[-1]
                    high_52 = closes.max()
                    if pd.isna(curr_price) or high_52 == 0: continue
                    dist = (high_52 - curr_price) / high_52
                    ret_3m = (curr_price / closes.iloc[0] - 1)
                    avg_v = (stock_df['Volume'].mean() * curr_price) / 1_000_000
                    
                    if dist < dist_threshold and ret_3m > spy_ret_3m and avg_v > min_volume:
                        winners.append({"Ticker": ticker, "Price": curr_price, "From High %": round(dist*100, 2), "3M Return %": round(ret_3m*100, 2), "Vol ($M)": round(avg_v, 1)})
                except: continue

            status_text.text("✅ Scan Complete!")
            if winners:
                df_w = pd.DataFrame(winners).sort_values("3M Return %", ascending=False).reset_index(drop=True)
                st.dataframe(df_w.style.format({"Price": "${:.2f}", "From High %": "{:.2f}%", "3M Return %": "{:.2f}%"}).background_gradient(subset=["3M Return %"], cmap="Greens"))
                st.code(", ".join(df_w['Ticker'].tolist()))
            else: st.warning("No matches found.")
        except Exception as e: st.error(f"Scanner Error: {e}")