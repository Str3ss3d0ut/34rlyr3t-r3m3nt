import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import io
import smtplib # For Email
import json
import os
from email.mime.text import MIMEText # For Email Body
from email.mime.multipart import MIMEMultipart # For Email Structure
from datetime import datetime, timedelta

# --- TRY IMPORTING GOOGLE SHEETS LIBRARIES ---
try:
    import gspread
    from google.oauth2.service_account import Credentials
    HAS_GSPREAD = True
except ImportError:
    HAS_GSPREAD = False

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

# --- NEW: WATCHLIST STATE MANAGEMENT ---
if "watchlist_content" not in st.session_state:
    st.session_state.watchlist_content = default_tickers

# --- NEW: SAVE / LOAD BUTTONS ---
with st.sidebar.expander("📂 Manage Watchlist (Save/Load)", expanded=False):
    col_load, col_save = st.columns(2)
    
    # LOAD
    uploaded_file = st.file_uploader("Load .txt", type=["txt", "json"])
    if uploaded_file is not None:
        string_data = uploaded_file.getvalue().decode("utf-8")
        # Only update if different to prevent loop
        if string_data != st.session_state.watchlist_content:
            st.session_state.watchlist_content = string_data
            st.rerun() # Refresh app with new tickers

    # SAVE
    st.download_button(
        label="💾 Save List",
        data=st.session_state.watchlist_content,
        file_name="my_watchlist.txt",
        mime="text/plain",
        help="Download current tickers as a text file."
    )

with st.sidebar.expander("📝 Edit Watchlist", expanded=False):
    # Using a form here prevents reload on every keystroke
    with st.form("ticker_form"):
        ticker_input = st.text_area("Enter Tickers:", value=st.session_state.watchlist_content, height=150)
        submit_tickers = st.form_submit_button("Update Watchlist")
        if submit_tickers:
            st.session_state.watchlist_content = ticker_input
            st.rerun()

# Note: This slider is now used primarily for the Backtest and Manual Tool.
# Tab 1 now uses 3xATR automatically.
stop_loss_pct = st.sidebar.slider("Trailing Stop Loss % (Backtest Only)", 5, 40, 30, 1) / 100
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

# --- EMAIL CONFIGURATION ---
st.sidebar.divider()
with st.sidebar.expander("📧 Email Settings (Optional)"):
    st.caption("Send Top 10 Picks to yourself.")
    st.caption("ℹ️ **GMAIL USERS:** You MUST use an 'App Password'. Go to Google Account > Security > 2-Step Verification > App Passwords.")
    email_sender = st.text_input("Sender Email (Gmail):")
    email_password = st.text_input("App Password:", type="password")
    email_recipient = st.text_input("Recipient Email:")
    email_host = st.text_input("SMTP Server:", value="smtp.gmail.com")
    email_port = st.number_input("SMTP Port:", value=587)

tickers = [x.strip().upper() for x in st.session_state.watchlist_content.split(',') if x.strip()]

# --- HELPER FUNCTIONS & CACHING ---
@st.cache_data(ttl=3600) # Cache Yahoo Data for 1 Hour
def get_data(ticker_list, period, interval="1d", group_by='ticker'):
    return yf.download(ticker_list, period=period, interval=interval, group_by=group_by, auto_adjust=True, threads=True)

@st.cache_data(ttl=86400) # Cache Wikipedia Data for 24 Hours
def get_wiki_tickers(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers)
        return pd.read_html(io.StringIO(response.text))[0]
    except Exception as e:
        return pd.DataFrame()

def calculate_kama(df, n=10, pow1=2, pow2=30):
    """Calculates Kaufman's Adaptive Moving Average (KAMA) for noise filtering."""
    try:
        if len(df) < n: return pd.Series([0]*len(df), index=df.index)
        df = df.copy()
        change = abs(df['Close'] - df['Close'].shift(n))
        volatility = abs(df['Close'] - df['Close'].shift(1)).rolling(window=n).sum()
        er = change / volatility
        sc_fatest = 2 / (pow1 + 1)
        sc_slowest = 2 / (pow2 + 1)
        sc = (er * (sc_fatest - sc_slowest) + sc_slowest) ** 2
        kama = pd.Series(index=df.index, dtype='float64')
        kama.iloc[n-1] = df['Close'].iloc[n-1]
        for i in range(n, len(df)):
            kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (df['Close'].iloc[i] - kama.iloc[i-1])
        return kama
    except:
        return pd.Series([0]*len(df), index=df.index)

def calculate_adx(df, n=14):
    """Calculates ADX to determine trend strength (0-100)."""
    try:
        df = df.copy()
        df['H-L'] = df['High'] - df['Low']
        df['H-C'] = abs(df['High'] - df['Close'].shift(1))
        df['L-C'] = abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['H-L', 'H-C', 'L-C']].max(axis=1)
        
        df['UpMove'] = df['High'] - df['High'].shift(1)
        df['DownMove'] = df['Low'].shift(1) - df['Low']
        
        df['+DM'] = np.where((df['UpMove'] > df['DownMove']) & (df['UpMove'] > 0), df['UpMove'], 0)
        df['-DM'] = np.where((df['DownMove'] > df['UpMove']) & (df['DownMove'] > 0), df['DownMove'], 0)
        
        df['+DI'] = 100 * (df['+DM'].rolling(window=n).mean() / df['TR'].rolling(window=n).mean())
        df['-DI'] = 100 * (df['-DM'].rolling(window=n).mean() / df['TR'].rolling(window=n).mean())
        df['DX'] = 100 * abs((df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI']))
        df['ADX'] = df['DX'].rolling(window=n).mean()
        return df['ADX'].iloc[-1]
    except:
        return 0

# --- ROBUST EMAIL FUNCTION ---
def send_email_report(df_top10, sender, password, recipient, host, port):
    try:
        msg = MIMEMultipart()
        msg['From'] = sender
        msg['To'] = recipient
        msg['Subject'] = f"🚀 Daily Alpha Scan: Top 10 - {datetime.now().strftime('%Y-%m-%d')}"
        
        # Simple HTML Styling for the table
        html_table = df_top10.to_html(index=False, border=0, justify='center')
        html_body = f"""
        <html>
          <body style="font-family: Arial, sans-serif;">
            <h2 style="color: #2E86C1;">🚀 Your Top 10 Alpha Stocks</h2>
            <p>Here are the best opportunities from today's scan:</p>
            {html_table}
            <br>
            <p><em>Generated by Your Personal Alpha Screener</em></p>
          </body>
        </html>
        """
        msg.attach(MIMEText(html_body, 'html'))
        
        # --- ROBUST CONNECTION BLOCK ---
        server = smtplib.SMTP(host, int(port))
        server.ehlo()          # 1. Identify ourselves
        server.starttls()      # 2. Encrypt the connection
        server.ehlo()          # 3. Re-identify as encrypted connection
        server.login(sender, password) # 4. Login
        server.sendmail(sender, recipient, msg.as_string()) # 5. Send
        server.quit()
        return True, "✅ Email Sent Successfully!"
    except Exception as e:
        return False, f"❌ Error: {str(e)}"

# --- STORAGE HANDLERS (GOOGLE SHEETS / CSV) ---
def connect_gsheet():
    """Attempts to connect to Google Sheets using Streamlit Secrets."""
    if not HAS_GSPREAD: return None
    if "gcp_service_account" not in st.secrets: return None
    try:
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds_dict = dict(st.secrets["gcp_service_account"])
        creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
        client = gspread.authorize(creds)
        return client
    except Exception as e:
        st.sidebar.error(f"GSheet Error: {e}")
        return None

def load_portfolio():
    """Loads portfolio from Google Sheets (preferred) or Local CSV."""
    # 1. Try Google Sheets
    client = connect_gsheet()
    if client:
        try:
            # Assumes a sheet named "AlphaPortfolio" exists
            sheet = client.open("AlphaPortfolio").sheet1
            data = sheet.get_all_records()
            return pd.DataFrame(data), "cloud"
        except:
            pass # Fallback to CSV if sheet doesn't exist or error

    # 2. Fallback to Local CSV
    if os.path.exists("portfolio.csv"):
        return pd.read_csv("portfolio.csv"), "local"
    
    # 3. Return Empty
    return pd.DataFrame(columns=["Symbol", "Shares", "Entry Price", "Date", "Stop Price"]), "none"

def save_trade(symbol, shares, price, stop_price):
    """Saves a new trade to the persistence layer."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    new_row = {"Symbol": symbol, "Shares": shares, "Entry Price": price, "Date": date_str, "Stop Price": stop_price}
    
    # 1. Try Google Sheets
    client = connect_gsheet()
    if client:
        try:
            sheet = client.open("AlphaPortfolio").sheet1
            # Check if headers exist, if not add them
            if not sheet.row_values(1):
                sheet.append_row(["Symbol", "Shares", "Entry Price", "Date", "Stop Price"])
            sheet.append_row([symbol, shares, price, date_str, stop_price])
            return True, "Saved to Google Cloud! ☁️"
        except Exception as e:
            return False, f"Cloud Save Failed: {e}"

    # 2. Fallback to Local CSV
    df, mode = load_portfolio()
    new_df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    new_df.to_csv("portfolio.csv", index=False)
    return True, "Saved to Local CSV 💾"

# --- DAILY TRADING TOOLS ---
st.sidebar.markdown("---")
st.sidebar.subheader("🧮 Daily Tools")
with st.sidebar.expander("Stop Loss Calculator", expanded=True):
    st.caption("Enter the Highest Price the stock has hit:")
    curr_high = st.number_input("Stock High ($)", value=0.0)
    if curr_high > 0:
        st.error(f"Set Broker Stop Order to: **${curr_high * (1 - stop_loss_pct):.2f}**")
        st.caption("Note: This simple tool uses the slider %. For ATR stops, check the Live Screener.")

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
# UPDATED: Added Tab 4 for Portfolio
tab1, tab2, tab3, tab4 = st.tabs(["📊 Live Screener & Sizing", "📉 Backtest & Wealth Sim", "🔎 S&P 500/400 Scanner", "💼 Live Portfolio"])

# ==========================================
# TAB 1: LIVE SCREENER (TRANSPARENT SIGNALS)
# ==========================================
with tab1:
    with st.expander("👮 Pre-Flight Checklist"):
        st.markdown("""
        1. **Earnings Check:** Does the stock report earnings in the next 5 days? (Now automated in Signal).
        2. **Time Check:** Is it the last trading day of the month?
        """)
    
    # --- FORM WRAPPER FIX: PREVENTS DOUBLE CLICK ISSUE ---
    with st.form("live_scan_form"):
        use_live_regime = st.checkbox("🛑 Enable Market Safety Lock (SPY > 200 SMA)?", value=True)
        
        # --- NEW: AUTO-EMAIL CHECKBOX ---
        col_scan, col_check = st.columns([1, 2])
        with col_check:
            auto_email = st.checkbox("📧 Auto-email Top 10 results?", value=False, help="If checked, results will be emailed immediately after scanning.")
        
        # This button is now a Submit Button - it waits for other inputs to settle
        run_scan = st.form_submit_button("Find Alpha (Live Scan)")

    if run_scan:
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
                # --- CACHED CALL ---
                data = get_data(tickers, period="6mo", group_by='ticker')
                status_text.text("✅ Data Downloaded. Processing Signals & Earnings...")
                
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
                        
                        # --- KAMA UPGRADE IN TAB 1 ---
                        kama_series = calculate_kama(hist)
                        kama_val = kama_series.iloc[-1]
                        trend_status = "UP" if current_price > kama_val else "DOWN"
                        
                        # --- ADX FILTER ---
                        adx_val = calculate_adx(hist)
                        
                        raw_momentum = (r1m * 2) + r3m
                        alpha_score = (raw_momentum / 100) / volatility_20
                        
                        signal = ""
                        earnings_warning = False
                        days_to_earn = None
                        
                        # --- SIGNAL LOGIC ---
                        if trend_status == "DOWN":
                            signal = "❄️ COLD"
                        elif alpha_score <= 0:
                            signal = "WAIT (Score)"
                        elif vol_ratio < vol_threshold:
                            signal = "WAIT (Low Vol)"
                        elif adx_val < 20:
                            signal = "WAIT (Choppy)" # ADX Logic
                        else:
                            # --- ROBUST MULTI-PASS EARNINGS LOGIC ---
                            try:
                                t_obj = yf.Ticker(symbol)
                                next_earn = None
                                
                                # Pass 1: DataFrame Calendar
                                cal = t_obj.calendar
                                if isinstance(cal, pd.DataFrame) and not cal.empty:
                                    if 'Earnings Date' in cal.columns: next_earn = cal['Earnings Date'].iloc[0]
                                    elif 0 in cal.columns: next_earn = cal.iloc[0, 0]
                                # Pass 2: Dictionary Calendar
                                elif isinstance(cal, dict) and 'Earnings Date' in cal:
                                    next_earn = cal['Earnings Date'][0]
                                
                                # Pass 3: Metadata / get_earnings_dates Fallback
                                if next_earn is None:
                                    try:
                                        dates_df = t_obj.get_earnings_dates(limit=2)
                                        if dates_df is not None and not dates_df.empty:
                                            future_dates = dates_df.index[dates_df.index >= pd.Timestamp.now().floor('D')]
                                            if not future_dates.empty: next_earn = future_dates[0]
                                    except: pass

                                if next_earn:
                                    # Normalize to simple datetime
                                    next_earn = pd.to_datetime(next_earn).tz_localize(None)
                                    days_to_earn = (next_earn - datetime.now()).days
                                    # Catch earnings today(0), tomorrow(1) or next week
                                    if 0 <= days_to_earn <= 7:
                                        earnings_warning = True
                            except:
                                pass 
                            
                            if earnings_warning:
                                signal = "⚠️ EARNINGS"
                            else:
                                signal = "🔥 HOT"

                        # --- NEW: ATR 3x STOP LOSS CALCULATION ---
                        h_l = hist['High'] - hist['Low']
                        h_pc = (hist['High'] - hist['Close'].shift(1)).abs()
                        l_pc = (hist['Low'] - hist['Close'].shift(1)).abs()
                        tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
                        atr_14 = tr.rolling(14).mean().iloc[-1] # 14-Day ATR

                        # --- SIZING LOGIC ---
                        base_alloc = allocation_pct
                        if use_vol_target and ann_volatility > 0:
                            vol_alloc = target_vol_ann / ann_volatility
                            final_alloc = min(base_alloc, vol_alloc)
                        else:
                            final_alloc = base_alloc
                        
                        capital_per_trade = (account_balance_input * final_alloc) / max_positions
                        shares_to_buy = capital_per_trade / current_price
                        
                        atr_stop_dist = atr_14 * 3 

                        # --- FIX: TRAILING STOP FROM 20-DAY HIGH ---
                        recent_highest_high = hist['High'].tail(20).max()
                        if pd.isna(recent_highest_high): recent_highest_high = current_price # Fallback
                        
                        trail_stop_high = recent_highest_high - atr_stop_dist
                        
                        # Fallback: if calculated stop is above current price (unlikely unless massive gap down), cap it
                        if trail_stop_high > current_price:
                             trail_stop_high = current_price - atr_stop_dist

                        stop_pct_equivalent = (atr_stop_dist / current_price) * 100

                        # --- NEW: EXPLICIT SELL SIGNAL COLUMN LOGIC ---
                        sell_reasons = []
                        if trend_status == "DOWN":
                            sell_reasons.append("Trend Brk")
                        if current_price < trail_stop_high:
                            sell_reasons.append("Stop Hit")
                        if earnings_warning and days_to_earn <= 2:
                            sell_reasons.append("Earn Risk")
                        
                        if len(sell_reasons) > 0:
                            action_col = f"SELL ({', '.join(sell_reasons)})"
                        elif signal == "🔥 HOT":
                            action_col = "BUY"
                        else:
                            action_col = "WAIT"

                        alpha_data.append({
                            "Symbol": symbol,
                            "Zone/Action": action_col, 
                            "Sector": SECTOR_MAP.get(symbol, "Growth"),
                            "Price": current_price,
                            "SHARES": shares_to_buy, 
                            "Stop Price": trail_stop_high,
                            "Trail Stop (High)": trail_stop_high, 
                            "Highest High (20D)": recent_highest_high, 
                            "Signal": signal,
                            "Vol Ratio": vol_ratio,
                            "Score (Risk Adj)": alpha_score,
                            "ADX": adx_val,
                            "Ann Vol": ann_volatility,
                            "Alloc %": final_alloc * 100,
                            "ATR Stop %": stop_pct_equivalent, 
                            "Days to Earn": days_to_earn,
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
                        st.error(f"🛑 **Set Initial Stop:** ${top_stock['Stop Price']:.2f} (This is a {top_stock['ATR Stop %']:.1f}% Trailing Stop)")
                    elif "EARNINGS" in top_stock['Signal']:
                        st.warning(f"⚠️ Top stock ({top_stock['Symbol']}) has EARNINGS coming up. Risk of 20% gap. SKIP.")
                    else:
                        st.warning(f"🏆 Top stock ({top_stock['Symbol']}) is {top_stock['Signal']}. Reason: {top_stock['Signal']}")
                    
                    st.dataframe(
                        df,
                        column_config={
                            "Link": st.column_config.LinkColumn("News"),
                            "Zone/Action": st.column_config.TextColumn("Zone/Action", help="BUY = Good Trend. SELL = Stop hit or Trend broken."),
                            "Score (Risk Adj)": st.column_config.ProgressColumn("Score", min_value=-5, max_value=5, format="%.2f"),
                            "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
                            "Vol Ratio": st.column_config.NumberColumn("Vol Ratio", format="%.2f"),
                            "ADX": st.column_config.NumberColumn("ADX (Trend)", format="%.0f"),
                            "ATR Stop %": st.column_config.NumberColumn("ATR Stop %", format="%.1f%%"),
                            "Ann Vol": st.column_config.NumberColumn("Ann Vol", format="%.1%"),
                            "Alloc %": st.column_config.NumberColumn("Alloc %", format="%.1f%%"),
                            "Days to Earn": st.column_config.NumberColumn("Days to Earn", format="%d"),
                            "Trail Stop (High)": st.column_config.NumberColumn("Trail Stop (High)", format="$%.2f"),
                            "Highest High (20D)": st.column_config.NumberColumn("Highest High (20D)", format="$%.2f"),
                        },
                        column_order=("Symbol", "Zone/Action", "Price", "Signal", "Days to Earn", "Alloc %", "SHARES", "Stop Price", "ATR Stop %", "Vol Ratio", "ADX", "Score (Risk Adj)", "Link")
                    )

                    # --- AUTO EMAIL LOGIC ---
                    if auto_email:
                        if not email_sender or not email_password or not email_recipient:
                            st.error("⚠️ Auto-email enabled but settings are missing in Sidebar!")
                        else:
                            with st.spinner("📧 Sending Auto-Email..."):
                                df_top10 = df.head(10).drop(columns=['Link'])
                                success, msg = send_email_report(df_top10, email_sender, email_password, email_recipient, email_host, email_port)
                                if success:
                                    st.success(msg)
                                else:
                                    st.error(msg)
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
            data = get_data(all_tickers, period="5y", interval="1d", group_by='column')
            
            if isinstance(data.columns, pd.MultiIndex):
                try: data = data['Close']
                except KeyError: pass
            
            # --- BACKTEST SPEED OPTIMIZATION (Pre-calculate KAMA) ---
            kama_data = {}
            for tick in tickers:
                if tick in data.columns:
                    kama_data[tick] = calculate_kama(data[[tick]].rename(columns={tick: 'Close'}))

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
                
                strat_bal += monthly_contribution
                save_bal += monthly_contribution
                bench_bal += monthly_contribution
                
                # --- FIX FOR KEYERROR: Use .asof to ensure weekend dates don't crash ---
                valid_dt = data.index.asof(dt)
                valid_nxt = data.index.asof(nxt_dt)
                
                scores = {}
                for tick in tickers:
                    if tick not in data.columns: continue
                    try:
                        # Use optimized pre-calculated KAMA
                        if data[tick].loc[valid_dt] > kama_data[tick].loc[valid_dt]:
                            ret_1m = (monthly_data.loc[dt, tick] - monthly_data.iloc[i-1][tick]) / monthly_data.iloc[i-1][tick]
                            ret_3m = (monthly_data.loc[dt, tick] - monthly_data.iloc[i-3][tick]) / monthly_data.iloc[i-3][tick]
                            vol = monthly_vol.loc[dt, tick]
                            if pd.isna(vol) or vol == 0: vol = 1.0
                            raw_momentum = (ret_1m * 2) + ret_3m
                            scores[tick] = (raw_momentum / vol) if raw_momentum > 0 else raw_momentum
                    except: scores[tick] = -999 
                
                top_picks = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:max_positions]
                
                global_bull = True
                if use_regime and data["SPY"].loc[valid_dt] < spy_200_sma.loc[valid_dt]: global_bull = False

                m_rets = []
                if global_bull:
                    for s_name, s_score in top_picks:
                        if s_score > 0:
                            curr_ann_vol = monthly_ann_vol.loc[dt, s_name]
                            if use_vol_target and not pd.isna(curr_ann_vol) and curr_ann_vol > 0:
                                vol_alloc = target_vol_ann / curr_ann_vol
                                dynamic_alloc = min(allocation_pct, vol_alloc)
                            else:
                                dynamic_alloc = allocation_pct
                            
                            buy_p = data[s_name].loc[valid_dt]
                            d_slice = data.loc[valid_dt:valid_nxt, s_name]
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
                spy_ret = (data["SPY"].loc[valid_nxt] / data["SPY"].loc[valid_dt]) - 1
                
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
    
    # --- FORM WRAPPER FIX: PREVENTS DOUBLE CLICK ISSUE ---
    with st.form("sp500_scanner_form"):
        index_choice = st.radio("Select Universe:", ["S&P 500 (Large Cap)", "S&P 400 (Mid Cap)"], index=0, horizontal=True)
        
        c1, c2 = st.columns(2)
        dist_threshold = c1.slider("Max % from 52W High", 1, 20, 10) / 100
        min_volume = c2.number_input("Min Daily Volume ($M)", value=50)

        # This button triggers the form
        run_scan_sp = st.form_submit_button("Scan Market (Batch Mode)")

    if run_scan_sp:
        status_text = st.empty()
        if "500" in index_choice:
            status_text.text("⏳ Step 1: Fetching S&P 500 list...")
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        else:
            status_text.text("⏳ Step 1: Fetching S&P 400 list...")
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_400_companies'
        
        try:
            # --- CACHED WIKIPEDIA CALL ---
            sp_df = get_wiki_tickers(url)
            tickers_list = [t.replace('.', '-') for t in sp_df['Symbol'].tolist()]
            
            status_text.text(f"✅ Found {len(tickers_list)} tickers. Step 2: Downloading data...")
            # --- CACHED DATA CALL ---
            data = get_data(tickers_list + ["SPY"], period="6mo", group_by='ticker')
            
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
                # --- RESTORED COPY-PASTE TEXT BOX ---
                st.code(", ".join(df_w['Ticker'].tolist()))
            else: st.warning("No matches found.")
        except Exception as e: st.error(f"Scanner Error: {e}")

# ==========================================
# TAB 4: PORTFOLIO & DEPLOYMENT (NEW)
# ==========================================
with tab4:
    st.header("💼 My Live Portfolio")
    
    # Load Data
    portfolio_df, mode = load_portfolio()
    
    # Header Info
    c1, c2 = st.columns([3, 1])
    with c1:
        if mode == "cloud":
            st.success("🟢 Status: **Connected to Google Cloud** (Permanent Storage)")
        elif mode == "local":
            st.warning("🟡 Status: **Local CSV Mode** (Files saved to PC)")
        else:
            st.info("⚪ Status: Empty (No trades recorded yet)")
            
    with c2:
        if st.button("🔄 Refresh"): st.rerun()

    # --- DISPLAY HOLDINGS ---
    if not portfolio_df.empty:
        st.subheader("Current Holdings")
        
        # Calculate Current Value (Live)
        port_tickers = portfolio_df["Symbol"].unique().tolist()
        if port_tickers:
            live_data = get_data(port_tickers, period="5d")
            
            # Helper to get price safely
            def get_live_price(sym):
                try:
                    if len(port_tickers) == 1: return live_data['Close'].iloc[-1]
                    return live_data[sym]['Close'].iloc[-1]
                except: return 0.0

            portfolio_df["Current Price"] = portfolio_df["Symbol"].apply(get_live_price)
            portfolio_df["Value"] = portfolio_df["Shares"] * portfolio_df["Current Price"]
            portfolio_df["P/L"] = (portfolio_df["Current Price"] - portfolio_df["Entry Price"]) * portfolio_df["Shares"]
            portfolio_df["P/L %"] = ((portfolio_df["Current Price"] - portfolio_df["Entry Price"]) / portfolio_df["Entry Price"]) * 100

            st.dataframe(portfolio_df.style.format({
                "Entry Price": "${:.2f}", 
                "Current Price": "${:.2f}", 
                "Value": "${:.0f}",
                "P/L": "${:.0f}",
                "P/L %": "{:.2f}%"
            }).background_gradient(subset=["P/L %"], cmap="RdYlGn", vmin=-10, vmax=20))

            total_val = portfolio_df["Value"].sum()
            total_pl = portfolio_df["P/L"].sum()
            st.metric("Total Portfolio Value", f"${total_val:,.2f}", delta=f"${total_pl:,.2f}")
    else:
        st.info("No active trades found. Use the form below to add one.")

    st.divider()

    # --- ADD TRADE FORM ---
    st.subheader("📝 Record New Trade")
    with st.form("add_trade"):
        c1, c2, c3, c4 = st.columns(4)
        sym = c1.text_input("Ticker Symbol").upper()
        shares = c2.number_input("Shares Bought", min_value=0.01, step=1.0)
        price = c3.number_input("Entry Price ($)", min_value=0.1, step=0.1)
        stop = c4.number_input("Stop Loss Price ($)", min_value=0.1, step=0.1)
        
        submitted = st.form_submit_button("💾 Save Trade")
        if submitted:
            if sym and shares > 0 and price > 0:
                success, msg = save_trade(sym, shares, price, stop)
                if success:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
            else:
                st.error("Please fill in all fields.")
    
    # --- SETUP INSTRUCTIONS ---
    with st.expander("ℹ️ How to set up Google Cloud for Permanent Storage"):
        st.markdown("""
        To make your portfolio survive restarts on Streamlit Cloud, you need **Google Sheets** integration.
        
        1. **Create a Google Cloud Project** & Enable "Google Sheets API" and "Google Drive API".
        2. **Create a Service Account:** Go to Credentials > Create Service Account.
        3. **Download JSON Key:** Create a key for that account and download the `.json` file.
        4. **Update Secrets:** Open your `.streamlit/secrets.toml` file and paste the JSON content like this:
        ```toml
        [gcp_service_account]
        type = "service_account"
        project_id = "..."
        ... (copy all fields from json) ...
        ```
        5. **Share the Sheet:** Create a new Google Sheet named `AlphaPortfolio` and **Share** it with the `client_email` found in your JSON key.
        """)
