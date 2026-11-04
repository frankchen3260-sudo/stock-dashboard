import contextlib
import datetime
import os
import time
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from concurrent.futures import ThreadPoolExecutor
from html import escape as html_escape
from io import StringIO

import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# 💡 中文對照字典（代號 → 中文名稱）
STOCK_ZH_DICT = {
    "2313.TW": "華通",
    "3167.TW": "大量",
    "2330.TW": "台積電",
    "2317.TW": "鴻海",
    "2454.TW": "聯發科",
    "2308.TW": "台達電",
    "2812.TW": "台中銀",
    "00881.TW": "國泰台灣科技龍頭",
}
# 分析用的歷史資料區間（固定近 1 年）
TIME_PERIOD = "1y"

# 反向字典（中文名稱 → 代號），讓使用者可以直接輸入中文
ZH_TO_TICKER = {name: sym for sym, name in STOCK_ZH_DICT.items()}

# 英文名稱關鍵字 → 強制中文名稱
ENGLISH_OVERRIDES = {
    "TAICHUNG": "台中銀",
    "COMPEQ": "華通",
    "CATHAY": "國泰台灣科技龍頭",
}


# ---------------------------------------------------------------------------
# 手機版排版輔助
# ---------------------------------------------------------------------------
def is_mobile():
    """依瀏覽器 User-Agent 判斷是否為手機（平板視為桌機排版）。"""
    try:
        ua = st.context.headers.get("User-Agent", "")
    except Exception:
        return False
    return "Mobile" in ua or "iPhone" in ua


def show_chart(fig, height=None):
    """統一圖表外觀；手機版縮小高度、圖例放下方，並關閉拖曳縮放，避免手指滑頁面時被圖表搶走。"""
    mobile = is_mobile()
    fig.update_layout(
        margin=dict(l=8, r=8, t=48, b=48),
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="left", x=0),
        height=height or (380 if mobile else 520),
    )
    if mobile:
        fig.update_layout(title=dict(font=dict(size=15)))
        fig.update_xaxes(title=None, fixedrange=True)
        fig.update_yaxes(title=None, fixedrange=True)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": not mobile})


# ---------------------------------------------------------------------------
# 資料抓取（加上快取，避免每次 rerun 都重打 API 被限流）
# ---------------------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner=False)
def fetch_history(symbol, period):
    """只抓日K價格（快）。"""
    return yf.Ticker(symbol).history(period=period)


def _load_info(symbol):
    try:
        return yf.Ticker(symbol).info or {}
    except Exception:
        return {}


def _load_dividends(symbol):
    try:
        return yf.Ticker(symbol).dividends
    except Exception:
        return pd.Series(dtype=float)


def _load_splits(symbol):
    try:
        return yf.Ticker(symbol).splits
    except Exception:
        return pd.Series(dtype=float)


def _load_eps_quarterly(symbol):
    """季度 EPS（優先稀釋 EPS，沒有再用基本 EPS），新 → 舊排序。"""
    try:
        q_stmt = yf.Ticker(symbol).quarterly_income_stmt
        if q_stmt is not None and not q_stmt.empty:
            for key in ("Diluted EPS", "Basic EPS"):
                if key in q_stmt.index:
                    row = q_stmt.loc[key].dropna().astype(float)
                    if not row.empty:
                        return row.sort_index(ascending=False)
    except Exception:
        pass
    return pd.Series(dtype=float)


def _load_monthly(symbol):
    try:
        m = yf.Ticker(symbol).history(period="10y", interval="1mo")
        return m.dropna(subset=["Open", "High", "Low", "Close"])
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=900, show_spinner=False)
def fetch_fundamentals(symbol):
    """基本面、股利、EPS、月K：用多執行緒同時抓，只跟股票代號有關。"""
    with ThreadPoolExecutor(max_workers=5) as pool:
        f_info = pool.submit(_load_info, symbol)
        f_div = pool.submit(_load_dividends, symbol)
        f_split = pool.submit(_load_splits, symbol)
        f_eps = pool.submit(_load_eps_quarterly, symbol)
        f_month = pool.submit(_load_monthly, symbol)
        return (
            f_info.result(),
            f_div.result(),
            f_split.result(),
            f_eps.result(),
            f_month.result(),
        )


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_search_name(symbol):
    """用 yf.Search 取得簡稱，失敗回傳空字串。"""
    try:
        quotes = yf.Search(symbol, max_results=1).quotes
        if quotes:
            return quotes[0].get("shortname") or quotes[0].get("longname") or ""
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------------------
# 代號與名稱解析
# ---------------------------------------------------------------------------
@st.cache_resource
def _industry_store():
    """全站共用：{代號.TW: 證交所產業別}，由 ISIN 頁面順便記錄。"""
    return {}


def _names_from_isin(industry_store=None):
    """來源一：證交所 ISIN 頁面（用 http，避免 https 憑證驗證失敗）。同時記錄產業別。"""
    result = {}
    for mode, suffix in ((2, ".TW"), (4, ".TWO")):
        try:
            resp = requests.get(
                f"http://isin.twse.com.tw/isin/C_public.jsp?strMode={mode}",
                timeout=8,
            )
            html = resp.content.decode("cp950", errors="ignore")
            table = pd.read_html(StringIO(html))[0]
            for row_i, cell in enumerate(table[0].astype(str)):
                parts = cell.split("\u3000")  # 格式：「2330　台積電」
                if len(parts) == 2:
                    code, name = parts[0].strip(), parts[1].strip()
                    if code.isalnum() and 4 <= len(code) <= 6:
                        result[code + suffix] = name
                        if industry_store is not None and table.shape[1] > 4:
                            industry = str(table.iloc[row_i, 4]).strip()
                            if industry and industry.lower() != "nan":
                                industry_store[code + suffix] = industry
        except Exception:
            continue
    return result


def _names_from_openapi():
    """來源二（備援）：證交所與櫃買中心的 OpenAPI。"""
    result = {}
    sources = (
        ("https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL", "Code", "Name", ".TW"),
        (
            "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes",
            "SecuritiesCompanyCode",
            "CompanyName",
            ".TWO",
        ),
    )
    for url, code_key, name_key, suffix in sources:
        try:
            for row in requests.get(url, timeout=8).json():
                code, name = str(row.get(code_key, "")).strip(), str(row.get(name_key, "")).strip()
                if code and name:
                    result.setdefault(code + suffix, name)
        except Exception:
            continue
    return result


@st.cache_data(ttl=86400, show_spinner=False)
def load_tw_stock_names():
    """台股 代號 → 中文名稱。全部來源都失敗時丟出例外（例外不會被快取，下次會重試）。"""
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_isin = pool.submit(_names_from_isin, _industry_store())
        f_api = pool.submit(_names_from_openapi)
        result = f_isin.result()
        for sym, name in f_api.result().items():
            result.setdefault(sym, name)
    if not result:
        raise RuntimeError("無法取得台股名稱清單")
    return result


def get_tw_industry(symbol):
    """證交所／櫃買的產業別（中文），例如「半導體業」。查不到回傳空字串。"""
    get_tw_names()  # 確保清單已載入
    return _industry_store().get(symbol, "")


def get_tw_names():
    try:
        return load_tw_stock_names()
    except Exception:
        return {}


def search_chinese_name(keyword):
    """用中文名稱找代號：先完全比對，再找名稱包含關鍵字且最短的。"""
    name_map = get_tw_names()
    exact = [sym for sym, name in name_map.items() if name == keyword]
    if exact:
        return exact[0]
    partial = [(len(name), sym) for sym, name in name_map.items() if keyword in name]
    return min(partial)[1] if partial else None


def get_ticker_from_name(input_text):
    """把使用者輸入（代號或名稱）轉成 Yahoo Finance 代號。"""
    text = input_text.strip()
    upper = text.upper()

    if upper.endswith(".TW") or upper.endswith(".TWO"):
        return upper
    if text in ZH_TO_TICKER:
        return ZH_TO_TICKER[text]
    if text.isdigit() and len(text) >= 4:
        return f"{text}.TW"

    # 含中文字：用台股名稱清單查詢
    if any("\u4e00" <= ch <= "\u9fff" for ch in text):
        found = search_chinese_name(text)
        if found:
            return found

    try:
        quotes = getattr(yf.Search(text, max_results=10), "quotes", [])
        for q in quotes:
            symbol = q.get("symbol", "").upper()
            if symbol.endswith(".TW") or symbol.endswith(".TWO"):
                return symbol
        if quotes and quotes[0].get("symbol"):
            return quotes[0]["symbol"].upper()
    except Exception:
        pass
    return upper


def resolve_display_name(symbol, keyword="", fallback_name=""):
    """決定要顯示的名稱，優先順序：字典 → 中文輸入 → 搜尋名稱 → 代號。"""
    base = symbol.split(".")[0]

    if symbol in STOCK_ZH_DICT:
        return STOCK_ZH_DICT[symbol]

    tw_name = get_tw_names().get(symbol)
    if tw_name:
        return tw_name

    if any("\u4e00" <= ch <= "\u9fff" for ch in keyword):
        return keyword.strip()

    name = fetch_search_name(symbol) or fallback_name or base

    for eng, zh in ENGLISH_OVERRIDES.items():
        if eng in name.upper():
            return zh
    return name


# ---------------------------------------------------------------------------
# 三大法人買賣超（資料來源：FinMind，原始資料為證交所 / 櫃買中心公開資訊）
# ---------------------------------------------------------------------------
INSTITUTION_GROUPS = {
    "Foreign_Investor": "外資",
    "Foreign_Dealer_Self": "外資",
    "Investment_Trust": "投信",
    "Dealer_self": "自營商",
    "Dealer_Hedging": "自營商",
    "Dealer": "自營商",
}


def _finmind_token():
    """選填：有 FinMind token 可提高每小時請求上限（secrets.toml 或環境變數 FINMIND_TOKEN）。"""
    try:
        token = st.secrets.get("FINMIND_TOKEN", "")
    except Exception:
        token = ""
    return token or os.environ.get("FINMIND_TOKEN", "")


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_institutional_raw(stock_id):
    """抓近 30 天三大法人買賣明細。失敗會丟出例外（例外不會被快取，下次會重試）。"""
    start = (datetime.date.today() - datetime.timedelta(days=30)).strftime("%Y-%m-%d")
    params = {
        "dataset": "TaiwanStockInstitutionalInvestorsBuySell",
        "data_id": stock_id,
        "start_date": start,
    }
    token = _finmind_token()
    if token:
        params["token"] = token
    resp = requests.get("https://api.finmindtrade.com/api/v4/data", params=params, timeout=10)
    resp.raise_for_status()
    body = resp.json()
    if body.get("status") != 200:
        raise RuntimeError(body.get("msg", "FinMind 回傳錯誤"))
    return pd.DataFrame(body.get("data", []))


def summarize_institutional(raw, days=10):
    """整理成「近 N 個交易日 × (外資 / 投信 / 自營商 / 合計)」的買賣超表，單位：張。"""
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    names = set(df["name"].unique())
    # 若已有自營商細項（自行買賣、避險），就不要再重複計入 Dealer 合計
    if "Dealer_self" in names or "Dealer_Hedging" in names:
        df = df[df["name"] != "Dealer"]
    df["group"] = df["name"].map(INSTITUTION_GROUPS)
    df = df.dropna(subset=["group"])
    df["net"] = (df["buy"].astype(float) - df["sell"].astype(float)) / 1000  # 股 → 張
    table = df.pivot_table(index="date", columns="group", values="net", aggfunc="sum").fillna(0)
    for col in ("外資", "投信", "自營商"):
        if col not in table.columns:
            table[col] = 0.0
    table = table[["外資", "投信", "自營商"]].sort_index().tail(days)
    table["三大法人合計"] = table.sum(axis=1)
    return table


def _streak(series):
    """從最新一天往回數，連續同方向（買超或賣超）的天數。回傳 (方向, 天數)，方向為 +1 / -1 / 0。"""
    values = list(series.values)
    if not values or values[-1] == 0:
        return 0, 0
    direction = 1 if values[-1] > 0 else -1
    count = 0
    for v in reversed(values):
        if v * direction > 0:
            count += 1
        else:
            break
    return direction, count


def judge_institutional(table):
    """依近 N 日買賣超判斷各法人偏買或偏賣，並找出連續買進／賣出的天數。"""
    rows = []
    for name in ["外資", "投信", "自營商", "三大法人合計"]:
        total = table[name].sum()
        buy_days = int((table[name] > 0).sum())
        sell_days = int((table[name] < 0).sum())
        if total > 0:
            verdict = "🔴 買 > 賣（偏買）"
        elif total < 0:
            verdict = "🟢 賣 > 買（偏賣）"
        else:
            verdict = "⚪ 買賣持平"

        direction, count = _streak(table[name])
        if count >= 3:
            streak_txt = f"🔥 連續買進 {count} 天" if direction > 0 else f"🔥 連續賣出 {count} 天"
        elif count == 2:
            streak_txt = "連續買進 2 天" if direction > 0 else "連續賣出 2 天"
        else:
            streak_txt = "無連續"
        rows.append(
            {
                "法人": name,
                f"{len(table)}日買賣超(張)": f"{total:+,.0f}",
                "買超天數": buy_days,
                "賣超天數": sell_days,
                "判斷": verdict,
                "近期連續": streak_txt,
            }
        )

    # 綜合提示
    sums = {n: table[n].sum() for n in ["外資", "投信", "自營商"]}
    buyers = [n for n, v in sums.items() if v > 0]
    sellers = [n for n, v in sums.items() if v < 0]
    if len(buyers) == 3:
        overall = "三大法人同步偏買，籌碼面偏多。"
    elif len(sellers) == 3:
        overall = "三大法人同步偏賣，籌碼面偏空。"
    elif "外資" in buyers and "投信" in buyers:
        overall = "外資與投信同步偏買，籌碼面偏多。"
    elif "外資" in sellers and "投信" in sellers:
        overall = "外資與投信同步偏賣，籌碼面偏空。"
    else:
        overall = "各法人方向不一致，籌碼面看法分歧。"

    notes = []
    for name in ["外資", "投信", "自營商"]:
        direction, count = _streak(table[name])
        if count >= 3:
            notes.append(f"{name}已連續{'買進' if direction > 0 else '賣出'} {count} 天")
    if notes:
        overall += "　" + "、".join(notes) + "。"
    level = "success" if (len(buyers) >= 2 and "外資" in buyers) else (
        "error" if (len(sellers) >= 2 and "外資" in sellers) else "info"
    )
    return pd.DataFrame(rows), overall, level


def _color_net(v):
    if isinstance(v, (int, float)):
        if v > 0:
            return "color: #ff4b4b"  # 買超：紅
        if v < 0:
            return "color: #21c354"  # 賣超：綠
    return ""


def render_institutional(symbol):
    st.divider()
    st.subheader("🏦 法人買賣超（近十個交易日）")

    if not symbol.endswith((".TW", ".TWO")):
        st.info("三大法人資料僅支援台灣上市櫃股票。")
        return

    stock_id = symbol.split(".")[0]
    source = "證交所／櫃買中心公開資料"
    table = None
    with st.spinner("讀取法人資料中（第一次約需數秒）..."):
        try:
            table = fetch_institutional_official(symbol, days=10)
        except Exception as official_err:
            # 官方來源失敗時，退回 FinMind（需要有額度或 token）
            try:
                table = summarize_institutional(fetch_institutional_raw(stock_id), days=10)
                source = "FinMind（備援）"
            except Exception as fallback_err:
                st.warning(
                    f"⚠️ 目前無法取得法人資料。官方來源：{official_err}；FinMind：{fallback_err}"
                )
                return

    if table is None or table.empty:
        st.info("查無近期法人買賣超資料（可能是新上市、停牌或資料尚未更新）。")
        return

    judge_df, overall_msg, overall_level = judge_institutional(table)
    getattr(st, overall_level)(f"🧭 法人判讀：{overall_msg}")

    # --- 10 日合計指標 ---
    cols = st.columns(4)
    for col, name in zip(cols, ["外資", "投信", "自營商", "三大法人合計"]):
        total = table[name].sum()
        buy_days = int((table[name] > 0).sum())
        sell_days = int((table[name] < 0).sum())
        color = "red" if total > 0 else ("green" if total < 0 else "gray")  # 台股：紅買綠賣
        col.caption(f"{name} 近{len(table)}日")
        col.markdown(f"### :{color}[{total:+,.0f} 張]")
        col.caption(f"買超 {buy_days} 天 / 賣超 {sell_days} 天")

    # --- 各法人判斷與連續買賣 ---
    def _color_signed_text(v):
        text = str(v)
        if text.startswith("+") and text.strip("+,0") not in ("", "."):
            return "color: #ff4b4b"  # 買超：紅
        if text.startswith("-"):
            return "color: #21c354"  # 賣超：綠
        return ""

    judge_styler = judge_df.style
    net_col = [c for c in judge_df.columns if "買賣超" in c][0]
    judge_styler = (getattr(judge_styler, "map", None) or judge_styler.applymap)(
        _color_signed_text, subset=[net_col]
    )
    judge_styler = (getattr(judge_styler, "map", None) or judge_styler.applymap)(
        lambda v: "color: #ff4b4b", subset=["買超天數"]
    )
    judge_styler = (getattr(judge_styler, "map", None) or judge_styler.applymap)(
        lambda v: "color: #21c354", subset=["賣超天數"]
    )
    st.dataframe(judge_styler, hide_index=True, width="stretch")

    # --- 每日買賣超長條圖 ---
    fig_i = go.Figure()
    palette = {"外資": "#4c9be8", "投信": "#f5a623", "自營商": "#b07cf0"}
    for name, color in palette.items():
        fig_i.add_trace(go.Bar(x=table.index, y=table[name], name=name, marker_color=color))
    fig_i.update_layout(
        title="三大法人每日買賣超（張）",
        barmode="group",
        template="plotly_dark",
        xaxis_title="日期",
        yaxis_title="買賣超（張）",
        xaxis_type="category",
    )
    show_chart(fig_i)

    # --- 明細表（新 → 舊），紅＝買超、綠＝賣超 ---
    shown = table.sort_index(ascending=False)
    styler = shown.style.format("{:+,.0f}")
    styler = (getattr(styler, "map", None) or styler.applymap)(_color_net)
    st.dataframe(styler, width="stretch")

    if len(table) < 10:
        st.caption(
            f"⚠️ 目前只取得近 {len(table)} 個交易日的資料（部分日期可能尚未公布或暫時抓取失敗，稍後重新整理可補齊）。"
        )
    st.caption(
        "📌 單位為「張」（1 張 = 1,000 股），正數為買超、負數為賣超，紅色買超、綠色賣超。"
        "外資含外資自營商；資料通常於交易日收盤後（約 16:00 至 17:00）才公布，資料來源為" + f"{source}" + "，僅供參考。"
    )

    return table


# ---------------------------------------------------------------------------
# 三大法人買賣超：官方免費資料（證交所 T86／櫃買中心），FinMind 為備援
# ---------------------------------------------------------------------------
@st.cache_resource
def _inst_store():
    """全站共用的快取（可在執行緒內安全使用）：{(市場, 日期): (抓取時間, {代號: (外資, 投信, 自營商)})}"""
    return {}


def _to_num(x):
    try:
        return float(str(x).replace(",", "").strip() or 0)
    except Exception:
        return 0.0


def _parse_t86(js):
    """證交所 T86：整個上市市場某一天的三大法人買賣超（股）。換算成張。"""
    if js.get("stat") != "OK" or not js.get("data"):
        return {}  # 假日或尚未公布
    fields = js["fields"]
    foreign_idx = [
        i for i, f in enumerate(fields)
        if f.startswith(("外陸資買賣超股數", "外資自營商買賣超股數")) or f == "外資買賣超股數"
    ]
    trust_idx = fields.index("投信買賣超股數")
    dealer_idx = fields.index("自營商買賣超股數")  # 第一個完全相符的是自營商合計
    out = {}
    for row in js["data"]:
        out[str(row[0]).strip()] = (
            sum(_to_num(row[i]) for i in foreign_idx) / 1000,
            _to_num(row[trust_idx]) / 1000,
            _to_num(row[dealer_idx]) / 1000,
        )
    return out


def _parse_tpex(js):
    """櫃買中心：整個上櫃市場某一天的三大法人買賣超（股）。依欄位名稱找欄位。"""
    tables = js.get("tables") or []
    if not tables or not tables[0].get("data"):
        return {}
    fields = [str(f) for f in tables[0]["fields"]]

    def find(pred):
        for i, f in enumerate(fields):
            if f.endswith("買賣超股數") and pred(f):
                return i
        raise ValueError("櫃買中心欄位格式與預期不符")

    foreign_i = find(lambda f: f.startswith("外資及陸資") and "不含" not in f)
    trust_i = find(lambda f: f.startswith("投信"))
    dealer_i = find(lambda f: f.startswith("自營商") and "自行買賣" not in f and "避險" not in f)
    out = {}
    for row in tables[0]["data"]:
        out[str(row[0]).strip()] = (
            _to_num(row[foreign_i]) / 1000,
            _to_num(row[trust_i]) / 1000,
            _to_num(row[dealer_i]) / 1000,
        )
    return out


def _fetch_market_day(market, day, store):
    """抓某市場某一天的全市場法人資料（有快取）。過去日期快取 24 小時，今天快取 10 分鐘。"""
    key = (market, day.isoformat())
    today = datetime.datetime.now(TW_TZ).date()
    ttl = 600 if day == today else 86400
    hit = store.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]

    headers = {"User-Agent": "Mozilla/5.0"}

    def _request():
        if market == "TW":
            resp = requests.get(
                "https://www.twse.com.tw/rwd/zh/fund/T86",
                params={"date": day.strftime("%Y%m%d"), "selectType": "ALL", "response": "json"},
                headers=headers,
                timeout=10,
            )
            resp.raise_for_status()
            return _parse_t86(resp.json())
        else:
            resp = requests.get(
                "https://www.tpex.org.tw/www/zh-tw/insti/dailyTrade",
                params={"type": "Daily", "sect": "EW", "date": day.strftime("%Y/%m/%d"), "id": "", "response": "json"},
                headers=headers,
                timeout=10,
            )
            resp.raise_for_status()
            return _parse_tpex(resp.json())

    for attempt in range(3):  # 偶爾被官方限流，稍等後重試
        try:
            data = _request()
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1.0 * (attempt + 1))

    if data or day != today:  # 今天還沒公布（空資料）時不要快取，晚點再試
        store[key] = (time.time(), data)
    time.sleep(0.2)  # 對官方網站客氣一點
    return data


def fetch_institutional_official(symbol, days=10):
    """從證交所／櫃買中心抓近 N 個交易日資料，整理成與 summarize_institutional 相同的表格（單位：張）。"""
    market = "OTC" if symbol.endswith(".TWO") else "TW"
    code = symbol.split(".")[0]
    store = _inst_store()
    today = datetime.datetime.now(TW_TZ).date()
    dates = [today - datetime.timedelta(days=i) for i in range(24)]
    dates = [d for d in dates if d.weekday() < 5]  # 只抓週一到週五，新 → 舊

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [(d, pool.submit(_fetch_market_day, market, d, store)) for d in dates]
        rows, errors = {}, 0
        for d, fut in futures:
            try:
                data = fut.result()
            except Exception:
                errors += 1
                continue
            if code in data:
                rows[d] = data[code]
    if errors > len(dates) // 2:
        raise RuntimeError("無法連線至證交所／櫃買中心")

    if not rows:
        return pd.DataFrame()
    table = pd.DataFrame.from_dict(rows, orient="index", columns=["外資", "投信", "自營商"]).sort_index()
    table.index = [d.strftime("%Y-%m-%d") for d in table.index]
    table["三大法人合計"] = table.sum(axis=1)
    return table.tail(days)


# ---------------------------------------------------------------------------
# 月K線趨勢分析（整理自網路上常見的月線技術分析經驗）
# ---------------------------------------------------------------------------
def add_monthly_indicators(m, windows=(5, 10, 20)):
    """計算均線（預設 5/10/20）、KD (9,3,3)、MACD (12,26,9)。月K與日K共用。"""
    m = m.copy()
    for n in windows:
        m[f"MA{n}"] = m["Close"].rolling(n).mean()

    low9 = m["Low"].rolling(9).min()
    high9 = m["High"].rolling(9).max()
    rsv = (m["Close"] - low9) / (high9 - low9).replace(0, float("nan")) * 100
    k_val, d_val, ks, ds = 50.0, 50.0, [], []
    for r in rsv:
        if pd.isna(r):
            ks.append(float("nan"))
            ds.append(float("nan"))
            continue
        k_val = k_val * 2 / 3 + r / 3
        d_val = d_val * 2 / 3 + k_val / 3
        ks.append(k_val)
        ds.append(d_val)
    m["K"], m["D"] = ks, ds

    ema12 = m["Close"].ewm(span=12, adjust=False).mean()
    ema26 = m["Close"].ewm(span=26, adjust=False).mean()
    m["DIF"] = ema12 - ema26
    m["MACD"] = m["DIF"].ewm(span=9, adjust=False).mean()
    m["OSC"] = m["DIF"] - m["MACD"]
    return m


def analyze_monthly_trend(m):
    """回傳 (訊號列表, 總分, 判讀標籤, 提示等級)。每個訊號分數為 +1 / 0 / -1。"""
    signals = []
    last, prev = m.iloc[-1], m.iloc[-2]
    close = last["Close"]

    # 1. 股價相對月均線
    mas = [(n, last[f"MA{n}"]) for n in (5, 10, 20) if not pd.isna(last[f"MA{n}"])]
    if mas:
        above = sum(1 for _, v in mas if close > v)
        score = 1 if above == len(mas) else (-1 if above == 0 else 0)
        signals.append(("股價與月均線", f"站上 {above}/{len(mas)} 條月均線 (5/10/20月)", score))

    # 2. 均線排列
    ma5, ma10, ma20 = last["MA5"], last["MA10"], last["MA20"]
    if not (pd.isna(ma5) or pd.isna(ma10) or pd.isna(ma20)):
        if ma5 > ma10 > ma20:
            signals.append(("均線排列", "月MA5 > MA10 > MA20，多頭排列", 1))
        elif ma5 < ma10 < ma20:
            signals.append(("均線排列", "月MA5 < MA10 < MA20，空頭排列", -1))
        else:
            signals.append(("均線排列", "均線糾結，方向未明", 0))

    # 3. 月MA10 方向（中期趨勢）
    if not (pd.isna(last["MA10"]) or pd.isna(prev["MA10"])):
        if last["MA10"] > prev["MA10"]:
            signals.append(("月MA10 方向", "月MA10 上揚", 1))
        elif last["MA10"] < prev["MA10"]:
            signals.append(("月MA10 方向", "月MA10 下彎", -1))
        else:
            signals.append(("月MA10 方向", "月MA10 走平", 0))

    # 4. 近三個月紅黑K
    recent3 = m.tail(3)
    reds = int((recent3["Close"] > recent3["Open"]).sum())
    if reds == 3:
        signals.append(("近3月K棒", "連續 3 根紅K，買盤延續", 1))
    elif reds == 0:
        signals.append(("近3月K棒", "連續 3 根黑K，賣壓沉重", -1))
    else:
        signals.append(("近3月K棒", f"3 根中有 {reds} 根紅K，多空拉鋸", 0))

    # 5. 高低點墊高 / 摜低
    h3, l3 = recent3["High"], recent3["Low"]
    if h3.is_monotonic_increasing and l3.is_monotonic_increasing and h3.nunique() == 3:
        signals.append(("高低點", "近3月高點、低點逐月墊高", 1))
    elif h3.is_monotonic_decreasing and l3.is_monotonic_decreasing and h3.nunique() == 3:
        signals.append(("高低點", "近3月高點、低點逐月下移", -1))
    else:
        signals.append(("高低點", "高低點無明顯墊高或下移", 0))

    # 6. 突破 / 跌破近 12 個月區間
    prior12 = m.iloc[:-1].tail(12)
    if len(prior12) >= 6:
        if close > prior12["High"].max():
            signals.append(("12月區間", "收盤突破前 12 個月高點", 1))
        elif close < prior12["Low"].min():
            signals.append(("12月區間", "收盤跌破前 12 個月低點", -1))
        else:
            signals.append(("12月區間", "仍在前 12 個月高低區間內", 0))

    # 7. 月KD
    if not (pd.isna(last["K"]) or pd.isna(last["D"])):
        zone = "（高檔 ≥80，留意鈍化或反轉）" if last["K"] >= 80 else ("（低檔 ≤20，留意超賣反彈）" if last["K"] <= 20 else "")
        if last["K"] > last["D"]:
            signals.append(("月KD", f"K {last['K']:.0f} > D {last['D']:.0f}，偏多{zone}", 1))
        else:
            signals.append(("月KD", f"K {last['K']:.0f} < D {last['D']:.0f}，偏空{zone}", -1))

    # 8. 月MACD
    if not pd.isna(last["OSC"]):
        if last["DIF"] > 0 and last["OSC"] > 0:
            signals.append(("月MACD", "DIF 在零軸上且柱狀體為正，多方控盤", 1))
        elif last["DIF"] < 0 and last["OSC"] < 0:
            signals.append(("月MACD", "DIF 在零軸下且柱狀體為負，空方控盤", -1))
        else:
            signals.append(("月MACD", "DIF 與柱狀體方向不一致，轉折觀察期", 0))

    # 9. 最新月K型態（上下影線）
    rng = last["High"] - last["Low"]
    if rng > 0:
        upper = (last["High"] - max(last["Open"], last["Close"])) / rng
        lower = (min(last["Open"], last["Close"]) - last["Low"]) / rng
        if upper > 0.5:
            signals.append(("最新月K型態", "長上影線，上檔賣壓重", -1))
        elif lower > 0.5:
            signals.append(("最新月K型態", "長下影線，下檔有支撐", 1))
        else:
            signals.append(("最新月K型態", "無明顯長影線", 0))

    total = sum(sc for _, _, sc in signals)
    if total >= 5:
        label, level = "強勢多頭", "success"
    elif total >= 2:
        label, level = "偏多", "success"
    elif total >= -1:
        label, level = "盤整觀望", "info"
    elif total >= -4:
        label, level = "偏空", "error"
    else:
        label, level = "空頭走勢", "error"
    return signals, total, label, level


def render_monthly_analysis(monthly, display_name):
    st.divider()
    st.subheader("📅 月K線趨勢分析")

    if monthly is None or len(monthly) < 6:
        st.info("月K資料不足（少於 6 個月），無法進行月線趨勢分析。")
        return

    m = add_monthly_indicators(monthly)
    signals, total, label, level = analyze_monthly_trend(m)

    getattr(st, level)(
        f"🧭 月K綜合判讀：**{label}**（多空分數 {total:+d}，共 {len(signals)} 項訊號）"
    )

    last = m.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    for col, n in zip((c1, c2, c3), (5, 10, 20)):
        v = last[f"MA{n}"]
        col.metric(f"月MA{n}", f"{v:.2f}" if not pd.isna(v) else "N/A")
    c4.metric(
        "月KD (K / D)",
        f"{last['K']:.0f} / {last['D']:.0f}" if not pd.isna(last["K"]) else "N/A",
    )

    icon = {1: "🔴 偏多", 0: "⚪ 中性", -1: "🟢 偏空"}  # 台股慣例：紅多綠空
    st.dataframe(
        pd.DataFrame(
            {
                "判斷項目": [n for n, _, _ in signals],
                "現況": [d for _, d, _ in signals],
                "訊號": [icon[sc] for _, _, sc in signals],
            }
        ),
        hide_index=True,
        width="stretch",
    )

    view = m.tail(36)
    fig_m = go.Figure()
    fig_m.add_trace(
        go.Candlestick(
            x=view.index,
            open=view["Open"],
            high=view["High"],
            low=view["Low"],
            close=view["Close"],
            name="月K",
            increasing_line_color="red",
            decreasing_line_color="green",
        )
    )
    for n in (5, 10, 20):
        fig_m.add_trace(go.Scatter(x=view.index, y=view[f"MA{n}"], mode="lines", name=f"{n}月均線"))
    fig_m.update_layout(
        title=f"{display_name} 月K線（近 36 個月）",
        xaxis_title="月份",
        yaxis_title="價格",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
    )
    show_chart(fig_m)

    st.caption(
        "📌 最後一根為本月（尚未收盤）的資料，訊號可能隨月底收盤改變。"
        "判讀規則整理自網路常見的月線技術分析經驗（均線排列、高低點、區間突破、KD、MACD、K棒型態），"
        "僅供參考，不構成投資建議。"
    )

    return signals, total


# ---------------------------------------------------------------------------
# 日K線趨勢分析
# ---------------------------------------------------------------------------
def analyze_daily_trend(d):
    """回傳 (訊號列表, 總分, 判讀標籤, 提示等級)。每個訊號分數為 +1 / 0 / -1。"""
    signals = []
    last, prev = d.iloc[-1], d.iloc[-2]
    close = last["Close"]

    # 1. 股價相對 5/10/20/60 日均線
    mas = [(n, last[f"MA{n}"]) for n in (5, 10, 20, 60) if not pd.isna(last[f"MA{n}"])]
    if mas:
        above = sum(1 for _, v in mas if close > v)
        score = 1 if above == len(mas) else (-1 if above == 0 else 0)
        signals.append(("股價與均線", f"站上 {above}/{len(mas)} 條均線 (5/10/20/60日)", score))

    # 2. 均線排列
    ma5, ma10, ma20, ma60 = (last[f"MA{n}"] for n in (5, 10, 20, 60))
    if not (pd.isna(ma5) or pd.isna(ma10) or pd.isna(ma20)):
        chain = [ma5, ma10, ma20] + ([] if pd.isna(ma60) else [ma60])
        if all(a > b for a, b in zip(chain, chain[1:])):
            signals.append(("均線排列", "短中長均線由上而下排列，多頭排列", 1))
        elif all(a < b for a, b in zip(chain, chain[1:])):
            signals.append(("均線排列", "短中長均線由下而上排列，空頭排列", -1))
        else:
            signals.append(("均線排列", "均線糾結，方向未明", 0))

    # 3. 月線（20日）方向、4. 季線（60日）方向
    for n, label in ((20, "月線"), (60, "季線")):
        cur, pre = last[f"MA{n}"], prev[f"MA{n}"]
        if not (pd.isna(cur) or pd.isna(pre)):
            if cur > pre:
                signals.append((f"{label}方向", f"{label} (MA{n}) 上揚", 1))
            elif cur < pre:
                signals.append((f"{label}方向", f"{label} (MA{n}) 下彎", -1))
            else:
                signals.append((f"{label}方向", f"{label} (MA{n}) 走平", 0))

    # 5. 近 5 日紅黑K
    recent5 = d.tail(5)
    reds = int((recent5["Close"] > recent5["Open"]).sum())
    if reds >= 4:
        signals.append(("近5日K棒", f"5 根中有 {reds} 根紅K，買盤強勢", 1))
    elif reds <= 1:
        signals.append(("近5日K棒", f"5 根中僅 {reds} 根紅K，賣壓偏重", -1))
    else:
        signals.append(("近5日K棒", f"5 根中有 {reds} 根紅K，多空拉鋸", 0))

    # 6. 高低點墊高 / 摜低（近 3 日）
    r3 = d.tail(3)
    if r3["High"].is_monotonic_increasing and r3["Low"].is_monotonic_increasing and r3["High"].nunique() == 3:
        signals.append(("高低點", "近3日高點、低點逐日墊高", 1))
    elif r3["High"].is_monotonic_decreasing and r3["Low"].is_monotonic_decreasing and r3["High"].nunique() == 3:
        signals.append(("高低點", "近3日高點、低點逐日下移", -1))
    else:
        signals.append(("高低點", "高低點無明顯墊高或下移", 0))

    # 7. 突破 / 跌破前 20 日區間
    prior20 = d.iloc[:-1].tail(20)
    if len(prior20) >= 10:
        if close > prior20["High"].max():
            signals.append(("20日區間", "收盤突破前 20 日高點", 1))
        elif close < prior20["Low"].min():
            signals.append(("20日區間", "收盤跌破前 20 日低點", -1))
        else:
            signals.append(("20日區間", "仍在前 20 日高低區間內", 0))

    # 8. 量價關係（今日量 vs 前 5 日均量）
    if "Volume" in d.columns and len(d) >= 7:
        vol_avg = d["Volume"].iloc[-6:-1].mean()
        if vol_avg and vol_avg > 0 and last["Volume"] > 0:
            ratio = last["Volume"] / vol_avg
            chg = (close - prev["Close"]) / prev["Close"] * 100
            if ratio >= 1.5 and chg > 0:
                signals.append(("量價關係", f"價漲量增（量為前5日均量 {ratio:.1f} 倍）", 1))
            elif ratio >= 1.5 and chg < 0:
                signals.append(("量價關係", f"價跌量增（量為前5日均量 {ratio:.1f} 倍），賣壓放大", -1))
            elif ratio <= 0.7 and chg > 0:
                signals.append(("量價關係", f"價漲量縮（量為前5日均量 {ratio:.1f} 倍），追價力道不足", 0))
            else:
                signals.append(("量價關係", f"量能無明顯異常（前5日均量 {ratio:.1f} 倍）", 0))

    # 9. 日KD
    if not (pd.isna(last["K"]) or pd.isna(last["D"])):
        zone = "（高檔 ≥80，留意鈍化或反轉）" if last["K"] >= 80 else ("（低檔 ≤20，留意超賣反彈）" if last["K"] <= 20 else "")
        if last["K"] > last["D"]:
            signals.append(("日KD", f"K {last['K']:.0f} > D {last['D']:.0f}，偏多{zone}", 1))
        else:
            signals.append(("日KD", f"K {last['K']:.0f} < D {last['D']:.0f}，偏空{zone}", -1))

    # 10. 日MACD
    if not pd.isna(last["OSC"]):
        if last["DIF"] > 0 and last["OSC"] > 0:
            signals.append(("日MACD", "DIF 在零軸上且柱狀體為正，多方控盤", 1))
        elif last["DIF"] < 0 and last["OSC"] < 0:
            signals.append(("日MACD", "DIF 在零軸下且柱狀體為負，空方控盤", -1))
        else:
            signals.append(("日MACD", "DIF 與柱狀體方向不一致，轉折觀察期", 0))

    # 11. RSI(14)
    if not pd.isna(last["RSI"]):
        rsi = last["RSI"]
        note = "（超買 ≥70，留意回檔）" if rsi >= 70 else ("（超賣 ≤30，留意反彈）" if rsi <= 30 else "")
        if rsi > 50:
            signals.append(("RSI(14)", f"RSI {rsi:.0f} 在 50 之上，偏多{note}", 1))
        elif rsi < 50:
            signals.append(("RSI(14)", f"RSI {rsi:.0f} 在 50 之下，偏空{note}", -1))
        else:
            signals.append(("RSI(14)", "RSI 剛好 50，多空平衡", 0))

    # 12. 最新日K型態
    rng = last["High"] - last["Low"]
    if rng > 0:
        upper = (last["High"] - max(last["Open"], last["Close"])) / rng
        lower = (min(last["Open"], last["Close"]) - last["Low"]) / rng
        if upper > 0.5:
            signals.append(("最新日K型態", "長上影線，上檔賣壓重", -1))
        elif lower > 0.5:
            signals.append(("最新日K型態", "長下影線，下檔有支撐", 1))
        else:
            signals.append(("最新日K型態", "無明顯長影線", 0))

    total = sum(sc for _, _, sc in signals)
    if total >= 6:
        label, level = "強勢多頭", "success"
    elif total >= 3:
        label, level = "偏多", "success"
    elif total >= -2:
        label, level = "盤整觀望", "info"
    elif total >= -5:
        label, level = "偏空", "error"
    else:
        label, level = "空頭走勢", "error"
    return signals, total, label, level


def add_daily_indicators(daily):
    d = add_monthly_indicators(daily, windows=(5, 10, 20, 60))
    diff = d["Close"].diff()
    gain = diff.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-diff.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    d["RSI"] = 100 - 100 / (1 + gain / loss.replace(0, float("nan")))
    d["RSI"] = d["RSI"].where(loss != 0, 100.0)  # 完全沒下跌時 RSI = 100
    return d


def render_daily_analysis(symbol):
    st.divider()
    st.subheader("📈 日K線趨勢分析")

    try:
        daily = fetch_history(symbol, "1y")  # 固定用近一年資料
    except Exception as e:
        st.warning(f"⚠️ 目前無法取得日K資料：{e}")
        return
    daily = daily.dropna(subset=["Open", "High", "Low", "Close"]) if daily is not None else daily
    if daily is None or len(daily) < 30:
        st.info("日K資料不足（少於 30 個交易日），無法進行日K趨勢分析。")
        return

    d = add_daily_indicators(daily)
    signals, total, label, level = analyze_daily_trend(d)

    getattr(st, level)(
        f"🧭 日K綜合判讀：**{label}**（多空分數 {total:+d}，共 {len(signals)} 項訊號）"
    )

    last = d.iloc[-1]
    cols = st.columns(5)
    for col, n, name in zip(cols[:3], (5, 20, 60), ("週線 MA5", "月線 MA20", "季線 MA60")):
        v = last[f"MA{n}"]
        col.metric(name, f"{v:.2f}" if not pd.isna(v) else "N/A")
    cols[3].metric(
        "日KD (K / D)",
        f"{last['K']:.0f} / {last['D']:.0f}" if not pd.isna(last["K"]) else "N/A",
    )
    cols[4].metric("RSI(14)", f"{last['RSI']:.0f}" if not pd.isna(last["RSI"]) else "N/A")

    icon = {1: "🔴 偏多", 0: "⚪ 中性", -1: "🟢 偏空"}  # 台股慣例：紅多綠空
    st.dataframe(
        pd.DataFrame(
            {
                "判斷項目": [n for n, _, _ in signals],
                "現況": [dt for _, dt, _ in signals],
                "訊號": [icon[sc] for _, _, sc in signals],
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "📌 最後一根為最新交易日資料，盤中尚未收盤時訊號可能隨收盤改變。"
        "判讀規則整理自常見的日線技術分析經驗（均線排列、區間突破、量價、KD、MACD、RSI、K棒型態），"
        "僅供參考，不構成投資建議。"
    )

    return signals, total, d


# ---------------------------------------------------------------------------
# 近期新聞（台股：Google 新聞 RSS 繁體中文；其他：yfinance 新聞）
# ---------------------------------------------------------------------------
TW_TZ = datetime.timezone(datetime.timedelta(hours=8))


def _news_from_google(query):
    url = "https://news.google.com/rss/search"
    params = {"q": f"{query} when:30d", "hl": "zh-TW", "gl": "TW", "ceid": "TW:zh-Hant"}
    resp = requests.get(url, params=params, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    items = []
    for it in root.findall("./channel/item"):
        title = (it.findtext("title") or "").strip()
        link = (it.findtext("link") or "").strip()
        source = (it.findtext("source") or "").strip()
        if source and title.endswith(f" - {source}"):
            title = title[: -len(f" - {source}")]
        try:
            published = parsedate_to_datetime(it.findtext("pubDate")).astimezone(TW_TZ)
        except Exception:
            published = None
        if title and link:
            items.append({"title": title, "link": link, "source": source, "time": published})
    return items


def _news_from_yfinance(symbol):
    items = []
    for n in yf.Ticker(symbol).news or []:
        c = n.get("content", n)  # 新舊兩種格式
        title = c.get("title", "")
        link = (
            (c.get("canonicalUrl") or {}).get("url")
            or (c.get("clickThroughUrl") or {}).get("url")
            or c.get("link", "")
        )
        source = (c.get("provider") or {}).get("displayName") or c.get("publisher", "")
        published = None
        try:
            if c.get("pubDate"):
                published = pd.Timestamp(c["pubDate"]).to_pydatetime().astimezone(TW_TZ)
            elif c.get("providerPublishTime"):
                published = datetime.datetime.fromtimestamp(c["providerPublishTime"], TW_TZ)
        except Exception:
            pass
        if title and link:
            items.append({"title": title, "link": link, "source": source, "time": published})
    return items


@st.cache_data(ttl=900, show_spinner=False)
def fetch_news(symbol, name):
    """抓近期新聞。失敗會丟出例外（不會被快取）。"""
    items = []
    if symbol.endswith((".TW", ".TWO")):
        code = symbol.split(".")[0]
        items = _news_from_google(f"{name} {code}")
    if not items:
        items = _news_from_yfinance(symbol)
    items = [i for i in items if i["link"].startswith(("http://", "https://"))]
    items.sort(key=lambda i: i["time"] or datetime.datetime.min.replace(tzinfo=TW_TZ), reverse=True)
    return items[:10]


def render_news(symbol, display_name):
    st.divider()
    st.subheader("📰 近期新聞")
    try:
        with st.spinner("讀取新聞中..."):
            items = fetch_news(symbol, display_name)
    except Exception as e:
        st.warning(f"⚠️ 目前無法取得新聞：{e}")
        return

    if not items:
        st.info("查無近期相關新聞。")
        return

    for it in items:
        title = it["title"].replace("[", "［").replace("]", "］")
        meta = " · ".join(
            x for x in (it["source"], it["time"].strftime("%Y-%m-%d %H:%M") if it["time"] else "") if x
        )
        st.markdown(f"[{title}]({it['link']})")
        if meta:
            st.caption(meta)

    st.caption("📌 新聞標題與連結來自 Google 新聞（台股）或 Yahoo Finance，點擊後會前往原始媒體網站；內容僅供參考，不構成投資建議。")


# ---------------------------------------------------------------------------
# 即時報價（台股：證交所 MIS 即時行情，約每 5 秒更新；其他：Yahoo，可能延遲）
# ---------------------------------------------------------------------------
def is_market_open():
    """台股交易時間：週一至週五 09:00–13:30（含收盤試撮緩衝）。不含國定假日判斷。"""
    now = datetime.datetime.now(TW_TZ)
    return now.weekday() < 5 and datetime.time(8, 55) <= now.time() <= datetime.time(13, 35)


def _to_float(x):
    try:
        return float(str(x).replace(",", ""))
    except Exception:
        return None


def _first_price(text):
    """五檔報價字串「123.0_122.5_...」取第一個有效價格。"""
    for part in str(text or "").split("_"):
        v = _to_float(part)
        if v:
            return v
    return None


def _fmt_mis_time(d, t):
    d = str(d)
    if len(d) == 8 and d.isdigit():
        d = f"{d[:4]}-{d[4:6]}-{d[6:]}"
    return f"{d} {t}".strip()


def _parse_mis(item):
    last = _to_float(item.get("z"))
    bid, ask = _first_price(item.get("b")), _first_price(item.get("a"))
    note = ""
    if last is None:  # 尚無成交（例如剛開盤）：用最佳買賣價中間值當參考
        if bid and ask:
            last, note = (bid + ask) / 2, "尚無成交，顯示買賣價中間值"
        else:
            last = bid or ask
            note = "尚無成交，顯示最佳委買或委賣價" if last else ""
    prev = _to_float(item.get("y"))
    return {
        "price": last,
        "prev_close": prev,
        "open": _to_float(item.get("o")),
        "high": _to_float(item.get("h")),
        "low": _to_float(item.get("l")),
        "volume": _to_float(item.get("v")),  # 累計成交量（張）
        "time": _fmt_mis_time(item.get("d", ""), item.get("t", "")),
        "source": "臺灣證交所即時行情",
        "note": note,
    }


@st.cache_resource(ttl=600, show_spinner=False)
def _mis_session():
    sess = requests.Session()
    sess.headers.update({"User-Agent": "Mozilla/5.0", "Referer": "https://mis.twse.com.tw/stock/index.jsp"})
    try:
        sess.get("https://mis.twse.com.tw/stock/index.jsp", timeout=5)  # 取得 cookie
    except Exception:
        pass
    return sess


def _quote_from_mis(symbol):
    prefix = "otc" if symbol.endswith(".TWO") else "tse"
    ex_ch = f"{prefix}_{symbol.split('.')[0].lower()}.tw"
    resp = _mis_session().get(
        "https://mis.twse.com.tw/stock/api/getStockInfo.jsp",
        params={"ex_ch": ex_ch, "json": "1", "delay": "0", "_": int(time.time() * 1000)},
        timeout=6,
    )
    resp.raise_for_status()
    arr = resp.json().get("msgArray") or []
    if not arr:
        raise RuntimeError("證交所即時行情查無此代號")
    q = _parse_mis(arr[0])
    if q["price"] is None:
        raise RuntimeError("證交所即時行情尚無報價")
    return q


def _quote_from_yahoo(symbol):
    fi = yf.Ticker(symbol).fast_info
    price = fi.get("last_price")
    if price is None:
        raise RuntimeError("Yahoo 無即時報價")
    vol = fi.get("last_volume")
    is_tw = symbol.endswith((".TW", ".TWO"))
    return {
        "price": float(price),
        "prev_close": fi.get("previous_close"),
        "open": fi.get("open"),
        "high": fi.get("day_high"),
        "low": fi.get("day_low"),
        "volume": (vol / 1000 if is_tw else vol) if vol else None,
        "time": datetime.datetime.now(TW_TZ).strftime("%Y-%m-%d %H:%M:%S"),
        "source": "Yahoo Finance（可能延遲約 15–20 分鐘）",
        "note": "",
    }


@st.cache_data(ttl=4, show_spinner=False)
def fetch_realtime_quote(symbol):
    """短暫快取 4 秒：多位使用者同時看同一檔時共用結果，減少對來源的請求。失敗會丟出例外。"""
    if symbol.endswith((".TW", ".TWO")):
        try:
            return _quote_from_mis(symbol)
        except Exception:
            pass  # 證交所失敗就退回 Yahoo
    return _quote_from_yahoo(symbol)


def _fmt(v, nd=2):
    return f"{v:,.{nd}f}" if isinstance(v, (int, float)) else "N/A"


def _live_quote_body(symbol):
    try:
        q = fetch_realtime_quote(symbol)
    except Exception as e:
        st.warning(f"⚠️ 目前無法取得即時報價：{e}")
        return

    cols = st.columns(5)
    price, prev = q["price"], q["prev_close"]
    if isinstance(prev, (int, float)) and prev:
        chg = price - prev
        delta = f"{chg:+.2f} ({chg / prev * 100:+.2f}%)"
    else:
        delta = None
    cols[0].metric("成交價", _fmt(price), delta, delta_color="inverse")  # 台股：紅漲綠跌
    cols[1].metric("開盤", _fmt(q["open"]))
    cols[2].metric("最高", _fmt(q["high"]))
    cols[3].metric("最低", _fmt(q["low"]))
    cols[4].metric("累計成交量", f"{q['volume']:,.0f} 張" if q["volume"] else "N/A")
    extra = f"　昨收 {_fmt(prev)}" if prev else ""
    note = f"　（{q['note']}）" if q["note"] else ""
    st.caption(f"⚡ 資料時間 {q['time']}　來源：{q['source']}{extra}{note}")


# 同一個函式做成兩種版本：交易時間每 5 秒自動更新，其餘時間只在載入時更新一次
live_quote_auto = st.fragment(run_every=5)(_live_quote_body)
live_quote_static = st.fragment(_live_quote_body)


def render_live_quote(symbol):
    st.markdown("##### ⚡ 即時報價")
    st.toggle("🔄 交易時間自動更新（每 5 秒）", value=True, key="auto_refresh")
    if st.session_state.get("auto_refresh", True) and is_market_open():
        live_quote_auto(symbol)
    else:
        live_quote_static(symbol)
        if not is_market_open():
            st.caption("目前非交易時間，顯示最近一次成交資料。")


# ---------------------------------------------------------------------------
# 購買指標燈號（放在結果最上方）：達到條件亮紅燈、未達到亮綠燈、資料不足亮白燈
# ---------------------------------------------------------------------------
def _beacon_items(pe_ratio, eps_last4, inst_table, daily_res, monthly_res):
    """回傳 [(指標名稱, 狀態, 說明)]，狀態：True=達到、False=未達到、None=資料不足。"""
    items = []

    # 2. 連續四季 EPS 為正
    if eps_last4 is not None and len(eps_last4) >= 4:
        last4 = list(eps_last4.head(4).values)
        ok = all(v > 0 for v in last4)
        items.append(("連續四季 EPS 為正", ok, "近四季（新→舊）：" + "、".join(f"{v:.2f}" for v in last4)))
    else:
        items.append(("連續四季 EPS 為正", None, "取得的季度 EPS 不足四季"))

    # 3~5. 外資、投信、自營商：各自連續買進兩天以上
    for inst in ("外資", "投信", "自營商"):
        name = f"{inst}連續買進 ≥ 2 天"
        if inst_table is not None and len(inst_table) >= 2 and inst in inst_table.columns:
            direction, count = _streak(inst_table[inst])
            if direction > 0 and count >= 2:
                items.append((name, True, f"已連續買超 {count} 天"))
            elif direction > 0:
                items.append((name, False, "最近一日買超，但尚未連續兩天"))
            elif direction < 0:
                items.append((name, False, f"已連續賣超 {count} 天"))
            else:
                items.append((name, False, "最近一日買賣持平"))
        else:
            items.append((name, None, "無法人資料"))

    # 4 / 5. 日K、月K：偏多訊號數 > 偏空訊號數
    for name, res in (("日K趨勢 偏多 > 偏空", daily_res), ("月K趨勢 偏多 > 偏空", monthly_res)):
        if res is None:
            items.append((name, None, "資料不足，無法判讀"))
        else:
            signals = res[0]
            bull = sum(1 for _, _, sc in signals if sc > 0)
            bear = sum(1 for _, _, sc in signals if sc < 0)
            items.append((name, bull > bear, f"偏多 {bull} 項、偏空 {bear} 項"))

    # 8. 近期突破 / 跌破月線（日K的 20 日均線）：站上月線亮紅燈、跌破月線亮綠燈
    name = "近期突破／跌破月線"
    items.append((name, *_ma20_cross_state(daily_res)))
    return items


def _ma20_cross_state(daily_res):
    """回傳 (狀態, 說明)。狀態：True=站上月線（含近期往上突破）、False=跌破月線（含近期往下跌破）、None=資料不足。"""
    if daily_res is None or len(daily_res) < 3:
        return None, "日K資料不足，無法判斷月線"
    d = daily_res[2]
    valid = d["Close"].notna() & d["MA20"].notna()
    d = d[valid]
    if len(d) < 3:
        return None, "資料不足，無法計算月線"

    above = (d["Close"] > d["MA20"]).tolist()
    recent = above[-6:]  # 今天加上前 5 個交易日
    now = recent[-1]
    gap = (d["Close"].iloc[-1] - d["MA20"].iloc[-1]) / d["MA20"].iloc[-1] * 100

    cross_at = None  # 最近一次站在「相反側」的位置
    for i in range(len(recent) - 2, -1, -1):
        if recent[i] != now:
            cross_at = i
            break

    if cross_at is None:
        side = "站在月線之上" if now else "在月線之下"
        return now, f"近 5 日持續{side}（距月線 {gap:+.1f}%）"
    days_ago = len(recent) - 2 - cross_at
    when = "今日" if days_ago == 0 else f"{days_ago} 個交易日前"
    if now:
        return True, f"{when}由下往上突破月線（距月線 {gap:+.1f}%）"
    return False, f"{when}由上往下跌破月線（距月線 {gap:+.1f}%）"


def render_beacons(slot, pe_ratio, eps_last4, inst_table, daily_res, monthly_res, fund_items=None, tech_items=None):
    # 全部燈號整合成同一份清單：(名稱, 狀態, 說明, 判斷標準)
    items = [(n, ok, d, "") for n, ok, d in _beacon_items(pe_ratio, eps_last4, inst_table, daily_res, monthly_res)]
    for n, abbr, ok, d, std in (fund_items or []) + (tech_items or []):
        items.append((f"{n}（{abbr}）", ok, d, std))
    # 統一轉成 Python 布林（numpy 的 True/False 用 `is True` 判斷會落空，導致統計漏算）
    items = [(n, None if ok is None else bool(ok), d, std) for n, ok, d, std in items]

    red = sum(1 for _, ok, _, _ in items if ok is True)
    green = sum(1 for _, ok, _, _ in items if ok is False)
    white = sum(1 for _, ok, _, _ in items if ok is None)

    with slot.container():
        st.markdown("##### 🚦 購買指標燈號")
        st.markdown(
            f"🔴 紅燈 **{red}** 個　🟢 綠燈 **{green}** 個"
            + (f"　⚪ 資料不足 **{white}** 個" if white else "")
        )
        per_row = 4
        for start in range(0, len(items), per_row):
            cols = st.columns(per_row)
            for col, (name, ok, detail, std) in zip(cols, items[start:start + per_row]):
                lamp = "⚪" if ok is None else ("🔴" if ok else "🟢")
                col.markdown(f"### {lamp}")
                col.markdown(f"**{name}**")
                col.caption(detail, help=std or None)


# ---------------------------------------------------------------------------
# 基本面／技術面指標燈號（放在購買指標燈號下方）
# 判斷門檻集中在這裡，想調整標準直接改這幾個數字即可
# ---------------------------------------------------------------------------
FUND_PER_MAX = 20       # 本益比 ≤ 此值 → 紅燈
FUND_PBR_MAX = 1.0      # 股價淨值比 < 此值 → 紅燈
FUND_YIELD_MIN = 4.0    # 現金股利殖利率 > 此值(%) → 紅燈
FUND_ROE_MIN = 15.0     # ROE > 此值(%) → 紅燈
BIAS_OVERHEAT = 10.0    # 乖離率(20日) 未達此值(%)（未過熱）→ 紅燈


def _trailing_cash_dividend(dividends_series, info):
    """近一年累計現金股利；抓不到再退回 Yahoo 的年度股利。"""
    try:
        if dividends_series is not None and not dividends_series.empty:
            idx = pd.to_datetime(dividends_series.index).tz_localize(None)
            cutoff = pd.Timestamp.now().tz_localize(None) - pd.Timedelta(days=365)
            total = float(dividends_series[idx >= cutoff].sum())
            if total > 0:
                return total
    except Exception:
        pass
    backup = info.get("trailingAnnualDividendRate", info.get("dividendRate", 0))
    return float(backup) if isinstance(backup, (int, float)) and backup > 0 else 0.0


def _fundamental_items(info, pe_ratio, eps_quarterly, dividends_series, latest_close):
    """基本面：回傳 [(名稱, 英文簡稱, 狀態, 說明, 判斷標準)]，狀態 True=達標(紅燈)、False=未達標(綠燈)、None=資料不足。"""
    items = []

    # 1. 本益比 PER
    std = f"≤ {FUND_PER_MAX} 倍視為合理偏低；過高可能代表泡沫化或成長性極強"
    if isinstance(pe_ratio, (int, float)) and pe_ratio > 0:
        items.append(("本益比", "PER", pe_ratio <= FUND_PER_MAX, f"目前 {pe_ratio:.2f} 倍", std))
    else:
        items.append(("本益比", "PER", None, "無本益比資料（可能為虧損或資料缺漏）", std))

    # 2. 股價淨值比 PBR
    std = f"< {FUND_PBR_MAX:g} 代表股價低於每股淨值，具資產防禦價值"
    pbr = info.get("priceToBook")
    if isinstance(pbr, (int, float)) and pbr > 0:
        items.append(("股價淨值比", "PBR", pbr < FUND_PBR_MAX, f"目前 {pbr:.2f} 倍", std))
    else:
        items.append(("股價淨值比", "PBR", None, "無股價淨值比資料", std))

    # 3. 現金股利殖利率
    std = f"> {FUND_YIELD_MIN:g}% 具長期存股吸引力"
    cash = _trailing_cash_dividend(dividends_series, info)
    if isinstance(latest_close, (int, float)) and latest_close > 0:
        dy = cash / latest_close * 100
        detail = f"近一年配息 {cash:.2f} 元，殖利率 {dy:.2f}%" if cash > 0 else "近一年無現金股利"
        items.append(("現金股利殖利率", "Dividend Yield", dy > FUND_YIELD_MIN, detail, std))
    else:
        items.append(("現金股利殖利率", "Dividend Yield", None, "無股價資料，無法計算", std))

    # 4. 每股盈餘 EPS：最新一季較去年同期成長
    std = "EPS 持續成長代表獲利能力上升（比較最新一季與去年同期）"
    eps_ok, eps_detail = None, "EPS 資料不足，無法判斷成長"
    try:
        if eps_quarterly is not None and len(eps_quarterly) >= 5:
            cur, yoy = float(eps_quarterly.iloc[0]), float(eps_quarterly.iloc[4])
            eps_ok = cur > yoy
            eps_detail = f"最新一季 {cur:.2f}，去年同期 {yoy:.2f}"
        else:
            g = info.get("earningsQuarterlyGrowth")
            if isinstance(g, (int, float)):
                eps_ok = g > 0
                eps_detail = f"Yahoo 季獲利年增率 {g * 100:+.1f}%"
    except Exception:
        pass
    items.append(("每股盈餘", "EPS", eps_ok, eps_detail, std))

    # 5. 股東權益報酬率 ROE
    std = f"> {FUND_ROE_MIN:g}% 通常代表為股東創造利潤的能力優異"
    roe = info.get("returnOnEquity")
    if isinstance(roe, (int, float)):
        items.append(("股東權益報酬率", "ROE", roe * 100 > FUND_ROE_MIN, f"目前 ROE {roe * 100:.2f}%", std))
    else:
        items.append(("股東權益報酬率", "ROE", None, "無 ROE 資料", std))

    return items


def _technical_items(d):
    """技術面：d 為含 K、D、DIF、MACD、OSC、RSI、MA20、MA60 的日K資料表。"""
    names = [
        ("移動平均線", "MA"), ("隨機指標", "KD (KDJ)"), ("相對強弱指標", "RSI"),
        ("指數平滑異同移動平均線", "MACD"), ("乖離率", "BIAS"),
    ]
    stds = [
        "紅燈：股價同時站上季線(60日)與年線(240日)視為中長期多頭。\n\n"
        "• 多頭排列（極佳，全面作多訊號）｜20MA > 60MA > 240MA 且均線同步勾頭向上：無論長短線投資人都在賺錢，是市場最強烈、最健康的「好趨勢」，應順勢做多。\n\n"
        "• 空頭排列（極壞，絕對觀望/放空訊號）｜240MA > 60MA > 20MA 且均線全面勾頭向下：市場上套牢賣壓沉重，走勢極「壞」，千萬不可盲目攤平接刀。\n\n"
        "• 黃金交叉（好訊號）｜短天期均線由下往上突破長天期均線（例如 20MA 突破 60MA）：短期買盤力道超越長期平均，中長期波段極大機率即將啟動上漲。\n\n"
        "• 死亡交叉（壞訊號）｜短天期均線由上往下跌破長天期均線（例如 20MA 跌破 60MA）：短期持股成本快速跌破長期平均，多頭行情結束，可能迎來深度的崩盤或回檔。",
        "紅燈：K > D 且 K < 80（未進入超買區）視為偏多。\n\n"
        "• 80 以上｜超買區（短線過熱）：短期內買方力道極強，通常暗示股價面臨追高風險，隨時可能拉回修正，屬於短線需提高警覺的「壞」位置。\n\n"
        "• 20～80｜多空平衡區：股價處於正常波動範圍，此時以順勢操作或參考交叉訊號為主。\n\n"
        "• 20 以下｜超賣區（短線過冷）：短期內股價被過度拋售，代表賣壓可能已經竭盡，隨時容易觸底反彈，屬於值得留意買點的「好」機會。",
        "紅燈：RSI 落在 52～70「多頭強勢」區（買方占優但未過熱）。\n\n"
        "• 70 以上｜超買區（過熱）：買方力道極強，但接近 80 需警惕追高風險，短期可能高檔折返或修正。\n\n"
        "• 50～70｜多頭強勢：買方大於賣方，多方控盤，屬穩健上漲趨勢，走勢相對偏好。\n\n"
        "• 50 附近（48～52）｜多空平衡：買賣力道持平，可能處於盤整、觀望期或方向選擇中。\n\n"
        "• 30～50｜空頭弱勢：賣方大於買方，空方控盤，股價偏向盤跌或弱勢。\n\n"
        "• 30 以下｜超賣區（過冷）：賣方力道極強，股價已連續拋售，但也暗示短期內具反彈機會，是尋找潛在買點的觀察區。",
        "紅燈：柱狀體 OSC > 0（DIF 在 MACD 之上，紅柱）。\n\n"
        "• 快慢線在 0 軸之上｜絕對多頭市場：走勢偏好，以做多為主，每一次回檔波段往往是尋找買點的機會。\n\n"
        "• 快慢線在 0 軸之下｜絕對空頭市場：走勢偏壞，宜觀望或做空為主，反彈往往是強弩之末。\n\n"
        "• 柱狀體由綠翻紅（由負變正）｜買方動能增強：下跌力道減弱、上漲動能開始爆發，屬於波段「轉好」的起點。\n\n"
        "• 柱狀體由紅翻綠（由正變負）｜賣方動壓湧現：上漲力道竭盡、下跌動能開始增強，屬於波段「轉壞」的起點。",
        f"紅燈：乖離率(20日) < {BIAS_OVERHEAT:g}%（未過熱）視為合理。\n\n"
        f"• 正乖離過大（≥ {BIAS_OVERHEAT:g}%）｜短線超買、嚴重過熱：股價遠高於市場平均成本，短線獲利賣壓極重，隨時可能拉回修正，屬於追高風險極大的「壞」位置。\n\n"
        "• 接近 0% 附近｜多空成本平衡：股價貼近均線。多頭市場中，常是拉回均線有撐的「好買點」；空頭市場中，則是反彈遭遇均線壓力的「反轉點」。\n\n"
        f"• 負乖離過大（≤ -{BIAS_OVERHEAT:g}%）｜短線超賣、跌深恐慌：股價遠低於市場平均成本，市場陷入非理性拋售，但也暗示隨時會出現強烈反彈，是左側交易者尋找潛在「好買點」的觀察區。",
    ]
    if d is None or len(d) < 30:
        return [(n, a, None, "日K資料不足", s) for (n, a), s in zip(names, stds)]

    last = d.iloc[-1]
    close = last["Close"]
    items = []

    # 1. 移動平均線：排列 + 方向 + 20/60 黃金/死亡交叉
    ma60 = last["MA60"]
    ma240_s = d["Close"].rolling(240).mean()
    ma240 = ma240_s.iloc[-1]
    if pd.isna(ma60):
        items.append((*names[0], None, "資料不足 60 日，無法計算季線", stds[0]))
    else:
        a60 = close > ma60
        if pd.isna(ma240):
            detail = f"股價 {close:.2f}｜季線 {ma60:.2f}（{'之上' if a60 else '之下'}）｜年線資料不足，僅以季線判斷"
            ok = a60
        else:
            a240 = close > ma240
            detail = (f"股價 {close:.2f}｜季線 {ma60:.2f}（{'之上' if a60 else '之下'}）"
                      f"｜年線 {ma240:.2f}（{'之上' if a240 else '之下'}）")
            ok = a60 and a240

        ma20_v = last["MA20"]
        ma20_s, ma60_s = d["MA20"], d["MA60"]

        def _slope(series, n=5):
            """近 n 日均線方向：1 上揚、-1 下彎、0 走平"""
            sr = series.dropna()
            if len(sr) <= n or sr.iloc[-n - 1] == 0:
                return 0
            chg = (sr.iloc[-1] - sr.iloc[-n - 1]) / sr.iloc[-n - 1] * 100
            return 1 if chg > 0.1 else (-1 if chg < -0.1 else 0)

        # 排列（有年線用 20/60/240，否則退而用 20/60）
        up_all = down_all = False
        if not pd.isna(ma20_v):
            if not pd.isna(ma240):
                chain = [ma20_v, ma60, ma240]
                slopes = [_slope(ma20_s), _slope(ma60_s), _slope(ma240_s)]
                label = "20MA > 60MA > 240MA"
                label_r = "240MA > 60MA > 20MA"
            else:
                chain = [ma20_v, ma60]
                slopes = [_slope(ma20_s), _slope(ma60_s)]
                label = "20MA > 60MA"
                label_r = "60MA > 20MA"
            bull = all(chain[x] > chain[x + 1] for x in range(len(chain) - 1))
            bear = all(chain[x] < chain[x + 1] for x in range(len(chain) - 1))
            up_all = bull and all(sl > 0 for sl in slopes)
            down_all = bear and all(sl < 0 for sl in slopes)
            if up_all:
                arr = f"多頭排列（{label}，均線同步勾頭向上）＝全面作多訊號：無論長短線投資人都在賺錢，是最強烈、最健康的好趨勢，宜順勢做多"
            elif down_all:
                arr = f"空頭排列（{label_r}，均線全面勾頭向下）＝絕對觀望/放空訊號：套牢賣壓沉重，走勢極壞，不可盲目攤平接刀"
            elif bull:
                arr = f"均線呈多頭排列（{label}），但尚有均線未同步向上，強度待確認"
            elif bear:
                arr = f"均線呈空頭排列（{label_r}），但尚有均線未同步向下，仍偏弱勢"
            else:
                arr = "均線糾結、排列不明確＝盤整格局，方向待選擇"
        else:
            arr = "資料不足，無法判斷均線排列"

        # 近 5 個交易日內的 20MA / 60MA 交叉
        cross = ""
        diff = (ma20_s - ma60_s).dropna().tail(6)
        if len(diff) >= 2:
            sg = [1 if v > 0 else (-1 if v < 0 else 0) for v in diff]
            for x in range(len(sg) - 1, 0, -1):
                if sg[x] > 0 and sg[x - 1] <= 0:
                    cross = "近期出現黃金交叉（20MA 由下往上突破 60MA）＝短期買盤力道超越長期平均，中長期波段極大機率即將啟動上漲"
                    break
                if sg[x] < 0 and sg[x - 1] >= 0:
                    cross = "近期出現死亡交叉（20MA 由上往下跌破 60MA）＝短期持股成本快速跌破長期平均，多頭行情結束，可能迎來深度崩盤或回檔"
                    break
        if not cross:
            cross = "近 5 日無 20MA／60MA 交叉"

        items.append((*names[0], bool(ok), f"{detail}｜{arr}｜{cross}", stds[0]))

    # 2. KD：依數值區間對照市場含意與操作邏輯 + 黃金/死亡交叉
    k, dd = last["K"], last["D"]
    if pd.isna(k) or pd.isna(dd):
        items.append((*names[1], None, "資料不足", stds[1]))
    else:
        if k >= 80:
            zone, logic = "超買區（短線過熱）", "短期內買方力道極強，通常暗示股價面臨追高風險，隨時可能拉回修正，屬於短線需提高警覺的壞位置"
        elif k <= 20:
            zone, logic = "超賣區（短線過冷）", "短期內股價被過度拋售，賣壓可能已經竭盡，隨時容易觸底反彈，屬於值得留意買點的好機會"
        else:
            zone, logic = "多空平衡區（20～80）", "股價處於正常波動範圍，以順勢操作或參考交叉訊號為主"

        kd_cross = ""
        kd_diff = (d["K"] - d["D"]).dropna().tail(4)
        if len(kd_diff) >= 2:
            sg = [1 if v > 0 else (-1 if v < 0 else 0) for v in kd_diff]
            for x in range(len(sg) - 1, 0, -1):
                if sg[x] > 0 and sg[x - 1] <= 0:
                    kd_cross = "｜近期 K 由下往上穿越 D（黃金交叉），短線買盤轉強"
                    break
                if sg[x] < 0 and sg[x - 1] >= 0:
                    kd_cross = "｜近期 K 由上往下跌破 D（死亡交叉），短線賣壓轉強"
                    break
        items.append((*names[1], bool(k > dd and k < 80),
                      f"K {k:.0f}／D {dd:.0f}｜{zone}：{logic}{kd_cross}", stds[1]))

    # 3. RSI：依數值區間對照市場含意與操作邏輯
    rsi = last["RSI"]
    if pd.isna(rsi):
        items.append((*names[2], None, "資料不足", stds[2]))
    else:
        if rsi >= 70:
            zone, logic = "超買區（過熱）", "買方力道極強；接近 80 需警惕追高風險，短期可能高檔折返或修正"
        elif rsi <= 30:
            zone, logic = "超賣區（過冷）", "賣方力道極強；股價已連續拋售，但短期內具反彈機會，可列入觀察買點"
        elif 48 <= rsi <= 52:
            zone, logic = "多空平衡（50 附近）", "買賣力道持平；可能處於盤整、觀望期或方向選擇中"
        elif rsi > 52:
            zone, logic = "多頭強勢（50～70）", "買方大於賣方，多方控盤，屬穩健上漲趨勢，走勢相對偏好"
        else:
            zone, logic = "空頭弱勢（30～50）", "賣方大於買方，空方控盤，股價偏向盤跌或弱勢"
        items.append((*names[2], bool(52 < rsi < 70), f"RSI(14) {rsi:.0f}｜{zone}：{logic}", stds[2]))

    # 4. MACD：快慢線與 0 軸的位置 + 柱狀體翻色
    if pd.isna(last["OSC"]):
        items.append((*names[3], None, "資料不足", stds[3]))
    else:
        dif, sig, osc = last["DIF"], last["MACD"], last["OSC"]
        if dif > 0 and sig > 0:
            zone, logic = "快慢線在 0 軸之上＝絕對多頭市場", "走勢偏好，以做多為主，回檔波段往往是尋找買點的機會"
        elif dif < 0 and sig < 0:
            zone, logic = "快慢線在 0 軸之下＝絕對空頭市場", "走勢偏壞，宜觀望或做空為主，反彈往往是強弩之末"
        else:
            zone, logic = "快慢線在 0 軸附近（一上一下）", "多空轉換中，方向尚未明確"

        # 近 3 個交易日內柱狀體是否翻色（台股：正＝紅柱、負＝綠柱）
        osc_recent = d["OSC"].dropna().tail(4)
        flip = ""
        if len(osc_recent) >= 2:
            signs = [1 if v > 0 else (-1 if v < 0 else 0) for v in osc_recent]
            for i in range(len(signs) - 1, 0, -1):
                if signs[i] > 0 and signs[i - 1] <= 0:
                    flip = "柱狀體由綠翻紅（由負變正）＝買方動能增強，下跌力道減弱、上漲動能開始爆發，屬波段轉好的起點"
                    break
                if signs[i] < 0 and signs[i - 1] >= 0:
                    flip = "柱狀體由紅翻綠（由正變負）＝賣方壓力湧現，上漲力道竭盡、下跌動能開始增強，屬波段轉壞的起點"
                    break
        hist = flip or f"柱狀體 {osc:+.2f}，{'紅柱延續' if osc > 0 else '綠柱延續'}，未見翻色"
        items.append((*names[3], bool(osc > 0),
                      f"DIF {dif:.2f}／MACD {sig:.2f}｜{zone}：{logic}｜{hist}", stds[3]))

    # 5. 乖離率（對 20 日均線）
    ma20 = last["MA20"]
    if pd.isna(ma20) or ma20 == 0:
        items.append((*names[4], None, "資料不足 20 日", stds[4]))
    else:
        bias = (close - ma20) / ma20 * 100
        if bias >= BIAS_OVERHEAT:
            zone, logic = "正乖離過大＝短線超買、嚴重過熱", "股價遠高於市場平均成本，短線獲利賣壓極重，隨時可能拉回修正，屬追高風險極大的壞位置"
        elif bias <= -BIAS_OVERHEAT:
            zone, logic = "負乖離過大＝短線超賣、跌深恐慌", "股價遠低於市場平均成本，市場陷入非理性拋售，但隨時會出現強烈反彈，是左側交易者尋找潛在好買點的觀察區"
        elif abs(bias) <= 2:
            above_ma60 = (not pd.isna(ma60)) and close > ma60
            zone, logic = "接近 0%＝多空成本平衡", (
                "股價貼近均線；目前站在季線之上偏多頭，拉回均線有撐時常是好買點" if above_ma60 else
                "股價貼近均線；目前在季線之下偏空頭，反彈遇均線壓力時留意反轉點")
        elif bias > 0:
            zone, logic = "正乖離（未達過熱）", "股價高於均線但尚在合理範圍，留意乖離持續擴大"
        else:
            zone, logic = "負乖離（未達超跌）", "股價低於均線但尚在合理範圍，可觀察是否止跌回穩"
        items.append((*names[4], bool(bias < BIAS_OVERHEAT),
                      f"20日乖離率 {bias:+.2f}%｜{zone}：{logic}", stds[4]))

    return items


# ---------------------------------------------------------------------------
# 行業別
# ---------------------------------------------------------------------------
SECTOR_ZH = {
    "Technology": "科技",
    "Financial Services": "金融服務",
    "Healthcare": "醫療保健",
    "Consumer Cyclical": "循環性消費",
    "Consumer Defensive": "防禦性消費",
    "Industrials": "工業",
    "Basic Materials": "基礎材料",
    "Energy": "能源",
    "Utilities": "公用事業",
    "Real Estate": "房地產",
    "Communication Services": "通訊服務",
}


# 細分行業標籤：代號 → 標籤（以「、」分隔）。這是程式內建的對照表，可自行增修；未收錄的股票會改用 Yahoo / 證交所分類。
INDUSTRY_TAGS = {
    # --- 半導體：IC設計 ---
    "2454": "IC設計、半導體", "3034": "IC設計、半導體", "2379": "IC設計、半導體", "3035": "IC設計、半導體",
    "3443": "IC設計、半導體", "6533": "IC設計、半導體", "4966": "IC設計、半導體", "5269": "IC設計、半導體",
    "6415": "IC設計、半導體", "3529": "IC設計、半導體", "8299": "IC設計、記憶體", "2458": "IC設計、半導體",
    "3014": "IC設計、半導體", "6756": "IC設計、半導體", "3227": "IC設計、半導體", "5274": "IC設計、半導體",
    "4961": "IC設計、半導體", "6643": "IC設計、半導體", "3661": "IC設計、半導體", "2436": "IC設計、半導體",
    "8016": "IC設計、半導體", "6202": "IC設計、半導體", "4919": "IC設計、半導體", "2401": "IC設計、半導體",
    "3006": "IC設計、記憶體",
    # --- 半導體：晶圓代工 ---
    "2330": "晶圓代工、半導體", "2303": "晶圓代工、半導體", "6770": "晶圓代工、半導體", "5347": "晶圓代工、半導體",
    # --- 半導體：封裝測試 ---
    "3711": "封裝測試、半導體", "2449": "封裝測試、半導體", "6239": "封裝測試、半導體", "8150": "封裝測試、半導體",
    "6257": "封裝測試、半導體", "2441": "封裝測試、半導體", "6147": "封裝測試、半導體", "3264": "封裝測試、半導體",
    "3374": "封裝測試、半導體", "2369": "封裝測試、半導體",
    # --- 半導體：設備、材料、廠務 ---
    "3680": "半導體設備", "3131": "半導體設備", "3583": "半導體設備", "6196": "半導體設備、廠務工程",
    "2404": "半導體廠務工程", "3413": "半導體設備", "2360": "半導體測試設備", "6187": "封裝設備、PCB設備",
    "3030": "測試設備、PCB設備", "3167": "PCB設備、半導體設備",
    "1560": "半導體材料", "3532": "半導體材料", "6488": "半導體材料（矽晶圓）", "5483": "半導體材料（矽晶圓）",
    "3016": "半導體材料、功率半導體",
    # --- 記憶體、功率半導體 ---
    "2408": "記憶體、半導體", "2337": "記憶體、半導體", "3260": "記憶體模組", "4967": "記憶體模組", "5351": "記憶體、IC設計",
    "2481": "功率半導體", "5425": "功率半導體", "8261": "功率半導體",
    # --- PCB 相關產業 ---
    "3037": "PCB相關產業、ABF載板", "8046": "PCB相關產業、ABF載板", "3189": "PCB相關產業、ABF載板",
    "2313": "PCB相關產業", "3044": "PCB相關產業", "2368": "PCB相關產業", "4958": "PCB相關產業",
    "5469": "PCB相關產業", "8213": "PCB相關產業", "2316": "PCB相關產業", "6191": "PCB相關產業",
    "2383": "PCB相關產業、銅箔基板", "6213": "PCB相關產業、銅箔基板", "6274": "PCB相關產業、銅箔基板",
    "6672": "PCB相關產業、銅箔基板", "1815": "PCB相關產業、玻纖布",
    "6153": "PCB相關產業、軟板", "6269": "PCB相關產業、軟板", "2355": "PCB相關產業", "8021": "PCB相關產業、鑽針",
    # --- 電子代工、伺服器、散熱、電源 ---
    "2317": "電子代工、AI伺服器", "2382": "AI伺服器、電腦代工", "3231": "AI伺服器、電腦代工", "6669": "AI伺服器",
    "2356": "AI伺服器、電腦代工", "2376": "伺服器、主機板", "2377": "伺服器、主機板", "3706": "伺服器",
    "2324": "電腦代工", "4938": "電子代工", "2357": "電腦品牌", "2353": "電腦品牌",
    "2308": "電源供應、散熱", "2301": "電源供應、光電", "3017": "散熱", "3324": "散熱", "3653": "散熱", "6230": "散熱",
    # --- 網通、光通訊 ---
    "2345": "網通設備", "5388": "網通設備", "2332": "網通設備", "3596": "網通設備", "4906": "網通設備",
    "6285": "網通設備", "3380": "網通設備",
    "4979": "光通訊", "3363": "光通訊", "4977": "光通訊", "3450": "光通訊", "4908": "光通訊", "6442": "光通訊", "3081": "光通訊",
    # --- 被動元件、連接器、光學、面板 ---
    "2327": "被動元件", "2492": "被動元件", "3026": "被動元件", "6173": "被動元件", "2375": "被動元件",
    "3533": "連接器", "3023": "連接器、線束", "2392": "連接器", "6197": "連接器",
    "3008": "光學鏡頭", "3406": "光學鏡頭", "2409": "面板", "3481": "面板",
    # --- 航運、鋼鐵、塑化、水泥、食品、電信 ---
    "2603": "航運", "2609": "航運", "2615": "航運", "2605": "航運", "2606": "航運",
    "2002": "鋼鐵", "2014": "鋼鐵", "2027": "鋼鐵",
    "1301": "塑化", "1303": "塑化", "1326": "塑化", "6505": "塑化",
    "1101": "水泥", "1102": "水泥", "1216": "食品",
    "2412": "電信", "3045": "電信", "4904": "電信",
    # --- 金融 ---
    "2881": "金控", "2882": "金控", "2891": "金控", "2886": "金控", "2884": "金控", "2880": "金控",
    "2885": "金控", "2887": "金控", "2890": "金控", "2892": "金控", "5880": "金控", "2883": "金控",
    "2812": "銀行", "2801": "銀行", "2834": "銀行", "2845": "銀行",
    # --- 汽車、生技、ETF ---
    "2201": "汽車", "1319": "汽車零組件", "1536": "汽車零組件",
    "6446": "生技醫療", "4743": "生技醫療", "6547": "生技醫療",
    "0050": "ETF（台灣50）", "0056": "ETF（高股息）", "00878": "ETF（高股息）", "00919": "ETF（高股息）",
    "00929": "ETF（高股息）", "00881": "ETF（科技）",
}

# Yahoo 的細分行業（英文）→ 中文，供未收錄股票使用
YAHOO_INDUSTRY_ZH = {
    "Semiconductors": "半導體",
    "Semiconductor Equipment & Materials": "半導體設備與材料",
    "Electronic Components": "電子零組件",
    "Computer Hardware": "電腦硬體",
    "Consumer Electronics": "消費電子",
    "Communication Equipment": "通訊設備",
    "Electronics & Computer Distribution": "電子通路",
    "Scientific & Technical Instruments": "儀器設備",
    "Electrical Equipment & Parts": "電機設備",
    "Solar": "太陽能",
    "Auto Parts": "汽車零組件",
    "Auto Manufacturers": "汽車",
    "Steel": "鋼鐵",
    "Banks - Regional": "銀行",
    "Insurance - Life": "保險",
    "Asset Management": "資產管理",
    "Marine Shipping": "航運",
    "Airlines": "航空",
    "Biotechnology": "生技",
    "Drug Manufacturers - Specialty & Generic": "製藥",
    "Medical Devices": "醫療器材",
    "Packaged Foods": "食品",
    "Specialty Chemicals": "特用化學",
    "Chemicals": "化學",
    "Textile Manufacturing": "紡織",
    "Building Materials": "建材",
    "Telecom Services": "電信",
    "Real Estate - Development": "建設",
    "Engineering & Construction": "營建工程",
    "Software - Application": "軟體",
    "Information Technology Services": "資訊服務",
    "Display & Optics": "光電",
}


def render_company_intro(symbol, info, display_name):
    """一行顯示行業別（細分行業標籤）。標籤優先用內建對照表，其次 Yahoo 細分行業，最後才是證交所／Yahoo 大類。"""
    code = symbol.split(".")[0]
    sector_zh = SECTOR_ZH.get(info.get("sector") or "", info.get("sector") or "")
    try:
        tw_industry = get_tw_industry(symbol)
    except Exception:
        tw_industry = ""

    main = INDUSTRY_TAGS.get(code) or YAHOO_INDUSTRY_ZH.get(info.get("industry") or "", "")
    broad = tw_industry or sector_zh
    if main and broad and broad not in main:
        text = f"{main}（{broad}）"
    else:
        text = main or broad or "查無資料"
    st.markdown(f"🏢 **行業別**：{text}")


# ---------------------------------------------------------------------------
# 頁面設定與 Session State
# ---------------------------------------------------------------------------
st.set_page_config(page_title="股票分析", layout="wide")
st.title("📈 股票分析")

# 重新執行時 Streamlit 會把舊內容變淡（反白），這段 CSS 讓畫面維持原樣
st.markdown(
    """
    <style>
    [data-stale='true'] { opacity: 1 !important; }
    .block-container, [data-testid="stMainBlockContainer"] {
        max-width: 1150px;
        margin-left: auto;
        margin-right: auto;
    }

    /* 手機版：縮小邊距與字級、欄位改成每列 2 個、底部留白避免被平台徽章遮住 */
    @media (max-width: 640px) {
        .block-container, [data-testid="stMainBlockContainer"] {
            padding: 2.2rem 0.8rem 5rem 0.8rem !important;
        }
        h1 { font-size: 1.6rem !important; line-height: 1.3 !important; }
        h2, h3 { font-size: 1.2rem !important; }
        [data-testid="stMain"] [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            gap: 0.5rem 0.75rem !important;
        }
        [data-testid="stMain"] [data-testid="stColumn"] {
            min-width: calc(50% - 0.5rem) !important;
            flex: 1 1 calc(50% - 0.5rem) !important;
            width: calc(50% - 0.5rem) !important;
        }
        [data-testid="stMetricValue"] { font-size: 1.4rem !important; }
        [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {
            font-size: 0.8rem !important;
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
        }
        [data-testid="stMetricDelta"] { font-size: 0.75rem !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "search_history" not in st.session_state:
    st.session_state.search_history = []  # [[顯示名稱, 股票代號], ...]
if "input_val" not in st.session_state:
    st.session_state.input_val = ""
if "trigger_search" not in st.session_state:
    st.session_state.trigger_search = ""
if "current_name" not in st.session_state:
    st.session_state.current_name = ""
if "pending_search" not in st.session_state:
    st.session_state.pending_search = ""
if "search_stopped" not in st.session_state:
    st.session_state.search_stopped = False


def click_history_btn(stock_symbol, stock_name):
    st.session_state.input_val = stock_name
    st.session_state.search_stopped = False
    st.session_state.pending_search = ""
    st.session_state.trigger_search = stock_symbol
    st.session_state.current_name = stock_name


def delete_single_history(index):
    if 0 <= index < len(st.session_state.search_history):
        st.session_state.search_history.pop(index)


def handle_search_button_click():
    """按下搜尋只記下關鍵字；真正的查詢在主流程進行，這樣才能顯示載入中與停止按鈕。"""
    keyword = st.session_state.input_val.strip()
    if keyword:
        st.session_state.pending_search = keyword
        st.session_state.search_stopped = False


def stop_search():
    st.session_state.pending_search = ""
    st.session_state.trigger_search = ""
    st.session_state.search_stopped = True


def run_search(user_keyword):
    """把輸入轉成代號、驗證股票存在，並更新搜尋紀錄。"""
    symbol = get_ticker_from_name(user_keyword)

    # 驗證股票是否存在；台股 .TW 查不到時，再試上櫃 .TWO
    candidates = [symbol]
    if symbol.endswith(".TW"):
        candidates.append(symbol[:-3] + ".TWO")

    valid_symbol = None
    for cand in candidates:
        try:
            if not fetch_history(cand, TIME_PERIOD).empty:
                valid_symbol = cand
                break
        except Exception:
            continue

    if valid_symbol is None:
        st.session_state.trigger_search = "FAILED:" + user_keyword
        return

    display_name = resolve_display_name(valid_symbol, user_keyword)
    st.session_state.trigger_search = valid_symbol
    st.session_state.current_name = display_name

    # 去重後置頂，最多保留 8 筆
    st.session_state.search_history = [
        h for h in st.session_state.search_history if h[1] != valid_symbol
    ]
    st.session_state.search_history.insert(0, [display_name, valid_symbol])
    del st.session_state.search_history[8:]


def render_analysis(symbol_upper, time_period):
    beacon_slot = None
    try:
        with st.spinner("⏳ 載入資料中，請稍候…"):
            df = fetch_history(symbol_upper, time_period)
            info, dividends_series, splits_series, eps_quarterly, monthly_df = (
                fetch_fundamentals(symbol_upper)
            )

        if df is None or len(df) < 2:
            st.warning("⚠️ 近一年的歷史資料不足（少於 2 筆），可能是剛上市的股票，無法進行分析。")
            return

        display_name = st.session_state.current_name or resolve_display_name(
            symbol_upper, "", info.get("shortName", "")
        )

        latest_close = df["Close"].iloc[-1]
        prev_close = df["Close"].iloc[-2]
        high_52 = info.get("fiftyTwoWeekHigh", latest_close)

        change = latest_close - prev_close
        change_pct = (change / prev_close) * 100

        pe_ratio = info.get("trailingPE")
        pe_display = f"{pe_ratio:.2f}" if isinstance(pe_ratio, (int, float)) else "N/A"

        # --- 近一年權益 ---
        cash_dividend = "0.00 元"
        stock_dividend = "0.00 元"
        one_year_ago = pd.Timestamp.now().tz_localize(None) - pd.Timedelta(days=365)

        try:
            if dividends_series is not None and not dividends_series.empty:
                idx = pd.to_datetime(dividends_series.index).tz_localize(None)
                total_cash = dividends_series[idx >= one_year_ago].sum()
                if total_cash > 0:
                    cash_dividend = f"{total_cash:.2f} 元"
        except Exception:
            pass

        try:
            if splits_series is not None and not splits_series.empty:
                idx = pd.to_datetime(splits_series.index).tz_localize(None)
                total_stock = 0.0
                for val in splits_series[idx >= one_year_ago]:
                    if 1.0 < val < 2.0 and symbol_upper.endswith((".TW", ".TWO")):
                        total_stock += (val - 1.0) * 10.0  # 估算：以面額 10 元換算
                    elif val > 1.0:
                        total_stock += val
                if total_stock > 0:
                    stock_dividend = f"{total_stock:.2f} 元"
        except Exception:
            pass

        if cash_dividend == "0.00 元":
            backup_div = info.get("trailingAnnualDividendRate", info.get("dividendRate", 0))
            if backup_div and backup_div > 0:
                cash_dividend = f"{backup_div:.2f} 元"

        # --- 均線與月線位階診斷 ---
        df["MA20"] = df["Close"].rolling(window=20).mean()
        df["MA50"] = df["Close"].rolling(window=50).mean()
        latest_ma20 = df["MA20"].iloc[-1]

        if pd.isna(latest_ma20):
            ma20_display = "N/A"
            ma20_delta = None
            diag_level = "info"
            diagnostic_msg = "💡 現況診斷：歷史資料不足 20 天，暫時無法計算月線位階。"
        else:
            diff_pct = (latest_close - latest_ma20) / latest_ma20 * 100
            ma20_display = f"{latest_ma20:.2f}"
            ma20_delta = f"{diff_pct:+.2f}% 距月線"
            if latest_close > latest_ma20:
                diag_level = "success"
                diagnostic_msg = (
                    f"💡 現況診斷：目前最新股價 **高於月線 ({diff_pct:+.2f}%)**，屬於偏多軌道。"
                )
            elif latest_close < latest_ma20:
                diag_level = "error"
                diagnostic_msg = (
                    f"💡 現況診斷：目前最新股價 **低於月線 ({diff_pct:+.2f}%)**，需留意短線修正回檔壓力。"
                )
            else:
                diag_level = "info"
                diagnostic_msg = "💡 現況診斷：目前最新股價與月線平齊，處於多空關鍵平衡點。"

        # --- 近四季 EPS ---
        eps_last4 = eps_quarterly.head(4) if eps_quarterly is not None else pd.Series(dtype=float)
        trailing_eps = info.get("trailingEps")
        if len(eps_last4) == 4:
            eps_display = f"{eps_last4.sum():.2f} 元"
            eps_note = "近四季合計"
        elif isinstance(trailing_eps, (int, float)):
            eps_display = f"{trailing_eps:.2f} 元"
            eps_note = "Yahoo 近12個月 EPS"
        else:
            eps_display = "N/A"
            eps_note = None

        # --- 渲染 ---
        st.subheader(f"📊 {display_name} ({symbol_upper}) 分析結果")
        render_company_intro(symbol_upper, info, display_name)
        beacon_slot = st.empty()
        beacon_slot.info("🚦 購買指標燈號計算中…")
        render_live_quote(symbol_upper)
        getattr(st, diag_level)(diagnostic_msg)

        # 第一列：價格相關
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        r1c1.metric(
            "日K最新價 (Yahoo)",
            f"{latest_close:.2f}",
            f"{change:+.2f} ({change_pct:+.2f}%)",
            delta_color="inverse",  # 台股慣例：紅漲綠跌
        )
        r1c2.metric(
            "月線 (MA20)",
            ma20_display,
            ma20_delta,
            delta_color="inverse",  # 紅＝股價高於月線，綠＝低於月線
        )
        r1c3.metric("本益比 (PE)", pe_display)
        r1c4.metric(
            "52週高點",
            f"{high_52:.2f}" if isinstance(high_52, (int, float)) else "N/A",
        )

        # 第二列：獲利與股利
        r2c1, r2c2, r2c3 = st.columns(3)
        r2c1.metric(
            "近四季 EPS" if eps_note != "Yahoo 近12個月 EPS" else "近12個月 EPS (Yahoo)",
            eps_display,
        )
        r2c2.metric("現金股利 (近一年累計配息)", cash_dividend)
        r2c3.metric("股票股利 (近一年累計配股，估算)", stock_dividend)

        # 近四季 EPS 明細
        if len(eps_last4) > 0:
            with (st.container() if is_mobile() else st.columns([1, 1])[0]):
                with st.expander("📋 近四季 EPS 明細"):
                    eps_table = pd.DataFrame(
                        {
                            "季度": [f"{d.year} Q{(d.month - 1) // 3 + 1}" for d in eps_last4.index],
                            "財報截止日": [d.strftime("%Y-%m-%d") for d in eps_last4.index],
                            "EPS (元)": [f"{v:.2f}" for v in eps_last4.values],
                        }
                    )
                    st.dataframe(eps_table, hide_index=True, width="stretch")
                    if len(eps_last4) < 4:
                        st.caption("⚠️ Yahoo 只提供不到四季的資料，上方 EPS 可能改用 Yahoo 的近 12 個月數值。")

        # --- K 線圖 + 均線 ---
        fig = go.Figure()
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="K線",
                increasing_line_color="red",  # 台股：紅漲
                decreasing_line_color="green",  # 台股：綠跌
            )
        )
        fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], mode="lines", name="20日均線"))
        fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], mode="lines", name="50日均線"))
        fig.update_layout(
            title=f"{display_name} K 線走勢與移動平均線 (MA)",
            xaxis_title="日期",
            yaxis_title="價格",
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
        )
        show_chart(fig)

        inst_table = render_institutional(symbol_upper)
        daily_res = render_daily_analysis(symbol_upper)
        monthly_res = render_monthly_analysis(monthly_df, display_name)
        try:
            fund_items = _fundamental_items(info, pe_ratio, eps_quarterly, dividends_series, latest_close)
            tech_items = _technical_items(daily_res[2] if daily_res else None)
        except Exception:
            fund_items, tech_items = None, None
        render_beacons(
            beacon_slot, pe_ratio, eps_last4, inst_table, daily_res, monthly_res, fund_items, tech_items
        )
        render_news(symbol_upper, display_name)

    except Exception as e:
        if beacon_slot is not None:
            beacon_slot.empty()
        st.error(f"讀取資料發生錯誤: {e}")


# ---------------------------------------------------------------------------
# 頁籤二：即時市場排行（成交量前 20 名，每分鐘更新，附簡易走勢圖）
# 做法：證交所 MIS 即時行情分批查詢全部上市櫃股票／ETF，依累計成交量排序
# ---------------------------------------------------------------------------
RANK_TOP_N = 20
MIS_BATCH_SIZE = 100


def _rank_universe():
    """可排行的標的：一般股票（4 碼）與 ETF（00 開頭）。回傳 [(市場, 代號, 名稱)]。"""
    out = []
    for sym, name in get_tw_names().items():
        code, _, suffix = sym.partition(".")
        if (code.isdigit() and len(code) == 4) or code.startswith("00"):
            out.append(("otc" if suffix == "TWO" else "tse", code, name))
    return out


def _mis_batch(sess, batch):
    """一次查一批代號；失敗回傳 None。"""
    ex_ch = "|".join(f"{ex}_{code}.tw" for ex, code, _ in batch)
    for attempt in range(2):
        try:
            resp = sess.get(
                "https://mis.twse.com.tw/stock/api/getStockInfo.jsp",
                params={"ex_ch": ex_ch, "json": "1", "delay": "0", "_": int(time.time() * 1000)},
                timeout=10,
            )
            resp.raise_for_status()
            arr = resp.json().get("msgArray") or []
            if arr:
                time.sleep(0.3)  # 對官方網站客氣一點
                return arr
        except Exception:
            pass
        time.sleep(0.8)
    return None


def _rank_bucket():
    """快取版本：交易時間每分鐘換一次；非交易時間每 30 分鐘換一次（避免休市時一直重抓）。"""
    now = int(time.time())
    return f"open-{now // 60}" if is_market_open() else f"closed-{now // 1800}"


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_sparklines(symbols, bucket):
    """用 Yahoo 5 分 K 取得各檔「最近一個交易日」的走勢（收盤價清單）。失敗回傳空字典。"""
    out = {}
    if not symbols:
        return out
    try:
        raw = yf.download(
            list(symbols), period="5d", interval="5m", group_by="ticker",
            progress=False, threads=True, auto_adjust=False,
        )
    except Exception:
        raw = None
    for s in (symbols if raw is not None else ()):
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                if s not in raw.columns.get_level_values(0):
                    continue
                sub = raw[s]
            else:
                sub = raw
            close = sub["Close"].dropna()
            if close.empty:
                continue
            last_day = close.index[-1].date()
            close = close[[ts.date() == last_day for ts in close.index]]
            if len(close) >= 2:
                out[s] = [float(v) for v in close.values]
        except Exception:
            continue
    if _fugle_key():  # Yahoo 沒抓到的，改用 Fugle 當日 5 分 K 補上
        for s in symbols:
            if s in out:
                continue
            try:
                js = _fugle_get(f"intraday/candles/{s.split('.')[0]}", timeframe="5")
                closes = [float(c["close"]) for c in js.get("data") or [] if c.get("close")]
                if len(closes) >= 2:
                    out[s] = closes
            except Exception:
                continue
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_market_rank(bucket):
    """全市場累計成交量前 N 名。bucket 只用來控制快取更新頻率；失敗會丟出例外（不會被快取）。"""
    universe = _rank_universe()
    if not universe:
        raise RuntimeError("無法取得台股代號清單，請稍後再試")
    sess = _mis_session()  # 在主執行緒取得，再交給工作執行緒使用
    batches = [universe[i:i + MIS_BATCH_SIZE] for i in range(0, len(universe), MIS_BATCH_SIZE)]
    name_map = {(ex, code): name for ex, code, name in universe}

    items, failed = [], 0
    with ThreadPoolExecutor(max_workers=2) as pool:
        for arr in pool.map(lambda b: _mis_batch(sess, b), batches):
            if arr is None:
                failed += 1
            else:
                items.extend(arr)
    if failed > len(batches) // 2:
        raise RuntimeError("證交所即時行情暫時無法連線（可能被限流），請稍後再試")

    rows = []
    for it in items:
        vol = _to_float(it.get("v"))
        if not vol:
            continue
        q = _parse_mis(it)
        price, prev = q["price"] or q["prev_close"], q["prev_close"]
        if not price:
            continue
        ex = it.get("ex") or "tse"
        code = str(it.get("c", "")).strip()
        rows.append(
            {
                "symbol": code + (".TWO" if ex == "otc" else ".TW"),
                "code": code,
                "name": it.get("n") or name_map.get((ex, code), code),
                "price": price,
                "prev": prev,
                "volume": vol,
                "time": q["time"],
            }
        )
    rows.sort(key=lambda r: r["volume"], reverse=True)
    top = rows[:RANK_TOP_N]
    spark = fetch_sparklines(tuple(r["symbol"] for r in top), bucket)
    return {
        "rows": top,
        "spark": spark,
        "fetched_at": datetime.datetime.now(TW_TZ).strftime("%H:%M:%S"),
        "covered": len(items),
        "total": len(universe),
        "failed": failed,
        "source": "證交所即時行情",
        "backup": False,
    }


def _fugle_key():
    """讀取 Fugle API key：.streamlit/secrets.toml 裡名稱含 FUGLE 的欄位（或 [fugle] 區段），或同名環境變數。"""
    try:
        for k in st.secrets:
            if "FUGLE" not in str(k).upper():
                continue
            v = st.secrets[k]
            if isinstance(v, str) and v.strip():
                return v.strip()
            if hasattr(v, "keys"):  # [fugle] 區段：取第一個字串值
                for kk in v.keys():
                    if isinstance(v[kk], str) and v[kk].strip():
                        return v[kk].strip()
    except Exception:
        pass
    for k, v in os.environ.items():
        if "FUGLE" in k.upper() and v.strip():
            return v.strip()
    return ""


def _fugle_get(path, **params):
    key = _fugle_key()
    if not key:
        raise RuntimeError("未設定 Fugle API key")
    resp = requests.get(
        f"https://api.fugle.tw/marketdata/v1.0/stock/{path}",
        params=params,
        headers={"X-API-KEY": key},
        timeout=10,
    )
    if resp.status_code == 401:
        raise RuntimeError("Fugle API key 驗證失敗")
    if resp.status_code == 403:
        raise RuntimeError("Fugle 目前的方案不支援此功能（成交量排行需開發者以上方案）")
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_market_rank_fugle(bucket):
    """備援一：Fugle 成交量排行（上市＋上櫃，近即時）。"""
    rows, stamp = [], ""
    for market, suffix in (("TSE", ".TW"), ("OTC", ".TWO")):
        js = _fugle_get(f"snapshot/actives/{market}", trade="volume", type="ALLBUT0999")
        t = str(js.get("time", ""))
        if len(t) >= 6:
            stamp = f"{js.get('date', '')} {t[:2]}:{t[2:4]}:{t[4:6]}"
        for d in js.get("data") or []:
            vol, price = d.get("tradeVolume"), d.get("closePrice")
            if not vol or not price:
                continue
            code = str(d.get("symbol", "")).strip()
            rows.append(
                {
                    "symbol": code + suffix,
                    "code": code,
                    "name": d.get("name") or code,
                    "price": float(price),
                    "prev": float(price) - float(d.get("change") or 0),
                    "volume": float(vol),  # Fugle 成交量單位：張
                    "time": stamp,
                }
            )
    if not rows:
        raise RuntimeError("Fugle 沒有回傳成交資料")
    rows.sort(key=lambda r: r["volume"], reverse=True)
    top = rows[:RANK_TOP_N]
    return {
        "rows": top,
        "spark": fetch_sparklines(tuple(r["symbol"] for r in top), bucket),
        "fetched_at": datetime.datetime.now(TW_TZ).strftime("%H:%M:%S"),
        "covered": len(rows),
        "total": len(rows),
        "failed": 0,
        "source": "Fugle 行情（備援）",
        "backup": True,
        "realtime": True,
    }


def _finmind_get(dataset, **params):
    """FinMind 備援來源共用請求；token 從 .streamlit/secrets.toml 或環境變數讀取。"""
    p = {"dataset": dataset, **params}
    token = _finmind_token()
    if token:
        p["token"] = token
    resp = requests.get("https://api.finmindtrade.com/api/v4/data", params=p, timeout=20)
    resp.raise_for_status()
    body = resp.json()
    if body.get("status") != 200:
        raise RuntimeError(body.get("msg", "FinMind 回傳錯誤"))
    return pd.DataFrame(body.get("data", []))


def _recent_weekdays(n=14, include_today=True):
    day = datetime.datetime.now(TW_TZ).date()
    if not include_today:
        day -= datetime.timedelta(days=1)
    out = []
    while len(out) < n:
        if day.weekday() < 5:
            out.append(day)
        day -= datetime.timedelta(days=1)
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_market_rank_backup(bucket):
    """備援：FinMind 最近一個交易日的成交量排行（非即時）。"""
    df = pd.DataFrame()
    day = None
    for day in _recent_weekdays(10):
        df = _finmind_get("TaiwanStockPrice", start_date=day.isoformat(), end_date=day.isoformat())
        if not df.empty:
            break
    if df.empty:
        raise RuntimeError("FinMind 查無資料（可能需要在 .streamlit/secrets.toml 設定 FINMIND_TOKEN）")
    df["stock_id"] = df["stock_id"].astype(str)
    df = df[df["stock_id"].map(lambda c: (c.isdigit() and len(c) == 4) or c.startswith("00"))]
    df = df.sort_values("Trading_Volume", ascending=False).head(RANK_TOP_N)
    names = get_tw_names()
    rows = []
    for _, r in df.iterrows():
        code = r["stock_id"]
        suffix = ".TWO" if (code + ".TWO") in names and (code + ".TW") not in names else ".TW"
        close = float(r["close"])
        rows.append(
            {
                "symbol": code + suffix,
                "code": code,
                "name": names.get(code + suffix, code),
                "price": close,
                "prev": close - float(r.get("spread", 0) or 0),
                "volume": float(r["Trading_Volume"]) / 1000,  # 股 → 張
                "time": day.isoformat(),
            }
        )
    spark = fetch_sparklines(tuple(r["symbol"] for r in rows), bucket)
    return {
        "rows": rows,
        "spark": spark,
        "fetched_at": datetime.datetime.now(TW_TZ).strftime("%H:%M:%S"),
        "covered": len(df),
        "total": len(df),
        "failed": 0,
        "source": "FinMind（備援，非即時）",
        "backup": True,
        "realtime": False,
    }


def _change_badge(price, prev):
    """台股慣例：紅漲綠跌。回傳 (漲跌, 漲跌幅文字)。"""
    if not prev:
        return None, "N/A"
    chg = price - prev
    pct = chg / prev * 100
    icon = "🔴 ▲" if chg > 0 else ("🟢 ▼" if chg < 0 else "⚪ ")
    return chg, f"{icon}{abs(pct):.2f}%"


def _market_rank_body():
    bucket = _rank_bucket()
    data, errors = None, []
    first_load = not st.session_state.get("_rank_loaded", False)
    with st.spinner("⏳ 載入資料中，請稍候…") if first_load else contextlib.nullcontext():
        for label, fn in (
            ("證交所即時行情", fetch_market_rank),
            ("Fugle", fetch_market_rank_fugle),
            ("FinMind", fetch_market_rank_backup),
        ):
            try:
                data = fn(bucket)
                break
            except Exception as e:
                errors.append(f"{label}：{e}")
    if data is not None:
        st.session_state["_rank_loaded"] = True
    if data is None:
        st.warning("⚠️ 目前無法取得即時排行。" + "；".join(errors))
        return
    if data.get("backup"):
        tail = "" if data.get("realtime", True) else "，顯示最近一個交易日的排行（非即時）"
        st.warning(f"⚠️ 證交所即時行情暫時無法使用（{errors[0]}），已改用備援來源「{data['source']}」{tail}。")
    rows = data["rows"]
    if not rows:
        st.info("目前沒有成交資料（尚未開盤或休市）。")
        return

    table = pd.DataFrame(
        {
            "排名": list(range(1, len(rows) + 1)),
            "代號": [r["code"] for r in rows],
            "名稱": [r["name"] for r in rows],
            "成交價": [r["price"] for r in rows],
            "漲跌幅": [_change_badge(r["price"], r["prev"])[1] for r in rows],
            "成交量(張)": [r["volume"] for r in rows],
            "走勢": [data["spark"].get(r["symbol"]) for r in rows],
        }
    )
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        height=min(35 * (len(rows) + 1) + 3, 740),
        column_config={
            "排名": st.column_config.NumberColumn(width="small"),
            "成交價": st.column_config.NumberColumn(format="%.2f"),
            "成交量(張)": st.column_config.NumberColumn(format="%d"),
            "走勢": st.column_config.LineChartColumn("走勢（當日 5 分K）", width="medium"),
        },
    )
    latest = max((r["time"] for r in rows if r["time"]), default="")
    st.caption(
        f"⚡ 更新時間 {data['fetched_at']}　資料時間 {latest}　掃描 {data['covered']}/{data['total']} 檔　"
        f"來源：{data['source']}（成交量）、Yahoo Finance（走勢圖，可能延遲）"
    )
    if data["failed"]:
        st.caption(f"⚠️ 有 {data['failed']} 批資料暫時抓取失敗，排行可能不完整，下次更新會自動補上。")


market_rank_auto = st.fragment(run_every=60)(_market_rank_body)
market_rank_static = st.fragment(_market_rank_body)


def render_market_rank():
    st.markdown("##### 🔥 即時成交量排行（前 20 名）")
    c1, c2 = st.columns([3, 1]) if not is_mobile() else (st.container(), st.container())
    with c1:
        st.toggle("🔄 交易時間每分鐘自動更新", value=True, key="rank_auto_refresh")
    with c2:
        if st.button("立即更新", key="rank_refresh_btn"):
            fetch_market_rank.clear()
            fetch_sparklines.clear()
            st.session_state["_rank_loaded"] = False
    if st.session_state.get("rank_auto_refresh", True) and is_market_open():
        market_rank_auto()
    else:
        market_rank_static()
        if not is_market_open():
            st.caption("目前非交易時間，顯示最近一個交易日的成交量排行。")


# ---------------------------------------------------------------------------
# 頁籤三：前一交易日 三大法人買超／賣超前 20 名（證交所 T86 + 櫃買中心）
# ---------------------------------------------------------------------------
INST_RANK_TOP_N = 20


def _inst_rows(by_code, names, industries=None):
    """by_code: {(代號, 後綴): (外資, 投信, 自營商)} → DataFrame"""
    rows = []
    for (code, suffix), (foreign, trust, dealer) in by_code.items():
        rows.append(
            {
                "代號": code,
                "名稱": names.get(code + suffix, ""),
                "產業別": (industries or {}).get(code + suffix, ""),
                "合計": foreign + trust + dealer,
                "外資": foreign,
                "投信": trust,
                "自營商": dealer,
            }
        )
    return pd.DataFrame(rows)


def _inst_rank_official():
    """主要來源：證交所 T86 + 櫃買中心。找「今天以前」最近一個有資料的交易日。"""
    store = _inst_store()
    data_tw, day = {}, None
    for day in _recent_weekdays(14, include_today=False):
        data_tw = _fetch_market_day("TW", day, store)
        if data_tw:
            break
    if not data_tw:
        raise RuntimeError("證交所查無最近的三大法人資料")
    otc_ok = True
    try:
        data_otc = _fetch_market_day("OTC", day, store)
    except Exception:
        data_otc, otc_ok = {}, False
    by_code = {(c, ".TW"): v for c, v in data_tw.items()}
    by_code.update({(c, ".TWO"): v for c, v in data_otc.items()})
    df = _inst_rows(by_code, get_tw_names(), _industry_store())
    return df, day.isoformat(), otc_ok, "證交所 T86、櫃買中心"


def _inst_rank_finmind():
    """備援：FinMind 全市場單日三大法人買賣超。"""
    raw, day = pd.DataFrame(), None
    for day in _recent_weekdays(14, include_today=False):
        raw = _finmind_get(
            "TaiwanStockInstitutionalInvestorsBuySell", start_date=day.isoformat(), end_date=day.isoformat()
        )
        if not raw.empty:
            break
    if raw.empty:
        raise RuntimeError("FinMind 查無資料（可能需要在 .streamlit/secrets.toml 設定 FINMIND_TOKEN）")
    raw["stock_id"] = raw["stock_id"].astype(str)
    if raw["name"].isin(["Dealer_self", "Dealer_Hedging"]).any():
        raw = raw[raw["name"] != "Dealer"]  # 已有自營商細項就不重複計入合計
    raw["group"] = raw["name"].map(INSTITUTION_GROUPS)
    raw = raw.dropna(subset=["group"])
    raw["net"] = (raw["buy"].astype(float) - raw["sell"].astype(float)) / 1000  # 股 → 張
    pivot = raw.pivot_table(index="stock_id", columns="group", values="net", aggfunc="sum").fillna(0)
    for col in ("外資", "投信", "自營商"):
        if col not in pivot.columns:
            pivot[col] = 0.0
    names = get_tw_names()
    by_code = {}
    for code, r in pivot.iterrows():
        suffix = ".TWO" if (code + ".TWO") in names and (code + ".TW") not in names else ".TW"
        by_code[(code, suffix)] = (r["外資"], r["投信"], r["自營商"])
    return _inst_rows(by_code, names), day.isoformat(), True, "FinMind（備援）"


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_prev_day_institutional_rank():
    """回傳 (資料表, 日期, 櫃買是否成功, 來源說明, 主要來源錯誤訊息)。主要來源失敗就改用 FinMind。"""
    try:
        return _inst_rank_official() + ("",)
    except Exception as e_main:
        try:
            return _inst_rank_finmind() + (str(e_main),)
        except Exception as e_backup:
            raise RuntimeError(f"主要來源：{e_main}；備援來源（FinMind）：{e_backup}")


def _render_rank_table(df, title, color):
    st.markdown(f"**{title}**")
    shown = df.drop(columns=["產業別"]).copy()  # 產業別資料常為空白，不顯示
    shown.insert(0, "排名", range(1, len(shown) + 1))
    num_cols = ["合計", "外資", "投信", "自營商"]
    styler = shown.style.format({c: "{:+,.0f}" for c in num_cols})
    styler = (getattr(styler, "map", None) or styler.applymap)(_color_net, subset=num_cols)
    st.dataframe(
        styler,
        hide_index=True,
        width="stretch",
        height=35 * (len(shown) + 1) + 3,
        column_config={
            "排名": st.column_config.NumberColumn(width="small"),
            "代號": st.column_config.TextColumn(width="small"),
            "名稱": st.column_config.TextColumn(width="large"),
            "合計": st.column_config.NumberColumn(width="small"),
            "外資": st.column_config.NumberColumn(width="small"),
            "投信": st.column_config.NumberColumn(width="small"),
            "自營商": st.column_config.NumberColumn(width="small"),
        },
    )


def render_prev_day_institutional_rank():
    st.markdown("##### 🏦 前一交易日 三大法人買超／賣超 前 20 名")
    try:
        with st.spinner("⏳ 載入資料中，請稍候…"):
            df, day_str, otc_ok, source, main_err = fetch_prev_day_institutional_rank()
    except Exception as e:
        st.warning(f"⚠️ 目前無法取得法人買賣超資料：{e}")
        return
    if df.empty:
        st.info("該交易日沒有法人買賣超資料。")
        return

    if main_err:
        st.warning(f"⚠️ 證交所／櫃買中心資料暫時無法使用（{main_err}），已改用備援來源 FinMind。")
    day = datetime.date.fromisoformat(day_str)
    weekday = "一二三四五六日"[day.weekday()]
    st.caption(f"📅 資料日期：{day_str}（週{weekday}）　單位：張（1 張 = 1,000 股）　紅色買超、綠色賣超")

    top_buy = df[df["合計"] > 0].nlargest(INST_RANK_TOP_N, "合計")
    top_sell = df[df["合計"] < 0].nsmallest(INST_RANK_TOP_N, "合計")
    # 上下排列、各佔滿寬度，欄位才不會被擠壓而截斷文字
    _render_rank_table(top_buy, "🔴 買超前 20 名", "red")
    _render_rank_table(top_sell, "🟢 賣超前 20 名", "green")

    if not otc_ok:
        st.caption("⚠️ 櫃買中心資料暫時抓取失敗，目前排行只含上市股票，稍後重新整理可補齊。")
    st.caption("📌 「合計」為外資（含外資自營商）＋投信＋自營商；資料來源：" + source + "，僅供參考。")


# ---------------------------------------------------------------------------
# 頁籤四：K線分析（用 K 線型態判斷進出場時機）
# 進場：錘形線、多頭吞噬（含量縮轉折紅K）、早晨之星、拉回買進、突破買進（含頸線）、回測頸線不破、回測大量支撐區
# 出場：上吊線／射擊之星、空頭吞噬、黃昏之星、跌破支撐／均線（含大量區）、高檔爆量長黑、高檔量價背離
# 每個訊號都附「建議進場價」與「停損／出場價」
# 台股慣例：紅K＝收盤高於開盤，黑K＝收盤低於開盤
# ---------------------------------------------------------------------------
KLINE_RANGES = {"近 3 個月": 63, "近 6 個月": 126, "近 1 年": 250}

KLINE_TIPS = {
    "錘形線": "出現在下跌段尾端，代表下方承接力道轉強、止跌回升機會高；隔日收紅或站上錘形線高點再進場較穩。",
    "多頭吞噬": "紅K完全包覆前一根黑K，多方力道扭轉空勢，常是底部翻轉訊號；可分批進場。若發生在量縮打底之後、今日帶量（低量轉折紅K），可靠度更高，可視為試單進場時機。",
    "早晨之星": "長黑、小K、長紅三根組合，空方力竭、多方反攻；第三根長紅確認時，或隔日逢低切入。",
    "拉回買進": "上升趨勢中股價回檔至 10 日線或 20 日線附近且未破線，可趁回檔進場。",
    "突破買進": "紅K帶量穿越前波高點，確認突破有效、成交量放大時順勢買進；若同時帶量突破長期區間／W底頸線，為經典的積極進場點。",
    "上吊線／射擊之星": "出現在波段高點，代表上方賣壓沉重、買盤追價無力，視為警訊、準備減碼。",
    "空頭吞噬": "黑K完全吞沒前一根紅K，多方攻勢被強力蓋過，往往是短線由多轉空的賣出訊號。",
    "黃昏之星": "長紅、小K、長黑三根組合，多頭動能耗盡；長黑確認後應考慮獲利了結或退場。",
    "跌破支撐／均線": "長黑K貫穿重要支撐、均線或下檔大量支撐區，原先的支撐邏輯失效、轉為壓力；必須執行紀律停損或出場，不可抱持僥倖心態。",
    "回測頸線不破": "股價帶量突破區間／W底頸線後，隔天或數日內回測頸線不跌破、量縮，是經典的積極進場點；若收盤跌回頸線之下代表突破失敗，立即停損。",
    "回測大量支撐區": "依分價量統計，股價回測至下檔累積成交量最大的價位區，且量縮止穩、賣壓減輕，屬於低風險切入點；若被長黑有效跌破，支撐失效須停損。",
    "高檔量價背離": "股價一波比一波高，但量能一波比一波小，連續出現三次以上價量背離，上攻動能耗盡；應分批獲利了結或調降持股水位。",
    "高檔爆量長黑": "高檔區突然出現長黑K並伴隨巨量（恐慌出貨或主力倒貨），若隨後跌破短中期均線或上升趨勢，為強烈出場訊號，應果斷停利或停損。",
}


# 頸線／分價量判斷用的參數（想調整標準改這裡）
VOL_PROFILE_WINDOW = 120   # 分價量統計：看近幾個交易日
VOL_PROFILE_BINS = 24      # 分價量統計：價格區間切成幾格
NECK_LOOKBACK = 60         # 頸線／大量區：前 60 日高點
NECK_VOL_MULT = 1.5        # 突破時成交量需 ≥ 月均量（20 日）的幾倍
NECK_MAX_RANGE = 0.45      # 前 60 日高低振幅 ≤ 45% 才視為「區間整理」後的突破
NECK_STOP_PCT = 0.97       # 頸線停損：頸線下 3%
ZONE_BREAK_PCT = 0.99      # 大量區有效跌破／停損：區間下緣再下 1%


def _volume_nodes(H, L, C, V, i, window=VOL_PROFILE_WINDOW, bins=VOL_PROFILE_BINS):
    """分價量統計：用 i 以前（不含 i）近 window 根K線，依典型價把成交量分到價格區間，
    回傳成交量最集中的「大量區」[(下緣, 上緣, 佔比)]，由大到小最多 3 個。"""
    s = max(0, i - window)
    if i - s < 40:
        return []
    lo, hi = float(np.min(L[s:i])), float(np.max(H[s:i]))
    vol = V[s:i]
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo or vol.sum() <= 0:
        return []
    tp = (H[s:i] + L[s:i] + C[s:i]) / 3
    hist, edges = np.histogram(tp, bins=bins, range=(lo, hi), weights=vol)
    mask = hist >= 1.5 * hist.mean()
    total = hist.sum()
    zones, k = [], 0
    while k < bins:
        if mask[k]:
            e = k
            while e + 1 < bins and mask[e + 1]:
                e += 1
            zones.append((float(edges[k]), float(edges[e + 1]), float(hist[k:e + 1].sum() / total)))
            k = e + 1
        else:
            k += 1
    zones.sort(key=lambda z: -z[2])
    return zones[:3]


def detect_kline_signals(df):
    """掃描整段日K，回傳 (含均線的資料表, 訊號清單)。
    訊號：{idx, date, kind(entry/exit), name, detail, close, entry(建議進場價區間或 None), stop(停損／出場價或 None)}。"""
    d = df[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
    if len(d) < 30:
        return d, []
    for w in (5, 10, 20):
        d[f"MA{w}"] = d["Close"].rolling(w).mean()
    O, H, L, C, V = (d[k].to_numpy(float) for k in ("Open", "High", "Low", "Close", "Volume"))
    MA5, MA10, MA20 = (d[f"MA{w}"].to_numpy(float) for w in (5, 10, 20))
    body = np.abs(C - O)
    top, bot = np.maximum(O, C), np.minimum(O, C)
    rng, upper, lower = H - L, H - top, bot - L
    avg_body = pd.Series(body).rolling(20).mean().to_numpy()
    vol5 = pd.Series(V).shift(1).rolling(5).mean().to_numpy()
    vol20 = pd.Series(V).shift(1).rolling(20).mean().to_numpy()  # 月均量（不含當天）

    # 頸線／大量區突破：收盤突破前 60 日高點、量 ≥ 月均量 N 倍、且前 60 日屬區間整理
    hi60 = pd.Series(H).shift(1).rolling(NECK_LOOKBACK).max().to_numpy()
    lo60 = pd.Series(L).shift(1).rolling(NECK_LOOKBACK).min().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        neck_range = (hi60 - lo60) / lo60
        neck_brk = (C > hi60) & (V >= NECK_VOL_MULT * vol20) & (neck_range <= NECK_MAX_RANGE)

    # 價量背離日：收盤創 20 日新高，但量低於月均量 0.9 倍，或能量潮 OBV 沒有同步創高
    obv = np.cumsum(np.sign(np.diff(C, prepend=C[0])) * V)
    div = np.zeros(len(d), dtype=bool)
    for k in range(21, len(d)):
        if C[k] >= C[k - 20:k].max() and np.isfinite(vol20[k]):
            div[k] = bool(V[k] < 0.9 * vol20[k] or obv[k] < obv[k - 20:k].max())

    sigs, last_sig = [], {}

    def add(i, kind, name, extra="", stop=None, entry=None, stop_label=None):
        last_sig[name] = i
        sigs.append(
            {
                "idx": i,
                "date": d.index[i],
                "kind": kind,
                "name": name,
                "detail": KLINE_TIPS[name] + (f"（{extra}）" if extra else ""),
                "close": float(C[i]),
                "entry": None if entry is None else (float(min(entry)), float(max(entry))),
                "stop": None if stop is None else float(stop),
                # 價位的意思：進場訊號＝停損；出場訊號＝跌破此價出場；「跌破支撐／均線」＝反彈站不回此價就出場
                "stop_label": stop_label or ("停損" if kind == "entry" else "跌破出場"),
            }
        )

    def cooled(name, i, gap):
        return i - last_sig.get(name, -999) >= gap

    for i in range(25, len(d)):
        if rng[i] <= 0 or not np.isfinite(MA20[i]) or not np.isfinite(avg_body[i]):
            continue
        ret5 = C[i - 1] / C[i - 6] - 1
        down = ret5 < -0.02 and C[i - 1] < MA10[i - 1]
        up = ret5 > 0.02 and C[i - 1] > MA10[i - 1]
        down_soft, up_soft = C[i - 1] < MA10[i - 1], C[i - 1] > MA10[i - 1]
        near_low = L[i] <= L[i - 10:i].min() * 1.03
        near_high = H[i] >= H[i - 20:i].max() * 0.98
        vr = V[i] / vol5[i] if np.isfinite(vol5[i]) and vol5[i] > 0 else np.nan
        long_lower = lower[i] >= 2 * body[i] and lower[i] >= 0.6 * rng[i] and upper[i] <= 0.2 * rng[i]
        long_upper = upper[i] >= 2 * body[i] and upper[i] >= 0.6 * rng[i] and lower[i] <= 0.2 * rng[i]
        nodes = _volume_nodes(H, L, C, V, i)  # 分價量大量區（只用 i 以前的資料）

        # ---------------- 進場 ----------------
        if long_lower and down and near_low:
            add(i, "entry", "錘形線", "長下影線出現在下跌段低檔", entry=(C[i], H[i]), stop=L[i])
        if (
            C[i - 1] < O[i - 1] and C[i] > O[i] and O[i] <= C[i - 1] and C[i] >= O[i - 1]
            and body[i] > body[i - 1] and down_soft
        ):
            extra = "紅K吞沒前一根黑K"
            if np.isfinite(vol20[i]) and V[i - 5:i].mean() <= 0.7 * vol20[i] and V[i] >= 1.3 * V[i - 1]:
                extra += (
                    f"；前 5 日量縮至月均量 {V[i - 5:i].mean() / vol20[i] * 100:.0f}%、今日帶量轉折，"
                    "屬量縮打底後的低量轉折紅K，可視為試單進場時機"
                )
            add(i, "entry", "多頭吞噬", extra, entry=((O[i] + C[i]) / 2, C[i]), stop=min(L[i], L[i - 1]))
        if (
            C[i - 2] < O[i - 2] and body[i - 2] >= avg_body[i] and C[i - 2] < MA10[i - 2]
            and body[i - 1] <= 0.5 * avg_body[i] and top[i - 1] <= C[i - 2]
            and C[i] > O[i] and body[i] >= avg_body[i] and C[i] > (O[i - 2] + C[i - 2]) / 2
        ):
            add(i, "entry", "早晨之星", "第三根長紅已確認", entry=((O[i] + C[i]) / 2, C[i]), stop=min(L[i - 2:i + 1]))
        uptrend = MA5[i] > MA10[i] > MA20[i] and MA20[i] > MA20[i - 3]
        near10 = L[i] <= MA10[i] * 1.015 and C[i] >= MA10[i] * 0.99
        near20 = L[i] <= MA20[i] * 1.015 and C[i] >= MA20[i]
        if uptrend and (near10 or near20) and C[i] < C[i - 5:i].max():
            ma_ = MA10[i] if near10 else MA20[i]
            add(i, "entry", "拉回買進", "回測 10 日線未破" if near10 else "回測 20 日線未破",
                entry=(ma_, ma_ * 1.01), stop=MA20[i] * 0.99)
        brk_buy = (
            C[i] > O[i] and C[i] > H[i - 20:i].max() and np.isfinite(vr) and vr >= 1.5
        )
        if brk_buy:
            lvl, stop_ = H[i - 20:i].max(), H[i - 20:i].max()
            extra = f"突破近 20 日高點，成交量為 5 日均量 {vr:.1f} 倍"
            if neck_brk[i]:
                lvl, stop_ = hi60[i], hi60[i] * NECK_STOP_PCT
                extra += (
                    f"；同時帶量突破近 {NECK_LOOKBACK} 日區間頸線 {hi60[i]:.2f}"
                    f"（成交量為月均量 {V[i] / vol20[i]:.1f} 倍），為積極進場點，停損設頸線下 3%"
                )
            add(i, "entry", "突破買進", extra, entry=(lvl, lvl * 1.02), stop=stop_)

        # 新增：突破頸線／大量區後，隔天或數日內回測頸線不破（突破當天已由「突破買進」涵蓋，故不重複）
        if not brk_buy:
            for k in (1, 2, 3):
                j = i - k
                if neck_brk[j]:
                    neck = hi60[j]
                    if C[j + 1:i + 1].min() >= neck and L[i] <= neck * 1.02 and V[i] < V[j]:
                        add(
                            i, "entry", "回測頸線不破",
                            f"{k} 個交易日前帶量（月均量 {V[j] / vol20[j]:.1f} 倍）突破頸線 {neck:.2f}，"
                            f"今日回測收盤未破、量縮",
                            entry=(neck, neck * 1.01), stop=neck * NECK_STOP_PCT,
                        )
                        break

        # 新增：分價量統計表支撐區（回測下檔大量區且量縮止穩）
        if nodes and np.isfinite(vr) and vr <= 0.8 and cooled("回測大量支撐區", i, 5):
            for zlo, zhi, share in nodes:
                if (
                    L[i] <= zhi * 1.01 and C[i] >= zlo and L[i] >= zlo * ZONE_BREAK_PCT * 0.99
                    and C[i - 5:i].max() > zhi * 1.02
                ):
                    add(
                        i, "entry", "回測大量支撐區",
                        f"近 {VOL_PROFILE_WINDOW} 日大量區 {zlo:.2f}～{zhi:.2f}（約佔成交量 {share * 100:.0f}%），"
                        f"今日量為 5 日均量 {vr:.1f} 倍、收盤守在區間之上",
                        entry=(zlo, zhi), stop=zlo * ZONE_BREAK_PCT,
                    )
                    break

        # ---------------- 出場（停損／出場價＝跌破此價出場；「跌破支撐／均線」則為反彈站不回此價就出場） ----------------
        if up and near_high and (long_lower or long_upper):
            add(i, "exit", "上吊線／射擊之星", "長下影（上吊線）" if long_lower else "長上影（射擊之星）", stop=L[i])
        if (
            C[i - 1] > O[i - 1] and C[i] < O[i] and O[i] >= C[i - 1] and C[i] <= O[i - 1]
            and body[i] > body[i - 1] and up_soft
        ):
            add(i, "exit", "空頭吞噬", "黑K吞沒前一根紅K", stop=L[i])
        if (
            C[i - 2] > O[i - 2] and body[i - 2] >= avg_body[i] and C[i - 2] > MA10[i - 2]
            and body[i - 1] <= 0.5 * avg_body[i] and bot[i - 1] >= C[i - 2]
            and C[i] < O[i] and body[i] >= avg_body[i] and C[i] < (O[i - 2] + C[i - 2]) / 2
        ):
            add(i, "exit", "黃昏之星", "第三根長黑已確認", stop=min(L[i - 1], L[i]))
        if C[i] < O[i] and body[i] >= avg_body[i]:
            broke, levels = [], []
            if C[i] < MA10[i] and C[i - 1] >= MA10[i - 1]:
                broke.append("跌破 10 日線")
                levels.append(MA10[i])
            if C[i] < L[i - 10:i].min():
                broke.append("跌破近 10 日低點")
                levels.append(L[i - 10:i].min())
            if C[i] < MA5[i] and C[i - 1] >= MA5[i - 1] and np.isfinite(vr) and vr >= 1.5:
                broke.append("爆量跌破 5 日線")
                levels.append(MA5[i])
            # 新增：原先底部的大量支撐區被長黑有效跌破
            for zlo, zhi, share in nodes:
                if C[i - 1] >= zlo and C[i] < zlo * ZONE_BREAK_PCT:
                    broke.append(f"長黑有效跌破下檔大量支撐區 {zlo:.2f}～{zhi:.2f}")
                    levels.append(zlo)
                    break
            # 新增：高檔區出現長黑K＋巨量（恐慌出貨／主力倒貨）
            hv_black = (
                np.isfinite(vr) and vr >= 2 and C[i - 1] > MA20[i - 1]
                and C[i - 1] >= C[max(0, i - 60):i].max() * 0.92
            )
            hv_txt = f"高檔爆量長黑（成交量為 5 日均量 {vr:.1f} 倍，疑似恐慌出貨／主力倒貨）" if hv_black else ""
            if broke:  # 與原本「跌破」訊號重複者合併成同一筆
                add(i, "exit", "跌破支撐／均線", "、".join(([hv_txt] if hv_black else []) + broke), stop=max(levels), stop_label="反彈出場")
            elif hv_black:
                add(i, "exit", "高檔爆量長黑", hv_txt, stop=L[i])

        # 新增：高檔價漲量縮／價量背離連續出現三次以上
        if div[i] and cooled("高檔量價背離", i, 8):
            cnt, last_j = 0, None
            for j in range(i, max(20, i - 40), -1):
                if div[j] and (last_j is None or last_j - j >= 3):
                    cnt += 1
                    last_j = j
            if cnt >= 3:
                add(
                    i, "exit", "高檔量價背離",
                    f"近 40 個交易日已出現 {cnt} 次「創高但量縮／能量潮未同步創高」",
                    stop=MA10[i] if MA10[i] < C[i] else L[i],
                )
    return d, sigs


def _pz(v):
    """建議價格：粗體、放大，方便一眼看到（需搭配 unsafe_allow_html=True）。"""
    return f"<span style='font-size:1.35em;font-weight:800'>{v:.2f}</span>"


def _pz_range(lo, hi):
    return _pz(lo) if abs(hi - lo) <= 0.005 else f"{_pz(lo)}～{_pz(hi)}"


def _describe_last_bar(d):
    o, h, l, c = (float(d[k].iloc[-1]) for k in ("Open", "High", "Low", "Close"))
    body, rng = abs(c - o), h - l
    avg_body = float((d["Close"] - d["Open"]).abs().tail(20).mean())
    if rng <= 0:
        return "一字K（無波動）"
    if body <= 0.1 * rng:
        kind = "十字K（多空拉鋸）"
    elif c > o:
        kind = "長紅K（買盤強勁）" if body >= avg_body else "紅K"
    else:
        kind = "長黑K（賣壓沉重）" if body >= avg_body else "黑K"
    notes = []
    if (min(o, c) - l) >= 2 * max(body, rng * 0.05):
        notes.append("長下影線")
    if (h - max(o, c)) >= 2 * max(body, rng * 0.05):
        notes.append("長上影線")
    return kind + ("，" + "、".join(notes) if notes else "")


# ---------------------------------------------------------------------------
# 下個交易日預測：把 K 線訊號、均線、MACD、RSI、量價、乖離率加權成 -100～+100 的分數
# （規則式評分，不是保證漲跌；並用這檔股票過去一年的資料回測同樣判斷的隔日表現當參考）
# ---------------------------------------------------------------------------
PRED_THRESHOLD = 25


def _prepare_indicators(d):
    c = d["Close"]
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    d["RSI"] = (100 - 100 / (1 + rs)).fillna(100)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    d["MACD_H"] = macd - macd.ewm(span=9, adjust=False).mean()
    d["VR"] = d["Volume"] / d["Volume"].shift(1).rolling(5).mean()
    d["OBV"] = (np.sign(c.diff()).fillna(0) * d["Volume"]).cumsum()  # 能量潮：價漲加量、價跌減量
    return d


def _pv_score(A, i):
    """量價關係評分（最多 ±20）：價量配合、價量背離（OBV）、爆量長影線。回傳 (分數, [(加減分, 理由)])。"""
    vr = A["VR"][i]
    if i < 21 or not np.isfinite(vr):
        return 0, []
    out = []
    c, pc = A["Close"][i], A["Close"][i - 1]
    o, h, l = A["Open"][i], A["High"][i], A["Low"][i]
    rng = h - l
    prev20_hi, prev20_lo = A["Close"][i - 20:i].max(), A["Close"][i - 20:i].min()
    near_high, near_low = c >= prev20_hi * 0.98, c <= prev20_lo * 1.02

    # 價量配合
    if c > pc and vr >= 1.3:
        out.append((10 if vr >= 2 else 8, f"價漲量增（成交量為 5 日均量 {vr:.1f} 倍），買盤積極、價量配合"))
    elif c < pc and vr >= 1.3:
        out.append((-10 if vr >= 2 else -8, f"價跌量增（成交量為 5 日均量 {vr:.1f} 倍），賣壓沉重"))
    elif c > pc and vr <= 0.7 and near_high:
        out.append((-6, f"價漲量縮（僅 5 日均量 {vr:.1f} 倍）且逼近高點，追價意願不足，留意背離"))
    elif c < pc and vr <= 0.7 and c > A["MA20"][i]:
        out.append((3, f"價跌量縮（僅 5 日均量 {vr:.1f} 倍）且仍在 20 日線上，賣壓輕、屬健康回檔"))
    elif c < pc and vr <= 0.7 and near_low:
        out.append((4, f"價跌量縮（僅 5 日均量 {vr:.1f} 倍）且貼近 20 日低點，賣壓竭盡、籌碼趨穩，進入築底觀察期（搭配 W 底，等量增長紅K確認再進場）"))
    elif c > pc and vr <= 0.7:
        out.append((-3, f"價漲量縮（僅 5 日均量 {vr:.1f} 倍），虛漲／動能衰退，追價意願不足，不宜貿然追高"))

    # 價量背離：價創 20 日新高／新低，但 OBV 沒跟上
    obv, obv_win = A["OBV"][i], A["OBV"][i - 20:i]
    if c >= prev20_hi and obv < obv_win.max():
        out.append((-6, "股價創 20 日新高，但能量潮（OBV）未同步創高，價量背離，漲勢恐難持續"))
    elif c <= prev20_lo and obv > obv_win.min():
        out.append((6, "股價創 20 日新低，但能量潮（OBV）未同步創低，賣壓遞減，有止跌跡象"))

    # 爆量長影線
    if rng > 0 and vr >= 2:
        upper, lower = h - max(o, c), min(o, c) - l
        if upper >= 0.5 * rng and near_high:
            out.append((-6, f"高檔爆量（{vr:.1f} 倍）收長上影線，上檔賣壓大、疑似出貨"))
        elif lower >= 0.5 * rng and near_low:
            out.append((6, f"低檔爆量（{vr:.1f} 倍）收長下影線，下檔承接力強、疑似止跌"))

    total = max(-20, min(20, sum(p for p, _ in out)))
    return total, out


def _inst_score(table, vol_lots_by_date=None):
    """三大法人買賣超評分（最多 ±20）：外資／投信連買連賣、當日買賣超佔成交量比、近 5 日方向、外資投信同向。"""
    if table is None or table.empty or len(table) < 3:
        return 0, []
    out = []

    def streak_pts(n, full, mid, low):
        return full if n >= 5 else mid if n >= 3 else low

    f_dir, f_n = _streak(table["外資"])
    if f_dir and f_n >= 2:
        pts = f_dir * streak_pts(f_n, 8, 6, 3)
        out.append((pts, f"外資連續 {f_n} 日{'買超' if f_dir > 0 else '賣超'}"))
    t_dir, t_n = _streak(table["投信"])
    if t_dir and t_n >= 2:
        pts = t_dir * (7 if t_n >= 3 else 4)
        out.append((pts, f"投信連續 {t_n} 日{'買超' if t_dir > 0 else '賣超'}（投信作帳力道常帶動中小型股）"))

    total = table["三大法人合計"]
    last = float(total.iloc[-1])
    lots = (vol_lots_by_date or {}).get(table.index[-1])
    if lots and lots > 0:
        ratio = last / lots * 100
        if abs(ratio) >= 5:
            out.append((6 if ratio > 0 else -6, f"{table.index[-1]} 三大法人{'買超' if ratio > 0 else '賣超'}佔成交量 {abs(ratio):.1f}%，籌碼明顯{'集中' if ratio > 0 else '鬆動'}"))
        elif abs(ratio) >= 2:
            out.append((3 if ratio > 0 else -3, f"{table.index[-1]} 三大法人{'買超' if ratio > 0 else '賣超'}佔成交量 {abs(ratio):.1f}%"))
    s5 = float(total.tail(5).sum())
    if s5:
        out.append((3 if s5 > 0 else -3, f"近 5 日三大法人合計{'買超' if s5 > 0 else '賣超'} {abs(s5):,.0f} 張"))
    f_last, t_last = float(table["外資"].iloc[-1]), float(table["投信"].iloc[-1])
    if f_last > 0 and t_last > 0:
        out.append((4, "最新一日外資與投信同步買超，土洋合力"))
    elif f_last < 0 and t_last < 0:
        out.append((-4, "最新一日外資與投信同步賣超，土洋同步倒貨"))

    return max(-20, min(20, sum(p for p, _ in out))), out


def _score_at(A, by, i, inst=None):
    """第 i 根K線收盤後的評分。回傳 (分數, [(加減分, 理由)])。inst 為 (法人分數, 理由)，只在最新一天使用。"""
    reasons = []
    c, m5, m10, m20 = A["Close"][i], A["MA5"][i], A["MA10"][i], A["MA20"][i]

    # 1) K 線型態（最近 3 根，越新權重越高，合計最多 ±30）
    k_sum = 0
    for lag, w, label in ((0, 20, "今天"), (1, 12, "昨天"), (2, 6, "前天")):
        for s in by.get(i - lag, []):
            pts = w if s["kind"] == "entry" else -w
            k_sum += pts
            kind = "進場" if s["kind"] == "entry" else "出場"
            reasons.append((pts, f"{label}出現{kind}訊號「{s['name']}」"))
    k_total = max(-30, min(30, k_sum))

    total = k_total
    # 2) 均線排列與 20 日線方向
    if c > m5 > m10 > m20:
        pts, txt = 15, "收盤 > 5日線 > 10日線 > 20日線，多頭排列"
    elif c < m5 < m10 < m20:
        pts, txt = -15, "收盤 < 5日線 < 10日線 < 20日線，空頭排列"
    elif c > m20:
        pts, txt = 5, "收盤站在 20 日線之上"
    else:
        pts, txt = -5, "收盤在 20 日線之下"
    total += pts
    reasons.append((pts, txt))
    up20 = m20 > A["MA20"][i - 3]
    total += 5 if up20 else -5
    reasons.append((5 if up20 else -5, "20 日線向上，中期趨勢偏多" if up20 else "20 日線向下，中期趨勢偏空"))

    # 3) MACD 柱狀體
    h, hp = A["MACD_H"][i], A["MACD_H"][i - 1]
    if h > 0:
        pts = 8 + (4 if h > hp else 0)
        txt = "MACD 柱狀體為正" + ("且持續擴大，多方動能增強" if h > hp else "，但動能略為收斂")
    else:
        pts = -(8 + (4 if h < hp else 0))
        txt = "MACD 柱狀體為負" + ("且持續擴大，空方動能增強" if h < hp else "，但跌勢略為收斂")
    total += pts
    reasons.append((pts, txt))

    # 4) RSI
    r = A["RSI"][i]
    if r < 30:
        pts, txt = 8, f"RSI {r:.0f} 低於 30，超賣區有反彈機會"
    elif r >= 75:
        pts, txt = -10, f"RSI {r:.0f} 高於 75，嚴重過熱，回檔風險高"
    elif r >= 70:
        pts, txt = -6, f"RSI {r:.0f} 進入過熱區（70 以上）"
    else:
        pts = int(round((r - 50) / 4))
        txt = f"RSI {r:.0f}，{'偏強' if pts > 0 else '偏弱'}"
    if pts:
        total += pts
        reasons.append((pts, txt))

    # 5) 量價關係
    pv_pts, pv_reasons = _pv_score(A, i)
    total += pv_pts
    reasons.extend(pv_reasons)

    # 6) 與 20 日線的乖離
    bias = (c / m20 - 1) * 100
    if bias > 10:
        total -= 8
        reasons.append((-8, f"股價高於 20 日線 {bias:.1f}%，乖離過大，容易拉回"))
    elif bias < -10:
        total += 6
        reasons.append((6, f"股價低於 20 日線 {abs(bias):.1f}%，超跌乖離，有反彈空間"))

    # 7) 三大法人買賣超（只有近 10 個交易日資料，因此只用在最新一天的預測）
    if inst and inst[1]:
        total += inst[0]
        reasons.extend(inst[1])

    return int(max(-100, min(100, total))), reasons


def _prediction_pack(d, sigs):
    d = _prepare_indicators(d.copy())
    A = {k: d[k].to_numpy(float) for k in ("Open", "Close", "High", "Low", "MA5", "MA10", "MA20", "RSI", "MACD_H", "VR", "OBV")}
    by = {}
    for s in sigs:
        by.setdefault(s["idx"], []).append(s)
    return d, A, by


def _render_prediction(d, sigs, inst_table=None):
    d, A, by = _prediction_pack(d, sigs)
    n = len(d)
    i = n - 1
    vol_lots = {ix.strftime("%Y-%m-%d"): float(v) / 1000 for ix, v in d["Volume"].items()}
    inst = _inst_score(inst_table, vol_lots)
    score, reasons = _score_at(A, by, i, inst)

    # 下一個交易日（只跳過週末，不含國定假日）
    nd = d.index[-1] + datetime.timedelta(days=1)
    while nd.weekday() >= 5:
        nd += datetime.timedelta(days=1)
    nd_txt = f"{nd.strftime('%Y-%m-%d')}（週{'一二三四五六日'[nd.weekday()]}）"

    st.markdown("##### 🔮 下個交易日預測")
    strength = "強烈" if abs(score) >= 50 else "偏"
    if score >= PRED_THRESHOLD:
        st.success(f"🔼 {nd_txt}：{strength}多，**適合進場**（已持有者可續抱）")
        verdict = "bull"
    elif score <= -PRED_THRESHOLD:
        st.error(f"🔽 {nd_txt}：{strength}空，**適合出場／減碼**（空手者不宜進場）")
        verdict = "bear"
    else:
        st.info(f"⚪ {nd_txt}：多空訊號不明確，**建議觀望**")
        verdict = "neutral"

    close = float(A["Close"][i])
    sup = [v for v in (A["MA5"][i], A["MA10"][i], A["MA20"][i], A["Low"][i - 10:i].min()) if v < close]
    res = [v for v in (A["MA5"][i], A["MA10"][i], A["MA20"][i], A["High"][i - 20:i].max()) if v > close]
    # 已跌破所有均線與近期低點（或已突破所有壓力）時，下方沒有支撐（上方沒有壓力），改用今日最低（最高）價當參考
    support = max(sup) if sup else float(A["Low"][i])
    resist = min(res) if res else float(A["High"][i])
    m1, m2, m3 = st.columns(3)
    m1.metric("預測分數（-100～+100）", f"{score:+d}")
    m2.metric(
        "下方支撐（跌破轉弱）", f"{support:.2f}",
        help=None if sup else "收盤已跌破所有均線與近 10 日低點，下方暫無支撐，改以今日最低價當參考。",
    )
    m3.metric(
        "上方壓力（突破轉強）", f"{resist:.2f}",
        help=None if res else "收盤已站上所有均線與近 20 日高點，上方暫無壓力，改以今日最高價當參考。",
    )
    st.progress((score + 100) / 200)
    st.caption(f"分數 ≥ +{PRED_THRESHOLD} 偏多、≤ -{PRED_THRESHOLD} 偏空，其餘觀望。")

    # 因子分項
    pv_pts, _ = _pv_score(A, i)
    k_pts = score - pv_pts - inst[0]
    f1, f2, f3 = st.columns(3)
    f1.metric("技術面（K線／均線／指標）", f"{k_pts:+d}")
    f2.metric("量價關係", f"{pv_pts:+d}")
    f3.metric("三大法人籌碼", f"{inst[0]:+d}" if inst[1] else "無資料")
    if inst_table is not None and not inst_table.empty:
        if inst_table.index[-1] != d.index[-1].strftime("%Y-%m-%d"):
            st.caption(f"ℹ️ 法人資料最新為 {inst_table.index[-1]}（當日資料可能尚未公布），籌碼因子以此日為準。")
    else:
        st.caption("ℹ️ 未取得三大法人資料（非台股或來源暫時無法連線），本次預測只含技術面與量價。")

    # 隔日進出場參考（用近 14 日平均真實波幅 ATR 估算當日合理波動）
    H_, L_, C_ = A["High"], A["Low"], A["Close"]
    tr = np.maximum(H_[1:] - L_[1:], np.maximum(np.abs(H_[1:] - C_[:-1]), np.abs(L_[1:] - C_[:-1])))
    atr = float(tr[-14:].mean())
    st.markdown("**📌 隔日進出場參考**")
    if verdict == "bull":
        st.markdown(
            f"- 📈 **進場**：開盤後回測 {_pz_range(close - 0.5 * atr, close)} 分批買進；若開盤直接跳空超過 {_pz(close + atr)} 不追價。\n"
            f"- 🛡️ **停損**：跌破 {_pz(support)} 出場。\n"
            f"- 🎯 **第一目標／壓力**：{_pz(resist)} 附近，量縮不過可先減碼。",
            unsafe_allow_html=True,
        )
    elif verdict == "bear":
        st.markdown(
            f"- 📉 **出場／減碼**：開盤逢反彈 {_pz_range(close, close + 0.5 * atr)} 先賣；若開盤直接跌破 {_pz(close - atr)} 不宜再凹單。\n"
            f"- 🛡️ **全數出場**：跌破 {_pz(support)}。\n"
            f"- 🔄 **空手者**：站回 {_pz(resist)} 且量能放大，才重新考慮進場。",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"- 👀 **進場條件**：收盤站上 {_pz(resist)} 且成交量放大（≥ 5 日均量 1.5 倍）再進場。\n"
            f"- 🚪 **出場條件**：收盤跌破 {_pz(support)} 就減碼／出場。\n"
            f"- ⏳ 區間內（{support:.2f}～{resist:.2f}）不動作，等方向。",
            unsafe_allow_html=True,
        )

    # 分價量統計：近半年成交量最集中的價位區（下方的大量區＝支撐）
    nodes_now = _volume_nodes(A["High"], A["Low"], A["Close"], d["Volume"].to_numpy(float), n)
    below = [z for z in nodes_now if z[0] < close]
    if below:
        zlo, zhi, share = max(below, key=lambda z: z[1])
        where = "股價目前就在此區內" if zlo <= close <= zhi else f"在股價下方 {(close / zhi - 1) * 100:.1f}%"
        st.markdown(
            f"- 📊 **下檔大量支撐區 {zlo:.2f}～{zhi:.2f}**（約佔近 {VOL_PROFILE_WINDOW} 日成交量 {share * 100:.0f}%，{where}）："
            f"回測此區且量縮止穩，可在 {_pz_range(zlo, zhi)} 低風險切入；"
            f"收盤跌破 {_pz(zlo * ZONE_BREAK_PCT)}（被長黑有效跌破）代表支撐失效，必須停損。",
            unsafe_allow_html=True,
        )

    # 理由（依影響力大小排序）
    st.markdown("**判斷理由**")
    for pts, txt in sorted(reasons, key=lambda x: -abs(x[0])):
        st.markdown(f"- {'🔴 ➕' if pts > 0 else '🟢 ➖'} `{pts:+d}`　{txt}")

    # 回測：這檔股票過去同樣判斷，隔天實際表現
    stats = {"bull": [], "bear": []}
    for k in range(30, n - 1):
        sc, _ = _score_at(A, by, k)
        nxt = A["Close"][k + 1] / A["Close"][k] - 1
        if sc >= PRED_THRESHOLD:
            stats["bull"].append(nxt)
        elif sc <= -PRED_THRESHOLD:
            stats["bear"].append(nxt)
    if verdict in stats:
        arr = np.array(stats[verdict])
        if len(arr) >= 5:
            if verdict == "bull":
                hit, word = (arr > 0).mean() * 100, "上漲"
            else:
                hit, word = (arr < 0).mean() * 100, "下跌"
            st.caption(
                f"📊 歷史參考：這檔股票過去約一年，出現同樣判斷 {len(arr)} 次，隔天{word}的比例 {hit:.0f}%，"
                f"隔日平均漲跌 {arr.mean() * 100:+.2f}%。"
            )
        else:
            st.caption("📊 歷史參考：過去一年同樣判斷的次數太少（不足 5 次），無法回測參考。")
    st.caption("（回測只含技術面與量價因子；法人資料只取得近 10 個交易日，無法回測整年。）")
    if inst_table is not None and not inst_table.empty:
        with st.expander("🏦 三大法人近期買賣超明細（張）"):
            st.dataframe(inst_table.sort_index(ascending=False), width="stretch")
    st.caption(
        "⚠️ 這是依技術指標加權評分的「規則式判斷」，不是保證漲跌的預測；"
        "隔日走勢還受消息面、大盤與籌碼影響。回測為過去資料，不代表未來，請務必搭配停損。"
    )


def _kline_start():
    """按下「開始分析」（或在輸入框按 Enter）：記下要分析的對象，之後每次重繪都用這份紀錄。"""
    kw = st.session_state.get("kline_input", "").strip()
    default_symbol = st.session_state.get("trigger_search", "")
    if default_symbol.startswith("FAILED:"):
        default_symbol = ""
    st.session_state.kline_req = (kw, default_symbol)


def render_kline_analysis():
    st.markdown("##### 📐 K線分析：用型態判斷進出場時機")
    mobile_ = is_mobile()
    c1, c2 = (st.container(), st.container()) if mobile_ else st.columns([3, 1], vertical_alignment="bottom")
    with c1:
        st.text_input(
            "股票代號或中文名稱（留白＝使用「個股分析」目前查詢的股票）",
            key="kline_input",
            on_change=_kline_start,  # 按 Enter 也會開始分析
        )
    with c2:
        st.button("📐 開始分析", type="primary", key="kline_btn", on_click=_kline_start,
                  width="stretch" if mobile_ else "content")
    range_label = st.radio("圖表區間", list(KLINE_RANGES), index=1, horizontal=True, key="kline_range")

    req = st.session_state.get("kline_req")
    if req is None:
        st.info("請輸入股票代號或名稱後按「開始分析」；留白則分析「個股分析」目前查詢的股票。")
        return
    kw, default_symbol = req
    target = kw or default_symbol
    if not target:
        st.info("請輸入股票代號或名稱，或先在「個股分析」查詢一檔股票。")
        return

    with st.spinner("⏳ 載入資料中，請稍候…"):
        symbol = get_ticker_from_name(target) if kw else default_symbol
        hist, valid = pd.DataFrame(), None
        for cand in [symbol] + ([symbol[:-3] + ".TWO"] if symbol.endswith(".TW") else []):
            try:
                hist = fetch_history(cand, "1y")
            except Exception:
                continue
            if not hist.empty:
                valid = cand
                break
        if valid is None:
            st.error(f"❌ 找不到「{target}」的股價資料。")
            return
        name = resolve_display_name(valid, kw)
        d, sigs = detect_kline_signals(hist)

    if len(d) < 30:
        st.warning("歷史資料不足 30 個交易日，無法判斷 K 線型態。")
        return

    n, window = len(d), KLINE_RANGES[range_label]
    last, prev = d.iloc[-1], d.iloc[-2]
    chg = float(last["Close"] - prev["Close"])
    ma5, ma10, ma20 = (float(last[f"MA{w}"]) for w in (5, 10, 20))
    close = float(last["Close"])
    if close > ma5 > ma10 > ma20:
        trend = "多頭排列"
    elif close < ma5 < ma10 < ma20:
        trend = "空頭排列"
    else:
        trend = "盤整／趨勢未明"

    st.subheader(f"{name} ({valid})")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("最新收盤", f"{close:.2f}", f"{chg:+.2f} ({chg / float(prev['Close']) * 100:+.2f}%)", delta_color="inverse")
    m2.metric("5 / 10 日線", f"{ma5:.2f} / {ma10:.2f}")
    m3.metric("20 日線", f"{ma20:.2f}")
    m4.metric("均線型態", trend)
    st.caption(f"📌 最新一根K線（{d.index[-1].strftime('%Y-%m-%d')}）：{_describe_last_bar(d)}")

    # --- 目前判斷（看最近 3 個交易日的訊號）---
    recent = [s for s in sigs if s["idx"] >= n - 3]
    buys = [s for s in recent if s["kind"] == "entry"]
    sells = [s for s in recent if s["kind"] == "exit"]

    def _names(lst):
        return "、".join(f"{s['name']}（{s['date'].strftime('%m/%d')}）" for s in lst)

    def _price_txt(s, html=False):
        """建議進場價與停損／出場價的文字（沒有就回傳空字串）。html=True 時數字用粗體大字。"""
        num = _pz if html else (lambda v: f"{v:.2f}")
        rng_ = _pz_range if html else (lambda lo, hi: f"{lo:.2f}" + (f"～{hi:.2f}" if abs(hi - lo) > 0.005 else ""))
        parts = []
        if s.get("entry"):
            parts.append("建議進場 " + rng_(*s["entry"]))
        if s.get("stop"):
            parts.append(f"{s.get('stop_label') or '停損'} " + num(s["stop"]))
        return "　".join(parts)

    def _price_lines(lst):
        lines = [
            f"- 💰 **{s['name']}**（{s['date'].strftime('%m/%d')}）　{_price_txt(s, html=True)}"
            for s in lst if _price_txt(s)
        ]
        return "\n".join(lines)

    if buys and sells:
        st.warning(
            f"⚖️ 最近 3 個交易日多空訊號並存：進場 {_names(buys)}；出場 {_names(sells)}。"
            "方向不明，建議觀望，等下一根K線確認。"
        )
    elif sells:
        st.error(
            f"⚠️ 最近 3 個交易日出現出場／減碼訊號：{_names(sells)}。"
            "持股者可考慮減碼或獲利了結，並嚴守停損紀律。"
        )
    elif buys:
        st.success(
            f"🔔 最近 3 個交易日出現進場訊號：{_names(buys)}。建議等確認後分批進場。"
        )
    else:
        tips = {
            "多頭排列": "目前沒有新的K線訊號，趨勢仍偏多：持股續抱，等拉回 10 日線／20 日線不破再找進場點。",
            "空頭排列": "目前沒有新的K線訊號，趨勢偏空：不宜貿然進場，等待止跌型態（錘形線、多頭吞噬、早晨之星）。",
            "盤整／趨勢未明": "目前沒有明確的K線訊號，趨勢未明：先觀望，等突破或跌破區間再行動。",
        }
        st.info("💡 " + tips[trend])
    _pl = _price_lines(buys + sells)
    if _pl:
        st.markdown(_pl, unsafe_allow_html=True)

    inst_table = None
    if valid.endswith((".TW", ".TWO")):
        with st.spinner("⏳ 載入法人資料中，請稍候…"):
            try:
                inst_table = fetch_institutional_official(valid, days=10)
            except Exception:
                try:
                    inst_table = summarize_institutional(fetch_institutional_raw(valid.split(".")[0]), days=10)
                except Exception:
                    inst_table = None
    _render_prediction(d, sigs, inst_table)

    # --- K 線圖 + 訊號標記 ---
    start = max(0, n - window)
    shown = d.iloc[start:]
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=shown.index, open=shown["Open"], high=shown["High"], low=shown["Low"], close=shown["Close"],
            name="K線", increasing_line_color="red", decreasing_line_color="green",
        )
    )
    for w, color in ((5, "#ffd166"), (10, "#4cc9f0"), (20, "#c77dff")):
        fig.add_trace(
            go.Scatter(x=shown.index, y=shown[f"MA{w}"], mode="lines", name=f"{w}日線", line=dict(width=1.2, color=color))
        )
    for kind, label, symbol_, color, pos, y_mult, col in (
        ("entry", "進場訊號", "triangle-up", "#ff9f1c", "bottom center", 0.985, "Low"),
        ("exit", "出場訊號", "triangle-down", "#2ec4b6", "top center", 1.015, "High"),
    ):
        pts = {}
        for s in sigs:
            if s["kind"] == kind and s["idx"] >= start:
                pts.setdefault(s["idx"], []).append(s)
        if pts:
            xs = [d.index[i] for i in pts]
            ys = [float(d[col].iloc[i]) * y_mult for i in pts]
            fig.add_trace(
                go.Scatter(
                    x=xs, y=ys, mode="markers+text", name=label,
                    marker=dict(symbol=symbol_, size=12, color=color),
                    text=["/".join(s["name"] for s in v) for v in pts.values()],
                    textposition=pos, textfont=dict(size=10, color=color),
                    hovertext=[
                        "<br>".join(
                            f"{s['name']}：{s['detail']}" + (f"<br>💰 {_price_txt(s)}" if _price_txt(s) else "")
                            for s in v
                        )
                        for v in pts.values()
                    ],
                    hoverinfo="x+text",
                )
            )
    fig.update_layout(
        title=f"{name} K 線與進出場訊號", template="plotly_dark",
        xaxis_rangeslider_visible=False, xaxis_title="日期", yaxis_title="價格",
    )
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    show_chart(fig)

    # --- 訊號明細 ---
    shown_sigs = [s for s in sigs if s["idx"] >= start]
    st.markdown(f"**📋 {range_label}訊號明細**（共 {len(shown_sigs)} 筆，新 → 舊）")
    if shown_sigs:
        css = (
            "<style>.kl-wrap{max-height:520px;overflow:auto;border:1px solid rgba(128,128,128,.35);border-radius:8px}"
            ".kl-wrap table{border-collapse:collapse;width:100%;font-size:0.9rem}"
            ".kl-wrap th{text-align:left;padding:8px 10px;background:rgba(128,128,128,.18);white-space:nowrap}"
            ".kl-wrap td{padding:8px 10px;border-top:1px solid rgba(128,128,128,.25);vertical-align:top}"
            ".kl-wrap td.n{text-align:right;white-space:nowrap}"
            ".kl-wrap td.p{text-align:right;white-space:nowrap;font-weight:800;font-size:1.15rem}"
            ".kl-wrap td.t{min-width:280px;line-height:1.5}</style>"
        )
        rows_html = []
        for s_ in reversed(shown_sigs):
            entry = s_["kind"] == "entry"
            color = "#ff9f1c" if entry else "#2ec4b6"
            stop = "—" if s_["stop"] is None else (
                f"{s_['stop']:.2f}<div style='font-size:.7rem;font-weight:400;opacity:.7'>{s_['stop_label']}</div>"
            )
            if s_.get("entry"):
                e_lo, e_hi = s_["entry"]
                entry_txt = f"{e_lo:.2f}" if abs(e_hi - e_lo) <= 0.005 else f"{e_lo:.2f}～{e_hi:.2f}"
            else:
                entry_txt = "—"
            rows_html.append(
                f"<tr><td style='white-space:nowrap'>{s_['date'].strftime('%Y-%m-%d')}</td>"
                f"<td style='white-space:nowrap;color:{color}'>{'📈 進場' if entry else '📉 出場'}</td>"
                f"<td style='white-space:nowrap'>{html_escape(s_['name'])}</td>"
                f"<td class='n'>{s_['close']:.2f}</td><td class='p'>{entry_txt}</td><td class='p'>{stop}</td>"
                f"<td class='t'>{html_escape(s_['detail'])}</td></tr>"
            )
        head = "".join(f"<th>{h}</th>" for h in ("日期", "訊號", "型態", "收盤價", "建議進場價", "停損／出場價", "操作解讀"))
        st.markdown(
            css + f"<div class='kl-wrap'><table><thead><tr>{head}</tr></thead><tbody>{''.join(rows_html)}</tbody></table></div>",
            unsafe_allow_html=True,
        )
    else:
        st.caption("此區間內沒有偵測到明確的K線訊號。")

    with st.expander("📖 型態說明與判斷條件"):
        st.markdown(
            """
**進場時機**

| 型態 | 判斷條件（本頁的程式規則） | 操作 |
|---|---|---|
| 錘形線 | 下影線 ≥ 實體 2 倍且佔全日振幅 6 成以上、上影線很短；出現在下跌後的近期低點 | 下跌段尾端，止跌回升機會高 |
| 多頭吞噬 | 前一天黑K、當天紅K實體更大並完全包覆；前一天收在 10 日線下。若前 5 日均量 ≤ 月均量 7 成、今日量 ≥ 前一日 1.3 倍，標註為「量縮打底後的低量轉折紅K」 | 底部翻轉訊號；量縮轉折時可試單 |
| 早晨之星 | 長黑 → 小K（實體不高於第一根收盤）→ 長紅，且收在第一根實體中點之上 | 第三根長紅確認時或隔日逢低切入 |
| 拉回買進 | 5 > 10 > 20 日線多頭排列、20 日線上揚；回檔觸及 10 或 20 日線附近但收盤未破線 | 上升趨勢中回檔進場 |
| 突破買進 | 紅K收盤突破近 20 日最高價，且成交量 ≥ 5 日均量 1.5 倍；若同時收盤突破前 60 日高點（頸線）、量 ≥ 月均量 1.5 倍、前 60 日振幅 ≤ 45%，會標註為突破頸線 | 突破有效、量增時順勢買進 |
| 回測頸線不破 | 近 3 日內曾帶量突破頸線（條件同上）；今日回測頸線（最低價在頸線上 2% 內）、收盤未跌破頸線、量縮 | 突破後回測不破的積極進場點 |
| 回測大量支撐區 | 分價量統計（近 120 日，依成交量分 24 格價位）找出成交量最集中的價位區；股價自上方回測到該區、收盤守住區間下緣、量 ≤ 5 日均量 0.8 倍 | 量縮止穩的低風險切入點 |

**出場時機**

| 型態 | 判斷條件（本頁的程式規則） | 操作 |
|---|---|---|
| 上吊線／射擊之星 | 長下影（上吊線）或長上影（射擊之星），出現在上漲後的近期高點 | 賣壓沉重，準備減碼 |
| 空頭吞噬 | 前一天紅K、當天黑K實體更大並完全吞沒；前一天收在 10 日線上 | 短線由多轉空 |
| 黃昏之星 | 長紅 → 小K（實體不低於第一根收盤）→ 長黑，且收在第一根實體中點之下 | 獲利了結或退場 |
| 跌破支撐／均線 | 長黑K跌破 10 日線、近 10 日低點、爆量跌破 5 日線，或收盤有效跌破下檔大量支撐區（低於區間下緣 1%） | 紀律停損或出場 |
| 高檔爆量長黑 | 長黑K、量 ≥ 5 日均量 2 倍，前一日收在 20 日線上且距近 60 日高點 8% 內（若同時跌破均線或支撐，併入上一列） | 恐慌出貨／主力倒貨，果斷停利停損 |
| 高檔量價背離 | 近 40 日內至少 3 次「收盤創 20 日新高，但量 < 月均量 0.9 倍或 OBV 未同步創高」（兩次間隔 ≥ 3 根K） | 上攻動能耗盡，分批獲利了結／降低持股 |

「長紅／長黑」＝實體大於近 20 日平均實體。

**建議進場價／停損價怎麼算**

| 型態 | 建議進場價 | 停損／出場價 |
|---|---|---|
| 錘形線 | 收盤價～錘形線最高價（站上高點再進較穩） | 錘形線最低價 |
| 多頭吞噬／早晨之星 | 紅K實體中點～收盤價（隔日逢低） | 型態最低價 |
| 拉回買進 | 被回測的均線～均線 +1% | 20 日線下 1% |
| 突破買進 | 突破價～突破價 +2%（過高不追，等回測） | 突破價；突破頸線者為頸線下 3% |
| 回測頸線不破 | 頸線～頸線 +1% | 頸線下 3% |
| 回測大量支撐區 | 大量區上下緣之間 | 大量區下緣再下 1% |
| 上吊線／射擊之星、空頭吞噬 | — | 訊號K最低價（跌破即出場） |
| 黃昏之星、高檔爆量長黑 | — | 長黑K最低價（跌破即出場） |
| 跌破支撐／均線 | — | 被跌破的價位（反彈站不回就出場） |
| 高檔量價背離 | — | 10 日線（跌破即分批出場） |

**下個交易日預測的評分因子（-100～+100）**

| 因子 | 內容 | 最大影響 |
|---|---|---|
| K線型態 | 近 3 日進出場訊號，越新權重越高 | ±30 |
| 均線／MACD／RSI／乖離 | 趨勢排列、動能、過熱或超跌 | 約 ±40 |
| 量價關係 | 價漲量增、價跌量增、價漲量縮（高檔背離／虛漲）、價跌量縮（健康回檔／低檔築底）、OBV 背離、高低檔爆量長影線 | ±20 |
| 三大法人 | 外資／投信連買連賣、買賣超佔成交量比、近 5 日方向、土洋同向 | ±20 |
            """
        )
    st.caption("⚠️ K線型態只是技術面參考，容易有假訊號，請搭配成交量、大盤與基本面，並設好停損；本頁不構成投資建議。")


# ---------------------------------------------------------------------------
# 主畫面：輸入區
# ---------------------------------------------------------------------------
tab_stock, tab_kline, tab_rank, tab_inst = st.tabs(
    ["🔍 個股分析", "📐 K線分析", "🔥 即時市場排行", "🏦 前一交易日買超賣超"]
)

# K線分析頁籤：輸入欄位不需連線，先渲染，打開就看得到（不必等其他頁籤載入完）
with tab_kline:
    render_kline_analysis()

with tab_stock:
    mobile = is_mobile()
    if mobile:
        # 手機：輸入框與按鈕由上而下全寬排列
        in_col = btn_col = st.container()
    else:
        in_col, btn_col, _spacer = st.columns([3, 1.5, 5], vertical_alignment="bottom")

    with in_col:
        st.text_input("請輸入股票代號或中文名稱：", key="input_val")
    with btn_col:
        st.button(
            "🔍 開始搜尋並分析",
            type="primary",
            on_click=handle_search_button_click,
            width="stretch" if mobile else "content",
        )

    time_period = TIME_PERIOD  # 固定為近 1 年


    # ---------------------------------------------------------------------------
    # 載入中 / 停止搜尋 / 結果顯示
    # ---------------------------------------------------------------------------
    stop_slot = st.empty()  # 載入期間才會出現的「停止搜尋」按鈕
    result_slot = st.empty()  # 結果區：每次重新載入時整塊替換，舊畫面會立刻消失

    pending = st.session_state.pending_search
    trigger = st.session_state.trigger_search
    busy = bool(pending) or (
        bool(trigger) and not trigger.startswith("FAILED:") and not st.session_state.search_stopped
    )

    if busy:
        stop_slot.button("⏹ 停止搜尋", key="stop_search_btn", on_click=stop_search)

    if pending:
        st.session_state.pending_search = ""
        with result_slot.container():
            with st.spinner(f"⏳ 載入「{pending}」資料中，請稍候…"):
                run_search(pending)

# ---------------------------------------------------------------------------
# 側邊欄：搜尋紀錄
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🕒 搜尋紀錄")

    if st.session_state.search_history:
        for i, (hist_name, hist_symbol) in enumerate(st.session_state.search_history):
            col_btn, col_del = st.columns([4, 1])
            with col_btn:
                st.button(
                    f"🔍 {hist_name}",
                    key=f"btn_{hist_symbol}_{i}",
                    on_click=click_history_btn,
                    args=(hist_symbol, hist_name),
                    width="stretch",
                )
            with col_del:
                st.button(
                    "❌",
                    key=f"del_{hist_symbol}_{i}",
                    on_click=delete_single_history,
                    args=(i,),
                    width="stretch",
                )

        st.markdown("---")
        if st.button("🗑️ 清除所有紀錄", type="primary", width="stretch"):
            st.session_state.search_history = []
            st.rerun()
    else:
        st.info("目前沒有搜尋紀錄")


trigger = st.session_state.trigger_search

if st.session_state.search_stopped:
    result_slot.warning("⏹ 已停止搜尋。")

elif not trigger:
    result_slot.info("👋 歡迎使用股票分析儀！請在上方輸入股票代號或名稱，並點選「開始搜尋並分析」。")

elif trigger.startswith("FAILED:"):
    failed_keyword = trigger.replace("FAILED:", "", 1)
    with result_slot.container():
        st.error(f"❌ 找不到「{failed_keyword}」的相關股票資料。請檢查輸入是否正確。")
        if any("\u4e00" <= ch <= "\u9fff" for ch in failed_keyword) and not get_tw_names():
            st.warning("⚠️ 目前無法連線取得台股中文名稱清單，請改輸入股票代號（例如 2383），或檢查網路連線。")

else:
    with result_slot.container():
        render_analysis(trigger.upper(), time_period)

stop_slot.empty()  # 載入完成，移除停止按鈕


# ---------------------------------------------------------------------------
# 較慢的頁籤放最後渲染（需要連線抓大量資料），避免擋住前面頁籤的畫面
# ---------------------------------------------------------------------------
with tab_inst:
    render_prev_day_institutional_rank()

with tab_rank:
    render_market_rank()