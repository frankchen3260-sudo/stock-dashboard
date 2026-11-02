import datetime
import os
import time
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from concurrent.futures import ThreadPoolExecutor
from io import StringIO

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
        "股價同時站上季線(60日)與年線(240日)視為中長期多頭",
        "K > D 且 K < 80（未進入超買區）視為偏多",
        "RSI 介於 50～70（買方占優但未過熱）視為偏多",
        "柱狀體 OSC > 0（DIF 在 MACD 之上）視為偏多",
        f"乖離率(20日) < {BIAS_OVERHEAT:g}%（未過熱）視為合理",
    ]
    if d is None or len(d) < 30:
        return [(n, a, None, "日K資料不足", s) for (n, a), s in zip(names, stds)]

    last = d.iloc[-1]
    close = last["Close"]
    items = []

    # 1. 移動平均線
    ma60 = last["MA60"]
    ma240 = d["Close"].rolling(240).mean().iloc[-1]
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
        items.append((*names[0], bool(ok), detail, stds[0]))

    # 2. KD
    k, dd = last["K"], last["D"]
    if pd.isna(k) or pd.isna(dd):
        items.append((*names[1], None, "資料不足", stds[1]))
    else:
        zone = "，高檔超買" if k >= 80 else ("，低檔超賣" if k <= 20 else "")
        items.append((*names[1], bool(k > dd and k < 80), f"K {k:.0f}／D {dd:.0f}{zone}", stds[1]))

    # 3. RSI
    rsi = last["RSI"]
    if pd.isna(rsi):
        items.append((*names[2], None, "資料不足", stds[2]))
    else:
        zone = "，超買" if rsi >= 70 else ("，超賣" if rsi <= 30 else "")
        items.append((*names[2], bool(50 < rsi < 70), f"RSI(14) {rsi:.0f}{zone}", stds[2]))

    # 4. MACD
    if pd.isna(last["OSC"]):
        items.append((*names[3], None, "資料不足", stds[3]))
    else:
        axis = "零軸上" if last["DIF"] > 0 else "零軸下"
        items.append((*names[3], bool(last["OSC"] > 0),
                      f"DIF {last['DIF']:.2f}／MACD {last['MACD']:.2f}（{axis}），柱狀體 {last['OSC']:+.2f}", stds[3]))

    # 5. 乖離率（對 20 日均線）
    ma20 = last["MA20"]
    if pd.isna(ma20) or ma20 == 0:
        items.append((*names[4], None, "資料不足 20 日", stds[4]))
    else:
        bias = (close - ma20) / ma20 * 100
        zone = "，正乖離過大，留意回檔" if bias >= BIAS_OVERHEAT else ("，負乖離過大，留意超跌反彈" if bias <= -BIAS_OVERHEAT else "")
        items.append((*names[4], bool(bias < BIAS_OVERHEAT), f"20日乖離率 {bias:+.2f}%{zone}", stds[4]))

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
        with st.spinner("⏳ 載入中，請稍候…"):
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
# 主畫面：輸入區
# ---------------------------------------------------------------------------
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
        with st.spinner(f"⏳ 搜尋「{pending}」中，請稍候…"):
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