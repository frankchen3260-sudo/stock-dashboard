"""K線分析核心（純計算，不依賴 Streamlit／網路）。

app.py 負責畫面與資料抓取；這裡只放可以單獨測試的部分：
技術指標、日／月K趨勢判讀、K線訊號偵測、18 型態、預測評分與回測、法人籌碼評分、休市日計算。
測試：python tests/test_kline_core.py
"""
import datetime

import numpy as np
import pandas as pd


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


# ---------------------------------------------------------------------------
# K線技術指標與趨勢分析（日K、月K共用同一套引擎）
# 指標公式只在這裡定義一次：MA、KD(9,3,3)、MACD(12,26,9)、RSI(14)。
# 趨勢判讀規則也只有一份（analyze_trend），日K與月K只是套用不同的參數表（_TREND_CFG）。
# ---------------------------------------------------------------------------
def _rsi(close, n=14):
    """RSI（Wilder 平滑）。完全沒有下跌時視為 100。"""
    diff = close.diff()
    gain = diff.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-diff.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, float("nan")))
    return rsi.where(loss != 0, 100.0)


def _kd(df, n=9):
    """KD (9,3,3)：K、D 起始值 50，RSV 無法計算（資料不足）時為 NaN。回傳 (K, D) 兩個 Series。"""
    low_n = df["Low"].rolling(n).min()
    high_n = df["High"].rolling(n).max()
    rsv = (df["Close"] - low_n) / (high_n - low_n).replace(0, float("nan")) * 100
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
    return pd.Series(ks, index=df.index), pd.Series(ds, index=df.index)


def _macd(close, fast=12, slow=26, signal=9):
    """MACD：回傳 (DIF, MACD 訊號線, OSC 柱狀體) 三個 Series。"""
    dif = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    sig = dif.ewm(span=signal, adjust=False).mean()
    return dif, sig, dif - sig


def add_indicators(df, windows=(5, 10, 20), with_rsi=False):
    """幫 K 線資料表加上 MA／K／D／DIF／MACD／OSC（with_rsi=True 時再加 RSI）。日K、月K共用。"""
    d = df.copy()
    for n in windows:
        d[f"MA{n}"] = d["Close"].rolling(n).mean()
    d["K"], d["D"] = _kd(d)
    d["DIF"], d["MACD"], d["OSC"] = _macd(d["Close"])
    if with_rsi:
        d["RSI"] = _rsi(d["Close"])
    return d


def _recent_cross(series, tail):
    """近 tail 筆內最近一次穿越零軸的方向：1＝由負（或零）轉正、-1＝由正（或零）轉負、0＝沒有穿越。
    例：MA20 − MA60 的黃金／死亡交叉、K − D 的交叉、MACD 柱狀體翻紅／翻綠。"""
    s = series.dropna().tail(tail)
    if len(s) < 2:
        return 0
    sg = np.sign(s.to_numpy(float))
    for x in range(len(sg) - 1, 0, -1):
        if sg[x] > 0 and sg[x - 1] <= 0:
            return 1
        if sg[x] < 0 and sg[x - 1] >= 0:
            return -1
    return 0


# 日K／月K的趨勢判讀參數（想調整規則或門檻，只要改這裡）
_TREND_CFG = {
    "月": dict(
        mas=(5, 10, 20), ma_label="月均線", ma_names="5/10/20月",
        arr_bull="月MA5 > MA10 > MA20，多頭排列", arr_bear="月MA5 < MA10 < MA20，空頭排列",
        dirs=[(10, "月MA10 方向", "月MA10")],
        k_title="近3月K棒", k_n=3,
        k_bull=lambda r: "連續 3 根紅K，買盤延續" if r == 3 else None,
        k_bear=lambda r: "連續 3 根黑K，賣壓沉重" if r == 0 else None,
        k_mixed=lambda r: f"3 根中有 {r} 根紅K，多空拉鋸",
        hl_up="近3月高點、低點逐月墊高", hl_down="近3月高點、低點逐月下移",
        range_n=12, range_min=6, range_title="12月區間", range_unit="個月",
        volume=False, rsi=False, kd_title="月KD", macd_title="月MACD", shape_title="最新月K型態",
        cuts=(5, 2, -1, -4),  # 強勢多頭 / 偏多 / 盤整 / 偏空 的分數下限
    ),
    "日": dict(
        mas=(5, 10, 20, 60), ma_label="均線", ma_names="5/10/20/60日",
        arr_bull="短中長均線由上而下排列，多頭排列", arr_bear="短中長均線由下而上排列，空頭排列",
        dirs=[(20, "月線方向", "月線 (MA20)"), (60, "季線方向", "季線 (MA60)")],
        k_title="近5日K棒", k_n=5,
        k_bull=lambda r: f"5 根中有 {r} 根紅K，買盤強勢" if r >= 4 else None,
        k_bear=lambda r: f"5 根中僅 {r} 根紅K，賣壓偏重" if r <= 1 else None,
        k_mixed=lambda r: f"5 根中有 {r} 根紅K，多空拉鋸",
        hl_up="近3日高點、低點逐日墊高", hl_down="近3日高點、低點逐日下移",
        range_n=20, range_min=10, range_title="20日區間", range_unit="日",
        volume=True, rsi=True, kd_title="日KD", macd_title="日MACD", shape_title="最新日K型態",
        cuts=(6, 3, -2, -5),
    ),
}


def _trend_label(total, cuts):
    strong, bull, flat, bear = cuts
    if total >= strong:
        return "強勢多頭", "success"
    if total >= bull:
        return "偏多", "success"
    if total >= flat:
        return "盤整觀望", "info"
    if total >= bear:
        return "偏空", "error"
    return "空頭走勢", "error"


def analyze_trend(d, tf):
    """日K（tf='日'）、月K（tf='月'）共用的趨勢判讀。d 需先用 add_indicators 加好指標（日K要 with_rsi=True）。
    回傳 (訊號列表, 總分, 判讀標籤, 提示等級)；每個訊號分數為 +1 / 0 / -1。"""
    cfg = _TREND_CFG[tf]
    sig = []
    last, prev = d.iloc[-1], d.iloc[-2]
    close = last["Close"]

    def add(title, text, score):
        sig.append((title, text, score))

    # 1. 股價相對均線
    mas = [(n, last[f"MA{n}"]) for n in cfg["mas"] if not pd.isna(last[f"MA{n}"])]
    if mas:
        above = sum(1 for _, v in mas if close > v)
        score = 1 if above == len(mas) else (-1 if above == 0 else 0)
        add(f"股價與{cfg['ma_label']}", f"站上 {above}/{len(mas)} 條{cfg['ma_label']} ({cfg['ma_names']})", score)

    # 2. 均線排列（前三條必備；第四條（季線）有資料才納入）
    core = [last[f"MA{n}"] for n in cfg["mas"][:3]]
    if not any(pd.isna(v) for v in core):
        chain = core + [last[f"MA{n}"] for n in cfg["mas"][3:] if not pd.isna(last[f"MA{n}"])]
        if all(a > b for a, b in zip(chain, chain[1:])):
            add("均線排列", cfg["arr_bull"], 1)
        elif all(a < b for a, b in zip(chain, chain[1:])):
            add("均線排列", cfg["arr_bear"], -1)
        else:
            add("均線排列", "均線糾結，方向未明", 0)

    # 3. 均線方向（月K：MA10；日K：月線 MA20、季線 MA60）
    for n, title, name in cfg["dirs"]:
        cur, pre = last[f"MA{n}"], prev[f"MA{n}"]
        if not (pd.isna(cur) or pd.isna(pre)):
            if cur > pre:
                add(title, f"{name} 上揚", 1)
            elif cur < pre:
                add(title, f"{name} 下彎", -1)
            else:
                add(title, f"{name} 走平", 0)

    # 4. 近幾根紅黑K
    recent = d.tail(cfg["k_n"])
    reds = int((recent["Close"] > recent["Open"]).sum())
    if (txt := cfg["k_bull"](reds)):
        add(cfg["k_title"], txt, 1)
    elif (txt := cfg["k_bear"](reds)):
        add(cfg["k_title"], txt, -1)
    else:
        add(cfg["k_title"], cfg["k_mixed"](reds), 0)

    # 5. 高低點墊高 / 摜低（近 3 根）
    r3 = d.tail(3)
    h3, l3 = r3["High"], r3["Low"]
    if h3.is_monotonic_increasing and l3.is_monotonic_increasing and h3.nunique() == 3:
        add("高低點", cfg["hl_up"], 1)
    elif h3.is_monotonic_decreasing and l3.is_monotonic_decreasing and h3.nunique() == 3:
        add("高低點", cfg["hl_down"], -1)
    else:
        add("高低點", "高低點無明顯墊高或下移", 0)

    # 6. 突破 / 跌破前 N 根區間
    rn, ru = cfg["range_n"], cfg["range_unit"]
    prior = d.iloc[:-1].tail(rn)
    if len(prior) >= cfg["range_min"]:
        if close > prior["High"].max():
            add(cfg["range_title"], f"收盤突破前 {rn} {ru}高點", 1)
        elif close < prior["Low"].min():
            add(cfg["range_title"], f"收盤跌破前 {rn} {ru}低點", -1)
        else:
            add(cfg["range_title"], f"仍在前 {rn} {ru}高低區間內", 0)

    # 7. 量價關係（只有日K；今日量 vs 前 5 日均量）
    if cfg["volume"] and "Volume" in d.columns and len(d) >= 7:
        vol_avg = d["Volume"].iloc[-6:-1].mean()
        if vol_avg and vol_avg > 0 and last["Volume"] > 0:
            ratio = last["Volume"] / vol_avg
            chg = (close - prev["Close"]) / prev["Close"] * 100
            if ratio >= 1.5 and chg > 0:
                add("量價關係", f"價漲量增（量為前5日均量 {ratio:.1f} 倍）", 1)
            elif ratio >= 1.5 and chg < 0:
                add("量價關係", f"價跌量增（量為前5日均量 {ratio:.1f} 倍），賣壓放大", -1)
            elif ratio <= 0.7 and chg > 0:
                add("量價關係", f"價漲量縮（量為前5日均量 {ratio:.1f} 倍），追價力道不足", 0)
            else:
                add("量價關係", f"量能無明顯異常（前5日均量 {ratio:.1f} 倍）", 0)

    # 8. KD
    if not (pd.isna(last["K"]) or pd.isna(last["D"])):
        zone = "（高檔 ≥80，留意鈍化或反轉）" if last["K"] >= 80 else ("（低檔 ≤20，留意超賣反彈）" if last["K"] <= 20 else "")
        if last["K"] > last["D"]:
            add(cfg["kd_title"], f"K {last['K']:.0f} > D {last['D']:.0f}，偏多{zone}", 1)
        else:
            add(cfg["kd_title"], f"K {last['K']:.0f} < D {last['D']:.0f}，偏空{zone}", -1)

    # 9. MACD
    if not pd.isna(last["OSC"]):
        if last["DIF"] > 0 and last["OSC"] > 0:
            add(cfg["macd_title"], "DIF 在零軸上且柱狀體為正，多方控盤", 1)
        elif last["DIF"] < 0 and last["OSC"] < 0:
            add(cfg["macd_title"], "DIF 在零軸下且柱狀體為負，空方控盤", -1)
        else:
            add(cfg["macd_title"], "DIF 與柱狀體方向不一致，轉折觀察期", 0)

    # 10. RSI(14)（只有日K）
    if cfg["rsi"] and not pd.isna(last["RSI"]):
        rsi = last["RSI"]
        note = "（超買 ≥70，留意回檔）" if rsi >= 70 else ("（超賣 ≤30，留意反彈）" if rsi <= 30 else "")
        if rsi > 50:
            add("RSI(14)", f"RSI {rsi:.0f} 在 50 之上，偏多{note}", 1)
        elif rsi < 50:
            add("RSI(14)", f"RSI {rsi:.0f} 在 50 之下，偏空{note}", -1)
        else:
            add("RSI(14)", "RSI 剛好 50，多空平衡", 0)

    # 11. 最新一根K棒的上下影線
    rng = last["High"] - last["Low"]
    if rng > 0:
        upper = (last["High"] - max(last["Open"], last["Close"])) / rng
        lower = (min(last["Open"], last["Close"]) - last["Low"]) / rng
        if upper > 0.5:
            add(cfg["shape_title"], "長上影線，上檔賣壓重", -1)
        elif lower > 0.5:
            add(cfg["shape_title"], "長下影線，下檔有支撐", 1)
        else:
            add(cfg["shape_title"], "無明顯長影線", 0)

    total = sum(sc for _, _, sc in sig)
    label, level = _trend_label(total, cfg["cuts"])
    return sig, total, label, level


# ---------------------------------------------------------------------------
# 即時報價（台股：證交所 MIS 即時行情，約每 5 秒更新；其他：Yahoo，可能延遲）
# ---------------------------------------------------------------------------
def next_trading_day(last_day, holidays=frozenset()):
    """last_day（date）之後的下一個交易日：跳過週末與 holidays（休市日集合）。"""
    nd = last_day + datetime.timedelta(days=1)
    while nd.weekday() >= 5 or nd in holidays:
        nd += datetime.timedelta(days=1)
    return nd


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


BIAS_OVERHEAT = 10.0    # 乖離率(20日) 未達此值(%)（未過熱）→ 紅燈


# 技術面燈號的名稱與「判斷標準」說明文字（放在函式外，函式只剩判斷邏輯）
_TECH_NAMES = [
    ("移動平均線", "MA"), ("隨機指標", "KD (KDJ)"), ("相對強弱指標", "RSI"),
    ("指數平滑異同移動平均線", "MACD"), ("乖離率", "BIAS"),
]


_TECH_STDS = [
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


def _technical_items(d):
    """技術面：d 為含 K、D、DIF、MACD、OSC、RSI、MA20、MA60 的日K資料表。"""
    names, stds = _TECH_NAMES, _TECH_STDS
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
        cross = {
            1: "近期出現黃金交叉（20MA 由下往上突破 60MA）＝短期買盤力道超越長期平均，中長期波段極大機率即將啟動上漲",
            -1: "近期出現死亡交叉（20MA 由上往下跌破 60MA）＝短期持股成本快速跌破長期平均，多頭行情結束，可能迎來深度崩盤或回檔",
            0: "近 5 日無 20MA／60MA 交叉",
        }[_recent_cross(ma20_s - ma60_s, 6)]

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

        kd_cross = {
            1: "｜近期 K 由下往上穿越 D（黃金交叉），短線買盤轉強",
            -1: "｜近期 K 由上往下跌破 D（死亡交叉），短線賣壓轉強",
            0: "",
        }[_recent_cross(d["K"] - d["D"], 4)]
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
        flip = {
            1: "柱狀體由綠翻紅（由負變正）＝買方動能增強，下跌力道減弱、上漲動能開始爆發，屬波段轉好的起點",
            -1: "柱狀體由紅翻綠（由正變負）＝賣方壓力湧現，上漲力道竭盡、下跌動能開始增強，屬波段轉壞的起點",
            0: "",
        }[_recent_cross(d["OSC"], 4)]
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


ZONE_MAX_WIDTH = 0.06      # 單一大量區寬度上限：約股價的 6%（避免把整片區間當成一個支撐）


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
    mean, total = hist.mean(), hist.sum()
    bw = (hi - lo) / bins
    max_bins = max(1, int(ZONE_MAX_WIDTH * (lo + hi) / 2 / bw))  # 大量區寬度上限（約股價的 6%）
    used = np.zeros(bins, dtype=bool)
    zones = []
    for k in np.argsort(hist)[::-1]:  # 由成交量最大的價位格開始
        if hist[k] < 1.5 * mean:
            break
        if used[k]:
            continue
        s = e = int(k)
        while e - s + 1 < max_bins:  # 向兩側擴張：只併入量能 ≥ 峰值 6 成、且尚未被別的大量區用掉的鄰格
            left = hist[s - 1] if s > 0 and not used[s - 1] else -1
            right = hist[e + 1] if e + 1 < bins and not used[e + 1] else -1
            nxt = max(left, right)
            if nxt < 0.6 * hist[k]:
                break
            if left >= right:
                s -= 1
            else:
                e += 1
        used[s:e + 1] = True
        zones.append((float(edges[s]), float(edges[e + 1]), float(hist[s:e + 1].sum() / total)))
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
    # 矛盾過濾：近 3 個交易日（含當天）剛出現出場訊號時，支撐型／順勢型進場訊號不成立
    # （例如昨天長黑跌破支撐，今天不能又說「回測支撐可進場」）。錘形線、吞噬、早晨之星屬止跌反轉型，不受影響。
    conflict_names = {"回測大量支撐區", "回測頸線不破", "拉回買進"}
    exit_idx = [s["idx"] for s in sigs if s["kind"] == "exit"]
    sigs = [
        s for s in sigs
        if not (s["name"] in conflict_names and any(0 <= s["idx"] - j <= 3 for j in exit_idx))
    ]
    return d, sigs


def _pz(v):
    """建議價格：粗體、放大，方便一眼看到（需搭配 unsafe_allow_html=True）。"""
    return f"<span style='font-size:1.35em;font-weight:800'>{v:.2f}</span>"


def _pz_range(lo, hi):
    return _pz(lo) if abs(hi - lo) <= 0.005 else f"{_pz(lo)}～{_pz(hi)}"


def _sig_price_txt(s, html=False):
    """某個K線訊號的建議進場價與停損／出場價文字（沒有就回傳空字串）。html=True 時數字用粗體大字。"""
    num = _pz if html else (lambda v: f"{v:.2f}")
    rng_ = _pz_range if html else (lambda lo, hi: f"{lo:.2f}" + (f"～{hi:.2f}" if abs(hi - lo) > 0.005 else ""))
    parts = []
    if s.get("entry"):
        parts.append("建議進場 " + rng_(*s["entry"]))
    if s.get("stop"):
        parts.append(f"{s.get('stop_label') or '停損'} " + num(s["stop"]))
    return "　".join(parts)


def _sig_price_lines(lst):
    return "\n".join(
        f"- 💰 **{s['name']}**（{s['date'].strftime('%m/%d')}）　{_sig_price_txt(s, html=True)}"
        for s in lst if _sig_price_txt(s)
    )


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
    d["RSI"] = _rsi(c).fillna(100)  # 公式與日K趨勢分析共用同一份（_rsi）
    d["MACD_H"] = _macd(c)[2]       # MACD 柱狀體（OSC）
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


# 評分權重集中在這裡（想調整某個因子的影響力，只改這個表；各因子的歷史有效性可看「K線分析」頁的因子回測表）
SCORE_W = dict(
    k_cap=30, k_lags=((0, 20, "今天"), (1, 12, "昨天"), (2, 6, "前天")),   # K線訊號：最近 3 根、越新權重越高
    ma_full=15, ma_part=5, ma_dir=5,                                        # 均線排列／站上月線／月線方向
    macd_base=8, macd_extra=4,                                              # MACD 柱狀體（同向擴大再加）
    rsi_oversold=8, rsi_hot=-6, rsi_very_hot=-10,                           # RSI
    bias_thr=10, bias_hot=-8, bias_cold=6,                                  # 與 20 日線乖離
    total_cap=100,
)


SCORE_GROUPS = ("K線訊號", "均線趨勢", "動能(MACD/RSI)", "量價", "乖離", "法人")


def _score_parts(A, by, i, inst=None):
    """第 i 根K線收盤後各因子群組的分數。回傳 ({群組: 分數}, [(加減分, 理由)])。inst 為 (法人分數, 理由)，只在最新一天使用。"""
    W = SCORE_W
    reasons = []
    c, m5, m10, m20 = A["Close"][i], A["MA5"][i], A["MA10"][i], A["MA20"][i]

    # 1) K 線型態（最近 3 根，越新權重越高）
    k_sum = 0
    for lag, w, label in W["k_lags"]:
        for s in by.get(i - lag, []):
            pts = w if s["kind"] == "entry" else -w
            k_sum += pts
            kind = "進場" if s["kind"] == "entry" else "出場"
            reasons.append((pts, f"{label}出現{kind}訊號「{s['name']}」"))
    parts = {"K線訊號": max(-W["k_cap"], min(W["k_cap"], k_sum))}

    # 2) 均線排列與 20 日線方向
    if c > m5 > m10 > m20:
        pts, txt = W["ma_full"], "收盤 > 5日線 > 10日線 > 20日線，多頭排列"
    elif c < m5 < m10 < m20:
        pts, txt = -W["ma_full"], "收盤 < 5日線 < 10日線 < 20日線，空頭排列"
    elif c > m20:
        pts, txt = W["ma_part"], "收盤站在 20 日線之上"
    else:
        pts, txt = -W["ma_part"], "收盤在 20 日線之下"
    trend = pts
    reasons.append((pts, txt))
    up20 = m20 > A["MA20"][i - 3]
    pts = W["ma_dir"] if up20 else -W["ma_dir"]
    trend += pts
    reasons.append((pts, "20 日線向上，中期趨勢偏多" if up20 else "20 日線向下，中期趨勢偏空"))
    parts["均線趨勢"] = trend

    # 3) MACD 柱狀體
    h, hp = A["MACD_H"][i], A["MACD_H"][i - 1]
    if h > 0:
        pts = W["macd_base"] + (W["macd_extra"] if h > hp else 0)
        txt = "MACD 柱狀體為正" + ("且持續擴大，多方動能增強" if h > hp else "，但動能略為收斂")
    else:
        pts = -(W["macd_base"] + (W["macd_extra"] if h < hp else 0))
        txt = "MACD 柱狀體為負" + ("且持續擴大，空方動能增強" if h < hp else "，但跌勢略為收斂")
    mom = pts
    reasons.append((pts, txt))

    # 4) RSI
    r = A["RSI"][i]
    if r < 30:
        pts, txt = W["rsi_oversold"], f"RSI {r:.0f} 低於 30，超賣區有反彈機會"
    elif r >= 75:
        pts, txt = W["rsi_very_hot"], f"RSI {r:.0f} 高於 75，嚴重過熱，回檔風險高"
    elif r >= 70:
        pts, txt = W["rsi_hot"], f"RSI {r:.0f} 進入過熱區（70 以上）"
    else:
        pts = int(round((r - 50) / 4))
        txt = f"RSI {r:.0f}，{'偏強' if pts > 0 else '偏弱'}"
    if pts:
        mom += pts
        reasons.append((pts, txt))
    parts["動能(MACD/RSI)"] = mom

    # 5) 量價關係
    pv_pts, pv_reasons = _pv_score(A, i)
    parts["量價"] = pv_pts
    reasons.extend(pv_reasons)

    # 6) 與 20 日線的乖離
    bias = (c / m20 - 1) * 100
    parts["乖離"] = 0
    if bias > W["bias_thr"]:
        parts["乖離"] = W["bias_hot"]
        reasons.append((W["bias_hot"], f"股價高於 20 日線 {bias:.1f}%，乖離過大，容易拉回"))
    elif bias < -W["bias_thr"]:
        parts["乖離"] = W["bias_cold"]
        reasons.append((W["bias_cold"], f"股價低於 20 日線 {abs(bias):.1f}%，超跌乖離，有反彈空間"))

    # 7) 三大法人買賣超（只有近 10 個交易日資料，因此只用在最新一天的預測）
    parts["法人"] = 0
    if inst and inst[1]:
        parts["法人"] = inst[0]
        reasons.extend(inst[1])
    return parts, reasons


# 各因子群組的倍率（預設全部 1.0＝不調整）。用 tools/calibrate_weights.py 跑多檔多年資料後，
# 把建議值貼在這裡：某因子沒有預測力就調成 0 或 0.5，不用去改 SCORE_W 裡的細部分數。
SCORE_SCALE = {g: 1.0 for g in SCORE_GROUPS}


def _total_score(parts):
    cap = SCORE_W["total_cap"]
    total = sum(v * SCORE_SCALE.get(g, 1.0) for g, v in parts.items())
    return int(max(-cap, min(cap, round(total))))


def _score_at(A, by, i, inst=None):
    """第 i 根K線收盤後的總分（-100～+100）。回傳 (分數, [(加減分, 理由)])。"""
    parts, reasons = _score_parts(A, by, i, inst)
    return _total_score(parts), reasons


def backtest_edge(scores, nxt, verdict, threshold=None):
    """某個判斷（bull／bear）在歷史上的表現，並和「所有日子」的基準比較。
    scores、nxt：同長度；nxt 為隔日報酬。回傳 dict(n, hit, base, edge, mean, base_mean)，判斷為觀望時回傳 None。"""
    thr = PRED_THRESHOLD if threshold is None else threshold
    if verdict not in ("bull", "bear") or len(nxt) == 0:
        return None
    s, r = np.asarray(scores, float), np.asarray(nxt, float)
    sel = r[s >= thr] if verdict == "bull" else r[s <= -thr]
    hit_of = (lambda x: (x > 0).mean() * 100) if verdict == "bull" else (lambda x: (x < 0).mean() * 100)
    out = dict(n=int(len(sel)), base=float(hit_of(r)), base_mean=float(r.mean() * 100))
    out["hit"] = float(hit_of(sel)) if len(sel) else float("nan")
    out["edge"] = out["hit"] - out["base"] if len(sel) else float("nan")
    out["mean"] = float(sel.mean() * 100) if len(sel) else float("nan")
    return out


def factor_effectiveness(parts_hist, nxt):
    """各因子群組的歷史有效性：該因子偏多（>0）的日子隔天上漲比例、偏空（<0）的日子隔天下跌比例，並附全期基準。
    回傳 [(群組, 偏多日數, 偏多後上漲%, 偏空日數, 偏空後下跌%)]；樣本為 0 時比例為 None。"""
    r = np.asarray(nxt, float)
    rows = []
    for g in SCORE_GROUPS:
        p = np.array([x.get(g, 0) for x in parts_hist], float)
        nb, ns = int((p > 0).sum()), int((p < 0).sum())
        rows.append((
            g, nb, float((r[p > 0] > 0).mean() * 100) if nb else None,
            ns, float((r[p < 0] < 0).mean() * 100) if ns else None,
        ))
    return rows


def _prediction_pack(d, sigs):
    d = _prepare_indicators(d.copy())
    A = {k: d[k].to_numpy(float) for k in ("Open", "Close", "High", "Low", "MA5", "MA10", "MA20", "RSI", "MACD_H", "VR", "OBV")}
    by = {}
    for s in sigs:
        by.setdefault(s["idx"], []).append(s)
    return d, A, by


K2_PAT_PTS, K2_PAT_CAP = 10, 15  # 18 種型態加減分：最近 3 根日K，每組偏多 +10／偏空 -10，合計上限 ±15（權重為經驗設定，未回測）


# 「K線分析」的訊號與「18 型態」有 7 組是同一種型態（例如錘形線＝錘子線）。
# 預測分數裡 K線訊號已經算過一次（權重較高），所以最近 3 根內 K線訊號已經出現的型態，18 型態這邊不再重複加減分。
K2_SAME_AS_K1 = {
    "錘子線": "錘形線", "吊人線": "上吊線／射擊之星", "射擊之星": "上吊線／射擊之星",
    "多頭吞噬": "多頭吞噬", "空頭吞噬": "空頭吞噬", "晨星": "早晨之星", "黃昏之星": "黃昏之星",
}


def k2_counted_patterns(recent_clusters, k1_recent_names):
    """最近 3 根內的型態群組 → 真正要計分的型態：每組重疊型態取最主要、且沒被 K線訊號計過分的一個。"""
    counted = []
    for c in recent_clusters:
        fresh = [m for m in c["members"] if K2_SAME_AS_K1.get(m["name"]) not in k1_recent_names]
        if fresh:
            counted.append(max(fresh, key=lambda m: (m["g"], -m["s"])))  # 與 k2_cluster 選主要型態的規則相同
    return counted


# ---------------------------------------------------------------------------
# 頁籤：K線分析2
# 參考「處置王－K線技術分析教學」的分析邏輯：
#   18 種型態（單根 6、兩根 6、三根 6）＋前置趨勢（3／5／10 根）＋日週月季各自重新比對
#   ＋重疊型態合併標註（+N）＋條件比對（每項條件的實際值與門檻）
# 注意：該網站只公開型態的結構說明，沒有公開精確門檻；下面的門檻是本程式採用的常見慣例，
#       所以結果可能和該 APP 不完全相同，一切以頁面內的「條件比對」為準。
# 台股慣例：紅K＝收盤高於開盤（上漲），綠K＝收盤低於開盤（下跌）；紅箭頭＝偏多分類，綠箭頭＝偏空分類
# ---------------------------------------------------------------------------
# (名稱, 幾根K, 方向分類, 前置趨勢需求)  需求：down 前置下跌／up 前置上漲／notup 非上漲／notdown 非下跌／None 不限
K2_PATTERNS = [
    ("長紅線", 1, "bull", None), ("長黑線", 1, "bear", None),
    ("錘子線", 1, "bull", "down"), ("吊人線", 1, "bear", "up"),
    ("倒錘子線", 1, "bull", "down"), ("射擊之星", 1, "bear", "up"),
    ("多頭吞噬", 2, "bull", "down"), ("空頭吞噬", 2, "bear", "up"),
    ("貫穿線", 2, "bull", "down"), ("烏雲罩頂", 2, "bear", "up"),
    ("多頭母子線", 2, "bull", "down"), ("空頭母子線", 2, "bear", "up"),
    ("晨星", 3, "bull", "down"), ("黃昏之星", 3, "bear", "up"),
    ("紅三兵", 3, "bull", "notup"), ("三隻烏鴉", 3, "bear", "notdown"),
    ("三內升", 3, "bull", "down"), ("三內降", 3, "bear", "up"),
]


# 前置趨勢門檻：前 N 根的收盤漲跌幅達此比例才算上漲／下跌（依週期放大，並隨 N 調整）
K2_TREND_BASE = {"日": 0.015, "週": 0.03, "月": 0.06, "季": 0.10}


K2_LONG_BODY = 1.5         # 長實體：實體 ≥ 前 20 根平均實體的 1.5 倍


K2_MID_BODY = 1.0          # 兩根／三根型態中「較明顯」的實體：≥ 平均實體 1.0 倍


K2_SHORT_SHADOW = 0.3      # 長紅／長黑：上下影線各 ≤ 實體的 30%


K2_SMALL_BODY_RNG = 0.35   # 錘子等小實體：實體 ≤ 全距的 35%


K2_LONG_SHADOW_RNG = 0.6   # 長影線：≥ 全距的 60%（且 ≥ 實體 2 倍）


K2_SHORT_SHADOW_RNG = 0.15  # 另一側短影線：≤ 全距的 15%


K2_STAR_BODY = 0.5         # 星線：實體 ≤ 第一根實體的 50%


K2_SOLDIER_BODY = 0.6      # 紅三兵／三隻烏鴉：每根實體 ≥ 平均實體的 60%


class _K2Chk:
    """記錄每一項「通過」的判斷條件（項目、本次實際值、門檻），給條件比對表使用。"""

    def __init__(self):
        self.rows = []

    def ok(self, label, actual, thresh, cond):
        if not bool(cond):
            return False
        self.rows.append((label, actual, thresh, "✅ 通過"))
        return True


class _K2Ctx:
    def __init__(self, df, n_trend, period):
        self.O, self.H, self.L, self.C = (df[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
        self.body = np.abs(self.C - self.O)
        self.rng = self.H - self.L
        self.top, self.bot = np.maximum(self.O, self.C), np.minimum(self.O, self.C)
        self.upper, self.lower = self.H - self.top, self.bot - self.L
        self.avgb = pd.Series(self.body).shift(1).rolling(20, min_periods=5).mean().to_numpy()  # 前 20 根平均實體
        self.n = n_trend
        self.thr = K2_TREND_BASE[period] * float(np.sqrt(n_trend / 5))

    def trend(self, s):
        """型態第一根（索引 s）之前 N 根的趨勢：回傳 (up/down/flat, 漲跌幅)；資料不足回傳 (None, None)。"""
        j0 = s - 1 - self.n
        if j0 < 0 or self.C[j0] <= 0:
            return None, None
        ret = self.C[s - 1] / self.C[j0] - 1
        return ("up" if ret >= self.thr else "down" if ret <= -self.thr else "flat"), ret


_K2_TREND_NAME = {"up": "上漲", "down": "下跌", "flat": "整理"}


def _k2_trend_ok(X, s, need, c):
    if need is None:
        return True
    cls, ret = X.trend(s)
    if cls is None:
        return False
    cond = {"down": cls == "down", "up": cls == "up", "notup": cls != "up", "notdown": cls != "down"}[need]
    thr = f"{X.thr * 100:.1f}%"
    rule = {
        "down": f"前置下跌（≤ -{thr}）", "up": f"前置上漲（≥ +{thr}）",
        "notup": f"前置非上漲（< +{thr}）", "notdown": f"前置非下跌（> -{thr}）",
    }[need]
    return c.ok(f"前置趨勢（型態前 {X.n} 根）", f"{ret * 100:+.1f}%（{_K2_TREND_NAME[cls]}）", rule, cond)


def _k2_red(X, k, c, label):
    return c.ok(label, f"開 {X.O[k]:.2f} → 收 {X.C[k]:.2f}", "收盤 > 開盤（紅K）", X.C[k] > X.O[k])


def _k2_green(X, k, c, label):
    return c.ok(label, f"開 {X.O[k]:.2f} → 收 {X.C[k]:.2f}", "收盤 < 開盤（綠K）", X.C[k] < X.O[k])


def _k2_bodyx(X, k, c, label, mult):
    r = X.body[k] / X.avgb[k]
    return c.ok(label, f"實體 {X.body[k]:.2f}＝平均實體 {r:.1f} 倍", f"≥ 平均實體 × {mult}", r >= mult)


def _k2_hammer_shape(X, i, c, inverted):
    rng = X.rng[i]
    if rng <= 0:
        return False
    long_s, short_s = (X.upper[i], X.lower[i]) if inverted else (X.lower[i], X.upper[i])
    ln, sn = ("上影線", "下影線") if inverted else ("下影線", "上影線")
    b = max(X.body[i], 1e-9)
    return (
        c.ok("小實體", f"實體 {X.body[i]:.2f}＝全距 {X.body[i] / rng * 100:.0f}%", f"≤ 全距 {K2_SMALL_BODY_RNG * 100:.0f}%",
             X.body[i] <= K2_SMALL_BODY_RNG * rng)
        and c.ok(f"長{ln}", f"{long_s:.2f}＝實體 {long_s / b:.1f} 倍、全距 {long_s / rng * 100:.0f}%",
                 f"≥ 實體 2 倍，且 ≥ 全距 {K2_LONG_SHADOW_RNG * 100:.0f}%",
                 long_s >= 2 * X.body[i] and long_s >= K2_LONG_SHADOW_RNG * rng)
        and c.ok(f"短{sn}", f"{short_s:.2f}＝全距 {short_s / rng * 100:.0f}%", f"≤ 全距 {K2_SHORT_SHADOW_RNG * 100:.0f}%",
                 short_s <= K2_SHORT_SHADOW_RNG * rng)
    )


def _k2_long_line(X, i, c, need, red):
    if X.rng[i] <= 0:
        return False
    first = _k2_red(X, i, c, "收紅") if red else _k2_green(X, i, c, "收黑")
    return (
        first
        and _k2_bodyx(X, i, c, "實體夠長", K2_LONG_BODY)
        and c.ok("上下影線短", f"上 {X.upper[i]:.2f}／下 {X.lower[i]:.2f}", f"各 ≤ 實體 × {K2_SHORT_SHADOW}",
                 X.upper[i] <= K2_SHORT_SHADOW * X.body[i] and X.lower[i] <= K2_SHORT_SHADOW * X.body[i])
    )


def _k2_engulf(X, i, c, need, bull):
    a, b = i - 1, i
    if bull:
        ok = _k2_green(X, a, c, "第 1 根收黑") and _k2_red(X, b, c, "第 2 根收紅") and c.ok(
            "第 2 根實體包住第 1 根", f"開 {X.O[b]:.2f} ≤ 前收 {X.C[a]:.2f}；收 {X.C[b]:.2f} ≥ 前開 {X.O[a]:.2f}",
            "開盤 ≤ 前一根收盤，且收盤 ≥ 前一根開盤", X.O[b] <= X.C[a] and X.C[b] >= X.O[a])
    else:
        ok = _k2_red(X, a, c, "第 1 根收紅") and _k2_green(X, b, c, "第 2 根收黑") and c.ok(
            "第 2 根實體包住第 1 根", f"開 {X.O[b]:.2f} ≥ 前收 {X.C[a]:.2f}；收 {X.C[b]:.2f} ≤ 前開 {X.O[a]:.2f}",
            "開盤 ≥ 前一根收盤，且收盤 ≤ 前一根開盤", X.O[b] >= X.C[a] and X.C[b] <= X.O[a])
    return ok and c.ok("實體比前一根大", f"{X.body[b]:.2f} > {X.body[a]:.2f}", "第 2 根實體 > 第 1 根實體",
                       X.body[b] > X.body[a]) and _k2_trend_ok(X, a, need, c)


def _k2_pierce(X, i, c, need, bull):
    a, b = i - 1, i
    mid = (X.O[a] + X.C[a]) / 2
    if bull:
        ok = (
            _k2_green(X, a, c, "第 1 根收黑") and _k2_bodyx(X, a, c, "第 1 根實體明顯", K2_MID_BODY)
            and _k2_red(X, b, c, "第 2 根收紅")
            and c.ok("開盤不高於前收", f"開 {X.O[b]:.2f} ≤ 前收 {X.C[a]:.2f}", "開盤 ≤ 前一根收盤", X.O[b] <= X.C[a])
            and c.ok("收盤深入前一根實體過半", f"收 {X.C[b]:.2f}；實體中點 {mid:.2f}；前開 {X.O[a]:.2f}",
                     "中點 < 收盤 < 前一根開盤（未形成吞噬）", mid < X.C[b] < X.O[a])
        )
    else:
        ok = (
            _k2_red(X, a, c, "第 1 根收紅") and _k2_bodyx(X, a, c, "第 1 根實體明顯", K2_MID_BODY)
            and _k2_green(X, b, c, "第 2 根收黑")
            and c.ok("開盤不低於前收", f"開 {X.O[b]:.2f} ≥ 前收 {X.C[a]:.2f}", "開盤 ≥ 前一根收盤", X.O[b] >= X.C[a])
            and c.ok("收盤深入前一根實體過半", f"收 {X.C[b]:.2f}；實體中點 {mid:.2f}；前開 {X.O[a]:.2f}",
                     "前一根開盤 < 收盤 < 中點（未形成吞噬）", X.O[a] < X.C[b] < mid)
        )
    return ok and _k2_trend_ok(X, a, need, c)


def _k2_harami(X, i, c, need, bull):
    a, b = i - 1, i
    first = _k2_green(X, a, c, "第 1 根收黑") if bull else _k2_red(X, a, c, "第 1 根收紅")
    return (
        first and _k2_bodyx(X, a, c, "第 1 根實體明顯", K2_MID_BODY)
        and c.ok("第 2 根實體收進第 1 根實體內", f"{X.bot[b]:.2f}～{X.top[b]:.2f} 在 {X.bot[a]:.2f}～{X.top[a]:.2f} 內",
                 "第 2 根實體上下緣都在第 1 根實體內", X.top[b] <= X.top[a] and X.bot[b] >= X.bot[a])
        and c.ok("第 2 根實體較小", f"{X.body[b]:.2f} < {X.body[a]:.2f}", "第 2 根實體 < 第 1 根實體", X.body[b] < X.body[a])
        and _k2_trend_ok(X, a, need, c)
    )


def _k2_star(X, i, c, need, bull):
    a, m, b = i - 2, i - 1, i
    mid1 = (X.O[a] + X.C[a]) / 2
    if bull:
        ok = _k2_green(X, a, c, "第 1 根收黑") and _k2_bodyx(X, a, c, "第 1 根實體明顯", K2_MID_BODY)
        star_pos = X.top[m] <= mid1
        pos_txt = f"星線實體上緣 {X.top[m]:.2f} ≤ 第 1 根實體中點 {mid1:.2f}"
        third = _k2_red(X, b, c, "第 3 根收紅") and c.ok(
            "第 3 根回到第 1 根實體內", f"收 {X.C[b]:.2f} > 中點 {mid1:.2f}", "收盤 > 第 1 根實體中點", X.C[b] > mid1)
    else:
        ok = _k2_red(X, a, c, "第 1 根收紅") and _k2_bodyx(X, a, c, "第 1 根實體明顯", K2_MID_BODY)
        star_pos = X.bot[m] >= mid1
        pos_txt = f"星線實體下緣 {X.bot[m]:.2f} ≥ 第 1 根實體中點 {mid1:.2f}"
        third = _k2_green(X, b, c, "第 3 根收黑") and c.ok(
            "第 3 根回到第 1 根實體內", f"收 {X.C[b]:.2f} < 中點 {mid1:.2f}", "收盤 < 第 1 根實體中點", X.C[b] < mid1)
    return (
        ok
        and c.ok("中間為小實體", f"{X.body[m]:.2f}＝第 1 根的 {X.body[m] / max(X.body[a], 1e-9) * 100:.0f}%",
                 f"≤ 第 1 根實體 × {K2_STAR_BODY}", X.body[m] <= K2_STAR_BODY * X.body[a])
        and c.ok("星線位置", pos_txt, "星線落在第 1 根實體靠反向的一半", star_pos)
        and third
        and _k2_bodyx(X, b, c, "第 3 根實體明顯", K2_MID_BODY)
        and _k2_trend_ok(X, a, need, c)
    )


def _k2_three_same(X, i, c, need, bull):
    ks = [i - 2, i - 1, i]
    for n_, k in enumerate(ks, 1):
        if not (_k2_red(X, k, c, f"第 {n_} 根收紅") if bull else _k2_green(X, k, c, f"第 {n_} 根收黑")):
            return False
        if not c.ok(f"第 {n_} 根實體夠大", f"{X.body[k]:.2f}＝平均 {X.body[k] / X.avgb[k]:.1f} 倍",
                    f"≥ 平均實體 × {K2_SOLDIER_BODY}", X.body[k] >= K2_SOLDIER_BODY * X.avgb[k]):
            return False
        sh = X.upper[k] if bull else X.lower[k]
        if not c.ok(f"第 {n_} 根{'上' if bull else '下'}影線短", f"{sh:.2f}", f"≤ 實體 × {K2_SHORT_SHADOW}",
                    sh <= K2_SHORT_SHADOW * X.body[k]):
            return False
    for n_ in (1, 2):
        p, k = ks[n_ - 1], ks[n_]
        if bull:
            good = X.C[k] > X.C[p] and X.C[p] >= X.O[k] >= X.O[p]
            thr = "收盤逐根墊高，且開盤落在前一根實體內"
        else:
            good = X.C[k] < X.C[p] and X.C[p] <= X.O[k] <= X.O[p]
            thr = "收盤逐根走低，且開盤落在前一根實體內"
        if not c.ok(f"第 {n_ + 1} 根相對第 {n_} 根", f"開 {X.O[k]:.2f}／收 {X.C[k]:.2f}（前收 {X.C[p]:.2f}）", thr, good):
            return False
    return _k2_trend_ok(X, ks[0], need, c)


def _k2_three_inside(X, i, c, need, bull):
    a, m, b = i - 2, i - 1, i
    if bull:
        ok = _k2_green(X, a, c, "第 1 根收黑") and _k2_red(X, m, c, "第 2 根收紅")
    else:
        ok = _k2_red(X, a, c, "第 1 根收紅") and _k2_green(X, m, c, "第 2 根收黑")
    if not ok or not _k2_bodyx(X, a, c, "第 1 根實體明顯", K2_MID_BODY):
        return False
    if not c.ok("前兩根為母子結構", f"{X.bot[m]:.2f}～{X.top[m]:.2f} 在 {X.bot[a]:.2f}～{X.top[a]:.2f} 內",
                "第 2 根實體收進第 1 根實體內且較小", X.top[m] <= X.top[a] and X.bot[m] >= X.bot[a] and X.body[m] < X.body[a]):
        return False
    if bull:
        third = _k2_red(X, b, c, "第 3 根收紅") and c.ok(
            "第 3 根向上延伸", f"收 {X.C[b]:.2f}（第 2 根收 {X.C[m]:.2f}、第 1 根開 {X.O[a]:.2f}）",
            "收盤 > 第 2 根收盤，且 > 第 1 根開盤", X.C[b] > X.C[m] and X.C[b] > X.O[a])
    else:
        third = _k2_green(X, b, c, "第 3 根收黑") and c.ok(
            "第 3 根向下延伸", f"收 {X.C[b]:.2f}（第 2 根收 {X.C[m]:.2f}、第 1 根開 {X.O[a]:.2f}）",
            "收盤 < 第 2 根收盤，且 < 第 1 根開盤", X.C[b] < X.C[m] and X.C[b] < X.O[a])
    return third and _k2_trend_ok(X, a, need, c)


K2_DETECTORS = {
    "長紅線": lambda X, i, c, n: _k2_long_line(X, i, c, n, True),
    "長黑線": lambda X, i, c, n: _k2_long_line(X, i, c, n, False),
    "錘子線": lambda X, i, c, n: _k2_hammer_shape(X, i, c, False) and _k2_trend_ok(X, i, n, c),
    "吊人線": lambda X, i, c, n: _k2_hammer_shape(X, i, c, False) and _k2_trend_ok(X, i, n, c),
    "倒錘子線": lambda X, i, c, n: _k2_hammer_shape(X, i, c, True) and _k2_trend_ok(X, i, n, c),
    "射擊之星": lambda X, i, c, n: _k2_hammer_shape(X, i, c, True) and _k2_trend_ok(X, i, n, c),
    "多頭吞噬": lambda X, i, c, n: _k2_engulf(X, i, c, n, True),
    "空頭吞噬": lambda X, i, c, n: _k2_engulf(X, i, c, n, False),
    "貫穿線": lambda X, i, c, n: _k2_pierce(X, i, c, n, True),
    "烏雲罩頂": lambda X, i, c, n: _k2_pierce(X, i, c, n, False),
    "多頭母子線": lambda X, i, c, n: _k2_harami(X, i, c, n, True),
    "空頭母子線": lambda X, i, c, n: _k2_harami(X, i, c, n, False),
    "晨星": lambda X, i, c, n: _k2_star(X, i, c, n, True),
    "黃昏之星": lambda X, i, c, n: _k2_star(X, i, c, n, False),
    "紅三兵": lambda X, i, c, n: _k2_three_same(X, i, c, n, True),
    "三隻烏鴉": lambda X, i, c, n: _k2_three_same(X, i, c, n, False),
    "三內升": lambda X, i, c, n: _k2_three_inside(X, i, c, n, True),
    "三內降": lambda X, i, c, n: _k2_three_inside(X, i, c, n, False),
}


def k2_detect(df, n_trend, period, selected=None):
    """掃描整段K線，回傳所有命中的型態：{name, g(幾根), direction, s, e, trend_txt, checks}。
    selected 不為 None 時，只比對被選取的型態（未選取者不會出現在標註、重疊清單或條件比對）。"""
    X = _K2Ctx(df, n_trend, period)
    hits = []
    for i in range(len(df)):
        for name, g, direction, need in K2_PATTERNS:
            if selected is not None and name not in selected:
                continue
            s = i - g + 1
            if s < 1 or np.isnan(X.avgb[s:i + 1]).any() or (X.avgb[s:i + 1] <= 0).any():
                continue
            c = _K2Chk()
            if not K2_DETECTORS[name](X, i, c, need):
                continue
            tr = "—"
            if need is not None:
                cls, ret = X.trend(s)
                tr = f"{_K2_TREND_NAME[cls]} {ret * 100:+.1f}%"
            hits.append({"name": name, "g": g, "direction": direction, "s": s, "e": i, "trend_txt": tr, "checks": c.rows})
    return hits


def k2_cluster(hits):
    """把互相重疊的型態合成一組標註：以根數最多者為主要型態，其餘列為「+N」。"""
    order = sorted(range(len(hits)), key=lambda k: (-hits[k]["g"], hits[k]["s"]))
    taken, clusters = set(), []
    for k in order:
        if k in taken:
            continue
        h = hits[k]
        taken.add(k)
        members = [h]
        for m in range(len(hits)):
            if m not in taken and hits[m]["s"] <= h["e"] and hits[m]["e"] >= h["s"]:
                taken.add(m)
                members.append(hits[m])
        clusters.append({
            "primary": h, "members": members,
            "s": min(x["s"] for x in members), "e": max(x["e"] for x in members),
        })
    clusters.sort(key=lambda x: (x["e"], x["s"]))
    return clusters


def k2_resample(hist, period, tail=400):
    """日K直接用；週K（週五結算）、月K、季K 先各自彙整 OHLC，再重新偵測型態（不是沿用日K的結果）。"""
    d = hist[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Open", "High", "Low", "Close"])
    if period == "日":
        return d.tail(tail) if tail else d
    rules = {"週": ["W-FRI"], "月": ["ME", "M"], "季": ["QE", "Q"]}[period]
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    for rule in rules:
        try:
            r = d.resample(rule).agg(agg)
            break
        except ValueError:
            continue
    return r.dropna(subset=["Open", "Close"])


K2_HORIZONS = (5, 10, 20)


def k2_pattern_stats(df, hits):
    """每種型態在這檔股票歷史上出現後，往後 5／10／20 根的平均報酬與勝率（偏多＝上漲、偏空＝下跌算勝）。"""
    C = df["Close"].to_numpy(float)
    n = len(C)
    by = {}
    for h in hits:
        by.setdefault(h["name"], {"direction": h["direction"], "ends": []})["ends"].append(h["e"])
    rows = {}
    for name, v in by.items():
        row = {"name": name, "direction": v["direction"], "count": len(v["ends"])}
        for hz in K2_HORIZONS:
            rets = np.array([C[e + hz] / C[e] - 1 for e in v["ends"] if e + hz < n and C[e] > 0])
            if len(rets):
                win = (rets > 0).mean() if v["direction"] == "bull" else (rets < 0).mean()
                row[hz] = {"n": int(len(rets)), "avg": float(rets.mean()), "win": float(win)}
            else:
                row[hz] = None
        rows[name] = row
    return rows


def _k2_stat_txt(stats, name, period):
    r = (stats or {}).get(name)
    if not r or not r.get(10):
        return ""
    s = r[10]
    word = "上漲" if r["direction"] == "bull" else "下跌"
    warn = "（樣本少，僅供參考）" if s["n"] < 10 else ""
    return (f"📊 這檔股票歷史上「{name}」共出現 {r['count']} 次，之後 10 根{period}K 內{word}的比例約 "
            f"{s['win'] * 100:.0f}%、平均報酬 {s['avg'] * 100:+.1f}%{warn}")


K2_BT_MAX_WAIT = 5    # 訊號出現後，最多等幾根K線讓掛單成交


K2_BT_MAX_HOLD = 20   # 成交後最多持有幾根K線，到期就以收盤價出場


def k2_backtest(df, hits, fee_disc=6.0):
    """策略回測（只做多）：偏多型態出現後，依「建議進場區」掛單，之後用「型態停損價」與「獲利價」出場。
    規則與畫面上的價位一致（只用訊號當下已知的資料，不偷看未來）：
    進場＝訊號後 5 根內價格碰到進場區就成交；同一根K線同時碰到停損與獲利時，保守地視為先停損；
    跳空時以開盤價成交；持倉期間不重複進場；成本含手續費（買賣各一次）與證交稅 0.3%。"""
    O, H, L, C = (df[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
    n = len(C)
    fee, tax = 0.001425 * fee_disc / 10, 0.003
    prev = np.concatenate([[C[0]], C[:-1]])
    tr = np.maximum(H - L, np.maximum(np.abs(H - prev), np.abs(L - prev)))
    atr = pd.Series(tr).rolling(14).mean().to_numpy()
    res20 = pd.Series(H).rolling(20).max().to_numpy()
    trades, busy = [], -1
    for h in sorted((x for x in hits if x["direction"] == "bull"), key=lambda x: x["e"]):
        e = h["e"]
        if e <= busy or e + 1 >= n or not np.isfinite(atr[e]) or not np.isfinite(res20[e]):
            continue
        lv = k2_hit_levels(df, h)
        lo, hi, stop = lv["lo"], lv["hi"], lv["stop"]
        tgt = res20[e] if res20[e] > hi + 0.5 * atr[e] else hi + 2 * atr[e]
        fill = None
        for j in range(e + 1, min(n, e + 1 + K2_BT_MAX_WAIT)):
            if L[j] <= hi and H[j] >= lo:
                fill = (j, min(max(O[j], lo), hi))
                break
            if L[j] < stop:
                break  # 還沒成交就先跌破停損：取消這張單
        if not fill or fill[1] <= stop or tgt <= fill[1]:
            continue
        j0, entry = fill
        exit_j, exit_px, reason = None, None, "時間到"
        last = min(n - 1, j0 + K2_BT_MAX_HOLD)
        for j in range(j0, last + 1):
            if L[j] <= stop:
                exit_j, exit_px, reason = j, (min(stop, O[j]) if j > j0 else stop), "停損"
                break
            if H[j] >= tgt:
                exit_j, exit_px, reason = j, (max(tgt, O[j]) if j > j0 else tgt), "獲利"
                break
        if exit_j is None:
            exit_j, exit_px = last, C[last]
        cost = entry * (1 + fee)
        net = exit_px * (1 - fee - tax) - cost
        risk = cost - stop * (1 - fee - tax)
        trades.append({
            "name": h["name"], "signal": e, "entry_i": j0, "exit_i": exit_j, "entry": float(entry),
            "exit": float(exit_px), "ret": float(net / cost), "R": float(net / risk) if risk > 0 else 0.0,
            "reason": reason, "bars": int(exit_j - j0 + 1),
        })
        busy = exit_j
    return trades


def k2_bt_summary(trades, df):
    """把交易明細整理成績效：勝率、期望值、獲利因子、最大回檔，並和買進持有比較。"""
    n = len(trades)
    C = df["Close"].to_numpy(float)
    out = {"n": n, "buyhold": float(C[-1] / C[0] - 1)}
    if n == 0:
        return out
    rets = np.array([x["ret"] for x in trades])
    wins, losses = rets[rets > 0], rets[rets <= 0]
    equity = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(np.concatenate([[1.0], equity]))[1:]
    out.update({
        "win": float((rets > 0).mean()), "avg": float(rets.mean()),
        "avg_win": float(wins.mean()) if len(wins) else 0.0, "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "avg_R": float(np.mean([x["R"] for x in trades])),
        "pf": float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else float("inf"),
        "total": float(equity[-1] - 1), "mdd": float(((equity - peak) / peak).min()),
        "bars": float(np.mean([x["bars"] for x in trades])),
        "exposure": float(sum(x["bars"] for x in trades) / len(C)),
        "equity": equity.tolist(),
    })
    return out


def _k2_fmt(ts, period):
    if period == "月":
        return ts.strftime("%Y-%m")
    if period == "季":
        return f"{ts.year}Q{(ts.month - 1) // 3 + 1}"
    return ts.strftime("%Y-%m-%d")


def k2_hit_levels(df, h):
    """依型態的K棒位置，算出「想購買者」的進場區間與停損價（偏多型態），
    或「持有者」的反彈減碼區間與跌破出場價（偏空型態）。回傳 {lo, hi, stop}。"""
    seg = df.iloc[h["s"]:h["e"] + 1]
    O, H, L, C = (seg[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
    mid = lambda k: (O[k] + C[k]) / 2
    low = float(L.min())
    nm = h["name"]
    # 每條規則用 lambda 包起來，只計算命中的那一個（避免單根／兩根型態去讀不存在的第 3 根K棒）
    if h["direction"] == "bull":
        rule = {
            "長紅線": lambda: (mid(0), C[0]),                        # 回測實體中點～收盤
            "錘子線": lambda: (C[0], H[0]),                          # 收盤～錘子線最高價
            "倒錘子線": lambda: (H[0], H[0] * 1.01),                 # 需向上確認：站上最高價
            "多頭吞噬": lambda: (mid(1), C[1]),                      # 第 2 根紅K實體中點～收盤
            "貫穿線": lambda: (mid(1), C[1]),
            "多頭母子線": lambda: (max(O[0], C[0]), max(O[0], C[0]) * 1.01),  # 需向上確認：站上第 1 根實體上緣
            "晨星": lambda: (mid(2), C[2]),
            "紅三兵": lambda: (O[2], C[2]),                          # 第 3 根開盤～收盤
            "三內升": lambda: (C[1], C[2]),                          # 第 2 根收盤～第 3 根收盤
        }[nm]()
        stop = low                                                   # 跌破型態最低價，型態失效
    else:
        rule = {
            "長黑線": lambda: (C[0], mid(0)),                        # 反彈到實體中點
            "吊人線": lambda: (C[0], H[0]),
            "射擊之星": lambda: (C[0], H[0]),
            "空頭吞噬": lambda: (C[1], mid(1)),
            "烏雲罩頂": lambda: (C[1], (O[0] + C[0]) / 2),           # 反彈到第 1 根實體中點
            "空頭母子線": lambda: (C[1], H[1]),
            "黃昏之星": lambda: (C[2], mid(2)),
            "三隻烏鴉": lambda: (C[2], O[2]),                        # 反彈到第 3 根開盤
            "三內降": lambda: (C[2], C[1]),
        }[nm]()
        stop = min(O[0], C[0]) if nm == "空頭母子線" else low        # 母子線以第 1 根實體下緣為轉弱確認
    lo, hi = float(min(rule)), float(max(rule))
    return {"lo": lo, "hi": hi, "stop": float(stop)}


def _k2_range_txt(lo, hi):
    return _pz(lo) if abs(hi - lo) <= 0.005 else f"{_pz(lo)}～{_pz(hi)}"


def _period_trend(df):
    """用週期K線的收盤與 5／10 根均價判斷方向：1 偏多、-1 偏空、0 整理；資料不足回傳 None。"""
    if len(df) < 10:
        return None
    c = df["Close"]
    px, ma5, ma10 = float(c.iloc[-1]), float(c.tail(5).mean()), float(c.tail(10).mean())
    return 1 if px > ma5 > ma10 else -1 if px < ma5 < ma10 else 0


# ---------------------------------------------------------------------------
# 大盤環境（加權指數）：個股訊號在大盤空頭時失敗率較高，所以把大盤方向獨立判斷出來
# ---------------------------------------------------------------------------
MARKET_REGIME_TXT = {
    "bull": ("🔴", "多頭", "大盤站穩月線與季線，個股的進場訊號相對可靠；仍要看個股自己的停損。"),
    "bear": ("🟢", "空頭", "大盤在月線與季線之下，個股多方訊號容易失敗，建議降低部位、拉高進場標準，或等大盤止穩。"),
    "mixed": ("⚪", "盤整／轉折", "大盤多空未明（站上其中一條均線或均線糾結），個股訊號的可信度打折，進場宜保守、停損要設好。"),
}


def market_regime(close):
    """用加權指數收盤序列（舊→新）判斷大盤環境。資料少於 60 筆回傳 None。
    多頭：收盤在月線（20 日均價）與季線（60 日均價）之上，且月線 > 季線；
    空頭：收盤在兩條之下，且月線 < 季線；其餘為盤整／轉折。
    回傳 dict：regime(bull/bear/mixed)、vote(+1/-1/0)、last、ma20、ma60、ret20（近 20 日報酬）、from_high（距近半年高點）、detail（文字）。"""
    c = pd.Series(close).dropna().astype(float)
    if len(c) < 60:
        return None
    last = float(c.iloc[-1])
    ma20, ma60 = float(c.tail(20).mean()), float(c.tail(60).mean())
    a20, a60 = last > ma20, last > ma60
    if a20 and a60 and ma20 > ma60:
        regime = "bull"
    elif (not a20) and (not a60) and ma20 < ma60:
        regime = "bear"
    else:
        regime = "mixed"
    ret20 = last / float(c.iloc[-21]) - 1 if len(c) > 21 else float("nan")
    from_high = last / float(c.tail(120).max()) - 1
    detail = (f"{'站上' if a20 else '跌破'}月線 {ma20:,.0f}、{'站上' if a60 else '跌破'}季線 {ma60:,.0f}，"
              f"月線{'在' if ma20 > ma60 else '低於'}季線{'之上' if ma20 > ma60 else ''}")
    return dict(regime=regime, vote={"bull": 1, "bear": -1, "mixed": 0}[regime], last=last, ma20=ma20, ma60=ma60,
                ret20=ret20, from_high=from_high, detail=detail)


def regime_conflict(verdict, regime):
    """個股預測（bull/bear/neutral）與大盤環境衝突時回傳提醒文字，否則回傳空字串。"""
    if not regime:
        return ""
    if verdict == "bull" and regime["regime"] == "bear":
        return "⚠️ 預測偏多，但大盤目前是空頭環境：逆勢做多的失敗率較高，建議減碼或等大盤止穩再進場。"
    if verdict == "bull" and regime["regime"] == "mixed":
        return "ℹ️ 預測偏多，但大盤多空未明：可以進場，但部位宜保守、停損要設好。"
    if verdict == "bear" and regime["regime"] == "bull":
        return "ℹ️ 預測偏空，但大盤目前是多頭環境：個股轉弱時仍要出場，但可留意是否只是短線回檔。"
    return ""


# ---------------------------------------------------------------------------
# 購買指標燈號：基本面／籌碼 與 技術面 分開（技術面放在 K線分析頁籤）
# ---------------------------------------------------------------------------
def fund_beacon_items(eps_last4, inst_table):
    """基本面／籌碼燈號：[(指標名稱, 狀態, 說明)]，狀態 True=達到、False=未達到、None=資料不足。"""
    items = []
    if eps_last4 is not None and len(eps_last4) >= 4:
        last4 = list(eps_last4.head(4).values)
        ok = all(v > 0 for v in last4)
        items.append(("連續四季 EPS 為正", ok, "近四季（新→舊）：" + "、".join(f"{v:.2f}" for v in last4)))
    else:
        items.append(("連續四季 EPS 為正", None, "取得的季度 EPS 不足四季"))
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
    return items


def tech_beacon_items(daily_res, monthly_res):
    """技術面燈號：日K／月K 趨勢偏多是否多於偏空，以及近期突破／跌破月線。"""
    items = []
    for name, res in (("日K趨勢 偏多 > 偏空", daily_res), ("月K趨勢 偏多 > 偏空", monthly_res)):
        if res is None:
            items.append((name, None, "資料不足，無法判讀"))
        else:
            signals = res[0]
            bull = sum(1 for _, _, sc in signals if sc > 0)
            bear = sum(1 for _, _, sc in signals if sc < 0)
            items.append((name, bull > bear, f"偏多 {bull} 項、偏空 {bear} 項"))
    items.append(("近期突破／跌破月線", *_ma20_cross_state(daily_res)))
    return items
