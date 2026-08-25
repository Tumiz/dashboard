#!/usr/bin/env python3
"""Fetch economic/financial series and update the dashboard_monthly.csv dashboard data."""

import argparse
import io
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as TaskTimeoutError

import akshare as ak
import pandas as pd
import requests

TASK_TIMEOUT = 180


def http_get(url: str, timeout: int = 30) -> bytes:
    """Download a URL with an explicit timeout so a stalled endpoint cannot hang forever."""
    resp = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0", "Accept": "*/*"},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.content


def update_dashboard(month_df: pd.DataFrame, output: str = "dashboard_monthly.csv"):
    """Merge a monthly series (month + value columns) into the wide dashboard CSV."""
    if os.path.exists(output):
        merged = pd.read_csv(output).set_index("month")
    else:
        merged = pd.DataFrame(index=pd.Index([], name="month"))
    upd = month_df.set_index("month")
    merged = merged.reindex(merged.index.union(upd.index)).sort_index()
    for col in upd.columns:
        if col in merged.columns:
            merged[col] = upd[col].reindex(merged.index).combine_first(merged[col])
        else:
            merged[col] = upd[col].reindex(merged.index)
    merged = merged.reset_index()
    merged = merged[["month"] + [c for c in merged.columns if c != "month"]]
    merged.to_csv(output, index=False)
    print(f"Updated {len(merged)} rows in {output}")


def fetch_gold(symbol: str = "Au99.99", months: bool = False, output: str = "sge_gold.csv"):
    df = ak.spot_hist_sge(symbol)
    df["date"] = pd.to_datetime(df["date"])

    if months:
        df = df.set_index("date").resample("ME").mean(numeric_only=True).dropna().reset_index()
        df["month"] = df["date"].dt.strftime("%Y-%m")
        update_dashboard(df[["month", "close"]].rename(columns={"close": "gold_close"}).round(1))
        return

    df = df.drop(columns=["date"]).round(1)
    df.to_csv(output, index=False)
    print(f"Saved {len(df)} rows to {output}")


def fetch_usd_index(months: bool = False, output: str = "usd_index.csv"):
    df = ak.index_global_hist_em(symbol="美元指数")
    df = df.rename(columns={
        "日期": "date",
        "今开": "open",
        "最新价": "close",
        "最高": "high",
        "最低": "low",
    })
    df["date"] = pd.to_datetime(df["date"])
    df = df[["date", "open", "high", "low", "close"]]

    if months:
        df = df.set_index("date").resample("ME").mean(numeric_only=True).dropna().reset_index()
        df["month"] = df["date"].dt.strftime("%Y-%m")
        update_dashboard(df[["month", "close"]].rename(columns={"close": "usd_close"}).round(1))
        return

    df = df.drop(columns=["date"]).round(1)
    df.to_csv(output, index=False)
    print(f"Saved {len(df)} rows to {output}")


def fetch_us_cpi():
    df = ak.macro_usa_cpi_yoy()
    df = df.rename(columns={"时间": "date", "现值": "cpi_yoy_pct"})
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.strftime("%Y-%m")
    update_dashboard(df[["month", "cpi_yoy_pct"]].round(1))


def fetch_us_real_yield():
    url = (
        "https://fred.stlouisfed.org/graph/fredgraph.csv"
        "?id=DFII10&cosd=2003-01-02&coed=9999-12-31"
    )
    df = pd.read_csv(io.BytesIO(http_get(url)))
    df.columns = ["date", "real_yield_pct"]
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").resample("ME").mean(numeric_only=True).dropna().reset_index()
    df["month"] = df["date"].dt.strftime("%Y-%m")
    update_dashboard(df[["month", "real_yield_pct"]].round(1))


def fetch_gpr():
    url = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"
    df = pd.read_excel(io.BytesIO(http_get(url, timeout=120)))
    df = df[["month", "GPR"]].copy()
    df = df.rename(columns={"month": "date", "GPR": "gpr"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["gpr"]).reset_index(drop=True)
    df["month"] = df["date"].dt.strftime("%Y-%m")
    update_dashboard(df[["month", "gpr"]].round(1))


def fetch_btc():
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CBBTCUSD"
    df = pd.read_csv(io.BytesIO(http_get(url)))
    df.columns = ["date", "btc_usd"]
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").resample("ME").mean(numeric_only=True).dropna().reset_index()
    df["month"] = df["date"].dt.strftime("%Y-%m")
    update_dashboard(df[["month", "btc_usd"]].round(1))


def fetch_aux():
    df = ak.futures_foreign_hist(symbol="XAU")
    df = df[["date", "close"]].rename(columns={"close": "aux_usd"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").resample("ME").mean(numeric_only=True).dropna().reset_index()
    df["month"] = df["date"].dt.strftime("%Y-%m")
    update_dashboard(df[["month", "aux_usd"]].round(1))


def fetch_xag():
    df = ak.futures_foreign_hist(symbol="XAG")
    df = df[["date", "close"]].rename(columns={"close": "xag_usd"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").resample("ME").mean(numeric_only=True).dropna().reset_index()
    df["month"] = df["date"].dt.strftime("%Y-%m")
    update_dashboard(df[["month", "xag_usd"]].round(1))


def fetch_au9999():
    df = ak.spot_hist_sge("Au99.99")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").resample("ME").mean(numeric_only=True).dropna().reset_index()
    df["month"] = df["date"].dt.strftime("%Y-%m")
    update_dashboard(df[["month", "close"]].rename(columns={"close": "au9999_close"}).round(1))


def fetch_fed_rates():
    url = (
        "https://fred.stlouisfed.org/graph/fredgraph.csv"
        "?id=FEDFUNDS&cosd=1954-07-01&coed=9999-12-31"
    )
    df = pd.read_csv(io.BytesIO(http_get(url)))
    df.columns = ["date", "fed_rate_pct"]
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").resample("ME").mean(numeric_only=True).dropna().reset_index()
    df["month"] = df["date"].dt.strftime("%Y-%m")
    update_dashboard(df[["month", "fed_rate_pct"]].round(1))


def fetch_nasdaq():
    url = (
        "https://fred.stlouisfed.org/graph/fredgraph.csv"
        "?id=NASDAQCOM&cosd=1971-02-05&coed=9999-12-31"
    )
    df = pd.read_csv(io.BytesIO(http_get(url)))
    df.columns = ["date", "nasdaq_close"]
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").resample("ME").mean(numeric_only=True).dropna().reset_index()
    df["month"] = df["date"].dt.strftime("%Y-%m")
    update_dashboard(df[["month", "nasdaq_close"]].round(1))


def fetch_twexb():
    url = (
        "https://fred.stlouisfed.org/graph/fredgraph.csv"
        "?id=DTWEXBGS&cosd=2006-01-02&coed=9999-12-31"
    )
    df = pd.read_csv(io.BytesIO(http_get(url)))
    df.columns = ["date", "twexb"]
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").resample("ME").mean(numeric_only=True).dropna().reset_index()
    df["month"] = df["date"].dt.strftime("%Y-%m")
    update_dashboard(df[["month", "twexb"]].round(2))


def fetch_usd_cny():
    url = (
        "https://fred.stlouisfed.org/graph/fredgraph.csv"
        "?id=DEXCHUS&cosd=1981-01-02&coed=9999-12-31"
    )
    df = pd.read_csv(io.BytesIO(http_get(url)))
    df.columns = ["date", "usd_cny"]
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").resample("ME").mean(numeric_only=True).dropna().reset_index()
    df["month"] = df["date"].dt.strftime("%Y-%m")
    update_dashboard(df[["month", "usd_cny"]].round(2))


def fetch_global_index(symbol: str, column: str):
    df = ak.index_global_hist_em(symbol=symbol)
    df = df.rename(columns={"日期": "date", "最新价": column})
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").resample("ME").mean(numeric_only=True).dropna().reset_index()
    df["month"] = df["date"].dt.strftime("%Y-%m")
    update_dashboard(df[["month", column]].round(1))


def fetch_cac40():
    fetch_global_index("法国CAC40", "cac40_close")


def fetch_dax():
    fetch_global_index("德国DAX30", "dax_close")


def fetch_sp500():
    fetch_global_index("标普500", "sp500_close")


def fetch_nikkei():
    fetch_global_index("日经225", "nikkei_close")


def fetch_shanghai():
    df = ak.stock_zh_index_daily(symbol="sh000001")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").resample("ME").mean(numeric_only=True).dropna().reset_index()
    df["month"] = df["date"].dt.strftime("%Y-%m")
    update_dashboard(df[["month", "close"]].rename(columns={"close": "shanghai_close"}).round(1))


def fetch_kospi():
    fetch_global_index("韩国KOSPI", "kospi_close")


def fetch_tencent():
    df = ak.stock_hk_daily(symbol="00700")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").resample("ME").mean(numeric_only=True).dropna().reset_index()
    df["month"] = df["date"].dt.strftime("%Y-%m")
    update_dashboard(df[["month", "close"]].rename(columns={"close": "tencent_close"}).round(2))


def fetch_cb_demand():
    url = "https://fsapi.gold.org/api/v11/charts/supply-and-demand/42"
    data = json.loads(http_get(url, timeout=60))
    qdata = None
    qcats = None
    for chart_type, chart_data in data["chartData"].items():
        if chart_type == "Demand_Quarterly":
            for series in chart_data["series"]:
                if series["name"] == "Central banks":
                    qdata = series["data"]
                    qcats = chart_data["categories"]
                    break
    if qdata is None:
        raise ValueError("Central banks data not found in API response")
    rows = []
    for cat, val in zip(qcats, qdata):
        parts = cat.replace("'", "").split()
        year = int(parts[1]) + 2000
        q = int(parts[0][1])
        end_month = q * 3
        start_month = end_month - 2
        for m in range(start_month, end_month + 1):
            month_str = f"{year}-{m:02d}"
            rows.append({"month": month_str, "cb_demand": round(val, 1)})
    update_dashboard(pd.DataFrame(rows))


def update_all(symbol: str, timeout: int = TASK_TIMEOUT):
    """Run every fetcher to refresh all series in dashboard_monthly.csv."""
    tasks = [
        ("gold", fetch_gold, (symbol,), {"months": True}),
        ("usd_index", fetch_usd_index, (), {"months": True}),
        ("us_cpi", fetch_us_cpi, (), {}),
        ("real_yield", fetch_us_real_yield, (), {}),
        ("gpr", fetch_gpr, (), {}),
        ("btc", fetch_btc, (), {}),
        ("aux", fetch_aux, (), {}),
        ("xag", fetch_xag, (), {}),
        ("au9999", fetch_au9999, (), {}),
        ("fed_rates", fetch_fed_rates, (), {}),
        ("nasdaq", fetch_nasdaq, (), {}),
        ("usd_cny", fetch_usd_cny, (), {}),
        ("twexb", fetch_twexb, (), {}),
        ("tencent", fetch_tencent, (), {}),
        ("sp500", fetch_sp500, (), {}),
        ("nikkei", fetch_nikkei, (), {}),
        ("dax", fetch_dax, (), {}),
        ("cac40", fetch_cac40, (), {}),
        ("kospi", fetch_kospi, (), {}),
        ("shanghai", fetch_shanghai, (), {}),
        ("cb_demand", fetch_cb_demand, (), {}),
    ]
    failures = []
    ctx = multiprocessing.get_context("fork")
    for name, func, args, kwargs in tasks:
        print(f"[{name}] fetching ...", flush=True)
        executor = ProcessPoolExecutor(max_workers=1, mp_context=ctx)
        try:
            executor.submit(func, *args, **kwargs).result(timeout=timeout)
            print(f"[{name}] done", flush=True)
        except TaskTimeoutError:
            failures.append(name)
            print(f"[{name}] timed out after {timeout}s", flush=True)
        except Exception as exc:
            failures.append(name)
            print(f"[{name}] failed: {exc}", flush=True)
        finally:
            for proc in list(getattr(executor, "_processes", {}).values()):
                if proc.is_alive():
                    proc.terminate()
            executor.shutdown(wait=False)
    if failures:
        print(f"Finished with failures: {', '.join(failures)}")
    else:
        print("All series updated successfully")


def list_gold_symbols():
    syms = ak.spot_symbol_table_sge()
    print(syms.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch series and update dashboard_monthly.csv (or write daily files for gold/USD index)"
    )
    parser.add_argument("--all", action="store_true", help="Update all series in dashboard_monthly.csv")
    parser.add_argument("--symbol", default="Au99.99", help="Gold contract symbol (default: Au99.99)")
    parser.add_argument("--symbols", action="store_true", help="List all available gold symbols")
    parser.add_argument("--monthly", action="store_true", help="Resample gold/USD index to monthly and update dashboard")
    parser.add_argument("-o", "--output", default="", help="Output CSV file (daily gold/USD index only)")
    parser.add_argument("-u", "--usd-index", action="store_true", help="Fetch USD index instead of gold")
    parser.add_argument("--us-cpi", action="store_true", help="Fetch monthly US CPI YoY")
    parser.add_argument("--real-yield", action="store_true", help="Fetch monthly US real 10-year yield (TIPS)")
    parser.add_argument("--gpr", action="store_true", help="Fetch monthly Geopolitical Risk Index")
    parser.add_argument("--btc", action="store_true", help="Fetch monthly BTC/USD price")
    parser.add_argument("--aux", action="store_true", help="Fetch monthly XAU/USD (gold spot) price")
    parser.add_argument("--xag", action="store_true", help="Fetch monthly XAG/USD (silver spot) price")
    parser.add_argument("--au9999", action="store_true", help="Fetch monthly SGE Au99.99 gold price (CNY/g)")
    parser.add_argument("--cb-demand", action="store_true", help="Fetch quarterly central-bank gold demand (tonnes)")
    parser.add_argument("--nasdaq", action="store_true", help="Fetch monthly NASDAQ Composite index")
    parser.add_argument("--fed-rates", action="store_true", help="Fetch monthly Fed Funds rate")
    parser.add_argument("--usd-cny", action="store_true", help="Fetch monthly USD/CNY exchange rate")
    parser.add_argument("--twexb", action="store_true", help="Fetch monthly Broad Trade-Weighted USD Index (FRED DTWEXBGS)")
    parser.add_argument("--tencent", action="store_true", help="Fetch monthly Tencent HK (0700.HK) stock price")
    parser.add_argument("--cac40", action="store_true", help="Fetch monthly CAC 40 index")
    parser.add_argument("--dax", action="store_true", help="Fetch monthly DAX index")
    parser.add_argument("--sp500", action="store_true", help="Fetch monthly S&P 500 index")
    parser.add_argument("--nikkei", action="store_true", help="Fetch monthly Nikkei 225 index")
    parser.add_argument("--shanghai", action="store_true", help="Fetch monthly Shanghai Composite index")
    parser.add_argument("--kospi", action="store_true", help="Fetch monthly KOSPI index")
    args = parser.parse_args()

    if args.symbols:
        list_gold_symbols()
    elif args.all:
        update_all(args.symbol)
    elif args.cac40:
        fetch_cac40()
    elif args.dax:
        fetch_dax()
    elif args.sp500:
        fetch_sp500()
    elif args.nikkei:
        fetch_nikkei()
    elif args.shanghai:
        fetch_shanghai()
    elif args.kospi:
        fetch_kospi()
    elif args.nasdaq:
        fetch_nasdaq()
    elif args.fed_rates:
        fetch_fed_rates()
    elif args.cb_demand:
        fetch_cb_demand()
    elif args.btc:
        fetch_btc()
    elif args.aux:
        fetch_aux()
    elif args.xag:
        fetch_xag()
    elif args.au9999:
        fetch_au9999()
    elif args.gpr:
        fetch_gpr()
    elif args.us_cpi:
        fetch_us_cpi()
    elif args.real_yield:
        fetch_us_real_yield()
    elif args.usd_index:
        fetch_usd_index(args.monthly, args.output or "usd_index.csv")
    elif args.tencent:
        fetch_tencent()
    elif args.usd_cny:
        fetch_usd_cny()
    elif args.twexb:
        fetch_twexb()
    else:
        fetch_gold(args.symbol, args.monthly, args.output or "sge_gold.csv")
