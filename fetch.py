#!/usr/bin/env python3
"""Fetch Au99.99 daily prices from Shanghai Gold Exchange, USD index, US CPI, or real 10-year yield."""

import argparse

import akshare as ak
import pandas as pd
import requests


def fetch_gold(symbol: str = "Au99.99", months: bool = False, output: str = "sge_gold.csv"):
    df = ak.spot_hist_sge(symbol)
    df["date"] = pd.to_datetime(df["date"])

    if months:
        df = df.set_index("date").resample("ME").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
        }).dropna().reset_index()
        df["month"] = df["date"].dt.strftime("%Y-%m")

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
        df = df.set_index("date").resample("ME").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
        }).dropna().reset_index()
        df["month"] = df["date"].dt.strftime("%Y-%m")

    df.to_csv(output, index=False)
    print(f"Saved {len(df)} rows to {output}")


def fetch_us_cpi(output: str = "us_cpi_monthly.csv"):
    df = ak.macro_usa_cpi_yoy()
    df = df.rename(columns={"时间": "date", "现值": "cpi_yoy_pct"})
    df["date"] = pd.to_datetime(df["date"])
    df[["date", "cpi_yoy_pct"]].to_csv(output, index=False)
    print(f"Saved {len(df)} rows to {output}")


def fetch_us_real_yield(output: str = "us_real_yield_monthly.csv"):
    url = (
        "https://fred.stlouisfed.org/graph/fredgraph.csv"
        "?id=DFII10&cosd=2003-01-02&coed=9999-12-31"
    )
    df = pd.read_csv(url)
    df.columns = ["date", "real_yield_pct"]
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").resample("ME").last().dropna().reset_index()
    df["month"] = df["date"].dt.strftime("%Y-%m")
    df[["date", "month", "real_yield_pct"]].to_csv(output, index=False)
    print(f"Saved {len(df)} rows to {output}")


def list_gold_symbols():
    syms = ak.spot_symbol_table_sge()
    print(syms.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch SGE gold prices, USD index, US CPI, or US real 10-year yield"
    )
    parser.add_argument("--symbol", default="Au99.99", help="Gold contract symbol (default: Au99.99)")
    parser.add_argument("--symbols", action="store_true", help="List all available gold symbols")
    parser.add_argument("--monthly", action="store_true", help="Resample to monthly OHLC")
    parser.add_argument("-o", "--output", default="", help="Output CSV file")
    parser.add_argument("-u", "--usd-index", action="store_true", help="Fetch USD index instead of gold")
    parser.add_argument("--us-cpi", action="store_true", help="Fetch monthly US CPI YoY")
    parser.add_argument("--real-yield", action="store_true", help="Fetch monthly US real 10-year yield (TIPS)")
    args = parser.parse_args()

    if args.symbols:
        list_gold_symbols()
    elif args.us_cpi:
        output = args.output or "us_cpi_monthly.csv"
        fetch_us_cpi(output)
    elif args.real_yield:
        output = args.output or "us_real_yield_monthly.csv"
        fetch_us_real_yield(output)
    elif args.usd_index:
        output = args.output or "usd_index.csv"
        fetch_usd_index(args.monthly, output)
    else:
        output = args.output or "sge_gold.csv"
        fetch_gold(args.symbol, args.monthly, output)
