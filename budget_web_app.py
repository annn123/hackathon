"""Local web interface for the A-company budget regression example.

Run with: python budget_web_app.py
Only Python's standard library is required. The page is served at localhost:8765.
"""

from __future__ import annotations

import csv
import json
import random
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "A_monthly_data.csv"
HOST, PORT = "127.0.0.1", 8765


def read_rows(csv_text: str) -> list[dict]:
    reader = csv.DictReader(csv_text.lstrip("\ufeff").splitlines())
    required = {"西元年", "月份", "月預算", "月支出_模擬"}
    if not reader.fieldnames or not required.issubset(reader.fieldnames):
        raise ValueError("CSV 必須包含：西元年、月份、月預算、月支出_模擬")
    rows = []
    for row in reader:
        try:
            budget = float(row["月預算"])
            spend = float(row["月支出_模擬"])
            year = int(float(row["西元年"]))
            month = int(float(row["月份"]))
        except (TypeError, ValueError, KeyError):
            continue
        if budget <= 0 or not 1 <= month <= 12:
            continue
        rows.append({"year": year, "month": month, "budget": budget,
                     "spend": spend, "rate": spend / budget})
    if not rows:
        raise ValueError("CSV 中沒有有效的月資料。")
    return rows


def solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    n = len(vector)
    augmented = [matrix[i][:] + [vector[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(augmented[row][col]))
        if abs(augmented[pivot][col]) < 1e-12:
            raise ValueError("目前選擇的變數無法估計，請開啟月份效果或年度趨勢後重算。")
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        divisor = augmented[col][col]
        augmented[col] = [value / divisor for value in augmented[col]]
        for row in range(n):
            if row == col:
                continue
            factor = augmented[row][col]
            augmented[row] = [augmented[row][j] - factor * augmented[col][j]
                              for j in range(n + 1)]
    return [augmented[i][-1] for i in range(n)]


def make_features(year_index: int, month: int, use_trend: bool, use_month: bool) -> list[float]:
    features = [1.0]
    if use_trend:
        features.append(float(year_index))
    if use_month:
        features.extend(1.0 if month == m else 0.0 for m in range(2, 13))
    return features


def analyze(csv_text: str, use_trend: bool, use_month: bool) -> dict:
    rows = read_rows(csv_text)
    years = sorted({row["year"] for row in rows})
    year_index = {year: i for i, year in enumerate(years)}
    xs = [make_features(year_index[row["year"]], row["month"], use_trend, use_month)
          for row in rows]
    ys = [row["rate"] for row in rows]
    p = len(xs[0])
    xtx = [[sum(row[i] * row[j] for row in xs) for j in range(p)] for i in range(p)]
    xty = [sum(row[i] * y for row, y in zip(xs, ys)) for i in range(p)]
    beta = solve(xtx, xty)
    fitted = [sum(a * b for a, b in zip(x, beta)) for x in xs]
    residuals = [actual - pred for actual, pred in zip(ys, fitted)]
    mean_rate = sum(ys) / len(ys)
    sse = sum(error * error for error in residuals)
    sst = sum((y - mean_rate) ** 2 for y in ys)
    forecasts = []
    next_year = max(years) + 1
    for month in range(1, 13):
        x = make_features(len(years), month, use_trend, use_month)
        rate = sum(a * b for a, b in zip(x, beta))
        forecasts.append({"month": month, "rate": max(0.0, min(1.0, rate))})
    annual = []
    for year in years:
        year_rows = [row for row in rows if row["year"] == year]
        annual.append({"year": year, "rate": sum(r["rate"] for r in year_rows) / len(year_rows)})
    month_summary = []
    for month in range(1, 13):
        month_rows = [row for row in rows if row["month"] == month]
        month_summary.append({"month": month,
                              "actual": sum(r["rate"] for r in month_rows) / len(month_rows),
                              "forecast": forecasts[month - 1]["rate"]})
    result_rows = []
    for row, fit in zip(rows, fitted):
        result_rows.append({**row, "fitted": fit})
    coefficient_names = ["截距"]
    if use_trend:
        coefficient_names.append("年度趨勢（每年）")
    if use_month:
        coefficient_names.extend(f"{m}月相對1月效果" for m in range(2, 13))
    return {
        "rowCount": len(rows), "years": years, "nextYear": next_year,
        "simpleAverage": mean_rate,
        "weightedRate": sum(r["spend"] for r in rows) / sum(r["budget"] for r in rows),
        "rmse": (sse / len(rows)) ** 0.5, "r2": 1 - sse / sst if sst else 1.0,
        "coefficients": [{"name": name, "value": value}
                         for name, value in zip(coefficient_names, beta)],
        "trend": beta[1] if use_trend else None,
        "forecasts": forecasts, "annual": annual, "monthSummary": month_summary,
        "rows": result_rows,
    }


def analyze_overrun(csv_text: str, future_monthly_budget: float, simulations: int = 5000) -> dict:
    reader = csv.DictReader(csv_text.lstrip("\ufeff").splitlines())
    aliases = {
        "month": ("月份", "月", "month"),
        "budget": ("月預算", "預算", "budget", "monthly_budget"),
        "actual": ("實際支出", "支出", "actual_spend", "actual"),
    }
    fields = {}
    names = reader.fieldnames or []
    for key, candidates in aliases.items():
        fields[key] = next((candidate for candidate in candidates if candidate in names), None)
    if not all(fields.values()):
        raise ValueError("六個月 CSV 必須有欄位：月份、月預算、實際支出。")
    actual_rows = []
    for row in reader:
        if not row.get(fields["month"], "").strip():
            continue
        try:
            month = int(float(row[fields["month"]]))
            budget = float(row[fields["budget"]])
            actual = float(row[fields["actual"]])
        except (TypeError, ValueError):
            raise ValueError("月份、月預算、實際支出都必須填入數字。")
        if budget <= 0 or actual < 0 or not 1 <= month <= 6:
            raise ValueError("請提供 1–6 月資料，月預算需大於 0，實際支出不可小於 0。")
        actual_rows.append({"month": month, "budget": budget, "actual": actual})
    actual_rows.sort(key=lambda row: row["month"])
    if len(actual_rows) != 6 or [row["month"] for row in actual_rows] != list(range(1, 7)):
        raise ValueError("請剛好匯入 1–6 月各一筆資料（共六列）。")
    try:
        future_monthly_budget = float(future_monthly_budget)
    except (TypeError, ValueError):
        raise ValueError("請輸入後六個月的預估月預算。")
    if future_monthly_budget <= 0:
        raise ValueError("後六個月的預估月預算需大於 0。")

    historical = read_rows(DATA_FILE.read_text(encoding="utf-8-sig"))
    rates = [row["rate"] for row in historical]
    history_mean = sum(rates) / len(rates)
    ytd_budget = sum(row["budget"] for row in actual_rows)
    ytd_spend = sum(row["actual"] for row in actual_rows)
    ytd_rate = ytd_spend / ytd_budget
    annual_budget = ytd_budget + future_monthly_budget * 6
    rng = random.Random(20261031)
    totals = []
    future_spend_by_month = [0.0] * 6
    for _ in range(simulations):
        total = ytd_spend
        for i in range(6):
            # Bootstrap historical rates as residual scenarios around their 90% baseline.
            projected_rate = max(0.0, ytd_rate + rng.choice(rates) - history_mean)
            month_spend = future_monthly_budget * projected_rate
            future_spend_by_month[i] += month_spend
            total += month_spend
        totals.append(total)
    totals.sort()
    overrun_count = sum(total > annual_budget for total in totals)

    def quantile(q: float) -> float:
        return totals[min(len(totals) - 1, int(q * (len(totals) - 1)))]

    min_total, max_total = totals[0], totals[-1]
    bin_count = 24
    span = max_total - min_total
    histogram = [0] * bin_count
    for total in totals:
        index = min(bin_count - 1, int((total - min_total) / span * bin_count)) if span else 0
        histogram[index] += 1
    return {
        "months": actual_rows, "ytdBudget": ytd_budget, "ytdSpend": ytd_spend,
        "ytdRate": ytd_rate, "futureMonthlyBudget": future_monthly_budget,
        "annualBudget": annual_budget, "expectedYearEnd": sum(totals) / len(totals),
        "p10": quantile(0.10), "median": quantile(0.50), "p90": quantile(0.90),
        "overrunProbability": overrun_count / len(totals), "simulations": simulations,
        "futureMonthlyForecast": [value / simulations / future_monthly_budget
                                  for value in future_spend_by_month],
        "histogram": histogram, "histogramMin": min_total, "histogramMax": max_total,
        "historyMeanRate": history_mean,
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/" or path == "/index.html":
            file_path = ROOT / "index.html"
            if file_path.exists():
                self._send(200, file_path.read_bytes(), "text/html; charset=utf-8")
                return
        if path == "/A_data.js":
            file_path = ROOT / "A_data.js"
            if file_path.exists():
                self._send(200, file_path.read_bytes(), "application/javascript; charset=utf-8")
                return
        if path == "/api/data":
            try:
                data = DATA_FILE.read_text(encoding="utf-8-sig")
                self._send(200, data.encode("utf-8"), "text/csv; charset=utf-8")
                return
            except OSError as exc:
                self._send(500, str(exc).encode(), "text/plain; charset=utf-8")
                return
        self._send(404, "Not found".encode(), "text/plain; charset=utf-8")

    def do_POST(self) -> None:
        route = urlparse(self.path).path
        if route not in ("/api/model", "/api/overrun"):
            self._send(404, b"Not found", "text/plain; charset=utf-8")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            text = payload.get("csvText")
            if route == "/api/overrun":
                result = analyze_overrun(text or "", payload.get("futureMonthlyBudget"))
            else:
                if text is None:
                    text = DATA_FILE.read_text(encoding="utf-8-sig")
                result = analyze(text, bool(payload.get("useTrend", True)),
                                 bool(payload.get("useMonth", True)))
            body = json.dumps(result, ensure_ascii=False).encode("utf-8")
            self._send(200, body, "application/json; charset=utf-8")
        except Exception as exc:
            body = json.dumps({"error": str(exc)}, ensure_ascii=False).encode("utf-8")
            self._send(400, body, "application/json; charset=utf-8")

    def log_message(self, fmt: str, *args) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}")


if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"預算分析網頁已啟動：http://{HOST}:{PORT}")
    print("按 Ctrl+C 可停止。")
    threading.Timer(0.8, lambda: webbrowser.open(f"http://{HOST}:{PORT}/")).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n網頁伺服器已停止。")
        server.server_close()
