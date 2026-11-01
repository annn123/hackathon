"""A company budget execution-rate example: average + trend/month regression.

Run: python a_budget_model.py
Reads A_monthly_data.csv using Python's built-in csv module.
All amounts and rates are taken from the source workbook; monthly spends are simulated.
"""

from __future__ import annotations

import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "A_monthly_data.csv"
MONTHLY_OUTPUT = BASE_DIR / "A_model_monthly_results.csv"
COEF_OUTPUT = BASE_DIR / "A_model_coefficients.csv"


def solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Solve a small square system with Gaussian elimination and pivoting."""
    n = len(vector)
    a = [row[:] + [vector[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(a[row][col]))
        if abs(a[pivot][col]) < 1e-12:
            raise ValueError("迴歸設計矩陣不可逆，請檢查輸入欄位。")
        a[col], a[pivot] = a[pivot], a[col]
        scale = a[col][col]
        a[col] = [value / scale for value in a[col]]
        for row in range(n):
            if row == col:
                continue
            factor = a[row][col]
            a[row] = [a[row][j] - factor * a[col][j] for j in range(n + 1)]
    return [a[i][-1] for i in range(n)]


def fit_ols(features: list[list[float]], outcomes: list[float]) -> tuple[list[float], float, float]:
    """Estimate OLS coefficients, in-sample RMSE and R-squared."""
    p = len(features[0])
    xtx = [[sum(row[i] * row[j] for row in features) for j in range(p)] for i in range(p)]
    xty = [sum(row[i] * y for row, y in zip(features, outcomes)) for i in range(p)]
    beta = solve_linear_system(xtx, xty)
    predictions = [sum(x * b for x, b in zip(row, beta)) for row in features]
    residuals = [y - pred for y, pred in zip(outcomes, predictions)]
    mean_y = sum(outcomes) / len(outcomes)
    sse = sum(err * err for err in residuals)
    sst = sum((y - mean_y) ** 2 for y in outcomes)
    return beta, (sse / len(outcomes)) ** 0.5, (1 - sse / sst if sst else 1.0)


def make_features(year_index: int, month: int) -> list[float]:
    """Intercept + linear year trend + month dummies (January is baseline)."""
    return [1.0, float(year_index)] + [1.0 if month == m else 0.0 for m in range(2, 13)]


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"找不到輸入檔：{INPUT_FILE}")

    rows = []
    with INPUT_FILE.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"西元年", "月份", "月預算", "月支出_模擬", "執行率"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"CSV 必須包含欄位：{', '.join(sorted(required))}")
        for row in reader:
            rows.append({"year": int(row["西元年"]), "month": int(row["月份"]),
                         "budget": float(row["月預算"]), "spend": float(row["月支出_模擬"]),
                         "rate": float(row["執行率"])})
    if not rows:
        raise ValueError("A 檔沒有可用的年度/月資料。")

    years = sorted({row["year"] for row in rows})
    year_to_index = {year: i for i, year in enumerate(years)}
    features = [make_features(year_to_index[row["year"]], row["month"]) for row in rows]
    outcomes = [row["rate"] for row in rows]
    beta, rmse, r2 = fit_ols(features, outcomes)
    mean_rate = sum(outcomes) / len(outcomes)
    weighted_rate = sum(row["spend"] for row in rows) / sum(row["budget"] for row in rows)

    # One-year-ahead forecast: all 10 years train the model; 2026 follows index 10.
    next_year = max(years) + 1
    forecasts = []
    for month in range(1, 13):
        x = make_features(len(years), month)
        predicted_rate = sum(a * b for a, b in zip(x, beta))
        predicted_rate = max(0.0, min(1.0, predicted_rate))  # execution rate is bounded for display
        forecasts.append((month, predicted_rate))

    with MONTHLY_OUTPUT.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["年份", "月份", "月預算", "月支出(模擬)", "執行率", "模型擬合執行率"])
        for row, x in zip(rows, features):
            fitted = sum(a * b for a, b in zip(x, beta))
            writer.writerow([row["year"], row["month"], f"{row['budget']:.6f}",
                             f"{row['spend']:.6f}", f"{row['rate']:.8f}", f"{fitted:.8f}"])
        for month, rate in forecasts:
            writer.writerow([next_year, month, "", "", "", f"預測 {rate:.8f}"])

    coef_names = ["截距(1月、首年)", "年度趨勢(每年)" ] + [f"月份效果({m}月相對1月)" for m in range(2, 13)]
    with COEF_OUTPUT.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["參數", "估計值"])
        writer.writerows((name, f"{value:.10f}") for name, value in zip(coef_names, beta))
        writer.writerow(["整體平均月執行率", f"{mean_rate:.10f}"])
        writer.writerow(["加權執行率(總支出/總預算)", f"{weighted_rate:.10f}"])
        writer.writerow(["樣本內RMSE", f"{rmse:.10f}"])
        writer.writerow(["樣本內R平方", f"{r2:.10f}"])

    print(f"資料筆數：{len(rows)}（{len(years)} 年 × 月資料）")
    print(f"月執行率簡單平均：{mean_rate:.2%}")
    print(f"加權執行率（總支出 ÷ 總預算）：{weighted_rate:.2%}")
    print(f"迴歸式：執行率 = 截距 + 年度趨勢 × 年序 + 各月份效果")
    print(f"年度趨勢係數：{beta[1]:+.6f}（每年；乘 100 為百分點）")
    print(f"樣本內 RMSE：{rmse:.4%}；R2：{r2:.4f}")
    print(f"{next_year} 預測平均執行率：{sum(r for _, r in forecasts) / 12:.2%}")
    print(f"輸出：{MONTHLY_OUTPUT.name}、{COEF_OUTPUT.name}")
    print("注意：月支出是原檔的模擬值；此模型用於方法示範，不是實際月度歷史驗證。")


if __name__ == "__main__":
    main()
