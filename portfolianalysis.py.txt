# !pip install yfinance seaborn matplotlib pandas numpy

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Style des graphiques
plt.style.use("seaborn-v0_8-darkgrid")

# === Paramètres ===
tickers = ["AIR.PA", "BA", "SAF.PA", "HO.PA", "RTX", "RR.L", "LMT", "NOC"]
start_date = "2023-01-01"
end_date = "2024-12-31"

# === Téléchargement des données ===
try:
    data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=True)
    
    # Extraire uniquement les prix ajustés pour tous les tickers
    adj_close = pd.DataFrame({ticker: data[ticker]["Close"] for ticker in tickers})
    adj_close.dropna(inplace=True)
except Exception as e:
    print("Erreur lors du téléchargement des données :", e)
    raise SystemExit

# === Calculs financiers ===
daily_returns = adj_close.pct_change().dropna()
cumulative_returns = (1 + daily_returns).cumprod()
annual_returns = daily_returns.mean() * 252
volatility = daily_returns.std() * np.sqrt(252)
sharpe_ratio = annual_returns / volatility
correlation_matrix = daily_returns.corr()

# === Portefeuille à poids égaux ===
equal_weights = np.ones(len(tickers)) / len(tickers)
portfolio_daily_returns = daily_returns.dot(equal_weights)
portfolio_cumulative_returns = (1 + portfolio_daily_returns).cumprod()
portfolio_annual_return = portfolio_daily_returns.mean() * 252
portfolio_volatility = portfolio_daily_returns.std() * np.sqrt(252)
portfolio_sharpe = portfolio_annual_return / portfolio_volatility

# === Visualisations ===

# 1. Rendement cumulé
plt.figure(figsize=(12, 6))
cumulative_returns.plot(alpha=0.5)
portfolio_cumulative_returns.plot(label='Portefeuille global', color='black', linewidth=2)
plt.title("Rendement cumulé - Portefeuille vs Actions individuelles")
plt.xlabel("Date")
plt.ylabel("Indice de performance")
plt.legend()
plt.tight_layout()
plt.show()

# 2. Matrice de corrélation
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Corrélation entre les actions")
plt.tight_layout()
plt.show()

# === Résumé des indicateurs clés ===
print("=== Indicateurs individuels ===")
print("\nRendements annualisés :\n", annual_returns.round(4))
print("\nVolatilité annualisée :\n", volatility.round(4))
print("\nRatio de Sharpe :\n", sharpe_ratio.round(4))

print("\n=== Indicateurs du portefeuille ===")
print(f"Rendement annualisé : {portfolio_annual_return:.4f}")
print(f"Volatilité annualisée : {portfolio_volatility:.4f}")
print(f"Ratio de Sharpe : {portfolio_sharpe:.4f}")
