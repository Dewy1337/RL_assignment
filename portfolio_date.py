#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 15:07:15 2023

@author: dewy
"""

import pandas as pd
import numpy as np
np.set_printoptions(precision = 4, suppress = True)


def read_data():
    global dfReturns, dfVol
    dfReturns = pd.read_csv("returns.csv")
    dfVol = pd.read_csv("implied_vol_C_7_30.csv")
    
    
def generate_portfolio(iK):
    vRandom = np.random.normal(size = iK)
    vWeights = vRandom/sum(vRandom)
    return vWeights


def generate_portfolios(iN, iK):
    iK = len(vTickers)
    mWeights = np.array([generate_portfolio(iK) for i in range(iN)])
    
    return mWeights
    
    
def compute_portfolio_returns(mWeights):
    global dfReturns, dfPortfolio_returns
    vDates = dfReturns["Date"]
    dfPortfolio_returns = pd.DataFrame(index = vDates)
    mReturns = dfReturns.iloc[:,1:].to_numpy()
    
    for i, vWeights in enumerate(mWeights):
        vReturns = np.array([np.inner(vWeights, vReturn) for vReturn in mReturns])
        dfPortfolio_returns[str(i+1)] = vReturns
        
    return dfPortfolio_returns
    


def compute_portfolio_vol(mWeights):
    global dfVol, dfPortfolio_vol
    vDates = dfVol["Date"]
    dfPortfolio_vol = pd.DataFrame(index = vDates)
    mVol = dfVol.iloc[:,1:].to_numpy()
    
    for i, vWeights in enumerate(mWeights):
        vVol = np.array([np.sqrt(np.inner(vWeights**2, vVol**2)) for vVol in mVol])
        dfPortfolio_vol[str(i+1)] = vVol
        
    return dfPortfolio_vol


def compute_portfolio_RSI(dfPortfolio_returns):
    global dfPortfolio_RSI, mReturns
    vDates = dfPortfolio_returns.index
    dfPortfolio_RSI = pd.DataFrame(index = vDates[14:])
    mReturns = dfPortfolio_returns.to_numpy()
    
    for i in range(mReturns.shape[1]):
        dfPortfolio_RSI[str(i+1)] = compute_RSI(14, mReturns[:,i])
        
    return dfPortfolio_RSI


def compute_RSI(iTime_horizon, vReturns):
    iT = len(vReturns)
    vRSI = np.zeros(iT-iTime_horizon)
    for t in range(iTime_horizon, iT):
        dPos_avg = np.mean(np.array([dReturn for dReturn in vReturns[t-iTime_horizon:t] if dReturn > 0]))
        dNeg_avg = -np.mean(np.array([dReturn for dReturn in vReturns[t-iTime_horizon:t] if dReturn < 0]))
        if np.isnan(dNeg_avg):
            vRSI[t-iTime_horizon] = 100
        elif np.isnan(dPos_avg):
            vRSI[t-iTime_horizon] = 0
        else:
            vRSI[t-iTime_horizon] = 100-100/(1 + (dPos_avg/dNeg_avg))
    
    return vRSI


def SaveData(df, sFilename):
    df.to_csv(sFilename+".csv")
    
    
    
def main():
    global vTickers, dfPortfolio_vol, dfPortfolio_RSI, dfPortfolio_rewards, dfPortfolio_returns
    vTickers = np.array(["COST", "MU", "MSFT", "AMZN", "IBM", "AAPL", "ORCL", "KEY", "AEE", "MRO", "AMGN", "UNH",
                "LUV", "CSCO", "MMM", "MS", "NKE"])
    sStart_date = "1998-01-23"
    sEnd_date = "2021-12-31"
    
    iK = len(vTickers)
    iN = 17
    read_data()
    
    np.random.seed(1234)
    
    mWeights = generate_portfolios(iN, iK)
    dfPortfolio_returns = compute_portfolio_returns(mWeights)
    dfPortfolio_vol = compute_portfolio_vol(mWeights)
    dfPortfolio_RSI = compute_portfolio_RSI(dfPortfolio_returns)
    dfPortfolio_rewards = dfPortfolio_returns.iloc[1:,:]
    dfPortfolio_returns = dfPortfolio_returns.iloc[:-1,:]
    dfPortfolio_vol = dfPortfolio_vol.iloc[1:-1,:]    
    
    SaveData(dfPortfolio_returns, "portfolio_returns")
    SaveData(dfPortfolio_RSI, "portfolio_rsi")
    SaveData(dfPortfolio_rewards, "portfolio_rewards")
    SaveData(dfPortfolio_vol, "portfolio_vol")
    
    
if __name__ == "__main__":
    main()