#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 16:03:26 2023

@author: dewy
"""

from pandas_datareader import data as pdr
import pandas as pd
from copy import copy
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
yf.pdr_override() 


def preprocess_option_data():
    global dfOptions
    dfOptions = pd.read_csv("vjosikbcqztidljb.csv")
    #dfCall_A = dfOptions[(dfOptions["cp_flag"] == "C") & (dfOptions["exercise_style"] == "A") & (dfOptions["issuer"] == "AMEREN CORP.")]
    #dfCall_A = dfCall_A.dropna().reset_index(drop=True)
    return dfOptions
    

def getData(vTickers, sStart_date, sEnd_date):
    global dfReturns, dfData, mVolume, mReturns, mRSI, vData
    dfStart_date = pd.to_datetime(sStart_date).date()
    dfEnd_date = pd.to_datetime(sEnd_date).date()

    vData = [pdr.get_data_yahoo(sTicker, start=sStart_date, end=sEnd_date) for sTicker in vTickers]

    vStart_dates = np.array([data_asset.index.date[0] for data_asset in vData])
    vLater_start = np.where(vStart_dates != dfStart_date)[0]

    print("Following tickers are removed: ", vTickers[vLater_start], "With starting dates: ", vStart_dates[vLater_start])

    vTickers = list(vTickers)
    for index in sorted(vLater_start, reverse=True):
        del vData[index]
        del vTickers[index]
        
    vTickers = np.array(vTickers)
    vDates = vData[0].index[1:]
    
    mVolume = np.array([data_asset.loc[:,"Volume"].to_numpy()[1:] for data_asset in vData]).T
    mReturns = np.array([log_returns(data_asset) for data_asset in vData]).T
    mOBV = np.array([compute_OBV(mVolume[:,i], mReturns[:,i]) for i in range(mVolume.shape[1])]).T
    
    iHorizon_RSI = 14
    mRSI = np.array([compute_RSI(iHorizon_RSI, mReturns[:,i]) for i in range(mReturns.shape[1])]).T

    dfReturns = pd.DataFrame(mReturns[:-1,], columns = vTickers, index = vDates[:-1])
    dfReward = pd.DataFrame(mReturns[1:,], columns = vTickers, index = vDates[:-1])
    dfVolume = pd.DataFrame(mVolume[:-1,], columns = vTickers, index = vDates[:-1])
    dfOBV = pd.DataFrame(mOBV[:-1,], columns = vTickers, index = vDates[:-1])
    dfRSI = pd.DataFrame(mRSI[:-1,], columns = vTickers, index = vDates[iHorizon_RSI:-1])
    
    SaveData(dfReturns, "returns")
    SaveData(dfReward, "reward")
    SaveData(dfVolume, "volume")
    SaveData(dfOBV, "OBV")
    SaveData(dfRSI, "RSI")
    
    return vDates
    

def compute_OBV(vVolume, vReturns):
    iT = len(vVolume)
    vOBV = np.zeros(iT)
    vOBV[0] = vVolume[0]
    for t in range(1,iT):
        vOBV[t] = vOBV[t-1] + vVolume[t] if vReturns[t] >= 0 else vOBV[t-1] - vVolume[t]
    return vOBV
    
    
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

    
def log_returns(mData):
    vReturns = mData.loc[:,"Adj Close"].to_numpy()
    vLog_returns = 100*(np.log(vReturns[1:]) - np.log(vReturns[:-1]))
    return vLog_returns


def SaveData(df, sFilename):
    df.to_csv(sFilename+".csv")
    
    
def get_implied_vol(dfOptions, vDates):
    global dfOptions_7_14_days, dfImplied_vol
    dfOptions["date"] = pd.to_datetime(dfOptions["date"])
    dfOptions.index = dfOptions["date"]
    dfOptions["exdate"] = pd.to_datetime(dfOptions["exdate"])
    dfOptions["date_diff_days"] = np.array((dfOptions["exdate"].dt.date - dfOptions["date"].dt.date).dt.days)
    dfOptions_7_14_days = dfOptions[dfOptions["date_diff_days"].isin(np.array(np.arange(7,30)))]
    
    dfImplied_vol = pd.DataFrame(index = pd.to_datetime(vDates))
    dfImplied_vol["avg_implied_vol"] = np.nan
    
    for date in dfImplied_vol.index.to_series():
        try:
            dfImplied_vol.loc[date, "avg_implied_vol"] = dfOptions_7_14_days.loc[date, "impl_volatility"].mean()
        except KeyError:
            dfImplied_vol.loc[date, "avg_implied_vol"] = np.nan
            
    dfImplied_vol = dfImplied_vol.interpolate(method='linear', limit_direction='forward')
    
    return dfImplied_volxf
    
    
    
    
def main():
    global df_test, vDates, dfImplied_vol_C
    vCompanies = np.array(["COSTCO WHOLESALE CORP", "MICRON TECHNOLOGY INC.", "MICROSOFT CORPORATION", "AMAZON.COM INC.", 
                  "INTERNATIONAL BUSINESS MACHI", "APPLE INC", "ORACLE CORP.", "KEYCORP", "AMEREN CORP.",
                  "MARATHON OIL CORPORATION", "AMGEN INC.", "UNITEDHEALTH GROUP INC", "SOUTHWEST AIRLINES CO",
                  "CISCO SYSTEMS, INC.", "3M CO.", "MORGAN STANLEY", "NIKE, INC."])
    
    vTickers = np.array(["COST", "MU", "MSFT", "AMZN", "IBM", "AAPL", "ORCL", "KEY", "AEE", "MRO", "AMGN", "UNH",
                "LUV", "CSCO", "MMM", "MS", "NKE"])
    
    sStart_date = "1998-01-23"
    sEnd_date = "2021-12-31"
    
    vDates = getData(vTickers, sStart_date, sEnd_date)
    
    #dfOptions = preprocess_option_data()
    
    #dfImplied_vol_C = pd.DataFrame(index = vDates)
    
    #dfCall_A = dfOptions[(dfOptions["cp_flag"] == "C") & (dfOptions["exercise_style"] == "A") & (dfOptions["issuer"] == "AMEREN CORP.")]
    #dfCall_A = dfCall_A.dropna().reset_index(drop=True)
    
    #for i in range(len(vTickers)):
    #    dfOptions_ticker = dfOptions[(dfOptions["cp_flag"] == "C") & (dfOptions["exercise_style"] == "A") & (dfOptions["issuer"] == vCompanies[i])]
    #    dfImplied_vol_C[vTickers[i]] = get_implied_vol(dfOptions_ticker, vDates)
    
    #SaveData(dfImplied_vol_C, "implied_vol_C_7_30")
    
    
    
if __name__ == "__main__":
    main()
    
    
    
    
