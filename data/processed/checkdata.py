import os 
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import pathlib
import numpy as np

files = ["SE3prices", "SwedenGrid", "time_features", "weather_data", "holidays", "VillamichelinConsumption", "heat_pump_power_15min_20191211_to_20250513"]

def check_data(file_name: str):
    if file_name == "VillamichelinConsumption":
        file_path = pathlib.Path(__file__).parent / "villamichelin" / (file_name + ".csv")
    
    elif file_name.startswith("heat_pump_power_15min"): #any file that starts with heat_pump_power_15min
        file_path = pathlib.Path(__file__).parent / "villamichelin" / "Thermia" / (file_name + ".csv")
        
    else:
        file_path = pathlib.Path(__file__).parent / (file_name + ".csv")

    df = pd.read_csv(file_path)
    # Check for missing values
    if df.isnull().any().any():
        print(f"Missing values found in {file_path}")
    else:
        print(f"No missing values found in {file_name}")
    if df.index.duplicated().any():
        print(f"Duplicated indices found in {file_path}")
    else:
        print(f"No duplicated indices found in {file_name}")



def plot(file_name: str, amount: int):
    if file_name == "VillamichelinConsumption":
        file_path = pathlib.Path(__file__).parent  / "villamichelin" / (file_name + ".csv")
        
    elif file_name.startswith("heat_pump_power_15min"): #any file that starts with heat_pump_power_15min
        file_path = pathlib.Path(__file__).parent  / "villamichelin" / "Thermia" / (file_name + ".csv")
        
    else:
        file_path = pathlib.Path(__file__).parent  / (file_name + ".csv")

    df = pd.read_csv(file_path)
    
    print(df.head())
    xvalues = input("Enter the x-axis column name: ")
    yvalues = input("Enter the y-axis column name: ")
    
    #check if there are any missing values
    if df.isnull().any().any():
        df = df.ffill().bfill()
        
    #convert the x-axis column to datetime
    df[xvalues] = pd.to_datetime(df[xvalues])


    # Cut the timestamp to be the latest input amount of rows
    df = df.iloc[-amount:]
    
    plt.step(df[xvalues], df[yvalues])
    plt.xlabel("Timestamp")
    plt.xticks(rotation=25, ha="right")
    plt.ylabel(f"{str(df[yvalues].iloc[0])}")
    plt.title("Custom plot")
    plt.show()




#investigate negative prices

def investigate_negative_prices():
    # Load the SE3 prices CSV
    csv_path = r"C:\_Projects\home-energy-ai\data\processed\SE3prices.csv"
    df = pd.read_csv(
        csv_path,
        parse_dates=['HourSE'],    # adjust if your timestamp column is named differently
        index_col='HourSE'
    )

    # Filter for prices < -50 öre/kWh
    neg50 = df[df['SE3_price_ore'] < -50]

    # Compute metrics
    total_entries      = len(df)
    filtered_entries   = len(neg50)
    percent_of_total   = filtered_entries / total_entries * 100 if total_entries else 0
    mean_price         = neg50['SE3_price_ore'].mean()
    median_price       = neg50['SE3_price_ore'].median()
    std_price          = neg50['SE3_price_ore'].std()
    min_price          = neg50['SE3_price_ore'].min()
    earliest_timestamp = neg50.index.min()
    latest_timestamp   = neg50.index.max()

    print(f"Total data points:             {total_entries} hours")
    print(f"Total days:                    {np.round(total_entries/24)} days")
    print(f"Total months:                  {np.round(total_entries/24/30)} months")
    print(f"Total years:                   {np.round(total_entries/24/30/12)} years")
    print(f"Entries < -50 öre/kWh:         {filtered_entries} ({percent_of_total:.2f} %)")
    print(f"Mean of filtered prices:       {mean_price:.2f} öre/kWh")
    print(f"Median of filtered prices:     {median_price:.2f} öre/kWh")
    print(f"Std. dev. of filtered prices:  {std_price:.2f} öre/kWh")
    print(f"Most negative price:           {min_price:.2f} öre/kWh")
    print(f"Earliest negative timestamp:   {earliest_timestamp}")
    print(f"Latest negative timestamp:     {latest_timestamp}")



args = argparse.ArgumentParser()

args.add_argument("--neg_prices", action="store_true")
args.add_argument("--check-all", action="store_true")
args.add_argument("--check", action="store_true")
args.add_argument("--file", type=str, default="VillamichelinConsumption")

args.add_argument("--plot", action="store_true")
args.add_argument("--amount", type=int, default=1000)
args = args.parse_args()


if args.plot and args.file and args.amount :
    plot(args.file, args.amount)

elif args.check_all:
    for file in files:
        check_data(file)
        
elif args.check and args.file:
    check_data(args.file)

elif args.neg_prices:
    investigate_negative_prices()





