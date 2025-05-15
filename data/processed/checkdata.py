import os 
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import pathlib

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






args = argparse.ArgumentParser()

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





