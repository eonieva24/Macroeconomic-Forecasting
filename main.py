# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
import pandas as pd
from dbnomics import fetch_series

# fetch all countries, monthly CPI index 2015=100
data = fetch_series("OECD/MEI/CPALTT01.IXOB.M")
df = data.to_dataframe()

# reshape by country
df_pivot = df.pivot(index="period", columns="country", values="value")
df_pivot.to_csv("oecd_cpi_all_countries.csv")
