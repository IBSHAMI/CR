import pandas as pd


# read_file = pd.read_csv ("original_data/Kucoin_BCHUSDT_minute.csv")
# read_file.to_excel ("original_data/Kucoin_BCHUSDT_minute.xlsx", index = None, header=True)

from utils import *
# crypto_dict = {
#     "ADA":600000,
#     }
df = data_preparation("original_data/Kucoin_XRPUSDT_minute.xlsx", crypto_code="XRP")
# df = data_preparation("sample-data.xlsx")
print(df)

plot_data(df)




