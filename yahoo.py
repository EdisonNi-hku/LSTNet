import sys
from yahoo_finance_api2 import share
from yahoo_finance_api2.exceptions import YahooFinanceError
import yfinance as yf
import pandas as pd
'''
indices = ['^GSPC', '^DJI', '^IXIC', '^NYA', '^XAX', '^BUK100P', '^RUT', '^VIX', '^FTSE', '^GDAXI', '^FCHI', '^STOXX50E',
           '^N100', '^BFX', 'IMOEX.ME', '^N225', '^HSI', '000001.SS', '399001.SZ', '^STI', '^AXJO', '^AORD', '^BSESN',
           '^JKSE', '^KLSE', '^NZ50', '^KS11', '^TWII', '^GSPTSE', '^BVSP', '^MXX', '^MERV', '^TA125.TA', 'CASE30',
           '^JN0U.JO']
'''
indices1 = ['000001.SS', '000002.SS', '000003.SS', '000004.SS', '000005.SS', '000006.SS', '000007.SS', '000008.SS',
            '000010.SS', '000011.SS', '000012.SS', '000013.SS', '000015.SS', '000016.SS', '000017.SS', '000018.SS',
            '000019.SS', '000020.SS', '000021.SS', '000022.SS', '000025.SS', '000026.SS', '000027.SS', '000028.SS']
indices2 = ['399001.SZ', '399002.SZ', '399003.SZ', '399004.SZ', '399005.SZ', '399006.SZ', '399007.SZ', '399008.SZ',
            '399009.SZ', '399010.SZ', '399011.SZ', '399012.SZ', '399013.SZ', '399015.SZ', '399016.SZ', '399017.SZ',
            '399018.SZ', '399088.SZ', '399100.SZ', '399101.SZ', '399102.SZ', '399108.SZ', '399106.SZ', '399107.SZ']

df1 = yf.download('000010.SS', interval='1m', start="2020-08-03", end="2020-08-08")

if __name__ == '__main__':
    print(len(df1))
    '''
    df_ = pd.DataFrame()
    for i in range(len(indices1)):

        df = yf.download(indices1[i], interval='1m', start="2020-08-03", end="2020-08-08")

        df = df.reset_index()
        print(indices1[i] + ' =====================')
        # print(df[(df.Datetime > '2020-08-24') & (df.Datetime < '2020-08-24 11:31:00')])
        if i == 0:
            df_['Datetime'] = df.Datetime
            df_[indices1[i]] = df['Open']
        else:
            df = df[['Datetime', 'Open']]
            df.columns = ['Datetime', indices1[i]]
            df_ = df_.merge(df, on=['Datetime'], how='left')
        print(indices1[i] + 'finished')

    df_.to_csv('../SH_0803_0808.csv', index=False)
    '''