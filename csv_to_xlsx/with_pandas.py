import pandas as pd


def csv_to_xlsx_pd_train():
    csv = pd.read_csv('zhendong.csv', encoding='utf-8')
    csv.to_excel('zhendong.xlsx', sheet_name='data')
def csv_to_xlsx_pd_test():
    csv = pd.read_csv('test.csv', encoding='utf-8')
    csv.to_excel('test.xlsx', sheet_name='data')

if __name__ == '__main__':
    csv_to_xlsx_pd_train()
    # csv_to_xlsx_pd_test()
