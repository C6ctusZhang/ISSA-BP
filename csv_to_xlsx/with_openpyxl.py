from openpyxl import Workbook
import datetime


def csv_to_xlsx_pd(sourcePath:str,savePath:str,encode='utf-8'):
    """将csv 转为 excel（.xlsx格式）
    如果不需要可以把计时相关代码删除
    Args:
        sourcePath:str 来源文件路径
        savePath:str 保存文件路径，需要包含保存的文件名，文件名需要是 xlsx 格式的
        encode='utf-8' 默认编码，可以改为需要的编码如gbk
    """
    print('开始处理%s' % sourcePath)
    curr_time = datetime.datetime.now()
    print(curr_time)

    f = open(sourcePath, 'r', encoding=encode)
    # 创建一个workbook 设置编码
    workbook = Workbook()
    # 创建一个worksheet
    worksheet = workbook.active
    workbook.title = 'sheet'

    for line in f:
        row = line.split(',')
        worksheet.append(row)
        # if row[0].endswith('00'):    # 每一百行打印一次
        #     print(line, end="")

    workbook.save(savePath)
    print('处理完毕')
    curr_time2 = datetime.datetime.now()
    print(curr_time2-curr_time)


if __name__ == '__main__':
    source = 'source.csv'
    save = 'result.xlsx'
    csv_to_xlsx_pd(sourcePath=source, savePath=save, encode='utf-8')
