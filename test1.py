import pandas as pd

# CSV파일 불러옴
path = 'detect_results/7/7_file.csv'
data = pd.read_csv(path)

# "Execution Time" 열의 표준 편차 계산
std_dev = data['Execution Time'].std()

data.loc[475] = ['STDEV', None, None, std_dev]

data.to_csv(path, index=False)