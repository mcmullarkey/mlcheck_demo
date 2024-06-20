# ds501 coefficient analysis

import pandas as pd
import glob
import numpy as np
import sklearn.linear_model
import sklearn.feature_selection
import seaborn as sb
import matplotlib.pyplot as plt
import statistics

fileList = glob.glob("C:/Users/Alexander/Documents/DS501_Case2/weekly_data/*.csv")

#print(fileList)

names_list = []
chi_col_list = []
bango = 0

for file in fileList:

    print(file)

    stock_name = file[53:]
    stock_name = stock_name[:-4]

    print(stock_name)
    names_list.append(stock_name)

    stock_df = pd.read_csv(file)

    point = stock_df.shape[0]-520

    df = stock_df.iloc[point:]

    print(df.shape)

    # % increease = close - prev_close / prev_close
    # we want stocks with low VAR of this, high SUM of this

    init_prev_close = df.iloc[0, 4] #just get the first one
    chi_col = []

    for i in range(0,df.shape[0]):
        close = df.iloc[i,4]
        
        if (i == 0):
            prev_close = init_prev_close
            # make % increase
            _1_chi = (close - init_prev_close) / init_prev_close
        else:
            prev_close = df.iloc[i-1, 4]

            _1_chi = (close - prev_close) / prev_close

        chi_col.append(_1_chi)
        #print(_1_chi)

    chi_col_list.append(chi_col)

# now we have a name list and a chi_col_list, could make a cov matrix with this
print(names_list)
#print(chi_col_list)
print(len(chi_col_list))
print(len(chi_col_list[0]))
for i in range(0, len(chi_col_list)):
    print(len(chi_col_list[i]))
    print(names_list[i])


chi_df = np.array(chi_col_list).T
print(chi_df.shape)

chi_df = pd.DataFrame(chi_df, columns = names_list)
print(chi_df)

chi_cov = np.cov(chi_df.T)
print(chi_cov.shape)

chi_cor = np.corrcoef(chi_df.T)
chi_cor = pd.DataFrame(chi_cor, columns = names_list)
chi_cor.index = names_list

heat_map = sb.heatmap(chi_cov)
#plt.show()


# HERE IS THE CORRELATION MATRICES

heat_map = sb.heatmap(chi_cor, annot = True, xticklabels = 1, yticklabels = 1)
plt.show()

# which have lowest variance and highest mean?
sd_row = list(np.apply_along_axis(statistics.stdev, axis = 0, arr = chi_df))
print('lowest sd: ', names_list[sd_row.index(min(sd_row))])

mean_row = list(np.apply_along_axis(statistics.mean, axis = 0, arr = chi_df))
print('highest mean: ', names_list[mean_row.index(max(mean_row))])

q_stat = []

for i in range(0, chi_df.shape[1]):
    num = mean_row[i] / sd_row[i]
    q_stat.append(num)

print('highest q_stat: ', names_list[q_stat.index(max(q_stat))])

plt.plot(mean_row, sd_row)

plt.show()

print('done??')