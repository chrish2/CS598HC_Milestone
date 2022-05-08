import pandas as pd


df = pd.read_csv("../data/cleaned.txt", delimiter=">>><<<")

df.columns =['text', 'label']

print(len(df))

# train_data = df.iloc[:50]
# test_data = df.iloc[50:]
# train_data = df.iloc[:19987]
# test_data = df.iloc[19988:]
#
df.to_csv("../data/data_forest.csv", index = False, header=True)
# test_data.to_csv("../data/data_test2.csv", index = False, header=True)


# train_df = pd.read_csv("../data/data_train.csv")
print(train_df.iloc[:3])



#
# data = {'Product': ['Desktop Computer','Tablet','Printer','Laptop'],
#         'Price': [850,200,150,1300]
#         }
#
# df = pd.DataFrame(data, columns= ['Product', 'Price'])
#
# df.to_csv (r'../data/data_TRAIL2.csv', index = False, header=True)
#
# print (df)


# texts=[]
# label=[]
#
# counter0 = 0
# counter1 = 0
# with open('../data/cleaned.txt', 'r') as fr:
#     for line in fr:
#         if line is not None:
#             text = line.strip()
#             arr = text.split(">>><<<")
#             if int(arr[1].strip()) == 0:
#                 if counter0 < 6821:
#                     texts.append(arr[0])
#                     label.append(arr[1])
#                     counter0 += 1
#             else:
#                 counter1 += 1
#                 texts.append(arr[0])
#                 label.append(arr[1])
#
#
# print(len(texts), len(label))
#
# # print(counter0)
# # print(counter1)
# Data = pd.DataFrame({'text': texts, 'label':label})
# Data.to_csv("../data/data_train3.csv",  index = False, header=True)