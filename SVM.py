import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
import math
from sklearn.linear_model import LinearRegression

# In[178]:


cancer = datasets.load_breast_cancer() #load the dataset
type(cancer)


# In[191]:


data = np.c_[cancer.data, cancer.target]
columns = np.append(cancer.feature_names, ["target"])
data1 =pd.DataFrame(data, columns=columns)

print(columns)
#  

# In[192]:


data1.shape


# In[194]:


data_t = data1.to_numpy()
data = []
data2 = [0,0]
for i in range(len(data1)):
#     if data_t[i][4] == 1.0:
#         continue
    t_data = [data_t[i][0],data_t[i][2],data_t[i][-1]]
    data.append(t_data)
# data
data = np.array(data)
# plt.scatter(data[:,0],data[:,1])
# plt.show


# In[222]:




blue_data = []
for i in range(len(data)):
    if data[i][2] == 0:
        blue_data.append([data[i][0],data[i][1]])
blue_data = np.array(blue_data)
# plt.scatter(blue_data[:,0],blue_data[:,1])
# plt.show


# In[123]:


x1 = np.mean(blue_data[:,0])
y1 = np.mean(blue_data[:,1])
x1,y1


# In[225]:


# x1 = 8
# y1 = 10
temp = []
slope1 = math.tan(math.pi/6)
slope2 = math.tan(math.pi/2)
line1 = y1 + (slope1)*x1
line2 = y1 + (slope2)*(x1)
# line2 = -y1
for i in range(len(data)):
    if (slope1*data[i][0])+data[i][1]>=0 and ((-slope2)*data[i][0])+data[i][1]<0:
        temp.append(data[i])
# print(temp)
temp = np.array(temp)
print('temp',len(temp),len(data))
# plt.scatter(temp[:,0],temp[:,1])
# # plt.xlim([8, 20])
# # plt.ylim([10, 30])
# plt.show


# In[230]:


dict = {}
d = 10
step = 5
while d<=30:
    for i in range(len(temp)):
        if math.sqrt((x1-temp[i][0])**2+(y1-temp[i][1])**2)<=d and math.sqrt((x1-temp[i][0])**2+(y1-temp[i][1])**2) > d-step:
            if d not in dict:
                dict[d] = [[temp[i][0],temp[i][1],temp[i][2]]]
            else:
                dict[d].append([temp[i][0],temp[i][1],temp[i][2]])
    d+=step


# In[249]:

best_range = float('inf')
best_area = 0
for i in dict:
    red = 0
    blue = 0
    for j in range(len(dict[i])):
        if dict[i][j][2] == 1.0:
            red+=1
        else:
            blue+=1
    if best_range > 1.0*abs(red-blue)/(red+blue):
        best_range = 1.0*abs(red-blue)/(red+blue)
        best_area = i
        print(i,best_range)
print(best_area)


# In[254]:


array_linear_reg = []
for i in range(len(dict[best_area])):
    array_linear_reg.append(dict[best_area][i])

array_linear_reg = np.array(array_linear_reg)
# print(array_linear_reg)
input = array_linear_reg[:,0:2]
output = array_linear_reg[:,2]
print(input)
print(output)

regressor = LinearRegression()
regressor.fit(input, output)

#########################Important######################################
19

# Plug your weights into the general form (w0 + w1x + w2y = 0) and solve for x, x=0, y, y=0:

# x = -(w0 - w2y)/w1  
# x = 0 when y = -w0/w2  
# y = -(w0 - w1x)/w2  
# y = 0 when x = -w0/w1  
# Now we have two points that lie on the line: (0, -w0/w2) and (-w0/w1, 0)

# slope = -(w0/w2)/(w0/w1)  
# intercept = -w0/w2
############################################################################


print(regressor.intercept_)
print(regressor.coef_)
intercept = -1.0*regressor.intercept_/regressor.coef_[1]
slope = -1.0*regressor.coef_[0]/regressor.coef_[1]
temp_x = [[slope*input[i][0]+intercept,input[i][0]] for i in range(len(input))]
# temp_x = [[regressor.coef_[0]*input[i][0]+regressor.coef_[1]*input[i][1]+regressor.intercept_,output[i]] for i in range(len(input))]
temp_x = np.array(temp_x)

print('temp_x',temp_x)
plt.scatter(temp_x[:,0],temp_x[:,1])
plt.show
