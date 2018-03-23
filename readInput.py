import numpy as np
import matplotlib.pyplot as plt
temp =open("/home/ubuntu/cs230_project/deepnews/Classifier/print_filescopy/20180318_143210/run_result.txt")
dev_loss = []
dev_acc = []
average_epoch_loss = []
average_train_loss = []
average_train_acc = []
for line in temp :
    if "dev_loss" in line :
        num = float(line.split('[')[1].split(']')[0])
        dev_loss.append(num)
    
    if "dev_acc" in line :
        num = float(line.split('[')[1].split(']')[0])
        dev_acc.append(num)
        
    if "average_epoch_loss" in line :
        num = float(line.split(' ')[1])
        average_epoch_loss.append(num)
    
    if "average_train_loss" in line :
        num = line.split('[')[1].split(']')[0]
        num = np.mean(np.asarray(list(map(float,num.split(',')))))
        average_train_loss.append(num)
    
    if "average_train_acc" in line :
        num = line.split('[')[1].split(']')[0]
        num = np.mean(np.asarray(list(map(float,num.split(',')))))
        average_train_acc.append(num)

plt.plot(dev_loss)
plt.plot(dev_acc)
plt.plot(average_epoch_loss)
plt.plot(average_train_loss)
plt.plot(average_train_acc)
plt.show()
