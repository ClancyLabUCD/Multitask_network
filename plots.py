import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = np.load('mat.npz')
data_exp = np.load('mat_implement.npz')
x_train=data['x_train']
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1])
x_test=data['x_test']
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1])
y_train=data['y_train']
y_train = y_train.reshape(y_train.shape[0], y_train.shape[1])
y_test=data['y_test']
y_test = y_test.reshape(y_test.shape[0], y_test.shape[1])
train_pred_translation = data['train_pred_translation']
train_pred_translation = train_pred_translation.reshape(train_pred_translation.shape[0], train_pred_translation.shape[1])
test_pred_translation = data['test_pred_translation']
test_pred_translation = test_pred_translation.reshape(test_pred_translation.shape[0], test_pred_translation.shape[1])
APD90_real_train = data['APD90_real_train']
APD90_real_train = APD90_real_train.reshape(APD90_real_train.shape[0])
APD90_predict_train = data['APD90_pred_train']
APD90_predict_train = APD90_predict_train.reshape(APD90_predict_train.shape[0])
APD90_real_test = data['APD90_real_test']
APD90_real_test = APD90_real_test.reshape(APD90_real_test.shape[0])
APD90_predict_test = data['APD90_pred_test']
APD90_predict_test = APD90_predict_test.reshape(APD90_predict_test.shape[0])
Input_max=data['Input_max']
Input_mean = data['Input_mean']
y_reg_mean = data['y_reg_mean']
y_reg_max = data['y_reg_max']
Data_exp = data_exp['Data_exp']
pred_plot_exp = data_exp['pred_plot_exp']

a=80

#Fig4
num_fig = 6
for i in range(num_fig):
    plt.figure(num=None, figsize=(30, 18), dpi=300)
    plt.rc('axes', linewidth=4)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', pad=15)
    ax.tick_params(axis="y",direction="in", length=16, width=4)
    ax.tick_params(axis="x",direction="in", length=16, width=4)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(5)
        ax.spines[axis].set_color('black')
    if i == 0:
        #Training samples input
        plt.ylim(-1.2,1.2)
        plt.xlim(-40, 700)
        plt.xticks(np.arange(0,701, step=175),fontsize=a)
        plt.yticks(np.arange(-1,1.19, step=0.5),fontsize=a)
        plt.plot(x_train.transpose(),linewidth=5,color=[0,0,0])
        plt.xlabel("Time(ms)",fontsize=a)
        plt.ylabel("Membrane potential",fontsize=a)
        ttl=plt.title("iPSC-CM AP population (training set)",fontsize=a)
        ttl.set_position([.5, 1.05])
        plt.savefig("D:/Papers/Multitask_network/results/Fig5/input_data_train.png", dpi = 300, bbox_inches = 'tight')
    elif i == 1 :

        #Training samples target and network output
        plt.ylim(-1.2,1.2)
        plt.xlim(-40, 700)
        plt.xticks(np.arange(0,701, step=175),fontsize=a)
        plt.yticks(np.arange(-1,1.19, step=0.5),fontsize=a)
        plt.plot(y_train.transpose(),linewidth=3,color=[0.8,0,0],label='Simulated population')
        plt.plot(train_pred_translation.transpose(),linewidth=3,color=[24/256, 144/256, 130/256],label='Translated population')
        plt.xlabel("Time(ms)",fontsize=a)
        plt.ylabel("Membrane potential",fontsize=a)
        ttl=plt.title("adult-CM AP population (training set)",fontsize=a)
        ttl.set_position([.5, 1.05])
        plt.savefig("D:/Papers/Multitask_network/results/Fig5/output_data_train.png", dpi = 300, bbox_inches = 'tight')

    elif i==2:
        #APD90 parameter distribution train
        plt.xlim(180, 600)
        plt.ylim(0, 0.011)
        plt.xticks(np.arange(200,601, step=100),fontsize=a)
        plt.yticks(np.arange(0,0.011, step=0.002),fontsize=a)
        sns.kdeplot(APD90_real_train, color=[0.8,0,0],linewidth=15, label="Simulated population")
        sns.kdeplot(APD90_predict_train, color=[24/256, 144/256, 130/256],linewidth=5, label="Translated population")
        plt.legend(loc = 'best', prop={'size': a})
        plt.xlabel("$APD_{90}$",fontsize=a,labelpad=20)
        plt.ylabel("Frequency",fontsize=a,labelpad=20)
        ttl=plt.title("Distribution of adult-CM $APD_{90}$ (training set)",fontsize=a)
        ttl.set_position([.5, 1.05])
        plt.savefig("D:/Papers/Multitask_network/results/Fig5/input_data_test.png", dpi = 300, bbox_inches = 'tight')

    elif i == 3:
        #Test samples input
        plt.ylim(-1.2,1.2)
        plt.xlim(-40, 700)
        plt.xticks(np.arange(0,701, step=175),fontsize=a)
        plt.yticks(np.arange(-1,1.19, step=0.5),fontsize=a)
        plt.plot(x_test.transpose(),linewidth=5,color=[0,0,0])
        plt.xlabel("Time(ms)",fontsize=a)
        plt.ylabel("Membrane potential",fontsize=a)
        ttl=plt.title("iPSC-CM AP population (test set)",fontsize=a)
        ttl.set_position([.5, 1.05])
        plt.savefig("D:/Papers/Multitask_network/results/Fig5/fig21.pdf", dpi = 300, bbox_inches = 'tight')

    elif i == 4 :

        #Test samples target and network output
        plt.ylim(-1.2,1.2)
        plt.xlim(-40, 700)
        plt.xticks(np.arange(0,701, step=175),fontsize=a)
        plt.yticks(np.arange(-1,1.19, step=0.5),fontsize=a)
        plt.plot(y_test.transpose(),linewidth=3,color=[0.8,0,0],label='Simulated population')
        plt.plot(test_pred_translation.transpose(),linewidth=3,color=[[24/256, 144/256, 130/256],label='Translated population')
        plt.xlabel("Time(ms)",fontsize=a)
        plt.ylabel("Membrane potential ",fontsize=a)
        ttl=plt.title("adult-CM AP population (test set)",fontsize=a)
        ttl.set_position([.5, 1.05])
        plt.savefig("D:/Papers/Multitask_network/results/Fig5/output_data_test.png", dpi = 300, bbox_inches = 'tight')

    elif i==5:
        #APD90 parameter distribution test
        plt.xlim(180, 600)
        plt.ylim(0, 0.011)
        plt.xticks(np.arange(200,601, step=100),fontsize=a)
        plt.yticks(np.arange(0,0.011, step=0.002),fontsize=a)
        sns.kdeplot(APD90_real_test, color=[0.8,0,0],linewidth=15, label="Simulated population", legend=False)
        sns.kdeplot(APD90_predict_test, color=[24/256, 144/256, 130/256],linewidth=15, label="Translated population", legend=False)
        plt.legend(loc = 'best', prop={'size': a})
        plt.xlabel("$APD_{90}$",fontsize=a,labelpad=20)
        plt.ylabel("Frequency",fontsize=a,labelpad=20)
        ttl=plt.title("Distribution of adult-CM $APD_{90}$ (test set)",fontsize=a)
        ttl.set_position([.5, 1.05])
        plt.savefig("D:/Papers/Multitask_network/results/Fig5/test_APD90.png", dpi = 300, bbox_inches = 'tight')


#Fig7
num_fig = 2
a=80
for i in range(num_fig):
    plt.figure(num=None, figsize=(30, 18), dpi=300)
    plt.rc('axes', linewidth=4)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', pad=15)
    ax.tick_params(axis="y",direction="in", length=16, width=4)
    ax.tick_params(axis="x",direction="in", length=16, width=4)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(5)
        ax.spines[axis].set_color('black')
    plt.ylim(-1.2,1.2)
    plt.xlim(-40, 700)
    plt.xticks(np.arange(0,701, step=175),fontsize=a)
    plt.yticks(np.arange(-1,1.19, step=0.5),fontsize=a)

    if i == 0:
        #Experimental data
        plt.plot(Data_exp.transpose(),linewidth=6)
        plt.xlabel("Time(ms)",fontsize=a)
        plt.ylabel("Membrane potential",fontsize=a)
        ttl=plt.title("iPSC-CM AP population",fontsize=a)
        ttl.set_position([.5, 1.05])
        plt.savefig("D:/Multitask_network/results/Fig7/input_experiment.png", dpi = 300, bbox_inches = 'tight')

    elif i == 1 :
        #Translated adult AP from experimental data
        plt.plot(pred_plot_exp.transpose(),linewidth=6)
        plt.xlabel("Time(ms)",fontsize=a)
        plt.ylabel("Membrane potential ",fontsize=a)
        ttl=plt.title("Translated adult-CM AP population",fontsize=a)
        ttl.set_position([.5, 1.05])
        plt.savefig("D:/Papers/Multitask_network/results/Fig7/output_experiment.png", dpi = 300, bbox_inches = 'tight')
