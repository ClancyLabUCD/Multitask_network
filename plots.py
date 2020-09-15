import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = np.load('mat.npz')
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
Data_exp = data['Data_exp']
pred_plot_exp = data['pred_plot_exp']

x_train_plot=(x_train*Input_max)+Input_mean
x_test_plot =(x_test*Input_max)+Input_mean
y_train_plot = (y_train*y_reg_max)+y_reg_mean
y_test_plot = (y_test*y_reg_max)+y_reg_mean
pred_plot_train = (train_pred_translation *y_reg_max)+y_reg_mean
pred_plot_test = (test_pred_translation *y_reg_max)+y_reg_mean


a=50



#Fig4
num_fig = 6
for i in range(num_fig):
    plt.figure(num=None, figsize=(16, 10), dpi=300)
    plt.rc('axes', linewidth=4)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', pad=15)
    ax.tick_params(axis="y",direction="in", length=16, width=4)
    ax.tick_params(axis="x",direction="in", length=16, width=4)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if i == 0:
        #Training samples input
        plt.ylim(-100, 50,50)
        plt.xlim(-40, 700)
        plt.xticks(np.arange(0,701, step=175),fontsize=a)
        plt.yticks(np.arange(-100,51, step=50),fontsize=a)
        plt.plot(x_train_plot.transpose(),linewidth=3,color=[0,0,0])
        plt.xlabel("Time(ms)",fontsize=a)
        plt.ylabel("Membrane potential (mV)",fontsize=a)
        ttl=plt.title("iPSC-CM AP population (train)",fontsize=a)
        ttl.set_position([.5, 1.1])
        plt.savefig("D:/Papers/LSTM encoder_decoder/code/results/Fig4/fig11.pdf", dpi = 300, bbox_inches = 'tight')
    elif i == 1 :

        #Training samples target and network output
        plt.ylim(-100, 50,50)
        plt.xlim(-40, 700)
        plt.xticks(np.arange(0,701, step=175),fontsize=a)
        plt.yticks(np.arange(-100,51, step=50),fontsize=a)
        plt.plot(y_train_plot.transpose(),linewidth=3,color=[0.8,0,0],label='Simulated population')
        plt.plot(pred_plot_train.transpose(),linewidth=3,color=[0,0,1],label='Translated population')
        handles, labels = ax.get_legend_handles_labels()
        display = (0,400)
        plt.xlabel("Time(ms)",fontsize=a)
        plt.ylabel("Membrane potential (mV)",fontsize=a)
        ttl=plt.title("adult-CM AP population (train)",fontsize=a)
        ttl.set_position([.5, 1.1])
        plt.savefig("D:/Papers/LSTM encoder_decoder/code/results/Fig4/fig12.pdf", dpi = 300, bbox_inches = 'tight')

    elif i==2:
        #APD90 parameter distribution train
        plt.xlim(180, 640)
        plt.ylim(0, 0.15)
        plt.xticks(np.arange(200,641, step=110),fontsize=a)
        plt.yticks(np.arange(0,0.151, step=0.05),fontsize=a)
        sns.kdeplot(APD90_real_train, color=[0.8,0,0],linewidth=5, label="Simulated population", legend=False)
        sns.kdeplot(APD90_predict_train, color=[0,0,1],linewidth=5, label="Translated population", legend=False)
        plt.legend(loc = 'best', prop={'size': a})
        plt.xlabel("APD90",fontsize=a,labelpad=10)
        plt.ylabel("Frequency",fontsize=a,labelpad=15)
        ttl=plt.title("Distribution of adult-CM APD90 (train)",fontsize=a)
        ttl.set_position([.5, 1.1])
        plt.savefig("D:/Papers/LSTM encoder_decoder/code/results/Fig4/fig13.pdf", dpi = 300, bbox_inches = 'tight')

    elif i == 3:
        #Test samples input
        plt.ylim(-100, 50,50)
        plt.xlim(-40, 700)
        plt.xticks(np.arange(0,701, step=175),fontsize=a)
        plt.yticks(np.arange(-100,51, step=50),fontsize=a)
        plt.plot(x_test_plot.transpose(),linewidth=3,color=[0,0,0])
        plt.xlabel("Time(ms)",fontsize=a)
        plt.ylabel("Membrane potential (mV)",fontsize=a)
        ttl=plt.title("iPSC-CM AP population (test)",fontsize=a)
        ttl.set_position([.5, 1.1])
        plt.savefig("D:/Papers/LSTM encoder_decoder/code/results/Fig4/fig21.pdf", dpi = 300, bbox_inches = 'tight')

    elif i == 4 :

        #Test samples target and network output
        plt.ylim(-100, 50,50)
        plt.xlim(-40, 700)
        plt.xticks(np.arange(0,701, step=175),fontsize=a)
        plt.yticks(np.arange(-100,51, step=50),fontsize=a)
        plt.plot(y_test_plot.transpose(),linewidth=3,color=[0.8,0,0],label='Simulated population')
        plt.plot(pred_plot_test.transpose(),linewidth=3,color=[0,0,1],label='Translated population')
        handles, labels = ax.get_legend_handles_labels()
        display = (0,400)
        plt.xlabel("Time(ms)",fontsize=a)
        plt.ylabel("Membrane potential (mV)",fontsize=a)
        ttl=plt.title("adult-CM AP population (test)",fontsize=a)
        ttl.set_position([.5, 1.1])
        plt.savefig("D:/Papers/LSTM encoder_decoder/code/results/Fig4/fig22.pdf", dpi = 300, bbox_inches = 'tight')

    elif i==5:
        #APD90 parameter distribution test
        plt.xlim(180, 640)
        plt.ylim(0, 0.15)
        plt.xticks(np.arange(200,641, step=110),fontsize=a)
        plt.yticks(np.arange(0,0.151, step=0.05),fontsize=a)
        sns.kdeplot(APD90_real_test, color=[0.8,0,0],linewidth=5, label="Simulated population", legend=False)
        sns.kdeplot(APD90_predict_test, color=[0,0,1],linewidth=5, label="Translated population", legend=False)
        plt.legend(loc = 'best', prop={'size': a})
        plt.xlabel("APD90",fontsize=a,labelpad=10)
        plt.ylabel("Frequency",fontsize=a,labelpad=15)
        ttl=plt.title("Distribution of adult-CM APD90 (test)",fontsize=a)
        ttl.set_position([.5, 1.1])
        plt.savefig("D:/Papers/LSTM encoder_decoder/code/results/Fig4/fig23.pdf", dpi = 300, bbox_inches = 'tight')


#Fig7
num_fig = 2
for i in range(num_fig):
    plt.figure(num=None, figsize=(16, 10), dpi=300)
    plt.rc('axes', linewidth=4)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', pad=15)
    ax.tick_params(axis="y",direction="in", length=16, width=4)
    ax.tick_params(axis="x",direction="in", length=16, width=4)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.ylim(-100, 50,50)
    plt.xlim(-40, 700)
    plt.xticks(np.arange(0,701, step=175),fontsize=a)
    plt.yticks(np.arange(-100,51, step=50),fontsize=a)

    if i == 0:
        #Experimental data
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(3)
        plt.plot(Data_exp.transpose(),linewidth=6)
        plt.xlabel("Time(ms)",fontsize=a)
        plt.ylabel("Membrane potential (mV)",fontsize=a)
        ttl=plt.title("iPSC-CM AP population",fontsize=a)
        ttl.set_position([.5, 1.1])
        plt.savefig("D:/Papers/LSTM encoder_decoder/code/results/Fig7/fig71.pdf", dpi = 300, bbox_inches = 'tight')
    elif i == 1 :

        #Translated adult AP from experimental data
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(3)
        plt.plot(pred_plot_exp.transpose(),linewidth=6)
        plt.xlabel("Time(ms)",fontsize=a)
        plt.ylabel("Membrane potential (mV)",fontsize=a)
        ttl=plt.title("Translated adult-CM AP population",fontsize=a)
        ttl.set_position([.5, 1.1])
        plt.savefig("D:/Papers/LSTM encoder_decoder/code/results/Fig7/fig72.pdf", dpi = 300, bbox_inches = 'tight')
