from __future__ import print_function
import os
import re
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import tensorflow as tf
import  pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import matplotlib as mpl
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import _LRScheduler
import random
import math
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score , accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
import seaborn as sns
from matplotlib.lines import Line2D
import scikitplot as skplt
from scipy import signal
import time


import argparse
import yaml

import network_arch as network


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--config_path', type = str , help = 'Path to config for loading parameters')

    args = parser.parse_args()

    if args.config_path:
        args = load_parameters(args)

    execute(args)


def load_parameters(args):
    with open(args.config_path, 'r') as config_file:
        parameters = yaml.safe_load(config_file)
        args.batch_size = parameters['batch_size']
        args.lr = parameters['lr']
        args.num_epochs = parameters['num_epochs']
        args.input_dim = parameters['input_dim']
        args.lstm_hidden_dim = parameters['lstm_hidden_dim']
        args.lstm_layer_dim = parameters['lstm_layer_dim']
        args.layer1_class = parameters['layer1_class']
        args.layer2_class = parameters['layer2_class']
        args.layer3_class = parameters['layer3_class']
        args.layer1_reg = parameters['layer1_reg']
        args.seq_dim = parameters['seq_dim']
        args.path_data = parameters['path_data']
        args.path_results = parameters['path_results']
        args.mode = parameters['mode']
        args.train_from_scratch = parameters['train_from_scratch']
    return args

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def execute(args):
    Input = pd.read_csv(args.path_data+"input_without_noise.csv", header=None)
    y_reg = pd.read_csv(args.path_data+"target_without_noise.csv", header=None)
    y_class = pd.read_csv(args.path_data+"target_labels.csv", header=None)
    Input=Input.iloc[:, :].values
    y_reg=y_reg.iloc[:, :].values



    #Onehot encoding
    enc = preprocessing.OneHotEncoder()
    enc.fit(y_class)
    y_class = enc.transform(y_class).toarray()


    # Remove mean / max (Input/targets)
    Input_mean=np.mean(Input)
    Input_removed_mean=Input-Input_mean
    Input_max=np.max(Input_removed_mean)
    Input_preprocess =  Input_removed_mean/Input_max
    Input = Input_preprocess
    y_reg_mean=np.mean(y_reg)
    y_reg_removed_mean=y_reg-y_reg_mean
    y_reg_max=np.max(y_reg_removed_mean)
    y_reg_preprocess=y_reg_removed_mean/y_reg_max

    # Concatenate targets
    target=np.append(y_reg_preprocess,y_class,axis=1)

    x_train, x_test, y_train, y_test = train_test_split(Input, target, test_size=0.2, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

    # Regression
    train_x = Variable(torch.from_numpy(x_train).float())
    train_x = train_x.unsqueeze_(-1)
    val_x = Variable(torch.from_numpy(x_val).float())
    val_x = val_x.unsqueeze_(-1)
    test_x = Variable(torch.from_numpy(x_test).float())
    test_x = test_x.unsqueeze_(-1)
    train_y1 = Variable(torch.from_numpy(y_train[:,0:701]).float())
    train_y1 = train_y1.unsqueeze_(-1)
    val_y1 = Variable(torch.from_numpy(y_val[:,0:701]).float())
    val_y1 = val_y1.unsqueeze_(-1)
    test_y1 = Variable(torch.from_numpy(y_test[:,0:701]).float())
    test_y1 = test_y1.unsqueeze_(-1)

    #Classification
    train_y2 = Variable(torch.from_numpy(np.reshape(y_train[:,701:702], (1,np.product(y_train[:,701:702].shape)))[0]).long())
    val_y2 = Variable(torch.from_numpy(np.reshape(y_val[:,701:702], (1,np.product(y_val[:,701:702].shape)))[0]).long())
    test_y2 = Variable(torch.from_numpy(np.reshape(y_test[:,701:702], (1,np.product(y_test[:,701:702].shape)))[0]).long())

    #Data_Loader
    train_dataset = TensorDataset(train_x, train_y1,train_y2)
    val_dataset = TensorDataset(val_x, val_y1,val_y2)
    test_dataset = TensorDataset(test_x, test_y1 ,test_y2)

    loaders = {
        'train' : DataLoader(train_dataset, batch_size = args.batch_size,shuffle=True, num_workers=0),
        'validation' : DataLoader(val_dataset, batch_size = args.batch_size,shuffle=False, num_workers=0),
        'test' : DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False, num_workers=0)
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = network.Network(args)

    if torch.cuda.device_count() >0 :
        print('Lets use ', torch.cuda.device_count(), 'GPUs!')
        model.LSTMnet.cuda()
    else:
        print('Lets use', 1 , 'CPUs')

    if args.mode == 'test' or args.train_from_scratch is False:
        checkpoint = torch.load(os.path.join(args.path_results,'checkpoint.tar'))
        model.LSTMnet.load_state_dict(checkpoint['LSTM_weights_dict'])
        model.opt.load_state_dict(checkpoint['optimizer_dict'])
        print('Restore model from checkpoint')
    else:
        print('initializing the model from the scratch')

    if args.mode == 'train':
        print('Start model training')

        for epoch in range(args.num_epochs):

            train_loss = 0.0
            valid_loss = 0.0
            train_correct = 0
            train_total = 0
            val_correct = 0
            val_total = 0

            if epoch%100==0:
                args.lr /= 4

            for batch_idx, (x_batch1, y1_batch, y2_batch) in enumerate(loaders['train']):
                x_batch , y1_batch, y2_batch = x_batch1.cuda(), y1_batch.cuda(), y2_batch.cuda()

                short_term , out1 , out2, loss1 , loss2 = model.train_step(args, x_batch, y1_batch, y2_batch  )

                train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss2.data- train_loss))
                pred = out1.data.max(1, keepdim=True)[1]
                # compare predictions to true label
                train_correct += np.sum(np.squeeze(pred.eq(y2_batch.data.view_as(pred))).cpu().numpy())
                train_total += x_batch.size(0)
                train_accuracy = 100* train_correct / train_total


            for batch_idx, (x_batch1, y1_batch, y2_batch) in enumerate(loaders['validation']):
                x_batch , y1_batch, y2_batch = x_batch1.cuda(), y1_batch.cuda(), y2_batch.cuda()

                short_term , out1 , out2, loss1 , loss2 = model.val_step(args, x_batch, y1_batch, y2_batch  )
                valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss2.data- valid_loss))
                pred = out1.data.max(1, keepdim=True)[1]
                # compare predictions to true label
                val_correct += np.sum(np.squeeze(pred.eq(y2_batch.data.view_as(pred))).cpu().numpy())
                val_total += x_batch.size(0)
                val_accuracy = 100* val_correct / val_total

                if epoch % 1 == 0:
                    print('Epoch[{}/{}] \tTraining Loss: {:.3f} \tTrain acc: {:.3f} \tValidation Loss: {:.3f} \tValidation acc: {:.3f}'.format(
                    epoch  , args.num_epochs,
                    train_loss,
                    train_accuracy,
                    valid_loss,
                    val_accuracy
                    ))

                if epoch % 20 == 0 :
                    checkpoint ={
                    'LSTM_weights_dict' : model.LSTMnet.state_dict(),
                    'optimizer_dict' : model.opt.state_dict(),
                    }
                    torch.save(checkpoint, os.path.join(args.path_results,'checkpoint.tar'))
                    torch.save(model.LSTMnet, os.path.join(args.path_results,'model_LSTM.pth'))
                    print('checkpont and model saved')


    elif args.mode == 'test' :
        loss1_ave = []
        loss2_ave = []
        mse_test = []
        R2_score_test = []
        APD90_real_test=[]
        APD90_pred_test=[]
        Error=[]
        test_pred_classification = None
        test_pred_translation = None
        test_real_classification = None
        test_real_translation = None
        x_test = None
        y_test = None
        for test_input, test_target1, test_target2 in loaders['test']:
            test_input = test_input.cuda()
            test_target1 = test_target1.cuda()
            test_target2 = test_target2.cuda()
            short_term , pred1_test , pred2_test, loss1 , loss2 = model.test_step(args, test_input, test_target1, test_target2)
            loss1_ave.append(loss1.cpu().numpy())
            loss2_ave.append(loss2.cpu().numpy())
            pred1_test = F.softmax(pred1_test)
            pred1_test = pred1_test.data >= 0.5
            test_pred_classification_temp = pred1_test[:,1].long().cpu().data.numpy().astype(int)
            test_pred_translation_temp = pred2_test.cpu().data.numpy()
            test_real_classification_temp = test_target2.cpu().data.numpy()
            test_real_translation_temp = test_target1.cpu().data.numpy()
            x_test_temp = test_input.cpu().data.numpy()
            y_test_temp = test_target1.cpu().data.numpy()

            if test_pred_classification is None:
                test_pred_classification = test_pred_classification_temp
                test_pred_translation = test_pred_translation_temp
                test_real_classification = test_real_classification_temp
                test_real_translation = test_real_translation_temp
                x_test = x_test_temp
                y_test = y_test_temp

            else:
                test_pred_classification = np.concatenate((test_pred_classification, test_pred_classification_temp),0)
                test_pred_translation = np.concatenate((test_pred_translation, test_pred_translation_temp),0)
                test_real_classification = np.concatenate((test_real_classification, test_real_classification_temp),0)
                test_real_translation = np.concatenate((test_real_translation, test_real_translation_temp),0)
                x_test = np.concatenate((x_test, x_test_temp),0)
                y_test = np.concatenate((y_test, y_test_temp),0)

        Final_ave_loss1 = np.mean(loss1_ave)
        Final_ave_loss2 = np.mean(loss2_ave)

        #MSE & R2_score
        for i in range(len(test_real_translation)):
            #MSE
            MSE_test = mean_squared_error(test_real_translation[i] ,  test_pred_translation[i])
            mse_test.append((MSE_test))
            #R2_score
            R2_test = r2_score(test_real_translation[i] ,  test_pred_translation[i])
            R2_score_test.append((R2_test))
            #APD90 error
            mai = test_real_translation[i,7]
            mii = test_real_translation[i,700]
            l=mai-(0.9*(mai-mii))
            t=find_nearest(test_real_translation[i], l)
            p=np.where(test_real_translation[i]==t)
            APD90_real_test.append(p[0])

        for i in range(len(test_pred_translation)):
            mai = test_pred_translation[i,7]
            mii = test_pred_translation[i,700]
            l=mai-(0.9*(mai-mii))
            t=find_nearest(test_pred_translation[i], l)
            p=np.where(test_pred_translation[i]==t)
            APD90_pred_test.append(p[0])

        for i in range(len(test_pred_translation)):
            e=100*abs((np.array(APD90_pred_test[i])-np.array(APD90_real_test[i]))/np.array(APD90_real_test[i]))
            Error.append(e)

        test_mse=np.mean(mse_test)
        test_R2_score=np.mean(R2_score_test)
        acc_test=accuracy_score(test_real_classification, test_pred_classification)
        mean_APD90_error_test= np.mean(np.array(Error))
        print('MSE test set:', test_mse, 'R2_score test set:', test_R2_score, 'Classification Accuracy test set:', '%.2f'%acc_test, 'Error APD90 normal:' , mean_APD90_error_test )


        #Metrics for training set
        train_pred_classification = None
        train_pred_translation = None
        train_real_classification = None
        train_real_translation = None
        x_train = None
        y_train = None
        for train_input, train_target1, train_target2 in loaders['train']:
            train_input = train_input.cuda()
            train_target1 = train_target1.cuda()
            train_target2 = train_target2.cuda()
            short_term, pred1_train , pred2_train, loss1 , loss2 = model.test_step(args, train_input, train_target1, train_target2)
            pred1_train = F.softmax(pred1_train)
            pred1_train = pred1_train.data >= 0.5
            train_pred_classification_temp = pred1_train[:,1].cpu().data.numpy().astype(int)
            train_pred_translation_temp = pred2_train.cpu().data.numpy()
            train_real_classification_temp = train_target2.cpu().data.numpy()
            train_real_translation_temp = train_target1.cpu().data.numpy()
            x_train_temp = train_input.cpu().data.numpy()
            y_train_temp = train_target1.cpu().data.numpy()

            if train_pred_classification is None:
                train_pred_classification = train_pred_classification_temp
                train_pred_translation = train_pred_translation_temp
                train_real_classification = train_real_classification_temp
                train_real_translation = train_real_translation_temp
                x_train = x_train_temp
                y_train = y_train_temp
            else:
                train_pred_classification = np.concatenate((train_pred_classification, train_pred_classification_temp),0)
                train_pred_translation = np.concatenate((train_pred_translation, train_pred_translation_temp),0)
                train_real_classification = np.concatenate((train_real_classification, train_real_classification_temp),0)
                train_real_translation = np.concatenate((train_real_translation, train_real_translation_temp),0)
                x_train = np.concatenate((x_train, x_train_temp),0)
                y_train = np.concatenate((y_train, y_train_temp),0)




        #MSE & R2_score
        mse_train = []
        R2_score_train = []
        APD90_real_train=[]
        APD90_pred_train=[]
        Error=[]
        for i in range(len(train_real_translation)):
            #MSE
            MSE_train = mean_squared_error(train_real_translation[i] ,  train_pred_translation[i])
            mse_train.append((MSE_train))
            #R2_score
            R2_train = r2_score(train_real_translation[i] ,  train_pred_translation[i])
            R2_score_train.append((R2_train))
            #APD90 error
            mai = train_real_translation[i,7]
            mii = train_real_translation[i,700]
            l=mai-(0.9*(mai-mii))
            t=find_nearest(train_real_translation[i], l)
            p=np.where(train_real_translation[i]==t)
            APD90_real_train.append(p[0])

        for i in range(len(train_pred_translation)):
            mai = train_pred_translation[i,7]
            mii = train_pred_translation[i,700]
            l=mai-(0.9*(mai-mii))
            t=find_nearest(train_pred_translation[i], l)
            p=np.where(train_pred_translation[i]==t)
            APD90_pred_train.append(p[0])

        for i in range(len(train_pred_translation)):
            e=100*abs((np.array(APD90_pred_train[i])-np.array(APD90_real_train[i]))/np.array(APD90_real_train[i]))
            Error.append(e)

        train_mse=np.mean(mse_train)
        train_R2_score=np.mean(R2_score_train)
        acc_train=accuracy_score(train_real_classification, train_pred_classification)
        mean_APD90_error_train= np.mean(np.array(Error))
        print('MSE train set:', train_mse, 'R2_score train set:', train_R2_score,'Classification Accuracy train set:', '%.2f'%acc_train, 'Error APD90 normal:' , mean_APD90_error_train)

        #Save variables
        np.savez('mat.npz', x_train=x_train, x_test=x_test, y_train = y_train, y_test = y_test, Input_max=Input_max , Input_mean = Input_mean , y_reg_mean = y_reg_mean, y_reg_max = y_reg_max , train_pred_translation = train_pred_translation,
        test_pred_translation = test_pred_translation,APD90_real_train = APD90_real_train, APD90_pred_train = APD90_pred_train, APD90_real_test = APD90_real_test, APD90_pred_test = APD90_pred_test )

    elif args.mode == 'implement':
        Data_exp = pd.read_csv(args.path_data+"experiments.csv", header=None)
        Data_exp = Data_exp.iloc[:, :].values
        # Normalizing experimental Data_exp
        input_exp = Data_exp-Input_mean
        input_exp = input_exp/Input_max
        input_exp=Variable(torch.from_numpy(input_exp).float())
        input_exp = input_exp.unsqueeze_(-1)
        input_exp = input_exp.cuda()
        dummy_y1 = np.zeros((input_exp.shape[0],input_exp.shape[1]))
        dummy_y1 = Variable(torch.from_numpy(dummy_y1).float())
        dummy_y1 = dummy_y1.unsqueeze_(-1)
        dummy_y1 = dummy_y1.cuda()
        dummy_y2 = np.zeros((input_exp.shape[0],1))
        dummy_y2 = Variable(torch.from_numpy(dummy_y2).float())
        dummy_y2 = dummy_y2.unsqueeze_(-1)
        dummy_y2 = dummy_y2.cuda()
        exp_dataset = TensorDataset(input_exp, dummy_y1,dummy_y2)
        exp_iterator = DataLoader(exp_dataset, batch_size = input_exp.shape[0])
        for exp_input, exp_target1, exp_target2 in exp_iterator:
            history, pred1_exp , pred2_exp = model.implement(args, exp_input)
            # pred1_exp = F.softmax(pred1_exp)
            # pred1_exp = pred1_exp.data >= 0.5
            exp_pred_classification = pred1_exp[:,1].long().cpu().data.numpy().astype(int)
            exp_pred_translation = pred2_exp.cpu().data.numpy()
            pred_plot_exp = exp_pred_translation.reshape((exp_pred_translation.shape[0], exp_pred_translation.shape[1] ))
            pred_plot_exp=(pred_plot_exp*y_reg_max)+y_reg_mean

            np.savez('mat_implement.npz', Data_exp = Data_exp ,  pred_plot_exp = pred_plot_exp)



if __name__ == '__main__':
    main()
