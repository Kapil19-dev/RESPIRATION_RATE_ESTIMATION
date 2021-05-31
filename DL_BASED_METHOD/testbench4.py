from skorch import NeuralNetRegressor
import numpy as np
import pandas as pd
from data_extraction import *
from resp_signal_extraction import *
import torch
import re
from torch.utils.data import TensorDataset, DataLoader
from sklearn import utils
from model import BRUnet_mod,BRUnet_Multi,BRUnet_Encoder,BRUnet_raw, BRUnet_raw_mod,BRUnet_raw_Multi
import os
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from sklearn.preprocessing import MinMaxScaler
from torch.utils.tensorboard import SummaryWriter
import sys
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
import random
from rr_extration import *
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
### Ray tune related libraries
# from ray import tune
# from ray.tune import CLIReporter
# from ray.tune.schedulers import ASHAScheduler

model_basename = input("Enter the name for the model directory:")
model_type = input("Enter the model type:")
data_path = '/media/hticpose/drive1/Prithvi/PPG/PPG_FieldStudy/PPG_DALIA_DATA'
path = '../../../Prithvi/PPG/PPG_FieldStudy'
writer = SummaryWriter(os.path.join(path, "runs"))
srate = 700
win_length = 32*srate
data = extract_data(data_path , srate , win_length)

seed = 42
random.seed(seed)                                                                                                  
torch.manual_seed(seed)                                                                                            
torch.backends.cudnn.deterministic = True                                                                               
torch.backends.cudnn.benchmark = False     

for item in enumerate(data.keys()):
    patient_id = item[1]  
    ecg = data[patient_id]['ECG']['ECG_DATA']
    rpeaks = data[patient_id]['ECG']['RPEAKS']
    amps = data[patient_id]['ECG']['AMPLITUDES']
    acc = data[patient_id]['ACC']['ACC_DATA']
    resp = data[patient_id]['RESP']['RESP_DATA']
    activity_id = data[patient_id]['ACTIVITY_ID']
    scaler = MinMaxScaler()
    #import pdb;pdb.set_trace()

    #print(len(acc[0]))
    #print(len(acc[1]))
    edr_hrv , edr_rpeak , adr , ref_resp = edr_adr_extraction(acc, rpeaks , amps , resp)
    
    for i in range(len(edr_hrv)):
        edr_hrv[i] = np.append(edr_hrv[i] , np.zeros(128 - len(edr_hrv[i])))
        edr_rpeak[i] = np.append(edr_rpeak[i] , np.zeros(128 - len(edr_rpeak[i])))
        adr[i] = np.append(adr[i] , np.zeros(128 - len(adr[i])))
        ref_resp[i] = np.append(ref_resp[i] , np.zeros(128 - len(ref_resp[i])))
    ref_rr_duration, _ =  extremas_extraction(ref_resp)
    ref_rr = (60*4)/ref_rr_duration
       
    
    edr_hrv , edr_rpeak , adr , ref_resp = np.expand_dims(np.asarray(edr_hrv), axis = -1), np.expand_dims(np.asarray(edr_rpeak), axis = -1)\
                               , np.expand_dims(np.asarray(adr), axis =-1) , np.expand_dims(np.asarray(ref_resp), axis =-1)
    
    edr_hrv = scaler.fit_transform(edr_hrv.reshape(len(edr_hrv),len(edr_hrv[0])))
    edr_rpeak = scaler.fit_transform(edr_rpeak.reshape(len(edr_rpeak),len(edr_rpeak[0])))
    adr = scaler.fit_transform(adr.reshape(len(adr),len(adr[0])))
    ref_resp = scaler.fit_transform(ref_resp.reshape(len(ref_resp),len(ref_resp[0])))

    windowed_inp = np.concatenate((np.expand_dims(edr_hrv, 1), np.expand_dims(edr_rpeak, 1), np.expand_dims(adr, 1)), axis = 1)
    int_part  = re.findall(r'\d+', patient_id)
    sub_activity_ids = np.hstack((ref_rr.reshape(-1,1),np.array(activity_id).reshape(-1,1), np.array([int(int_part[0])]*len(edr_hrv)).reshape(-1,1)))

    if item[0] == 0:
        final_windowed_inp = windowed_inp
        final_windowed_op = np.array(ref_resp)
        final_sub_activity_ids = sub_activity_ids
    else:
        final_windowed_inp = np.vstack((final_windowed_inp , windowed_inp))
        final_windowed_op = np.vstack((final_windowed_op , ref_resp))
        final_sub_activity_ids = np.vstack((final_sub_activity_ids , sub_activity_ids))

activity_df = pd.DataFrame(final_sub_activity_ids , columns = ['Reference_RR' , 'activity_id','patient_id'])
activity_df.to_pickle('/media/hticpose/drive1/Prithvi/PPG/PPG_FieldStudy/annotation.pkl')
#print(final_sub_activity_ids.shape)
#     
torch_input = torch.from_numpy(final_windowed_inp)
torch_output = torch.from_numpy(final_windowed_op)
torch.save(torch_input , '/media/hticpose/drive1/Prithvi/PPG/PPG_FieldStudy/input_signal.pt')
torch.save(torch_output , '/media/hticpose/drive1/Prithvi/PPG/PPG_FieldStudy/output_signal.pt')
annotation = pd.read_pickle('/media/hticpose/drive1/Prithvi/PPG/PPG_FieldStudy/annotation.pkl')
input_data = torch.load('/media/hticpose/drive1/Prithvi/PPG/PPG_FieldStudy/input_signal.pt')
output_data = torch.load('/media/hticpose/drive1/Prithvi/PPG/PPG_FieldStudy/output_signal.pt')
reference_rr = (annotation['Reference_RR'].values).reshape(-1,1)
torch_ref_rr = torch.from_numpy(reference_rr)
raw_signal = torch.load('/media/hticpose/drive1/Prithvi/PPG/PPG_FieldStudy/raw_signals.pt')

training_ids = annotation['patient_id'] < 13
x_train_data = input_data[torch.from_numpy(training_ids.values)]
x_test_data = input_data[torch.from_numpy(~(training_ids.values))]
x_train_ref_rr = torch_ref_rr[torch.from_numpy(training_ids.values)]
x_test_ref_rr = torch_ref_rr[torch.from_numpy(~(training_ids.values))]
raw_train_signals = raw_signal[torch.from_numpy(training_ids.values)]
raw_test_signals = raw_signal[torch.from_numpy(~(training_ids.values))]

y_train_data = output_data[torch.from_numpy(training_ids.values)]
y_test_data = output_data[torch.from_numpy(~(training_ids.values))]
#
train_ids = annotation.loc[annotation['patient_id']<13]
test_ids = annotation.loc[annotation['patient_id']>=13]

if model_type.lower()=='raw_dl':
    #torch_train_data = TensorDataset(raw_train_signals , x_train_ref_rr)
    #torch_train_data = TensorDataset(raw_train_signals , y_train_data)
    torch_train_data = TensorDataset(raw_train_signals , y_train_data,x_train_ref_rr)
    trainloader = DataLoader(torch_train_data, batch_size = 128 , shuffle=True) 
    #torch_test_data = TensorDataset(raw_test_signals , x_test_ref_rr)
    #torch_test_data = TensorDataset(raw_test_signals , y_test_data)
    torch_test_data = TensorDataset(raw_test_signals , y_test_data,x_test_ref_rr)
    testloader = DataLoader(torch_test_data, batch_size = 128 , shuffle=False)

else:
    torch_train_data=TensorDataset(x_train_data,y_train_data,x_train_ref_rr,torch.from_numpy((train_ids['patient_id'].values).reshape(-1,1)),
                       torch.from_numpy((train_ids['activity_id'].values).reshape(-1,1))) #TensorDataset(torch_input, reference_respiration)

    #torch_train_data = TensorDataset(x_train_data, y_train_data, torch.from_numpy((train_ids['patient_id'].values).reshape(-1,1)),
    #                   torch.from_numpy((train_ids['activity_id'].values).reshape(-1,1)))                   
    #
    # config = {
    #     "l1": tune.sample_from(lambda _: 2**np.random.randint(2, 9)),
    #     "l2": tune.sample_from(lambda _: 2**np.random.randint(2, 9)),
    #     "lr": tune.loguniform(1e-4, 1e-1),
    #     "batch_size": tune.choice([64,128,256,512])
    # }
    #
    trainloader = DataLoader(torch_train_data, batch_size = 128 , shuffle=True) 
    #
    torch_test_data = TensorDataset(x_test_data, y_test_data,x_test_ref_rr, torch.from_numpy((test_ids['patient_id'].values).reshape(-1,1))
                     ,torch.from_numpy((test_ids['activity_id'].values).reshape(-1,1)))

                      #TensorDataset(torch_input, reference_respiration)

    #torch_test_data = TensorDataset(x_test_data, y_test_data,torch.from_numpy((test_ids['patient_id'].values).reshape(-1,1))
    #                 ,torch.from_numpy((test_ids['activity_id'].values).reshape(-1,1))) 
    testloader = DataLoader(torch_test_data, batch_size = 128 , shuffle=False)
#
#
model = BRUnet_raw_Multi((64,3,128)).cuda()
#model = BRUnet_raw_mod((64,3,128)).cuda()
#model = BRUnet_Multi((64,3,128)).cuda()
learning_rate = 0.005
criterion = torch.nn.SmoothL1Loss() # Loss
criterion_1 = torch.nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
num_epochs = 1000
train_loss_list= [] 
train_loss_list_rr= [] 
#ean_loss = []
#import pdb;pdb.set_trace()
saved_model = dt.datetime.now().strftime('%Y_%m_%d_%H_%M')
results_path = os.path.join('/media/hticpose/drive1/Prithvi/PPG/PPG_FieldStudy', model_basename)
if not(os.path.isdir(results_path)):
   os.mkdir(results_path)   
current_model_path = os.path.join(results_path, saved_model) 
os.mkdir(current_model_path)
writer = SummaryWriter(os.path.join(current_model_path, "runs"))
log_path = current_model_path + '_logs.txt' 
log_writer = open(log_path, 'w')
#
### Skorch based hyperparameter tuning
#import pdb;pdb.set_trace()
#scorer = make_scorer(mean_squared_error, greater_is_better=False)
#net = NeuralNetRegressor(model
#                          , max_epochs=1000
#                          , lr=0.001
#                          , verbose=1)


#params = {
#     'lr': [0.0001,0.001,0.005, 0.01, 0.05,0.1,0.5],
#     'max_epochs': list(range(500,5500, 500))
# }
#
#print('grid search is running')
#gs = RandomizedSearchCV(net , params , cv = 5 , scoring = scorer,random_state=0)
#gs.fit(x_train_data , y_train_data)
#best_parameters = gs.best_params_
#print(best_parameters)
#
#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer , milestones=[3000],gamma=0.1)
#
#

lamda = 0.01
for epoch in range(num_epochs):
    model.train()
    print("..............Training............")
    #for i, (br_input, br_gt,ref_rr,_, _) in enumerate(trainloader):      #ref_rr,
    #for i, (br_input, br_gt) in enumerate(trainloader):
    for i, (br_input, br_gt,ref_rr) in enumerate(trainloader):  
        #import pdb;pdb.set_trace()
        br_input = br_input.float().cuda()
        #import pdb;pdb.set_trace()
        br_gt = br_gt.float().view(-1,1,br_gt.shape[-1]).cuda()
        ref_rr = ref_rr.float().cuda() 
#       #br_input = br_input.permute(0,2,1)
        optimizer.zero_grad()
        model_output_resp,out_4 = model(br_input)
        #model_output_resp = model(br_input)
        #out_4 = model(br_input)
        loss_resp_signal = criterion(br_gt, model_output_resp)
        loss_rr_rate = criterion_1(ref_rr, out_4)
        loss_rr_rate = lamda * loss_rr_rate 
        total_loss = loss_resp_signal + loss_rr_rate
        train_loss_list.append(float(total_loss.cpu().data))
        #train_loss_list.append(float(loss_resp_signal.cpu().data))
        #train_loss_list = np.hstack((train_loss_list , np.squeeze(loss_resp_signal.detach().cpu().numpy(),1)))
        total_loss.backward()
        #loss_resp_signal.backward()
        #loss_rr_rate.backward()
        optimizer.step()
#       #scheduler.step()
        if (i+1) % 10 == 0:
            print('Epoch [%d/%d], lter [%d/%d] Loss: %.4f'
                    %(epoch+1, num_epochs, i+1, len(trainloader), total_loss.cpu().data))
    print("net loss -- {}".format(np.mean(np.array(train_loss_list))))
    #print("net loss rr -- {}".format(np.mean(np.array(train_loss_list_rr))))
#   
    log_writer.write('Epoch: {}, Training loss: {} \n'.format(epoch, np.mean(np.array(train_loss_list)))) 
    writer.add_scalar('Training_loss' ,np.mean(np.array(train_loss_list)) , epoch)
    #writer.add_scalar('Training_loss_rr' ,np.mean(np.array(train_loss_list_rr)) , epoch)
    writer.close()
    model.eval()
    test_loss_list = []
    best_loss = 10000
    print("..............Testing...............")
    with torch.no_grad():
        #for i, (br_input, br_gt,ref_rr,_, _) in enumerate(testloader):   #ref_rr,
        #for i, (br_input, br_gt) in enumerate(testloader):
        for i, (br_input, br_gt,ref_rr) in enumerate(testloader):
            #import pdb;pdb.set_trace()
            br_input = br_input.float().cuda()
            br_gt = br_gt.float().view(-1,1,br_gt.shape[-1]).cuda()
            ref_rr = ref_rr.float().cuda()  
#           #br_input = br_input.permute(0,2,1)
            model_output_resp,out_4 = model(br_input)
            #model_output_resp = model(br_input)
            #out_4 = model(br_input)
#           model_output_resp = model_output_resp.view(-1,1,model_output_resp.shape[-1])
            loss_resp = criterion(br_gt, model_output_resp)
            loss_rr = criterion_1(ref_rr,out_4)
            loss_rr = lamda * loss_rr
            total_loss = loss_resp + loss_rr
            test_loss_list.append(float(total_loss.cpu().data))
            #test_loss_list.append(float(loss_resp.cpu().data))
            #test_loss_list = np.hstack((test_loss_list , np.squeeze(loss_resp.detach().cpu().numpy(),1)))
            #test_loss_list.append(float(loss_resp.cpu().data))
#       #import pdb;pdb.set_trace()
        mean_loss = (sum(test_loss_list) / len(test_loss_list))         
        log_writer.write( 'Epoch: {}, Testing loss Resp: {}\n'.format(epoch, np.mean(np.array(test_loss_list)))) 
        writer.add_scalar('validation_loss' ,np.mean(np.array(test_loss_list)) , epoch)
        writer.close()
#
#   # with tune.checkpoint_dir(epoch) as checkpoint_dir:
#           
#   #         path = os.path.join(checkpoint_dir, "checkpoint")
#   #         torch.save((net.state_dict(), optimizer.state_dict()), path)
#
#   #     tune.report(loss=(val_loss / val_steps), accuracy=correct / total)
#
#
#   #import pdb;pdb.set_trace()
    if mean_loss<best_loss:
        best_loss = mean_loss
        log_writer.write( 'Epoch: {} - Saving model \n'.format(epoch)) 
        torch.save({'model_state_dict':model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),'loss':criterion, 'lr': learning_rate}
                    , os.path.join(current_model_path , 'parameters.pt'))
        torch.save(model , os.path.join(current_model_path , 'model.pt'))
#
# best_model = torch.load('/media/hticpose/drive1/Prithvi/PPG/PPG_FieldStudy/results/2021_03_18_13_16/model.pt')
# best_model_parameters = torch.load('/media/hticpose/drive1/Prithvi/PPG/PPG_FieldStudy/results/2021_03_18_13_16/parameters.pt')
# best_model.eval()
# test_loss = []
# final_output = torch.tensor([]).cuda()
# for i, (br_test_input, br_test_gt, _, _) in enumerate(testloader):
#     br_test_input = br_test_input.float().cuda()
#     br_test_gt = br_test_gt.float().view(-1,1,br_test_gt.shape[-1]).cuda() 
#     model_output = best_model(br_test_input)
#     model_output = model_output.view(-1,1,model_output.shape[-1])
#     #import pdb;pdb.set_trace()
#     loss = criterion(br_test_gt, model_output)
#     final_output = torch.cat((final_output , model_output) , dim= 0 )
#     test_loss.append(float(loss.cpu().data))
#
#
# final_resp_sig = final_output.detach().cpu().numpy()
# final_resp_sig = final_resp_sig.reshape(final_resp_sig.shape[0] , final_resp_sig.shape[2])
# print(final_resp_sig.shape)

