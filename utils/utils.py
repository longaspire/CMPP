import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
import math
import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error,mean_absolute_percentage_error
import scipy.linalg as la
import copy



class GCN(nn.Module):
    def __init__(self, sym_norm_Adj_matrix, in_channels, out_channels, dropout=0.0):
        super(GCN, self).__init__()
        self.sym_norm_Adj_matrix = sym_norm_Adj_matrix
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta0 = nn.Linear(in_channels, in_channels, bias=False)
        self.Theta = nn.Linear(in_channels, out_channels, bias=False)
    def forward(self, x):
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape
        x = x.permute(0, 2, 1, 3).reshape((-1, num_of_vertices, in_channels))  # (b*t,n,f_in)
        return F.relu(self.Theta(torch.matmul(self.sym_norm_Adj_matrix, x)).reshape((batch_size, num_of_timesteps, num_of_vertices, self.out_channels)).transpose(1, 2))

class gcn_gru_unit(nn.Module):
    def __init__(self,sym_norm_Adj_matrix, in_channels, out_channels,hidden_size,dropout=.0,):
        super(gcn_gru_unit,self).__init__()
        self.sa_gcn = GCN(sym_norm_Adj_matrix, in_channels, out_channels, dropout)
        self.gru = nn.GRU(input_size=out_channels,hidden_size=hidden_size,num_layers=1,batch_first=True)
        self.hidden_size = hidden_size
    def forward(self,x):
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape
        sa_gcn_out = self.sa_gcn(x) #(b,n,t,f_out)
        batch_size, num_of_vertices, num_of_timesteps, out_channels = sa_gcn_out.shape
        sa_gcn_out = sa_gcn_out.reshape((-1,num_of_timesteps,out_channels))
        gru_out, gru_hidden = self.gru(sa_gcn_out)
        gru_hidden = gru_hidden.squeeze()
        last_gru_out = gru_out[:,-1,:].squeeze() #(b*n, hidden_size)
        last_gru_out = last_gru_out.reshape(batch_size, num_of_vertices,self.hidden_size) # batch_size,num_of_vertices,hidden_size
        return last_gru_out #(b,n,hidden_size)

class gru_gcn_unit(nn.Module):
    def __init__(self, sym_norm_Adj_matrix, in_channels, out_channels, hidden_size, dropout=.0):
        super(gru_gcn_unit, self).__init__()
        self.gru = nn.GRU(input_size=in_channels, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.sa_gcn = GCN(sym_norm_Adj_matrix, hidden_size, out_channels, dropout)
        self.hidden_size = hidden_size
    def forward(self, x):
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape
        x = x.reshape((-1, num_of_timesteps, in_channels))
        gru_out, gru_hidden = self.gru(x)
        gru_out = gru_out.reshape(batch_size, num_of_vertices, num_of_timesteps,self.hidden_size)
        sa_gcn_out = self.sa_gcn(gru_out) #(b,n,t,f_out)
        sa_gcn_out = sa_gcn_out[:,:,-1,:].squeeze()
        return sa_gcn_out #(b,n,f_out)


class gru_unit(nn.Module):
    def __init__(self,in_channels,hidden_size):
        super(gru_unit, self).__init__()
        self.gru = nn.GRU(input_size=in_channels, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.hidden_size = hidden_size

    def forward(self,x):
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape
        x = x.reshape((-1, num_of_timesteps, in_channels))
        gru_out, gru_hidden = self.gru(x)
        last_gru_out = gru_out[:,-1,:].squeeze()
        last_gru_out = last_gru_out.reshape(batch_size, num_of_vertices,self.hidden_size) # batch_size,num_of_vertices,hidden_size
        return last_gru_out #(b,n,hidden_size)

class Attention(nn.Module):
    def __init__(self,att_input_size, att_out_size):
        super(Attention, self).__init__()
        self.k_weight = nn.Linear(att_input_size,att_out_size)
        self.q_weight = nn.Linear(att_input_size,att_out_size)
        self.v_weight = nn.Linear(att_input_size,att_out_size)

    def forward(self,x):
        gru_gcn_out, gcn_gru_out, gru_out, gcn_out = x
        b, n, f_out = gru_gcn_out.shape
        b, n, hidden_size = gcn_gru_out.shape
        gru_gcn_out = gru_gcn_out.reshape(-1,f_out)
        gru_gcn_out = torch.unsqueeze(gru_gcn_out,1)

        gcn_out = gcn_out.reshape(-1,f_out)
        gcn_out = torch.unsqueeze(gcn_out,1)

        gcn_gru_out = gcn_gru_out.reshape(-1,hidden_size)
        gcn_gru_out = torch.unsqueeze(gcn_gru_out,1)

        gru_out = gru_out.reshape(-1,hidden_size)
        gru_out = torch.unsqueeze(gru_out,1)

        att_input = torch.cat([gru_gcn_out,gcn_gru_out,gcn_out,gru_out],dim=1)

        k = self.k_weight(att_input)
        q = self.q_weight(att_input)
        v = self.v_weight(att_input)
        s = F.softmax(torch.matmul(k,q.transpose(1,2)), dim=-1)
        att_out = torch.matmul(s,v) #(b*n, 4, hidden_size)
        gru_gcn_out, gcn_gru_out, gru_out, gcn_out = att_out[:,0,:], att_out[:,1,:], att_out[:,2,:], att_out[:,3,:]
        return gru_gcn_out, gcn_gru_out, gru_out, gcn_out




class ParalleStGcn(nn.Module):
    def __init__(self,sym_norm_Adj_matrix, in_channels, out_channels, hidden_size,output_size, dropout=.0):
        super(ParalleStGcn, self).__init__()
        self.gru_gcn = gru_gcn_unit(sym_norm_Adj_matrix, in_channels, out_channels, hidden_size, dropout=.0)
        self.gcn_gru = gcn_gru_unit(sym_norm_Adj_matrix, in_channels, out_channels, hidden_size, dropout=.0)
        self.gru_unit = gru_unit(in_channels,hidden_size)
        self.gcn_unit = GCN(sym_norm_Adj_matrix, in_channels, out_channels, dropout)
        self.concat_size = 2*(hidden_size+out_channels)
        # self.concat_size = hidden_size
        self.att = Attention(att_input_size=hidden_size,att_out_size=hidden_size)
        self.fc1 = nn.Linear(self.concat_size,output_size)
        self.fc2 = nn.Linear(self.concat_size,output_size)


    def forward(self,x):
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape
        gru_gcn_out = self.gru_gcn(x) #(b,n,f_out)
        gru_gcn_out_unsq = torch.unsqueeze(gru_gcn_out,dim=2)
        gcn_gru_out = self.gcn_gru(x) #(b,n,hidden_size)
        # print('gcn_gru_out shape',gcn_gru_out.shape)
        gru_out = self.gru_unit(x) #(b,n,hidden_size)
        # print('gru_out shape',gru_out.shape)
        gcn_out = self.gcn_unit(x)
        gcn_out = gcn_out[:,:,-1,:].squeeze() #(b,n,f_out)
        # print('gcn_out shape',gcn_out.shape)

        # attention
        gru_gcn_out, gcn_gru_out, gru_out, gcn_out = self.att((gru_gcn_out, gcn_gru_out, gru_out, gcn_out))


        # concat
        # concat_out = torch.cat([gcn_gru_out],dim=-1)
        # print('concat_out shape',concat_out.shape)
        concat_out = torch.cat([gru_gcn_out,gcn_gru_out,gru_out,gcn_out],dim=-1)

        concat_out = concat_out.reshape(-1,self.concat_size)
        output_1 = self.fc1(concat_out)
        output_2 = self.fc2(concat_out)
        return output_1,output_2




class ParalleStGcnS(nn.Module):
    def __init__(self,sym_norm_Adj_matrix, in_channels, out_channels, hidden_size,output_size, dropout=.0):
        super(ParalleStGcnS, self).__init__()
        self.gru_gcn = gru_gcn_unit(sym_norm_Adj_matrix, in_channels, out_channels, hidden_size, dropout=.0)
        self.gcn_gru = gcn_gru_unit(sym_norm_Adj_matrix, in_channels, out_channels, hidden_size, dropout=.0)
        self.gru_unit = gru_unit(in_channels,hidden_size)
        self.gcn_unit = GCN(sym_norm_Adj_matrix, in_channels, out_channels, dropout)
        # self.concat_size = 2*(hidden_size+out_channels)
        self.concat_size = hidden_size*3
        self.att = GCN(att_input_size=hidden_size,att_out_size=hidden_size)
        self.fc1 = nn.Linear(self.concat_size,output_size)
        self.fc2 = nn.Linear(self.concat_size,output_size)
    def forward(self,x):
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape
        gru_gcn_out = self.gru_gcn(x) #(b,n,f_out)
        gcn_gru_out = self.gcn_gru(x) #(b,n,hidden_size)
        # print('gcn_gru_out shape',gcn_gru_out.shape)
        # gru_out = self.gru_unit(x) #(b,n,hidden_size)
        # print('gru_out shape',gru_out.shape)
        gcn_out = self.gcn_unit(x)
        gcn_out = gcn_out[:,:,-1,:].squeeze() #(b,n,f_out)

        # attention
        gru_gcn_out, gcn_gru_out, gcn_out = self.att((gru_gcn_out,gcn_gru_out, gcn_out))
        # concat
        # concat_out = torch.cat([gcn_gru_out],dim=-1)
        # print('concat_out shape',concat_out.shape)
        concat_out = torch.cat([gru_gcn_out, gcn_gru_out, gcn_out],dim=-1)

        concat_out = concat_out.reshape(-1,self.concat_size)
        output_1 = self.fc1(concat_out)
        output_2 = self.fc2(concat_out)
        return output_1,output_2


class ParalleStGcnT(nn.Module):
    def __init__(self,sym_norm_Adj_matrix, in_channels, out_channels, hidden_size,output_size, dropout=.0):
        super(ParalleStGcnT, self).__init__()
        self.gru_gcn = gru_gcn_unit(sym_norm_Adj_matrix, in_channels, out_channels, hidden_size, dropout=.0)
        self.gcn_gru = gcn_gru_unit(sym_norm_Adj_matrix, in_channels, out_channels, hidden_size, dropout=.0)
        self.gru_unit = gru_unit(in_channels,hidden_size)
        self.gcn_unit = GCN(sym_norm_Adj_matrix, in_channels, out_channels, dropout)
        # self.concat_size = 2*(hidden_size+out_channels)
        self.concat_size = hidden_size*3
        self.att = GCN(att_input_size=hidden_size,att_out_size=hidden_size)
        self.fc1 = nn.Linear(self.concat_size,output_size)
        self.fc2 = nn.Linear(self.concat_size,output_size)
    def forward(self,x):
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape
        gru_gcn_out = self.gru_gcn(x) #(b,n,f_out)
        gcn_gru_out = self.gcn_gru(x) #(b,n,hidden_size)
        # print('gcn_gru_out shape',gcn_gru_out.shape)
        gru_out = self.gru_unit(x) #(b,n,hidden_size)
        # print('gru_out shape',gru_out.shape)
        # gcn_out = self.gcn_unit(x)
        # gcn_out = gcn_out[:,:,-1,:].squeeze() #(b,n,f_out)

        # attention
        gru_gcn_out, gcn_gru_out, gru_out = self.att((gru_gcn_out,gcn_gru_out, gru_out))
        # concat
        # concat_out = torch.cat([gcn_gru_out],dim=-1)
        # print('concat_out shape',concat_out.shape)
        concat_out = torch.cat([gru_gcn_out, gcn_gru_out, gru_out],dim=-1)

        concat_out = concat_out.reshape(-1,self.concat_size)
        output_1 = self.fc1(concat_out)
        output_2 = self.fc2(concat_out)
        return output_1,output_2


class ParalleStGcnTS(nn.Module):
    def __init__(self,sym_norm_Adj_matrix, in_channels, out_channels, hidden_size,output_size, dropout=.0):
        super(ParalleStGcnTS, self).__init__()
        self.gru_gcn = gru_gcn_unit(sym_norm_Adj_matrix, in_channels, out_channels, hidden_size, dropout=.0)
        self.gcn_gru = gcn_gru_unit(sym_norm_Adj_matrix, in_channels, out_channels, hidden_size, dropout=.0)
        self.gru_unit = gru_unit(in_channels,hidden_size)
        self.gcn_unit = GCN(sym_norm_Adj_matrix, in_channels, out_channels, dropout)
        # self.concat_size = 2*(hidden_size+out_channels)
        self.concat_size = hidden_size*3
        self.att = GCN(att_input_size=hidden_size,att_out_size=hidden_size)
        self.fc1 = nn.Linear(self.concat_size,output_size)
        self.fc2 = nn.Linear(self.concat_size,output_size)
    def forward(self,x):
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape
        gru_gcn_out = self.gru_gcn(x) #(b,n,f_out)
        # gcn_gru_out = self.gcn_gru(x) #(b,n,hidden_size)
        # print('gcn_gru_out shape',gcn_gru_out.shape)
        gru_out = self.gru_unit(x) #(b,n,hidden_size)
        # print('gru_out shape',gru_out.shape)
        gcn_out = self.gcn_unit(x)
        gcn_out = gcn_out[:,:,-1,:].squeeze() #(b,n,f_out)

        # attention
        gru_gcn_out, gru_out, gcn_out = self.att((gru_gcn_out,gru_out, gcn_out))
        # concat
        # concat_out = torch.cat([gcn_gru_out],dim=-1)
        # print('concat_out shape',concat_out.shape)
        concat_out = torch.cat([gru_gcn_out, gru_out, gcn_out],dim=-1)

        concat_out = concat_out.reshape(-1,self.concat_size)
        output_1 = self.fc1(concat_out)
        output_2 = self.fc2(concat_out)
        return output_1,output_2

class ParalleStGcnST(nn.Module):
    def __init__(self,sym_norm_Adj_matrix, in_channels, out_channels, hidden_size,output_size, dropout=.0):
        super(ParalleStGcnST, self).__init__()
        self.gru_gcn = gru_gcn_unit(sym_norm_Adj_matrix, in_channels, out_channels, hidden_size, dropout=.0)
        self.gcn_gru = gcn_gru_unit(sym_norm_Adj_matrix, in_channels, out_channels, hidden_size, dropout=.0)
        self.gru_unit = gru_unit(in_channels,hidden_size)
        self.gcn_unit = GCN(sym_norm_Adj_matrix, in_channels, out_channels, dropout)
        # self.concat_size = 2*(hidden_size+out_channels)
        self.concat_size = hidden_size*3
        self.att = GCN(att_input_size=hidden_size,att_out_size=hidden_size)
        self.fc1 = nn.Linear(self.concat_size,output_size)
        self.fc2 = nn.Linear(self.concat_size,output_size)
    def forward(self,x):
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape
        # gru_gcn_out = self.gru_gcn(x) #(b,n,f_out)
        gcn_gru_out = self.gcn_gru(x) #(b,n,hidden_size)
        # print('gcn_gru_out shape',gcn_gru_out.shape)
        gru_out = self.gru_unit(x) #(b,n,hidden_size)
        # print('gru_out shape',gru_out.shape)
        gcn_out = self.gcn_unit(x)
        gcn_out = gcn_out[:,:,-1,:].squeeze() #(b,n,f_out)

        # attention
        gcn_gru_out, gru_out, gcn_out = self.att((gcn_gru_out,gru_out, gcn_out))
        # concat
        # concat_out = torch.cat([gcn_gru_out],dim=-1)
        # print('concat_out shape',concat_out.shape)
        concat_out = torch.cat([gcn_gru_out, gru_out, gcn_out],dim=-1)

        concat_out = concat_out.reshape(-1,self.concat_size)
        output_1 = self.fc1(concat_out)
        output_2 = self.fc2(concat_out)
        return output_1,output_2


class ParalleStGcnATT(nn.Module):
    def __init__(self,sym_norm_Adj_matrix, in_channels, out_channels, hidden_size,output_size, dropout=.0):
        super(ParalleStGcnATT, self).__init__()
        self.gru_gcn = gru_gcn_unit(sym_norm_Adj_matrix, in_channels, out_channels, hidden_size, dropout=.0)
        self.gcn_gru = gcn_gru_unit(sym_norm_Adj_matrix, in_channels, out_channels, hidden_size, dropout=.0)
        self.gru_unit = gru_unit(in_channels,hidden_size)
        self.gcn_unit = GCN(sym_norm_Adj_matrix, in_channels, out_channels, dropout)
        self.concat_size = 2*(hidden_size+out_channels)
        # self.concat_size = hidden_size
        # self.att = Attention(att_input_size=hidden_size,att_out_size=hidden_size)
        self.fc1 = nn.Linear(self.concat_size,output_size)
        self.fc2 = nn.Linear(self.concat_size,output_size)


    def forward(self,x):
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape
        gru_gcn_out = self.gru_gcn(x) #(b,n,f_out)
        gru_gcn_out_unsq = torch.unsqueeze(gru_gcn_out,dim=2)
        gcn_gru_out = self.gcn_gru(x) #(b,n,hidden_size)
        # print('gcn_gru_out shape',gcn_gru_out.shape)
        gru_out = self.gru_unit(x) #(b,n,hidden_size)
        # print('gru_out shape',gru_out.shape)
        gcn_out = self.gcn_unit(x)
        gcn_out = gcn_out[:,:,-1,:].squeeze() #(b,n,f_out)
        # print('gcn_out shape',gcn_out.shape)

        # attention
        # gru_gcn_out, gcn_gru_out, gru_out, gcn_out = self.att((gru_gcn_out, gcn_gru_out, gru_out, gcn_out))


        # concat
        # concat_out = torch.cat([gcn_gru_out],dim=-1)
        # print('concat_out shape',concat_out.shape)
        concat_out = torch.cat([gru_gcn_out,gcn_gru_out,gru_out,gcn_out],dim=-1)

        concat_out = concat_out.reshape(-1,self.concat_size)
        output_1 = self.fc1(concat_out)
        output_2 = self.fc2(concat_out)
        return output_1,output_2



class Gru(nn.Module):

    def __init__(self,in_channels,hidden_size,output_size):
        super(Gru, self).__init__()
        self.gru = nn.GRU(input_size=in_channels, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size

    def forward(self,x):
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape
        x = x.reshape((-1, num_of_timesteps, in_channels))
        gru_out, gru_hidden = self.gru(x)
        last_gru_out = gru_out[:,-1,:].squeeze()
        last_gru_out = last_gru_out.reshape(-1,self.hidden_size)
        last_gru_out_1 = self.fc1(last_gru_out)
        return last_gru_out_1


class GruMulti(nn.Module):

    def __init__(self,in_channels,hidden_size,output_size):
        super(GruMulti, self).__init__()
        self.gru = nn.GRU(input_size=in_channels, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, output_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size

    def forward(self,x):
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape
        x = x.reshape((-1, num_of_timesteps, in_channels))
        gru_out, gru_hidden = self.gru(x)
        last_gru_out = gru_out[:,-1,:].squeeze()
        last_gru_out = last_gru_out.reshape(-1,self.hidden_size)
        last_gru_out_1 = self.fc1(last_gru_out)
        last_gru_out_2 = self.fc2(last_gru_out)
        return last_gru_out_1, last_gru_out_2

def trans_form_features(df):
    df = df.loc[:,(df>0).any()]
    df1 = df[df>0].fillna(method='bfill').fillna(method='ffill')
    return df1



def read_data(seq_len, pre_len,train_rate=None, validation_rate=None):
    pop_mean_data_path = 'ten_secs_interval'
    pop_variance_data_path = 'ten_secs_interval/pop_variance_data'
    adj_data_path = 'adj.csv'
    hz_adj = pd.read_csv(adj_data_path, header=0, sep=',', index_col=0)
    dires = os.listdir(pop_mean_data_path)
    pop_mean_files = list(filter(lambda x:x.startswith('pop_2018'),dires))
    pop_mean_df = pd.DataFrame()
    pop_variance_df = pd.DataFrame()
    for pop_file in pop_mean_files:
        df= pd.read_csv(os.path.join(pop_mean_data_path,pop_file),sep=',',header=0,index_col=0)
        pop_mean_df = pd.concat([pop_mean_df,df],axis=0)
        pop_variance_file = '_'.join(['variance',pop_file.split('_')[1]])
        v_df = pd.read_csv(os.path.join(pop_variance_data_path,pop_variance_file),sep=',',header=0,index_col=0)
        pop_variance_df = pd.concat([pop_variance_df,v_df],axis=0)

    pop_mean_df = trans_form_features(pop_mean_df)
    pop_variance_df = pop_variance_df.where(~(pop_variance_df==0.001),0)
    pop_variance_df = trans_form_features(pop_variance_df)
    common_cols = [i for i in pop_mean_df.columns if i in pop_variance_df.columns]
    pop_mean_df = pop_mean_df.loc[:,common_cols]
    pop_variance_df = pop_variance_df.loc[:,common_cols]
    pop_mean_df.index.name = 'Time'
    pop_mean_df = pop_mean_df.sort_values(by='Time',ascending=True)
    pop_variance_df.index.name = 'Time'
    pop_variance_df = pop_variance_df.sort_values(by='Time',ascending=True)
    pop_mean_array = pop_mean_df.values
    pop_variance_array = pop_variance_df.values
    columns = pop_mean_df.columns
    int_columns = list(map(lambda x:int(x),columns))
    hz_adj = hz_adj.loc[int_columns,columns]
    adj = hz_adj.values
    features = []
    labels = []
    n = 0
    unit = 3601 #3601,601,121
    while list(pop_mean_array[n*unit:(n+1)*unit]):
        pop_mean_one_day = pop_mean_array[n*unit:(n+1)*unit]
        pop_variance_one_day = pop_variance_array[n*unit:(n+1)*unit]
        for i in range(len(pop_mean_one_day)-seq_len-pre_len):
            pop_mean_one_sample = pop_mean_one_day[i:i+seq_len]
            pop_mean_one_sample = np.expand_dims(pop_mean_one_sample,axis=2)
            pop_variance_one_sample = pop_variance_one_day[i:i+seq_len]
            pop_variance_one_sample = np.expand_dims(pop_variance_one_sample,axis=2)
            one_sample_feature = np.concatenate([pop_mean_one_sample,pop_variance_one_sample],axis=2)
            pop_mean_one_label = pop_mean_one_day[i+seq_len:i+seq_len+pre_len]
            pop_mean_one_label = np.expand_dims(pop_mean_one_label,axis=2)
            pop_variance_one_label = pop_variance_one_day[i+seq_len:i+seq_len+pre_len]
            pop_variance_one_label = np.expand_dims(pop_variance_one_label,axis=2)
            one_sample_label = np.concatenate([pop_mean_one_label,pop_variance_one_label],axis=2)
            features.append(one_sample_feature)
            labels.append(one_sample_label)
        n+=1
    features = np.array(features)
    labels = np.array(labels)
    idx = list(range(len(features)))
    idx_ordered = copy.copy(idx)
    idx_train = idx[:int(len(idx)*train_rate)]
    idx_validation = idx_ordered[int(len(idx)*train_rate):]
    idx_test = idx[int(len(idx)*(train_rate+validation_rate)):]
    np.random.shuffle(idx_train)
    np.random.shuffle(idx_test)
    return adj, features[:,:,:,:],labels,idx_train,idx_validation,idx_test, columns

def norm_Adj(W):
    D = np.diag(1/np.sqrt(np.sum(W, axis=1)))
    norm_Adj_matrix = np.dot(D, W)
    return norm_Adj_matrix

def evaluation_(a, b):
    rmse = math.sqrt(mean_squared_error(a, b))
    mae = mean_absolute_error(a, b)
    F_norm = la.norm(a - b, 'fro') / la.norm(a, 'fro')
    return rmse, mae, 1 - F_norm

def evaluation(a, b):
    rmse = math.sqrt(mean_squared_error(a, b))
    mae = mean_absolute_error(a, b)
    mape = mean_absolute_percentage_error(a,b)
    return rmse, mae, 1 - mape

def integrated_loss(out1, out2, label1,label2):
    loss1 = F.mse_loss(out1, label1)
    loss2 = F.mse_loss(out2, label2)
    loss = loss1 + loss2
    return loss

