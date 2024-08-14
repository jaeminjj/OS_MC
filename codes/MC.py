import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.nn.functional as F
import pickle
import datatable as dt
import scipy.spatial.distance as spd
import scipy.spatial.distance as spd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.environ["CUDA_VISIBLE_DEVICES"]="3" # 0, 1, 2, 3 중 하나

def compute_openmax_prob(scores, scores_u):
    prob_scores, prob_unknowns = [], []
    for s, su in zip(scores, scores_u):
        channel_scores = np.exp(s)
        channel_unknown = np.exp(np.sum(su))

        total_denom = np.sum(channel_scores) + channel_unknown
        prob_scores.append(channel_scores / total_denom)
        prob_unknowns.append(channel_unknown / total_denom)

    # Take channel mean
    scores = np.mean(prob_scores, axis=0)
    unknowns = np.mean(prob_unknowns, axis=0)
    modified_scores = scores.tolist() + [unknowns]
    return modified_scores

def openmax(weibull_model, categories, input_score, eu_weight, alpha=10, distance_type='eucos'):
    nb_classes = len(categories)

    ranked_list = input_score.argsort().ravel()[:-1][:alpha]
    alpha_weights = [((alpha + 1) - i) / float(alpha) for i in range(1, alpha + 1)]
    omega = np.zeros(nb_classes)
    omega[ranked_list] = alpha_weights

    scores, scores_u = [], []
    for channel, input_score_channel in enumerate(input_score):
        score_channel, score_channel_u = [], []
        for c, category_name in enumerate(categories):
            mav, dist, model = query_weibull(category_name, weibull_model, distance_type)
            channel_dist = calc_distance(input_score_channel, mav[channel], eu_weight, distance_type)
            wscore = model[channel].w_score(channel_dist)
            modified_score = input_score_channel[c] * (1 - wscore * omega[c])
            score_channel.append(modified_score)
            score_channel_u.append(input_score_channel[c] - modified_score)

        scores.append(score_channel)
        scores_u.append(score_channel_u)

    scores = np.asarray(scores)
    scores_u = np.asarray(scores_u)

    openmax_prob = np.array(compute_openmax_prob(scores, scores_u))
    softmax_prob = softmax(np.array(input_score.ravel()))
    return openmax_prob, softmax_prob
def calc_distance(query_score, mcv, eu_weight, distance_type='eucos'):
    if distance_type == 'eucos':
        query_distance = spd.euclidean(mcv, query_score) * eu_weight + \
            spd.cosine(mcv, query_score)
    elif distance_type == 'euclidean':
        query_distance = spd.euclidean(mcv, query_score)
    elif distance_type == 'cosine':
        query_distance = spd.cosine(mcv, query_score)
    else:
        print("distance type not known: enter either of eucos, euclidean or cosine")
    return query_distance

def query_weibull(category_name, weibull_model, distance_type='eucos'):
    return [weibull_model[category_name]['mean_vec'],
            weibull_model[category_name]['distances_{}'.format(distance_type)],
            weibull_model[category_name]['weibull_model']]
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
class Net(nn.Module):
        def __init__(self,D_in,H,D_out):
            super(Net,self).__init__()
            self.linear1=nn.Linear(D_in,H)
            #self.linear3=nn.Linear(H,H)
            #self.linear4=nn.Linear(H,H)
            #self.linear5=nn.Linear(H,H)
            self.linear2=nn.Linear(H,D_out)
            self.dropout = nn.Dropout(0.4)
            
        def forward(self,x,i):
            #x=func(self.linear1(x))
            if i==0:
                #x=torch.sigmoid(self.linear1(x))
                #x=torch.relu(self.linear1(x))
                #x=self.dropout(F.relu(self.linear1(x)))
                #x=self.dropout(F.relu(self.linear1(x)))
                x=F.relu(self.linear1(x))
                #x=self.dropout(F.relu(self.linear3(x)))
                #x=self.dropout(F.relu(self.linear4(x)))
                #x=self.dropout(F.relu(self.linear5(x)))
                #x=torch.softmax(selfß.linear1(x))
            if i==1:
                #x=self.dropout(F.tanh(self.linear1(x)))
                #x=self.dropout(F.tanh(self.linear1(x)))
                #x=self.dropout(F.tanh(self.linear1(x)))
                x=F.tanh(self.linear1(x))
                #x=self.dropout(F.tanh(self.linear3(x)))
                #x=self.dropout(F.tanh(self.linear4(x)))
                #x=self.dropout(F.tanh(self.linear5(x)))
                #torch.softmax(self.linear1(x))
            if i==4:
                #x=self.dropout(F.sigmoid(self.linear1(x)))
                x=F.sigmoid(self.linear1(x))
                #x=self.dropout(F.sigmoid(self.linear3(x)))
                #x=self.dropout(F.sigmoid(self.linear4(x)))
                #x=self.dropout(F.sigmoid(self.linear5(x)))
            if i==2:
                #x=self.dropout(F.softmax(self.linear1(x),dim=1))
                x = F.leaky_relu(self.linear1(x))

            x=self.linear2(x)
            #x=F.softmax(self.linear2(x),dim=1)
            return x
        
        
