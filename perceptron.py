#!/usr/bin/python

from scipy.sparse import csr_matrix, coo_matrix
import numpy as np
from constants import WEIGHT_DTYPE, CATEGORICAL_OFFSET, W2V_LENGTH, W2V_OFFSET_S0, W2V_OFFSET_A, W2V_OFFSET_B, W2V_PROD_AB, W2V_PROD_AS, W2V_PROD_SB

class Perceptron():
    
    #model = None
    #num_updates = 0
    #wstep = 1

    def __init__(self,model):
        self.model = model
        self.wstep = 1
        self.num_updates = 0# this is N - on page 87
        #self.reshape_rate = reshape_rate
        #self.weights = np.array([]); this vector is in the model for some reason

    def get_num_updates(self):
        return self.num_updates

    def no_update(self):
        self.wstep += 1

    def reshape_weight(self,act_idx,reshape_rate=10**5):
        print("RESHAPING");
        w = self.model.weight[act_idx];
        __u = self.model.u[act_idx];
        __q = self.model.q[act_idx];
        #aw = self.model.aux_weight[act_idx]
        #avgw = self.model.avg_weight[act_idx];

        self.model.weight[act_idx] = np.vstack((w,np.zeros(shape=(reshape_rate,w.shape[1]),dtype=WEIGHT_DTYPE)))
        new_u = np.zeros(shape=(reshape_rate,__u.shape[1]),dtype=WEIGHT_DTYPE);
        #new_u.fill(self.wstep);#i don't think this matters
        self.model.u[act_idx] = np.vstack((__u, new_u));
        self.model.q[act_idx] = np.vstack((__q,np.zeros(shape=(reshape_rate,__q.shape[1]),dtype=WEIGHT_DTYPE)))
        #self.model.aux_weight[act_idx] = np.vstack((aw,np.zeros(shape=(reshape_rate,aw.shape[1]),dtype=WEIGHT_DTYPE)))
        #self.model.avg_weight[act_idx] = np.vstack((avgw,np.zeros(shape=(reshape_rate,avgw.shape[1]),dtype=WEIGHT_DTYPE)))
        
    def update_weight_one_step(self,act_g,feat_g,act_l_g,act_b,feat_b,act_l_b):
        #print("trying to update...");
        self.num_updates += 1;
        
        act_g_idx = self.model.class_codebook.get_index(act_g)[0];
        act_b_idx = self.model.class_codebook.get_index(act_b)[0];
        #print("weight shape is: " + str(len(self.model.weight)));
        #print("weight indexed by action shape is: " + str(self.model.weight[act_g_idx].shape));
        #print("act label gold is : " + str(act_l_g));
        #print("act label best is : " + str(act_l_b));

        act_l_g = act_l_g if act_l_g else 0
        #act_t_g = act_t_g if act_t_g else 0
        act_l_b = act_l_b if act_l_b else 0
        #act_t_b = act_t_b if act_t_b else 0
        
        # this line says "since all features could potentially be new, if they are will we have enough space in the weight vector?"
        if self.model.weight[act_g_idx].shape[0] <= self.model.feature_codebook[act_g_idx].size()+len(feat_g):
            self.reshape_weight(act_g_idx)

        if self.model.weight[act_b_idx].shape[0] <= self.model.feature_codebook[act_b_idx].size()+len(feat_b):
            self.reshape_weight(act_b_idx)

        b_feats_indices, b_value_array = zip(*map(self.model.feature_codebook[act_b_idx].get_default_index,feat_b));
        g_feats_indices, g_value_array = zip(*map(self.model.feature_codebook[act_g_idx].get_default_index,feat_g));
        b_feats_indices = list(b_feats_indices);
        g_feats_indices = list(g_feats_indices);
        b_value_array = list(b_value_array);
        g_value_array = list(g_value_array);


        ############################# THIS NEEDS REFACTORING #############################
        #FOR B
        #if W2V_OFFSET_S0 in b_feats_indices and W2V_OFFSET_A in b_feats_indices:
        #    for i in range(0, W2V_LENGTH):
        #        b_feats_indices.append(W2V_PROD_AS + i);
        #        _a = b_value_array[b_feats_indices.index(W2V_OFFSET_S0 + i)];
        #        _b = b_value_array[b_feats_indices.index(W2V_OFFSET_A + i)];
        #        b_value_array.append(abs(_a-_b));
        #if W2V_OFFSET_S0 in b_feats_indices and W2V_OFFSET_B in b_feats_indices:
        #    for i in range(0, W2V_LENGTH):
        #        b_feats_indices.append(W2V_PROD_SB + i);
        #       _a = b_value_array[b_feats_indices.index(W2V_OFFSET_S0 + i)];
        #        _b = b_value_array[b_feats_indices.index(W2V_OFFSET_B + i)];
        #        b_value_array.append(abs(_a-_b));
        #if W2V_OFFSET_A in b_feats_indices and W2V_OFFSET_B in b_feats_indices:
        #    for i in range(0, W2V_LENGTH):
        #        b_feats_indices.append(W2V_PROD_AB + i);
        #        _a = b_value_array[b_feats_indices.index(W2V_OFFSET_A + i)];
        #        _b = b_value_array[b_feats_indices.index(W2V_OFFSET_B + i)];
        #        b_value_array.append(abs(_a-_b));
        #
        #FOR A
        #if W2V_OFFSET_S0 in g_feats_indices and W2V_OFFSET_A in g_feats_indices:
        #    for i in range(0, W2V_LENGTH):
        #        g_feats_indices.append(W2V_PROD_AS + i);
        #        _a = g_value_array[g_feats_indices.index(W2V_OFFSET_S0 + i)];
        #        _b = g_value_array[g_feats_indices.index(W2V_OFFSET_A + i)];
        #        g_value_array.append(abs(_a-_b));
        #if W2V_OFFSET_S0 in g_feats_indices and W2V_OFFSET_B in g_feats_indices:
        #    for i in range(0, W2V_LENGTH):
        #        g_feats_indices.append(W2V_PROD_SB + i);
        #        _a = g_value_array[g_feats_indices.index(W2V_OFFSET_S0 + i)];
        #        _b = g_value_array[g_feats_indices.index(W2V_OFFSET_B + i)];
        #        g_value_array.append(abs(_a-_b));
        #if W2V_OFFSET_A in g_feats_indices and W2V_OFFSET_B in g_feats_indices:
        #    for i in range(0, W2V_LENGTH):
        #        g_feats_indices.append(W2V_PROD_AB + i);
        #        _a = g_value_array[g_feats_indices.index(W2V_OFFSET_A + i)];
        #        _b = g_value_array[g_feats_indices.index(W2V_OFFSET_B + i)];
        #        g_value_array.append(abs(_a-_b));
        ##################################################################################

        b_feats_CONT = [];
        b_value_CONT = [];
        b_feats_CAT = [];
        g_feats_CONT = [];
        g_value_CONT = [];
        g_feats_CAT = [];
        ####### separate the lists into the continuous valued features and the categorical valued features
        #for i_ in range(0, len(b_feats_indices)):
        #    if b_feats_indices[i_] < CATEGORICAL_OFFSET:
        #        b_feats_CONT.append(b_feats_indices[i_]);
        #        b_value_CONT.append(b_value_array[i_]);
        #    else:
        #        b_feats_CAT.append(b_feats_indices[i_]);
        #for i_ in range(0, len(g_feats_indices)):
        #    if g_feats_indices[i_] < CATEGORICAL_OFFSET:
        #        g_feats_CONT.append(g_feats_indices[i_]);
        #        g_value_CONT.append(g_value_array[i_]);
        #    else:
        #        g_feats_CAT.append(g_feats_indices[i_]);

        #create the gradients
        print("Bidx: " + str(self.model.weight[act_b_idx].shape));
        print("Gidx: " + str(self.model.weight[act_g_idx].shape));
        gradients = list();
        for _weight_array_ in range(0, len(self.model.weight)):
            gradients.append(np.zeros(shape=self.model.weight[_weight_array_].shape));
        #print("gradients: " + str(gradients));
        #print("gtypes: " + str(type(gradients)));


        #print("type of weights: " + str(type(self.model.weight[act_g_idx])));
        #print("type of g_feats_indices: " + str(type(g_feats_indices)));
        #print("type of act_l_g: " + str(type(act_l_g)));
        # add gradients to gradient list first
        print("act_l_g: " + str(act_l_g));
        print("g_feats_indices: " + str(g_feats_indices));

        ############## calculate gradients for categorical features
        #gradients[act_g_idx][g_feats_CAT,act_l_g] -= 1.0;
        #self.model.q[act_g_idx][g_feats_CAT,act_l_g] += 1.0;
        gradients[act_g_idx][g_feats_indices,act_l_g] -= 1.0;
        self.model.q[act_g_idx][g_feats_indices,act_l_g] += 1.0;

        #gradients[act_b_idx][b_feats_CAT,act_l_b] += 1.0;
        #self.model.q[act_b_idx][b_feats_CAT,act_l_b] += 1.0;
        gradients[act_b_idx][b_feats_indices,act_l_b] += 1.0;
        self.model.q[act_b_idx][b_feats_indices,act_l_b] += 1.0;

        ############## calculate gradients for continuous features
        #b_value_CONT = np.array(b_value_CONT);
        #gradients[act_b_idx][b_feats_CONT,act_l_b] += b_value_CONT;
        #self.model.q[act_b_idx][b_feats_CONT,act_l_b] += b_value_CONT**2;

        #g_value_CONT = np.array(g_value_CONT);
        #gradients[act_g_idx][g_feats_CONT,act_l_g] -= g_value_CONT;
        #self.model.q[act_g_idx][g_feats_CONT,act_l_g] += g_value_CONT**2;
        
        ######## Make sure to not add gradients twice for any subset of the matrix
        print("Best act features of wrong label or action: " + str(self.model.weight[act_b_idx][b_feats_indices,act_l_b]));
        if act_b_idx != act_g_idx:
            #update b
            q_z = self.model.q[act_b_idx][b_feats_indices,act_l_b]**0.5;
            numer = self.model.weight[act_b_idx][b_feats_indices,act_l_b] * (q_z) - self.model.eta * gradients[act_b_idx][b_feats_indices,act_l_b];
            denom = self.model.eta*self.model.C + q_z;
            self.model.weight[act_b_idx][b_feats_indices,act_l_b] = np.divide(numer, denom);
            ############self.model.weight[act_b_idx][b_feats_indices,act_l_b] -= gradients[act_b_idx][b_feats_indices,act_l_b];############test for set coverage
            #then update g
            q_z = self.model.q[act_g_idx][g_feats_indices,act_l_g]**0.5;
            numer = self.model.weight[act_g_idx][g_feats_indices,act_l_g] * (q_z) - self.model.eta * gradients[act_g_idx][g_feats_indices,act_l_g];
            denom = self.model.eta*self.model.C + q_z;
            self.model.weight[act_g_idx][g_feats_indices,act_l_g] = np.divide(numer, denom);
            ############self.model.weight[act_g_idx][g_feats_indices,act_l_g] -= gradients[act_g_idx][g_feats_indices,act_l_g];############test for set coverage
        else:
            #take the union of feature lists and do it for each act label -- if we are in this function and the IF block condition is false then the labels MUST be different (unless theres a gold act violation?)
            a_feat_indicies = set();
            for _b_z in b_feats_indices:
                a_feat_indicies.add(_b_z);
            for _g_z in g_feats_indices:
                a_feat_indicies.add(_g_z);
            a_feat_indicies = list(a_feat_indicies);
            #label B
            q_z = self.model.q[act_b_idx][a_feat_indicies,act_l_b]**0.5;
            numer = self.model.weight[act_b_idx][a_feat_indicies,act_l_b] * (q_z) - self.model.eta * gradients[act_b_idx][a_feat_indicies,act_l_b];
            denom = self.model.eta*self.model.C + q_z;
            self.model.weight[act_b_idx][a_feat_indicies,act_l_b] = np.divide(numer, denom);
            #print("Gradients: " + str(gradients[act_b_idx][a_feat_indicies,act_l_b]));
            ############self.model.weight[act_b_idx][a_feat_indicies,act_l_b] -= gradients[act_b_idx][a_feat_indicies,act_l_b];############test for set coverage
            #label G
            if act_l_b != act_l_g:
                q_z = self.model.q[act_b_idx][a_feat_indicies,act_l_g]**0.5;
                numer = self.model.weight[act_b_idx][a_feat_indicies,act_l_g] * (q_z) - self.model.eta * gradients[act_b_idx][a_feat_indicies,act_l_g];
                denom = self.model.eta*self.model.C + q_z;
                self.model.weight[act_b_idx][a_feat_indicies,act_l_g] = np.divide(numer, denom);
                ############self.model.weight[act_b_idx][a_feat_indicies,act_l_g] -= gradients[act_b_idx][a_feat_indicies,act_l_g];############test for set coverage


        #if False in np.isfinite(self.model.weight[act_b_idx][b_feats_indices,act_l_b]):
        #    print("MADE AN INFINITY");

        #print("new weights: " + str(self.model.weight[act_b_idx][b_feats_indices,act_l_b]));

        #update the u vectors
        self.model.u[act_g_idx][g_feats_indices,act_l_g] = self.num_updates;
        self.model.u[act_b_idx][b_feats_indices,act_l_b] = self.num_updates;

        self.wstep += 1;
