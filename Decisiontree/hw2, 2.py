# Question 2
import math
import statistics
import numpy as np
import matplotlib.pyplot as plot


def read_and_convert(filename):
    with open(filename, mode='r') as f:
        data = []
        for line in f:
            terms = line.strip().split(',')
            data.append([float(term) if idx in {0, 5, 9, 11, 12, 13, 14} else term for idx, term in enumerate(terms)])
    return data

def binary_quantization(data, thresholds):
    for row in data:
        for idx, val in enumerate(row):
            if idx in thresholds and val >= thresholds[idx]:
                row[idx] = 'yes'
            elif idx in thresholds:
                row[idx] = 'no'
    return data

# Read and convert train and test data
mylist_train = read_and_convert('train.csv')
mylist_test = read_and_convert('test.csv')
train_thresholds = {idx: statistics.median([row[idx] for row in mylist_train]) for idx in {0, 5, 9, 11, 12, 13, 14}}
test_thresholds = {idx: statistics.median([row[idx] for row in mylist_test]) for idx in {0, 5, 9, 11, 12, 13, 14}}
mylist_train = binary_quantization(mylist_train, train_thresholds)
mylist_test = binary_quantization(mylist_test, test_thresholds)

Attr_dict ={'age':['yes','no'],
             'job':['admin.','unknown','unemployed','management',
                    'housemaid','entrepreneur','student','blue-collar',
                    'self-employed','retired','technician','services'],
                    'martial':['married','divorced','single'],
                    'education':['unknown','secondary','primary','tertiary'],
                     'default':['yes','no'],
                     'balance':['yes','no'],
                     'housing':['yes','no'],
                     'loan':['yes','no'],
                     'contact':['unknown','telephone','cellular'],
                     'day':['yes','no'],
                     'month':['jan', 'feb', 'mar', 'apr','may','jun','jul','aug','sep','oct', 'nov', 'dec'],
                     'duration': ['yes','no'],
                     'campaign':['yes','no'],
                     'pdays':['yes','no'],
                     'previous':['yes','no'],
                     'poutcome':[ 'unknown','other','failure','success']}
Attr_set = set(key for key in Attr_dict)

def pos(attr):
    pos=0
    if attr=='age':
        pos=0
    if attr=='job':
        pos=1
    if attr=='martial':
        pos=2
    if attr=='education':
        pos=3
    if attr=='default':
        pos=4
    if attr=='balance':
        pos=5
    if attr=='housing':
        pos=6
    if attr=='loan':
        pos=7
    if attr=='contact':
        pos=8
    if attr=='day':
        pos=9
    if attr=='month':
        pos=10
    if attr=='duration':
        pos=11
    if attr=='campaign':
        pos=12
    if attr=='pdays':
        pos=13
    if attr=='previous':
        pos=14
    if attr=='poutcome':
        pos=15
    if attr=='y':
        pos=16
    return pos        

def create_list(attr):
    obj={}
    for attr_val in Attr_dict[attr]:
        obj[attr_val]=[]
    return obj

def create_list_with_zero(attr):
    obj={}
    for attr_val in attr:
        obj[attr_val]=0
    return obj

def exp_entropy(groups, classes):
    Q = 0.0
    tp =0.0
    for attr_val in groups:
        tp = sum([row[-1] for row in groups[attr_val]])
        Q = Q + tp        
    exp_ent = 0.0
    for attr_val in groups:
        size = float(len(groups[attr_val]))
        if size == 0:
            continue
        score = 0
        q = sum([row[-1] for row in groups[attr_val]])
        for class_val in classes:
            p = sum([row[-1] for row in groups[attr_val] if row[-2] == class_val])/q
            if p==0:
                temp=0
            else:
                temp=p*math.log2(1/p)
            score +=temp
        exp_ent += score * sum([row[-1] for row in groups[attr_val]])/Q
    return exp_ent

def data_split(attr, dataset):
    branch_obj=create_list(attr)
    for row in dataset:
        for attr_val in Attr_dict[attr]:
           if row[pos(attr)] == attr_val:
               branch_obj[attr_val].append(row)
    return branch_obj

def find_best_split(dataset):
    if dataset==[]:
        return 
    label_values = list(set(row[-2] for row in dataset)) 
    metric_obj = create_list_with_zero(Attr_dict)
    for attr in Attr_dict:
        groups = data_split(attr, dataset)
        metric_obj[attr] = exp_entropy(groups, label_values)
    best_attr = min(metric_obj, key=metric_obj.get)
    best_groups = data_split(best_attr, dataset)  
    return {'best_attr':best_attr, 'best_groups':best_groups}

def leaf_node_label(group):
    majority_labels = [row[-2] for row in group]
    return max(set(majority_labels), key=majority_labels.count)

def child_node(node, max_depth, curr_depth):
    if curr_depth >= max_depth:
        for key in node['best_groups']:
            if  node['best_groups'][key]!= []:
                node[key] = leaf_node_label(node['best_groups'][key])   
            else:
                node[key] = leaf_node_label(sum(node['best_groups'].values(),[]))
        return  
    for key in node['best_groups']:
        if node['best_groups'][key]!= []:
            node[key] = find_best_split(node['best_groups'][key]) 
            child_node(node[key], max_depth, curr_depth + 1)
        else:
            node[key] = leaf_node_label(sum(node['best_groups'].values(),[]))  

def tree_build(train_data, max_depth):
	root = find_best_split(train_data)
	child_node(root, max_depth, 1)
	return root

def label_predict(node, inst):
    if isinstance(node[inst[pos(node['best_attr'])]],dict):
        return label_predict(node[inst[pos(node['best_attr'])]],inst)
    else:
        return node[inst[pos(node['best_attr'])]]

def sign_func(val):
    return 1.0 if val > 0 else -1.0

def label_return(dataset,tree):
    true_label = []
    pred_seq = []
    for row in dataset:
        true_label.append(row[-2])    
        pre = label_predict(tree, row)
        pred_seq.append(pre)
    return [true_label, pred_seq]
    
def list_obj(n):
    obj={}
    for i in range(n):
        obj[i] = []
    return obj

def convert_binary_to_num(llist):
    bin_list =[]
    for i in range(len(llist)):
        if llist[i] == 'yes':
            bin_list.append(1.0)
        else:
            bin_list.append(-1.0)
    return bin_list

def wt_update(curr_wt, vote, bin_true, bin_pred):
    next_wt=[]
    for i in range(len(bin_true)):
        next_wt.append(curr_wt[i]*math.e**(- vote*bin_true[i]*bin_pred[i]))
    next_weight = [x/sum(next_wt) for x in next_wt]
    return next_weight

def wt_append(mylist, weights):
    for i in range(len(mylist)):
        mylist[i].append(weights[i]) 
    return mylist 

def wt_update_to_data(data, weight):
    for i in range(len(data)):
        data[i][-1] = weight[i]
    return data
   
def wt_error(true_label, predicted, weights):
    count = 0
    for i in range(len(true_label)):
        if true_label[i] != predicted[i]:
            count += weights[i]
    return count

def _error(_true_lb, _pred_lb): 
    count = 0
    size = len(_true_lb)
    for i in range(size):
        if _true_lb[i] != _pred_lb[i]:
            count += 1
    return count/size

def get_final_decision(indiv_pred, vote, data_len, _T):
    fin_pred = []
    for j in range(data_len):
        score = sum([indiv_pred[i][0][j]*vote[i] for i in range(_T)])
        fin_pred.append(sign_func(score))
    return fin_pred


# =============================================================================
# Boosting algorithm

# ----- Parameters -----   
delta = 1e-8 # Delta value
T = 50 # Number of iterations
# ----------------------

def ada_boost(_T, _delta, train_data):
    pred_result = list_obj(_T)
    vote_say = []
    weights = [row[-1] for row in train_data]
    for i in range(_T):
        tree = tree_build(train_data, 1)
        [pp_true, qq_pred] = label_return(train_data, tree)
        pred_result[i].append(convert_binary_to_num(qq_pred))
        err = wt_error(pp_true, qq_pred, weights)
        print(tree['best_attr'], err, weights[0])
        vote_say.append( 0.5*math.log((1-err)/err))
        weights = wt_update(weights, 0.5*math.log((1-err)/err), convert_binary_to_num(pp_true), convert_binary_to_num(qq_pred))
        train_data = wt_update_to_data(train_data, weights) 
    return [pred_result, vote_say, weights]

W_1 = np.ones(len(mylist_train))/len(mylist_train)
mylist_train = wt_append(mylist_train, W_1) 
true_label_bin = convert_binary_to_num([row[-2] for row in mylist_train]) 


# =============================================================================
def iteration_error(T_max):
   ERR =[]
   for t in range(1,T_max):
       [aa_pred, bb_vote, weights] = ada_boost(t, .001, mylist_train)
       fin_pred = get_final_decision(aa_pred, bb_vote, len(mylist_train), t)
       ERR.append(_error(true_label_bin, fin_pred))
   return ERR

Err = iteration_error(5)       
plot.plot(Err)
plot.ylabel('Loss function value')
plot.xlabel('Number of iterations')
plot.title('tolerance= 0.000001, # passings =20000')
plot.show()
# =============================================================================
# W_1 = np.ones(len(mylist_train))/len(mylist_train)
# mylist_train = wt_append(mylist_train, W_1)      
# [aa_pred, bb_vote, weights] = ada_boost(T, delta, mylist_train)
# mylist_train = wt_update_to_data(mylist_train, weights) 
# tree_1 = tree_build(mylist_train, 1)
# [pp, qq] =label_return(mylist_train, tree_1)
# 
# 
# def compare(x,y):
#     count =0
#     for i in range(len(x)):
#         if x[i] != y[i]:
#             count += 1
#     return count
# print(compare(pp,qq))
# 
# print(wt_error(convert_binary_to_num(pp), convert_binary_to_num(qq), weights))
# =============================================================================

#fin_pred = get_final_decision(aa_pred, bb_vote, len(mylist_train), T)
#true_label =convert_binary_to_num([row[-2] for row in mylist_train])  
#print(_error(true_label, fin_pred))

        
# =============================================================================
# W_1 = np.random.random(len(mylist_train))
# MM = len(mylist_train)
# for i in range(MM):
#     if i <= 1000:
#         W_1[i] = 100
#     else:
#         W_1[i] = 80
# WW = [x/sum(W_1) for x in W_1]
# #W_1 = [x/sum(W_1) for x in W_1]
# mylist_train = wt_append(mylist_train, WW)
#tree = tree_build(mylist_train, 1)
#print(tree['best_attr'])
# =============================================================================
