import torch
import torch.nn.functional as F
import numpy as np
import os
import logging
from sklearn.metrics import roc_auc_score
# minimum float number
FLT_MIN = float(np.finfo(np.float32).eps)


def prepare_batches(model, dataset, num_of_batches, opt):
    """
    prepare some feature vectors for leaf node update.
    args:
        model: the neural decison forest to be trained
        dataset: the used dataset
        num_of_batches: total number of batches to prepare
        opt: experiment configuration object
    return: target vectors used for leaf node update
    """
    cls_onehot = torch.eye(opt.n_class)
    target_batches = []
    with torch.no_grad():
        # the features are prepared from the feature layer
        train_loader = torch.utils.data.DataLoader(dataset, 
                                                   batch_size = opt.batch_size, 
                                                   shuffle = True)
       
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx == num_of_batches:
                # enough batches
                break
            if opt.cuda:
                # move tensors to GPU if needed
                data, target, cls_onehot = data.cuda(), target.cuda(), \
                cls_onehot.cuda()
            # get the feature vectors
            feats = model.feature_layer(data)
            # release some memory
            del data
            feats = feats.view(feats.size()[0],-1)
            for tree in model.forest.trees:  
                # compute routing probability for each tree and cache them
                mu = tree(feats)
                mu += FLT_MIN
                tree.mu_cache.append(mu)  
            del feats
            target_batches.append(cls_onehot[target])    
    return target_batches

def evaluate(model, dataset, opt):
    """
    evaluate the neural decison forest.
    args:
        dataset: the evaluation dataset
        opt: experiment configuration object
    return: 
        record: evaluation statistics
    """
    # set the model in evaluation mode
    model.eval()
    # average evaluation loss
    test_loss = 0.0
    # total correct predictions
    correct = 0      
    # used for calculating AUC of ROC
    y_true = []
    y_score = []
    test_loader = torch.utils.data.DataLoader(dataset, 
                                              batch_size = opt.batch_size, 
                                              shuffle = False)
    for data, target in test_loader:
        with torch.no_grad():
            if opt.cuda:
                data, target = data.cuda(), target.cuda()
            # get the output vector
            output = model(data)
            # loss function                
            test_loss += F.nll_loss(torch.log(output), target, reduction='sum').data.item() # sum up batch loss
            # get class prediction
            pred = output.data.max(1, keepdim = True)[1] # get the index of the max log-probability
            # count correct prediction
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            if opt.eval_metric == "AUC":
                y_true.append(target.data.cpu().numpy())
                y_score.append(output.data.cpu().numpy()[:,1])
    test_loss /= len(test_loader.dataset)
    test_acc = int(correct) / len(dataset)
    # get AUC of ROC curve
    if opt.eval_metric == "AUC":
        y_true = np.concatenate(y_true, axis=0)
        y_score = np.concatenate(y_score, axis=0)
        auc = roc_auc_score(y_true, y_score)
    else:
        auc = None
    record = {'loss':test_loss, 'accuracy':test_acc, 
              'correct number':correct, 'AUC':auc}
    return record

def inference(model, dataset, opt, save=True):
    if dataset.name not in ['Nexperia']:
        raise NotImplementedError
    model.eval()
    all_preds = []
    test_loader = torch.utils.data.DataLoader(dataset, 
                                              batch_size = opt.batch_size, 
                                              shuffle = False)
    for data, target in test_loader:
        with torch.no_grad():
            if opt.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            pred = output.data.cpu().numpy()
            if dataset.name == 'Nexperia':
                pred = list(pred[:,1])
                #pred = list(np.argmax(pred, axis=1))
            all_preds += pred
    dataset.write_preds(all_preds)        
    return

def get_loss(output, target, dataset_name, loss_type='nll', reweight=True):
    if loss_type == 'nll' and reweight and dataset_name == 'Nexperia':
        # re-weight due to class imbalance 
        weight = torch.Tensor([1, 9])
        weight = weight.to(target.device)
        loss = F.nll_loss(torch.log(output), target, weight=weight)
    elif loss_type == 'nll':
        loss = F.nll_loss(torch.log(output), target)
    else:
        raise NotImplementedError
    return loss

def report(eval_record):
    logging.info('Evaluation summary:')
    for key in eval_record.keys():
        if eval_record[key] is not None:
            logging.info("{:s}: {:.3f}".format(key, eval_record[key]))    
    return

def metric_init(metric):
    if metric in ['accuracy', 'AUC']:
        value = 0.0
    else:
        raise NotImplementedError
    return value

def metric_comparison(current, best, metric):
    if metric in ['accuracy', 'AUC']:
        flag = current > best
    else:
        raise NotImplementedError
    return flag

def train(model, optim, sche, db, opt):
    """
    model training function.
    args:
        model: the neural decison forest to be trained
        optim: the optimizer
        sche: learning rate scheduler
        db: dataset object
        opt: experiment configuration object
    return:
        best_eval_acc: best evaluation accuracy
    """    
    # some initialization
    iteration_num = 0
    # number of batches to use for leaf node update
    num_of_batches = int(opt.label_batch_size/opt.batch_size)
    # number of images
    num_train = len(db['train'])
    # best evaluation metric
    best_eval_metric = metric_init(opt.eval_metric)
    # start training
    for epoch in range(1, opt.epochs + 1):
        # update learning rate by the scheduler
        sche.step()
    
        # Update leaf node prediction vector
        logging.info("Epoch %d : update leaf node distribution"%(epoch))

        # prepare feature vectors for leaf node update        
        target_batches = prepare_batches(model, db['train'],
                                         num_of_batches, opt)
        
        # update leaf node prediction vectors for every tree         
        for tree in model.forest.trees:
            for _ in range(20):
                tree.update_label_distribution(target_batches)
            # clear the cache for routing probabilities
            del tree.mu_cache
            tree.mu_cache = []
            
        # optimize decision functions
        model.train()
        train_loader = torch.utils.data.DataLoader(db['train'],
                                                   batch_size=opt.batch_size, 
                                                   shuffle=True)
        for batch_idx, (data, target) in enumerate(train_loader):
            if opt.cuda:
                # move tensors to GPU
                with torch.no_grad():
                    data, target = data.cuda(), target.cuda()
            iteration_num += 1
            optim.zero_grad()
            output = model(data)
            output = output.clamp(min=1e-6, max=1) # resolve some numerical issue
            # loss function
            loss = get_loss(output, target, opt.dataset)
            # compute gradients
            loss.backward()
            # update network parameters
            optim.step()
            # logging
            if batch_idx % opt.report_every == 0:
                logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(\
                    epoch, batch_idx * len(data), num_train,\
                    100. * batch_idx / len(train_loader), loss.data.item()))                    
                        
        # Evaluate after every epoch
        eval_record = evaluate(model, db['eval'], opt)
        if metric_comparison(eval_record[opt.eval_metric], best_eval_metric, opt.eval_metric):
            best_eval_metric = eval_record[opt.eval_metric]
            best_eval_acc = eval_record["accuracy"]
            # save prediction results for Nexperia testing set
            if opt.save and opt.dataset == "Nexperia":
                inference(model, db['test'], opt)            
            # save a snapshot of model when hitting a higher score
            if opt.save:
                save_path = os.path.join(opt.save_dir,
                           'depth_' + str(opt.tree_depth) +
                           'n_tree' + str(opt.n_tree) + \
                           'archi_type_' + opt.model_type + '_' + str(best_eval_acc) + \
                           '.pth')
                if not os.path.exists(opt.save_dir):
                    os.makedirs(opt.save_dir)
                torch.save(model, save_path)            
        # logging 
        report(eval_record)
    return best_eval_metric