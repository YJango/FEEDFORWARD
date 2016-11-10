import sys
from ASR import ASR
from FF import FF
import numpy as np
import theano

def trainwithPER(setID,typ='teacher',learning_rate=1e-4, drop_out=0.4, Layers=4, N_hidden=2048,L2_lambda=1e-4,patience=3, batch_size=256, continue_train=0,evalPER=0):
    N_EPOCHS = 100
    np.random.seed(55)
    path="/home/jango/distillation"
    dic=path+"/shared_babel/phone.dic"
    lm=path+"/shared_babel/pdnet"
    hlist=path+"/shared_babel/monophones1"
    opt="-sb 160 -b2 60 -s 1000 -m 4000 -quiet -1pass"
    julius='julius-4.3.1'
    hresults='HResults'
    priors_path=path+"/%s/StatPrior%s_train" %(typ,setID)
    testmfclist=path+"/%s/feature/list%s/testmfc.list" %(typ,setID)
    testdnnlist=path+"/%s/feature/list%s/testdnn.list" %(typ,setID)
    prob_path=path+"/%s/StatePro%s/" %(typ,setID)
    window_size=17
    results_mlf=path+"/%s/LSTMRec/rec.mlf" %typ
    
    label_mlf=path+'/shared_babel%s/mlf%s/alignedtest.mlf' %(typ[:3],setID)
    model=path+"/%s/HMM%s/hmmdefs" %(typ,setID)
    train_data={'feature':path+"/%s/LSTMFile%s/%s_train_lstm.npy" %(typ,setID,typ[:3]),
                'label':path+"/%s/LSTMFile%s/%s_train_target_lstm.npy" %(typ,setID,typ[:3])}

    #display training process
    def display(acc,loss,accvali,lossvali,b,n_samples,times,PER,t_loss,t_acc):
        sys.stdout.write('Epoch:%2.2s(%4.4s) | Train acc:%6.6s loss:%6.6s | Best acc:%6.6s loss:%6.6s | Cur acc:%6.6s loss:%6.6s | PER:%6.6s\r' %(times,round((float(b)/n_samples)*100,1),round(t_acc,4),round(t_loss,4),round(accvali,4),round(lossvali,4),round(acc,4),round(loss,4),round(PER,4)))
        sys.stdout.flush()
    #write log file
    def writer(fname,acc,loss,accvali,lossvali,b,n_samples,times,PER,t_loss,t_acc):
        f=open(fname+'.txt','a')
        f.write('Epoch:%2.2s | Train acc:%6.6s loss:%6.6s | Best acc:%6.6s loss:%6.6s | Cur acc:%6.6s loss:%6.6s | PER:%6.6s\n' %(times,round(t_acc,4),round(t_loss,4),round(accvali,4),round(lossvali,4),round(acc,4),round(loss,4),round(PER,4)))
        f.close()
    #instances
    asr=ASR(path, dic, lm, hlist, julius, hresults, priors_path,
        testmfclist, testdnnlist, setID, prob_path, window_size, results_mlf, label_mlf, opt, model)
    #make window
    x_train, x_vali, y_train, y_vali=asr.DNNDataLoader(window_size, train_data['feature'], train_data['label'], n_class=120)
    ff=FF(learning_rate=learning_rate, drop_out=drop_out, Layers=Layers, N_hidden=N_hidden, D_input=x_train.shape[1], D_out=120,
                            Task_type='classification', L2_lambda=L2_lambda, fixlayer=[])
    #where to store weights
    fname='%s/%s/DnnWeight%s/%s_L%s_N%s_D%s_L%s' %(path,typ,setID,typ[:3],Layers,N_hidden,drop_out,L2_lambda)
    #whether to retrain
    if continue_train:
        ff.loader(np.load(fname+'.npy'))
    if evalPER:
        ff.loader(np.load(fname+'.npy'))
        PER=asr.RecogWithStateProbs(ff.get_out,2,2)
        print(PER)
        return 0
    #init for training 
    b = 0
    epoch = 0
    p = 1
    t_l=0
    t_a=0
    lossvali=ff.get_loss(x_vali,y_vali)
    accvali=ff.get_acc(x_vali,y_vali)
    n_samples=len(y_train)
    x_train, y_train=asr.shufflelists([x_train,y_train])
    downlr=0 #how many times the learning rate will be reduced 
    #current preformences
    PER=asr.RecogWithStateProbs(ff.get_out)
    acc=accvali
    loss=lossvali
    BPER=PER
    # start to train
    while epoch < N_EPOCHS:
        #batch_size train step
        t_loss,t_acc=ff.train_loss_acc(x_train[b:b+batch_size],y_train[b:b+batch_size])
        b+=batch_size
        #calculate preformences for training set
        t_l+=t_loss*len(y_train[b:b+batch_size])
        t_a+=t_acc*len(y_train[b:b+batch_size])
        t_loss=t_l/b
        t_acc=t_a/b
        display(acc,loss,accvali,lossvali,b,n_samples,epoch,PER,t_loss,t_acc)
        if b >= n_samples:
            #update counters
            b = 0
            t_l=0
            t_a=0
            #current preformences
            PER=asr.RecogWithStateProbs(ff.get_out)
            acc=ff.get_acc(x_vali,y_vali)
            loss=ff.get_loss(x_vali,y_vali)
            #whether to change learning rate
            if loss<=lossvali:
                #save weights
                ff.saver(fname)
                #update the best acc and loss
                lossvali=loss
                accvali=acc
                BPER=PER
                #write log
                writer(fname,acc,loss,accvali,lossvali,b,n_samples,epoch,PER,t_loss,t_acc)
                #reset patience
                p=1
            else:
                p+=1
                #write log
                writer(fname,acc,loss,accvali,lossvali,b,n_samples,epoch,PER,t_loss,t_acc)
                if p>patience:
                    if downlr >= 2:
                        f=open(fname+'.txt','a')
                        f.write('\nlossvali%s, accvali%s, BPER%s, epoch%s' %(lossvali, accvali, BPER, epoch))
                        return 0
                    else:
                        #reduce learning_rate
                        learning_rate=learning_rate*0.1
                        ff=FF(learning_rate=learning_rate, drop_out=drop_out, Layers=Layers, N_hidden=N_hidden, D_input=x_train.shape[1], D_out=120,
                            Task_type='classification', L2_lambda=L2_lambda, fixlayer=[])
                        #reload the weights
                        ff.loader(np.load(fname+'.npy'))
                        #
                        p=1
                        downlr+=1   
            epoch += 1
            x_train, y_train=asr.shufflelists([x_train,y_train])


trainwithPER('3',typ='teacher',learning_rate=1e-4, drop_out=0.4, Layers=4, N_hidden=2048,L2_lambda=1e-4,patience=3, batch_size=256,continue_train=0,evalPER=1)

