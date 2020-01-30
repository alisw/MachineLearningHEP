from sklearn.metrics import f1_score 
import xgboost as xgb 
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials 

def hyperopt(x_train, y_train):
    space ={'max_depth': hp.quniform("x_max_depth", 1, 6, 1), 'min_child_weight': hp.quniform ('x_min_child', 1, 4, 1),
        'subsample': hp.uniform ('x_subsample', 0.5, 0.9), 'gamma' : hp.uniform ('x_gamma', 0.0,0.2),
        'colsample_bytree' : hp.uniform ('x_colsample_bytree', 0.5,0.9), 'reg_lambda' : hp.uniform ('x_reg_lambda', 0,1),
        'reg_alpha' : hp.uniform ('x_reg_alpha', 0,1), 'learning_rate' : hp.uniform ('x_learning_rate', 0.05, 0.35),
        'max_delta_step' : hp.quniform ('x_max_delta_step', 0, 8, 2)}

    def objectivefctn(space):
            X, Xcv, y, ycv = train_test_split(x_train, y_train, test_size = 0.2, random_state=0)

            def xgb_f1(y, t):
                t = t.get_label()
                y_bin = [1 if y_cont > 0.7 else 0 for y_cont in y] # binarizing your output
                return 'f1', f1_score(t, y_bin)

            def evalmcc(preds, dtrain):
                THRESHOLD = 0.7
                labels = dtrain.get_label()
                return 'MCC', matthews_corrcoef(labels, preds >= THRESHOLD)

            clf = xgb.XGBClassifier(n_estimators =1000, colsample_bytree=space['colsample_bytree'], learning_rate = space['learning_rate'],
                max_depth = int(space['max_depth']), min_child_weight = space['min_child_weight'], subsample = space['subsample'],
                gamma = space['gamma'], reg_lambda = space['reg_lambda'], max_delta_step = space['max_delta_step'], 
                reg_alpha = space['reg_alpha'], scale_pos_weight = 1, n_jobs = -1, objective = "binary:logistic")
            
            eval_set  = [(X, y), (Xcv, ycv)]
            clf.fit(X, y, eval_set=eval_set, eval_metric=xgb_f1, early_stopping_rounds=10, verbose=False)
            pred = clf.predict(Xcv)
            thresh = 0.7
            pred [pred > thresh] = 1
            pred [pred <= thresh] = 0
            f1score = f1_score(ycv, pred, average='weighted')
            mcc = matthews_corrcoef(ycv, pred)
            return {'loss': -mcc, 'status': STATUS_OK }


    trials = Trials()
    best = fmin(fn=objectivefctn, space=space, algo=tpe.suggest, max_evals=100, trials=trials)
    print(best)

