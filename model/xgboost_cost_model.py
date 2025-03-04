"""XGBoost as cost model"""

import time

import numpy as np


from tvm.autotvm import feature
from tvm.autotvm.utils import get_rank
from tvm.autotvm.tuner.metric import max_curve, recall_curve, cover_curve
from tvm.autotvm.tuner.model_based_tuner import FeatureCache
xgb = None

class XGBoostCostModel:
    def __init__(self, task, config, num_threads=None, log_interval=25):
        global xgb
        try:
            if xgb is None:
                xgb = __import__("xgboost")
        except ImportError:
            raise ImportError(
                "XGBoost is required for XGBoostCostModel. "
                "Please install its python package first. "
                "Help: (https://xgboost.readthedocs.io/en/latest/) "
            )

        self.task = task
        self.target = task.target
        self.loss_type = config.loss_type
        self.num_threads = num_threads
        self.log_interval = log_interval

        if self.loss_type == "reg":
            self.xgb_params = {
                "max_depth": 7,
                "gamma": 0.00001,
                "min_child_weight": 1,
                "subsample": 1.0,
                "eta": 0.3,
                "lambda": 1.00,
                "alpha": 0,
                "objective": "reg:linear",
            }
        elif self.loss_type == "rank":
            self.xgb_params = {
                "max_depth": 3,
                "gamma": 0.0001,
                "min_child_weight": 1,
                "subsample": 1.0,
                "eta": 0.3,
                "lambda": 1.00,
                "alpha": 0,
                "objective": "rank:pairwise",
            }
        else:
            raise RuntimeError("Invalid loss type: " + loss_type)

        self.xgb_params["verbosity"] = 0
        if num_threads:
            self.xgb_params["nthread"] = num_threads
        self.bst = None

        self._sample_size = 0


    def train(self, perf_buffer):
        x_train = np.array(perf_buffer.data_x)
        y_train = np.array(perf_buffer.data_y)
        self.fit(x_train, y_train)

    def fit(self, x_train, y_train, plan_size = 30):
        tic = time.time()
        y_train = y_train / (max(y_train) + 1e-8)

        valid_index = y_train > 1e-6
        index = np.random.permutation(len(x_train))
        dtrain = xgb.DMatrix(x_train[index], y_train[index])
        self._sample_size = len(x_train)

        callback = CustomCallback(
            stopping_rounds=20,
            metric="tr-a-recall@%d" % plan_size,
            evals=[(dtrain, "tr")],
            maximize=True,
            fevals=[
                xgb_average_recalln_curve_score(plan_size),
            ],
            verbose_eval=self.log_interval,
        )

        self.bst = xgb.train(
            self.xgb_params,
            dtrain,
            num_boost_round=8000,
            callbacks=[callback],
        )


    def predict(self, samples, output_margin=False):
        feas = np.array([s.point for s in samples])
        dtest = xgb.DMatrix(feas)
        if self.bst == None:
            return np.ones(len(samples))
        return self.bst.predict(dtest, output_margin=output_margin)

from xgboost.callback import TrainingCallback
from xgboost.callback import EvaluationMonitor

class CustomCallback(TrainingCallback):
    def __init__(self, stopping_rounds, metric, fevals, evals=(), log_file=None, maximize=None, verbose_eval=True):
        self.stopping_rounds = stopping_rounds
        self.metric = metric
        self.fevals = fevals
        self.evals = evals
        self.log_file = log_file
        self.maximize = maximize
        self.verbose_eval = verbose_eval
        self.state = {}
        self.metric_shortname = metric.split("-")[1]

    def before_training(self, model):
        self.state["maximize_score"] = self.maximize
        self.state["best_iteration"] = 0
        self.state["best_score"] = float("-inf") if self.maximize else float("inf")

        if model.attr("best_score") is not None:
            self.state["best_score"] = float(model.attr("best_score"))
            self.state["best_iteration"] = int(model.attr("best_iteration"))
            self.state["best_msg"] = model.attr("best_msg")
        else:
            model.set_attr(best_iteration=str(self.state["best_iteration"]))
            model.set_attr(best_score=str(self.state["best_score"]))

        return model

    def after_iteration(self, model, epoch, evals_log):
        try:
            from xgboost.training import aggcv
        except ImportError:
            from xgboost.callback import _aggcv as aggcv

        if not self.state:
            self.before_training(model)

        res_dict = {}

        for feval in self.fevals:
            bst_eval = []
            for dataset, name in self.evals:
                bst_eval.append(feval(model.predict(dataset), dataset))
            for key, val in bst_eval:
                res_dict[key] = [val]
        
        eval_res = []
        keys = list(res_dict.keys())
        keys.sort(key=lambda x: x if self.metric_shortname not in x else "a" + x)
        for key in keys:
            v = res_dict[key]
            eval_res.append([key] + v)

        if self.verbose_eval:
            infos = ["XGB iter: %3d" % epoch]
            for item in eval_res:
                if "null" in item[0]:
                    continue
                infos.append("%s: %.6f" % (item[0], item[1]))
            print("\t".join(infos))

        if self.log_file:
             with open(self.log_file, "a") as fout:
                fout.write("\t".join(infos) + "\n")

        score = None
        for item in eval_res:
            if item[0] == self.metric:
                score = item[1]
                break
        
        if score is None and eval_res:
            score = eval_res[0][1]
            print(f"Warning: Metric '{self.metric}' not found, using '{eval_res[0][0]}' instead.")

        if score is None:
            return False

        best_score = self.state["best_score"]
        best_iteration = self.state["best_iteration"]
        maximize_score = self.state["maximize_score"]

        if (maximize_score and score > best_score) or (not maximize_score and score < best_score):
            msg_parts = []
            for item in eval_res:
                try:
                    msg_parts.append(f"{item[0]}:{item[1]:.6f}")
                except:
                    pass

            msg = f"[{epoch}] {''.join(msg_parts)}"            

            self.state["best_msg"] = msg
            self.state["best_score"] = score
            self.state["best_iteration"] = epoch

            model.set_attr(
                best_score=str(self.state["best_score"]),
                best_iteration=str(self.state["best_iteration"]),
                best_msg=self.state["best_msg"],
            )
        elif epoch - best_iteration >= self.stopping_rounds:
            return True
        
        return False


# feval wrapper for xgboost
def xgb_max_curve_score(N):
    """evaluate max curve score for xgb"""

    def feval(preds, labels):
        labels = labels.get_label()
        trials = np.argsort(preds)[::-1]
        scores = labels[trials]
        curve = max_curve(scores)
        return "Smax@%d" % N, curve[N] / np.max(labels)

    return feval


def xgb_recalln_curve_score(N):
    """evaluate recall-n curve score for xgb"""

    def feval(preds, labels):
        labels = labels.get_label()
        trials = np.argsort(preds)[::-1]
        ranks = get_rank(labels[trials])
        curve = recall_curve(ranks)
        return "recall@%d" % N, curve[N]

    return feval


def xgb_average_recalln_curve_score(N):
    """evaluate average recall-n curve score for xgb"""

    def feval(preds, labels):
        labels = labels.get_label()
        trials = np.argsort(preds)[::-1]
        ranks = get_rank(labels[trials])
        curve = recall_curve(ranks)
        return "a-recall@%d" % N, np.sum(curve[:N]) / N

    return feval


def xgb_recallk_curve_score(N, topk):
    """evaluate recall-k curve score for xgb"""

    def feval(preds, labels):
        labels = labels.get_label()
        trials = np.argsort(preds)[::-1]
        ranks = get_rank(labels[trials])
        curve = recall_curve(ranks, topk)
        return "recall@%d" % topk, curve[N]

    return feval


def xgb_cover_curve_score(N):
    """evaluate cover curve score for xgb"""

    def feval(preds, labels):
        labels = labels.get_label()
        trials = np.argsort(preds)[::-1]
        ranks = get_rank(labels[trials])
        curve = cover_curve(ranks)
        return "cover@%d" % N, curve[N]

    return feval


def xgb_null_score(_):
    """empty score function for xgb"""

    def feval(__, ___):
        return "null", 0

    return feval
