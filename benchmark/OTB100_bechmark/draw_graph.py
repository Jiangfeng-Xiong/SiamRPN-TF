import matplotlib.pyplot as plt
plt.switch_backend('agg')

import numpy as np
import math
from config import *
from scripts import *

def draw_dict(dicts):
    from matplotlib import colors
    from scipy .interpolate import spline
    markers=['o','D','h','*','s','.','+','x','_','<','>']
    index = 0 
    fig = plt.figure(num=2, figsize=(18,12), dpi=150)
    for key in dicts:
        marker = markers[index%len(markers)]
        values = sorted(dicts[key],key = lambda x: int(x[0])) # values is a list of tuple
        values_x = [values[i][0] for i in range(len(values))]
        values_y = [values[i][1] for i in range(len(values))]
        max_index = np.argmax(values_y)
        mean_auc = np.mean(values_y)
        plt.plot(values_x,values_y,label="%s : max auc=%.2f(iter %s), mean auc=%.2f"%(key,values_y[max_index], values_x[max_index], mean_auc),linewidth=2,marker=marker)
        index = index + 1
    plt.legend(loc='lower right',prop={'size':8})
    plt.title('OTB100 AUC vs iteration')
    plt.xlabel('Iteration')
    plt.ylabel('AUC')
    plt.grid(color='b', alpha=0.8, ls=':')
    plt.savefig('validation.png',dpi=150, bbox_inches='tight')
    plt.show()

def main():
    evalTypes = ['OPE']
    testname = 'tb100'
    for i in range(len(evalTypes)):
        evalType = evalTypes[i]
        result_src = RESULT_SRC.format(evalType)
        trackers = os.listdir(result_src)
        scoreList = []
        for t in trackers:
            try:
                score = butil.load_scores(evalType, t, testname)
                scoreList.append(score)
            except: pass
        plt = get_graph(scoreList, i, evalType, testname)
    plt.show()


def get_graph(scoreList, fignum, evalType, testname):
    fig = plt.figure(num=fignum, figsize=(18,12), dpi=150)
    rankList = sorted(scoreList, key=lambda o: sum(o[0].successRateList), reverse=True)
    dicts = {}
    for i in range(len(rankList)):
        result = rankList[i]
        tracker = result[0].tracker
        attr = result[0]
        attr.successRateList = list(attr.successRateList)
        if len(attr.successRateList) == len(thresholdSetOverlap):
            if i < MAXIMUM_LINES:
                ls = '-'
                if i % 2 == 1:
                    ls = '--'
                ave = sum(attr.successRateList) / float(len(attr.successRateList))
                plt.plot(thresholdSetOverlap, attr.successRateList, 
                    c = LINE_COLORS[i%len(LINE_COLORS)], label='{0} [{1:.2f}]'.format(tracker, ave), lw=2.0, ls = ls)
                print("tracker %s with auc: %.2f"%(tracker, ave))
                key = '-'.join(tracker.split('-')[0:-1]) if len(tracker.split('-'))>1 else tracker
                value = tracker.split('-')[-1] if len(tracker.split('-'))>1 else 5000
                if dicts.get(key)==None:
                    dicts[key] = []
                dicts[key].append((value, ave))
            else:
                plt.plot(thresholdSetOverlap, attr.successRateList, 
                    label='', alpha=0.5, c='#202020', ls='--')
        else:
            print('err')
    plt.title('{0}_{1} (sequence average)'.format(evalType, testname.upper()))
    plt.rcParams.update({'axes.titlesize': 'medium'})
    plt.xlabel('thresholds')
    plt.xticks(np.arange(thresholdSetOverlap[0], thresholdSetOverlap[len(thresholdSetOverlap)-1]+0.1, 0.1))
    plt.grid(color='#101010', alpha=0.5, ls=':')
    plt.legend(fontsize='medium')
    plt.savefig('{0}_sq.png'.format(evalType), dpi=150, bbox_inches='tight')
    plt.show()  

    draw_dict(dicts)
    return plt

if __name__ == '__main__':
    main()
