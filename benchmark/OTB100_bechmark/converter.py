import scipy.io as sio
import os
import json

ORIGIN_PATH = 'mat_results/OPE'
EXPORT_PATH = 'mat_results/OPE_mat'

trackers = os.listdir(ORIGIN_PATH)
for tracker in trackers:
    files = os.listdir(os.path.join(ORIGIN_PATH, tracker))
    for filename in files:
        if filename[-4:] == 'json':
            fHandle = open(os.path.join(ORIGIN_PATH, tracker, filename))
            tmpfile = json.load(fHandle)
            fHandle.close()
            seqLen = len(tmpfile)
            extracted = []
            for i in range(seqLen):
                tmp_old = tmpfile[i]
                tmp_new = {}
                if 'tmplsize' in tmp_old.keys() and tmp_old['tmplsize'] is not None:
                    tmp_new['tmplsize'] = tmp_old['tmplsize']
                tmp_new['type'] = tmp_old['resType']
                tmp_new['len'] = tmp_old['endFrame'] - tmp_old['startFrame'] + 1
                if tracker != 'NCC':
                    tmp_new['res'] = tmp_old['res']
                else:
                    tmp_new['res'] = []
                    for rect in tmp_old['res']:
                        tmp_new['res'].append([float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3])])
                extracted.append(tmp_new)
            matFileName = filename[:-5].lower() + '_' + tracker
            sio.savemat(os.path.join(EXPORT_PATH, matFileName), {'results':extracted})
