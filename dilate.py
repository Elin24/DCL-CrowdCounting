# -*- conding: utf-8 -*-

import os.path as path
import os
import re

root = 'exp'
exps = os.listdir(root)
expreport = {}
for exp in exps:
    exprcd = path.join(root, exp, exp+'.txt')
    with open(exprcd) as f:
        exprcd = re.findall(r"\[mae (.*?) mse (.*?)\], \[val loss (.*?)\]", f.read())
    #print(zip(*exprcd))
    mae, mse, loss = zip(*exprcd)
    trans = lambda x: [float(_) for _ in x]
    mae, mse, loss = trans(mae), trans(mse), trans(loss)
    expreport[exp] = dict(
        mae=mae,
        mse=mse,
        loss=loss
    )
    if 'SHHA_VGG_SHUFFLE' in exp:
        print(exp, min(mae), min(mse))
import json
with open('report.json', 'w+') as f:
    json.dump(expreport, f)