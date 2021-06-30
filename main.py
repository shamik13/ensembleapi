from collections import defaultdict
from io import StringIO
from typing import List

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from sklearn.metrics import roc_auc_score

app = FastAPI()


@app.get("/")
async def index():
    return "Please head over to /docs for uploading csv files"


@app.post("/rocauc/")
async def rocauc(files: List[UploadFile] = File(...), weights: List[str] = None):
    """
    Get ROCAUC value of a single csv or ensembled multiple csv.

    Args:

    `files List[UploadFile]`: Upload CSV files. Each CSV must contain columns named
    `stem`, `score` and `label`.

    `weights (List[str], optional)`: weights for each csv file. if weights are not provided,
    average value will be considered.

    """

    stem_col = "stem"
    pred_col = "score"
    gt_col = "label"

    if len(files) == 1:  # for a single file, calculate ROCAUC directly
        df = pd.read_csv(StringIO(str(files[0].file.read(), "utf-8")), encoding="utf-8")
        y_true = df[gt_col].tolist()
        y_preds = df[pred_col].tolist()

        try:
            roc_auc = roc_auc_score(y_true, y_preds)
        except Exception as e:
            raise HTTPException(status_code=404, detail=str(e))
        return {"roc_auc": roc_auc}

    else:  # for multiple files create a dataframe with columns [stem, score1, score2, ... , label]
        d = defaultdict(list)
        for i in range(len(files)):
            df = pd.read_csv(
                StringIO(str(files[i].file.read(), "utf-8")), encoding="utf-8"
            )
            for index, row in df.iterrows():
                d[row[stem_col]].append(row[pred_col])
            if i == len(files) - 1:
                for index, row in df.iterrows():
                    d[row[stem_col]].append(row[gt_col])

        if np.sum([len(d[key]) == len(files) + 1 for key in d.keys()]) / len(d) != 1.0:
            raise HTTPException(
                status_code=404,
                detail="raw_stem column of each csv should have identical elements",
            )

        y_true, y_preds = [], []

        if len(weights[0]) == 0:  # if no weights are provided, use basic average
            weights = [1 / len(files)] * len(files)
        else:  # if weights are provided, convert them to float from str
            try:
                weights = list(map(float, weights[0].split(",")))
            except:
                raise HTTPException(
                    status_code=404,
                    detail="Please make sure provided weights are correct",
                )
            if len(weights) != len(files):
                raise HTTPException(
                    status_code=404, detail="Please provide weights for each input file"
                )
        try:
            for k, v in d.items():
                # y_preds.append(np.average(v[:-1]))
                y_preds.append(np.dot(weights, v[:-1]))
                y_true.append(int(v[-1]))
        except Exception as e:
            raise HTTPException(status_code=404, detail=str(e))
        return {"roc_auc": roc_auc_score(y_true, y_preds)}
