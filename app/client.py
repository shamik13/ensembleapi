import collections
import os
import shutil
import uuid
from pathlib import Path
from time import localtime, sleep
from zipfile import ZipFile

import click
import pandas as pd
import paramiko
import yaml
from paramiko import sftp
from paramiko.util import PFilter
from sklearn.metrics import roc_auc_score

REMOTE_DATASET_PATH = "/home/shamik/github/ENSEMBLE_DIRS/DATASETS"
TEMP_UPLOAD_DIR = "/home/shamik/github/ENSEMBLE_DIRS/TEMP"
REMOTE_ROC_PATH = "/home/shamik/github/ENSEMBLE_DIRS/ROC_CSVS"


def get_sftp_client():
    server = "192.168.2.244"
    username = "shamik"
    password = "Champions1!"

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.load_host_keys(os.path.expanduser(os.path.join("~", ".ssh", "known_hosts")))
    ssh.connect(server, username=username, password=password)
    sftp = ssh.open_sftp()
    return sftp, ssh


def download_remote_file(remotepath, localpath):
    sftp, ssh = get_sftp_client()
    try:
        stat = sftp.stat(remotepath)
        sftp.get(remotepath, localpath)
        sftp.close()
        ssh.close()
        return 1
    except Exception as e:
        return 0


def upload_remote_file(remotepath, localpath):
    try:
        sftp, ssh = get_sftp_client()
        sftp.put(localpath, remotepath)
        sftp.close()
        ssh.close()
        return 1
    except Exception as e:
        return 0


def checkformat(df, flag):
    COLUMNS_PRESENT_FLAG = 0
    CORRECT_VALUE_FLAG = 0
    if "stem" in df.columns and flag in df.columns:
        COLUMNS_PRESENT_FLAG = 1
        stem_isnan, label_isnan = df["stem"].isnull().any(), df[flag].isnull().any()
        if stem_isnan or label_isnan:
            click.echo("there might be NaN values in dataset!")
            return CORRECT_VALUE_FLAG, COLUMNS_PRESENT_FLAG
        if label_isnan is not True:
            if flag == "label":
                int_flag = (
                    df[flag]
                    .apply(lambda x: True if isinstance(x, int) else False)
                    .all()
                )
                if int_flag:
                    CORRECT_VALUE_FLAG = 1
                else:
                    click.echo("There might be non-integer values in label column!")
            elif flag == "score":
                float_flag = (
                    df[flag]
                    .apply(lambda x: True if isinstance(x, float) else False)
                    .all()
                )
                if float_flag:
                    CORRECT_VALUE_FLAG = 1
                else:
                    click.echo("There might be non-float values in score column!")
            return CORRECT_VALUE_FLAG, COLUMNS_PRESENT_FLAG
    return CORRECT_VALUE_FLAG, COLUMNS_PRESENT_FLAG


def upload_roc_csv_helper(dataset_name, model_name, roc_csv_path):
    LOCALSAVEDIR = os.path.expanduser(os.path.join("~", "ENSEMBLEAPI_TEMP"))
    os.makedirs(LOCALSAVEDIR, exist_ok=True)
    idx = str(uuid.uuid1())
    try:
        df = pd.read_csv(roc_csv_path)
    except Exception as e:
        click.echo(e)
        return
    CORRECT_VALUE_FLAG, COLUMNS_PRESENT_FLAG = checkformat(df, "score")
    if CORRECT_VALUE_FLAG == 0 or COLUMNS_PRESENT_FLAG == 0:
        click.echo(
            "Someting wrong with the CSV file! Please check for formatting errors!"
        )
        return
    remotedatafilepath = f"{REMOTE_DATASET_PATH}/{dataset_name}.csv"
    localdatafilepath = os.path.join(LOCALSAVEDIR, f"{dataset_name}___{idx}.csv")
    status_1 = download_remote_file(remotedatafilepath, localdatafilepath)
    remoterocpath = f"{REMOTE_ROC_PATH}/{dataset_name}___{model_name}.csv"
    localrocpath = os.path.join(LOCALSAVEDIR, f"{dataset_name}___{model_name}.csv")
    status_2 = download_remote_file(remoterocpath, localrocpath)
    if status_1 == 0:
        click.echo("Unknown dataset! Please upload the dataset first!")
        return
    if status_1:
        data_df = pd.read_csv(localdatafilepath)
        current_stem_list = df["stem"].tolist()
        dataset_stem_list = data_df["stem"].tolist()
        compare = lambda x, y: collections.Counter(x) == collections.Counter(y)
        if compare(current_stem_list, dataset_stem_list) is False:
            return
    if status_2 == 0:
        data_df = pd.read_csv(localdatafilepath)
        gt_list = data_df["label"].tolist()
        scores = df["score"].tolist()
        roc_auc = roc_auc_score(gt_list, scores)
        click.echo(f"ROCAUC of {model_name} model on {dataset_name} dataset: {roc_auc}")
        status = upload_remote_file(remoterocpath, roc_csv_path)
        if status:
            click.echo(
                "Current CSV file is uploaded successfully! It can be referenced later to calculate ROC score!"
            )
        else:
            click.echo("Something went wrong! Please try again!")
        file_to_rem = Path(localdatafilepath)
        file_to_rem.unlink()
        return
    if status_2 == 1:
        # existing ROC
        data_df = pd.read_csv(localdatafilepath)
        gt_list = data_df["label"].tolist()
        existing_df = pd.read_csv(localrocpath)
        existing_scores = existing_df["score"].tolist()
        existing_rocauc = roc_auc_score(gt_list, existing_scores)
        # current ROC
        current_scores = df["score"].tolist()
        current_rocauc = roc_auc_score(gt_list, current_scores)
        # comprare existing vs current
        if existing_rocauc >= current_rocauc:
            click.echo(
                f"Previously recoreded ROC score of {model_name} model on {dataset_name} is {existing_rocauc}"
            )
            if click.confirm("Do you still want to overwrite exisiting ROC file?"):
                status = upload_remote_file(remoterocpath, roc_csv_path)
                if status:
                    click.echo(
                        "Successfully overwritten previous ROC csv file on server!"
                    )
                else:
                    click.echo("Something went wrong! Please try again!")
        else:
            status = upload_remote_file(remoterocpath, roc_csv_path)
            if status:
                click.echo("Successfully overwritten previous ROC csv file on server!")
            else:
                click.echo("Something went wrong! Please try again!")
        file_to_rem = Path(localdatafilepath)
        file_to_rem.unlink()
        file_to_rem = Path(localrocpath)
        file_to_rem.unlink()
        return


def upload_dataset_helper(dataset_name, dataset_path):
    flag = 0
    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        click.echo(e)
        return -1
    CORRECT_VALUE_FLAG, COLUMNS_PRESENT_FLAG = checkformat(df, flag="label")
    if COLUMNS_PRESENT_FLAG == 0:
        click.echo(
            "Dataset missing required columns. Please make sure stem and label columns are present."
        )
        return -1
    if CORRECT_VALUE_FLAG == 0:
        return -1
    if COLUMNS_PRESENT_FLAG and CORRECT_VALUE_FLAG:
        sftp, ssh = get_sftp_client()
        try:
            stat = sftp.stat(f"{REMOTE_DATASET_PATH}/{dataset_name}.csv")
            if click.confirm(
                "Dataset with same name exists! Do you want to overwrite existing dataset?"
            ):
                flag = 1
        except:
            flag = 1
        if flag:
            sftp.put(dataset_path, f"{REMOTE_DATASET_PATH}/{dataset_name}.csv")
            sftp.close()
            ssh.close()
            click.echo("Dataset uploaded successfully")
        return 0
