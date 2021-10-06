import collections
import os
import shutil
import uuid
from collections import defaultdict
from pathlib import Path
from stat import S_ISDIR, S_ISREG
from time import localtime, sleep
from zipfile import ZipFile

import click
import numpy as np
import pandas as pd
import paramiko
import yaml
from colorama import Back, Fore, Style, init
from pandas.io.parsers import read_csv
from paramiko import sftp
from paramiko.util import PFilter
from sklearn.metrics import roc_auc_score
from sklearn.utils.extmath import weighted_mode

init()

REMOTE_DATASET_PATH = "/home/shamik/github/ENSEMBLE_DIRS/DATASETS"
TEMP_UPLOAD_DIR = "/home/shamik/github/ENSEMBLE_DIRS/TEMP"
REMOTE_ROC_PATH = "/home/shamik/github/ENSEMBLE_DIRS/ROC_CSVS"


def delete_file(filepath):
    try:
        if isinstance(filepath, list):
            for path in filepath:
                file_to_rem = Path(path)
                file_to_rem.unlink()
        else:
            file_to_rem = Path(filepath)
            file_to_rem.unlink()
    except Exception as e:
        click.echo(e)


def print_text(string, color="red"):
    if color == "red":
        click.echo(Fore.RED + string + Style.RESET_ALL)
    elif color == "green":
        click.echo(Fore.GREEN + string + Style.RESET_ALL)
    elif color == "yellow":
        click.echo(Fore.YELLOW + string + Style.RESET_ALL)


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
            click.echo(
                Fore.RED + "there might be NaN values in dataset!" + Style.RESET_ALL
            )
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
                    click.echo(
                        Fore.RED
                        + "There might be non-integer values in label column!"
                        + Style.RESET_ALL
                    )
            elif flag == "score":
                float_flag = (
                    df[flag]
                    .apply(lambda x: True if isinstance(x, float) else False)
                    .all()
                )
                if float_flag:
                    CORRECT_VALUE_FLAG = 1
                else:
                    click.echo(
                        Fore.RED
                        + "There might be non-float values in score column!"
                        + Style.RESET_ALL
                    )
            return CORRECT_VALUE_FLAG, COLUMNS_PRESENT_FLAG
    return CORRECT_VALUE_FLAG, COLUMNS_PRESENT_FLAG


def upload_roc_csv_helper(dataset_name, model_name, roc_csv_path):
    LOCALSAVEDIR = os.path.expanduser(os.path.join("~", ".ENSEMBLEAPI_TEMP"))
    os.makedirs(LOCALSAVEDIR, exist_ok=True)
    idx = str(uuid.uuid1())
    try:
        df = pd.read_csv(roc_csv_path)
    except Exception as e:
        click.echo(Fore.RED + str(e) + Style.RESET_ALL)
        return
    CORRECT_VALUE_FLAG, COLUMNS_PRESENT_FLAG = checkformat(df, "score")
    if CORRECT_VALUE_FLAG == 0 or COLUMNS_PRESENT_FLAG == 0:
        click.echo(
            Fore.RED
            + "Someting wrong with the CSV file! Please check for formatting errors!"
            + Style.RESET_ALL
        )
        return
    remotedatafilepath = f"{REMOTE_DATASET_PATH}/{dataset_name}.csv"
    localdatafilepath = os.path.join(LOCALSAVEDIR, f"{dataset_name}___{idx}.csv")
    status_1 = download_remote_file(remotedatafilepath, localdatafilepath)
    remoterocpath = f"{REMOTE_ROC_PATH}/{dataset_name}___{model_name}.csv"
    localrocpath = os.path.join(LOCALSAVEDIR, f"{dataset_name}___{model_name}.csv")
    status_2 = download_remote_file(remoterocpath, localrocpath)
    if status_1 == 0:
        click.echo(
            Fore.RED
            + "Unknown dataset! Please upload the dataset first!"
            + Style.RESET_ALL
        )
        return
    if status_1:
        data_df = pd.read_csv(localdatafilepath)
        current_stem_list = df["stem"].tolist()
        dataset_stem_list = data_df["stem"].tolist()
        compare = lambda x, y: collections.Counter(x) == collections.Counter(y)
        if compare(current_stem_list, dataset_stem_list) is False:
            delete_file(localdatafilepath)
            return
    if status_2 == 0:
        data_df = pd.read_csv(localdatafilepath)
        gt_list = data_df["label"].tolist()
        scores = df["score"].tolist()
        roc_auc = roc_auc_score(gt_list, scores)
        click.echo(
            f"ROCAUC of {Fore.CYAN}{model_name}{Style.RESET_ALL} model on {Fore.CYAN}{dataset_name}{Style.RESET_ALL} dataset: {Fore.GREEN}{roc_auc}{Style.RESET_ALL}"
        )
        status = upload_remote_file(remoterocpath, roc_csv_path)
        if status:
            click.echo(
                Fore.GREEN
                + "Current CSV file is uploaded successfully! It can be referenced later to calculate ROC score!"
                + Style.RESET_ALL
            )
        else:
            click.echo(
                Fore.RED + "Something went wrong! Please try again!" + Style.RESET_ALL
            )
        delete_file(localdatafilepath)
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
            if click.confirm(
                Fore.YELLOW
                + "Do you still want to overwrite exisiting ROC file?"
                + Style.RESET_ALL
            ):
                status = upload_remote_file(remoterocpath, roc_csv_path)
                if status:
                    click.echo(
                        Fore.GREEN
                        + "Successfully overwritten previous ROC csv file on server!"
                        + Style.RESET_ALL
                    )
                else:
                    click.echo(
                        Fore.RED
                        + "Something went wrong! Please try again!"
                        + Style.RESET_ALL
                    )
        else:
            status = upload_remote_file(remoterocpath, roc_csv_path)
            if status:
                click.echo(
                    Fore.GREEN
                    + "Successfully overwritten previous ROC csv file on server!"
                    + Style.RESET_ALL
                )
            else:
                click.echo(
                    Fore.RED
                    + "Something went wrong! Please try again!"
                    + Style.RESET_ALL
                )
        delete_file([localdatafilepath, localrocpath])
        return


def upload_dataset_helper(dataset_name, dataset_path):
    flag = 0
    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        click.echo(Fore.RED + str(e) + Style.RESET_ALL)
        return -1
    CORRECT_VALUE_FLAG, COLUMNS_PRESENT_FLAG = checkformat(df, flag="label")
    if COLUMNS_PRESENT_FLAG == 0:
        click.echo(
            Fore.RED
            + "Dataset missing required columns. Please make sure stem and label columns are present."
            + Style.RESET_ALL
        )
        return -1
    if CORRECT_VALUE_FLAG == 0:
        return -1
    if COLUMNS_PRESENT_FLAG and CORRECT_VALUE_FLAG:
        sftp, ssh = get_sftp_client()
        try:
            stat = sftp.stat(f"{REMOTE_DATASET_PATH}/{dataset_name}.csv")
            if click.confirm(
                Fore.YELLOW
                + "Dataset with same name exists! Do you want to overwrite existing dataset?"
                + Style.RESET_ALL
            ):
                flag = 1
        except:
            flag = 1
        if flag:
            sftp.put(dataset_path, f"{REMOTE_DATASET_PATH}/{dataset_name}.csv")
            sftp.close()
            ssh.close()
            click.echo(Fore.GREEN + "Dataset uploaded successfully" + Style.RESET_ALL)
        return 0


def ensemble_helper(cp, dn, w=""):
    LOCALSAVEDIR = os.path.expanduser(os.path.join("~", ".ENSEMBLEAPI_TEMP"))
    os.makedirs(LOCALSAVEDIR, exist_ok=True)
    idx = str(uuid.uuid1())
    remotedatafilepath = f"{REMOTE_DATASET_PATH}/{dn}.csv"
    localdatafilepath = os.path.join(LOCALSAVEDIR, f"{dn}___{idx}.csv")
    status_1 = download_remote_file(remotedatafilepath, localdatafilepath)
    if status_1 == 0:
        click.echo(
            Fore.RED
            + "Unknown dataset! Please upload the dataset first!"
            + Style.RESET_ALL
        )
        return

    roc_list = cp.split(",")
    if w:
        w_list = w.split(",")
        if len(roc_list) != len(w_list):
            click.echo(
                Fore.RED
                + "Each file/model should have its corresponding weights!"
                + Style.RESET_ALL
            )
            delete_file(localdatafilepath)
            return
    file_list, local_file_list, remote_file_list = [], [], []
    for item in roc_list:
        if len(item.split("/")) > 1 or item.split("/")[-1].split(".")[-1] == "csv":
            local_file_list.append(item)
            file_list.append(item)
        else:
            remote_file_list.append(item)
    localrocpathlist = []
    for item in remote_file_list:
        remoterocpath = f"{REMOTE_ROC_PATH}/{dn}___{item}.csv"
        localrocpath = os.path.join(LOCALSAVEDIR, f"{dn}___{item}.csv")
        status = download_remote_file(remoterocpath, localrocpath)
        if status == 0:
            click.echo(
                Fore.RED
                + "Something went wrong in getting records from server. Please check if the requested CSV is present in server!"
                + Style.RESET_ALL
            )
            delete_file(localrocpathlist + [localdatafilepath])
            return
        elif status == 1:
            localrocpathlist.append(localrocpath)
            file_list.append(localrocpath)

    if len(file_list) == 1:  # for a single file, calculate ROCAUC directly
        df = pd.read_csv(file_list[0])
        df_data = pd.read_csv(localdatafilepath)
        y_true = df_data["label"].tolist()
        y_preds = df["score"].tolist()
        try:
            roc_auc = roc_auc_score(y_true, y_preds)
            click.echo(
                f"Only one file detected! ROC-AUC: {Fore.GREEN}{roc_auc}{Style.RESET_ALL}"
            )
        except Exception as e:
            delete_file(localrocpathlist + [localdatafilepath])
            click.echo(Fore.RED + str(e) + Style.RESET_ALL)

    else:  # for multiple files create a dataframe with columns [stem, score1, score2, ... , label]
        d = defaultdict(list)
        try:
            for i in range(len(file_list)):
                df = pd.read_csv(file_list[i])
                for index, row in df.iterrows():
                    d[row["stem"]].append(row["score"])
            df_data = pd.read_csv(localdatafilepath)
            for index, row in df_data.iterrows():
                d[row["stem"]].append(row["label"])

            if (
                np.sum([len(d[key]) == len(file_list) + 1 for key in d.keys()]) / len(d)
                != 1.0
            ):
                click.echo(
                    Fore.RED
                    + "stem column of each csv should have identical elements"
                    + Style.RESET_ALL
                )
                delete_file(localrocpathlist + [localdatafilepath])
                return
        except Exception as e:
            click.echo(Fore.RED + str(e) + Style.RESET_ALL)
            delete_file(localrocpathlist + [localdatafilepath])
            return
        y_true, y_preds = [], []

        if len(w) == 0:  # if no weights are provided, use basic average
            weights = [1 / len(file_list)] * len(file_list)
        else:  # if weights are provided, convert them to float from str
            try:
                weights = list(map(float, w.split(",")))
            except:
                click.echo(
                    Fore.RED
                    + "Please make sure provided weights are correct"
                    + Style.RESET_ALL
                )
                delete_file(localrocpathlist + [localdatafilepath])
                return
        try:
            for k, v in d.items():
                # y_preds.append(np.average(v[:-1]))
                y_preds.append(np.dot(weights, v[:-1]))
                y_true.append(int(v[-1]))
        except Exception as e:
            click.echo(Fore.RED + str(e) + Style.RESET_ALL)
            delete_file(localrocpathlist + [localdatafilepath])
            return
        click.echo(
            f"Ensembled ROC_AUC: {Fore.GREEN}{roc_auc_score(y_true, y_preds)}{Style.RESET_ALL}"
        )
        delete_file(localrocpathlist + [localdatafilepath])


def get_model_list_helper(dn):
    roc_list = []
    roc_dict = defaultdict(list)
    sftp, ssh = get_sftp_client()
    for entry in sftp.listdir_attr(REMOTE_ROC_PATH):
        mode = entry.st_mode
        if S_ISREG(mode):
            roc_list.append(entry.filename)
    for item in roc_list:
        dataset = item.split("___")[0]
        model = item.split("___")[1][:-4]
        roc_dict[dataset].append(model)

    click.echo(f"\nAvailable models and datasets:")
    for k, v in roc_dict.items():
        if dn == "" or dn == k:
            click.echo(f"\ndataset: {Fore.BLUE}{k}{Style.RESET_ALL}")
            click.echo("Available models:")
            for item in v:
                click.echo(f"{Fore.CYAN}{item}{Style.RESET_ALL}")
