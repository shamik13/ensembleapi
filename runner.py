import datetime

import click

from app import ensemble_helper, upload_dataset_helper, upload_roc_csv_helper


@click.group()
def cli():
    pass


####### CHECK ENSEMBLE RESULT ########


@cli.command()
@click.option(
    "-cp",
    type=str,
    help="Absolute path of ROC CSVs / model name separated by comma",
    required=True,
)
@click.option(
    "-dn",
    type=str,
    help="Name of the dataset. New dataset can not be used without first uploading it.",
    required=True,
)
@click.option(
    "-w", type=str, help="comma separated weights to be used for ensemble.", default=""
)
def make_ensemble(cp, dn, w):
    ensemble_helper(cp, dn, w)


######### UPLOAD DATASET #############


@cli.command()
@click.option("-dn", type=str, help="Unique name for this dataset", required=True)
@click.option(
    "-cp",
    type=str,
    help="CSV Path for the dataset. Please ask the maintainer for proper format of the dataset",
    required=True,
)
def upload_dataset(dn, cp):
    upload_dataset_helper(dn, cp)


########## UPLOAD ROC CSV ##############


@cli.command()
@click.option("-dn", type=str, help="Enter dataset name.", required=True)
@click.option("-mn", type=str, help="Enter model name", required=True)
@click.option("-cp", type=str, help="Enter the CSV path", required=True)
def upload_roc_csv(dn, mn, cp):
    upload_roc_csv_helper(dn, mn, cp)


if __name__ == "__main__":
    cli()
