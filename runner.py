import datetime

import click

from app import upload_dataset_helper, upload_roc_csv_helper


@click.group()
def cli():
    pass


####### CHECK ENSEMBLE RESULT ########


@cli.command()
@click.option(
    "-cp", type=str, help="Absolute path of ROC CSVs", multiple=True, required=True
)
@click.option(
    "-dn",
    type=str,
    help="Name of the dataset. New dataset can not be used without first uploading it.",
    required=True,
)
@click.option(
    "-mn",
    type=str,
    help="Model name which needs to be ensembled with current CSV",
    multiple=True,
)
def make_ensemble(cp, dn, mn):
    print(cp)
    print(dn)
    print(mn)


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
