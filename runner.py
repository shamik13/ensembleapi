import datetime
import click
from app import download_files, set_output_path, upload_files




@click.group()
def cli():
    pass


@cli.command()
def hello():
    click.echo("Hello World")


@cli.command()
@click.option(
    "-dsp",
    "--datasetpath",
    type=str,
    help="Absolute path of Dataset [ZIP]",
)
@click.option(
    "-cp",
    "--configpath",
    type=str,
    help="Absolute path of Config file [YAML]",
)
def upload_csv(model, csvpath):


@cli.command()
@click.option(
    "-idx",
    type=str,
    help="Unique identifier of the previous experiment",
)
def get_weights(idx):
    download_files(idx)


@cli.command()
@click.option(
    "-op",
    type=str,
    help="Set the directory path where trained model weights will be saved.",
)
def set_output_dir(op):
    set_output_path(op)
    if op != "":
        click.echo(f"Output directory set to {op}")
    else:
        pass


if __name__ == "__main__":
    cli()
