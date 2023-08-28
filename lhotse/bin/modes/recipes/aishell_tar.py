import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.aishell_tar import prepare_aishell_tar
from lhotse.recipes.aishell import download_aishell
from lhotse.utils import Pathlike

__all__ = ["aishell_tar"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
def aishell_tar(corpus_dir: Pathlike, output_dir: Pathlike):
    """Aishell ASR data preparation."""
    prepare_aishell_tar(corpus_dir, output_dir=output_dir)

@download.command(context_settings=dict(show_default=True))
@click.argument("target_dir", type=click.Path())
def aishell_tar(target_dir: Pathlike):
    """Aishell download."""
    download_aishell(target_dir)