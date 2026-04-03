"""Command-line-interface to run SHARP model.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

import click

from . import finetune, finetune_ddp, predict, render, render_invisible_mask


@click.group()
def main_cli():
    """Run inference for SHARP model."""
    pass


main_cli.add_command(predict.predict_cli, "predict")
main_cli.add_command(render.render_cli, "render")
main_cli.add_command(render_invisible_mask.render_invisible_mask_cli, "render-invisible-mask")
main_cli.add_command(finetune.finetune_cli, "finetune")
main_cli.add_command(finetune_ddp.finetune_ddp_cli, "finetune-ddp")
