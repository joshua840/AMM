from lightning.pytorch.cli import (
    SaveConfigCallback,
    LightningArgumentParser,
    Namespace,
    Trainer,
    LightningModule,
    get_filesystem,
    os,
)


class MySaveConfigCallback(SaveConfigCallback):
    """Saves a LightningCLI config to the log_dir when training starts.

    Args:
        parser: The parser object used to parse the configuration.
        config: The parsed configuration that will be saved.
        config_filename: Filename for the config file.
        overwrite: Whether to overwrite an existing config file.
        multifile: When input is multiple config files, saved config preserves this structure.
        save_to_log_dir: Whether to save the config to the log_dir.

    Raises:
        RuntimeError: If the config file already exists in the directory to avoid overwriting a previous run

    """

    def __init__(
        self,
        parser: LightningArgumentParser,
        config: Namespace,
        config_filename: str = "config.yaml",
        overwrite: bool = True,
        multifile: bool = False,
        save_to_log_dir: bool = True,
    ) -> None:

        super().__init__(
            parser, config, config_filename, overwrite, multifile, save_to_log_dir
        )

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if self.already_saved:
            return

        assert trainer.log_dir is not None
        log_dir = os.path.join(
            trainer.log_dir, trainer.logger.name, trainer.logger.version
        )  # this broadcasts the directory
        if trainer.is_global_zero and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        config_path = os.path.join(log_dir, self.config_filename)

        fs = get_filesystem(log_dir)

        if not self.overwrite:
            # check if the file exists on rank 0
            file_exists = fs.isfile(config_path) if trainer.is_global_zero else False
            # broadcast whether to fail to all ranks
            file_exists = trainer.strategy.broadcast(file_exists)
            if file_exists:
                raise RuntimeError(
                    f"{self.__class__.__name__} expected {config_path} to NOT exist. Aborting to avoid overwriting"
                    " results of a previous run. You can delete the previous config file,"
                    " set `LightningCLI(save_config_callback=None)` to disable config saving,"
                    ' or set `LightningCLI(save_config_kwargs={"overwrite": True})` to overwrite the config file.'
                )

        if trainer.is_global_zero:
            # save only on rank zero to avoid race conditions.
            # the `log_dir` needs to be created as we rely on the logger to do it usually
            # but it hasn't logged anything at this point
            fs.makedirs(log_dir, exist_ok=True)
            self.parser.save(
                self.config,
                config_path,
                skip_none=False,
                overwrite=self.overwrite,
                multifile=self.multifile,
            )

        if trainer.is_global_zero:
            self.save_config(trainer, pl_module, stage)
            self.already_saved = True

        # broadcast so that all ranks are in sync on future calls to .setup()
        self.already_saved = trainer.strategy.broadcast(self.already_saved)
