from lightning.pytorch.cli import LightningCLI
from src.module.utils.mycallback import MySaveConfigCallback


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("trainer.max_epochs", "model.init_args.max_epochs")
        parser.link_arguments("seed_everything", "model.init_args.data_seed")


def cli_main():
    cli = MyLightningCLI(run=False, save_config_callback=MySaveConfigCallback)

    cli.trainer.fit(cli.model)
    cli.trainer.test(cli.model, ckpt_path="last")


if __name__ == "__main__":
    cli_main()
