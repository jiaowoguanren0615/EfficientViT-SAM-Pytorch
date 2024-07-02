import argparse
import os

import wandb

import setup
from util import dump_config, parse_unknown_args
from models import create_sam_model
from datasets import SAMDataProvider
from trainer import SAMRunConfig, SAMTrainer

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default='./configs/default.yaml', help="config file")
parser.add_argument("--path", type=str, metavar="DIR", help="run directory")
parser.add_argument("--manual_seed", type=int, default=0)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--amp", type=str, choices=["fp32", "fp16", "bf16"], default="fp32")
parser.add_argument("--rand_init", type=str, default="trunc_normal@0.02")
parser.add_argument("--last_gamma", type=float, default=0)


def main():
    args, opt = parser.parse_known_args()
    opt = parse_unknown_args(opt)

    setup.setup_dist_env()

    os.makedirs(args.path, exist_ok=True)
    dump_config(args.__dict__, os.path.join(args.path, "args.yaml"))

    setup.setup_seed(args.manual_seed, args.resume)

    config = setup.setup_exp_config(args.config, recursive=True, opt_args=opt)

    setup.save_exp_config(config, args.path)

    data_provider = setup.setup_data_provider(config, [SAMDataProvider], is_distributed=True)

    run_config = setup.setup_run_config(config, SAMRunConfig)

    model = create_sam_model(config["net_config"]["name"], False)

    trainer = SAMTrainer(
        path=args.path,
        model=model,
        data_provider=data_provider,
    )

    setup.init_model(
        trainer.network,
        init_from=config["net_config"]["ckpt"],
        rand_init=args.rand_init,
        last_gamma=args.last_gamma,
    )

    trainer.prep_for_training(run_config, args.amp)

    if args.resume:
        trainer.load_model()
        trainer.data_provider = setup.setup_data_provider(config, [SAMDataProvider], is_distributed=True)
    else:
        trainer.sync_model()

    trainer.train()


if __name__ == "__main__":
    main()