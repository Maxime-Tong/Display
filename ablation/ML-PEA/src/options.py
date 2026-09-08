import argparse

class Options:

    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # Loss function weights
        self.parser.add_argument(
            "--w_l1", type=float, default=0, help="w"
        )
        self.parser.add_argument(
            "--w_l2", type=float, default=0, help="w"
        )
        self.parser.add_argument(
            "--w_vgg", type=float, default=0, help="w"
        )
        self.parser.add_argument(
            "--w_power", type=float, default=0, help="w"
        )
        self.parser.add_argument(
            "--w_ssim", type=float, default=0, help="w"
        )
        
        # ablation parameters:
        self.parser.add_argument(
            "--method",
            type=str,
            default="MULT",
            help="ADD or MULT",
        )
        self.parser.add_argument(
            "--channels",
            type=int,
            default=1,
            help="number of output channels in dimming map",
        )
        
        # Target power saving rate:
        self.parser.add_argument(
            "--r", type=float, default=.8, help="power factor"
        )
        
        self.parser.add_argument(
            "--pathname", type=str, default="default", help="pathname for save dir"
        )
        self.parser.add_argument(
            "--savedir", type=str, default="default", help="path to saved model"
        )
        self.parser.add_argument(
            "--dataset", type=str, default="div2k", help="dataset name"
        )
        self.parser.add_argument(
            "--result_folder", type=str, default="results", help=""
        )

        # training options
        self.parser.add_argument(
            "--batch_size", type=int, default=1, help="batch size for training network."
        )
        self.parser.add_argument(
            "--epochs", type=int, default=60, help="number of epochs"
        )
        self.parser.add_argument(
            "--lr", type=float, default=0.0002, help="learning rate"
        )

        # debugging options
        self.parser.add_argument(
            "--print_model", action="store_true", help="print model"
        )
        self.parser.add_argument(
            "--save_ckpt_after",
            type=int,
            default=30,
            help="",
        )
        self.parser.add_argument(
            "--log_after",
            type=int,
            default=100,
            help="",
        )
        self.parser.add_argument(
            "--save_results_after",
            type=int,
            default=3,
            help="",
        )
        self.parser.add_argument(
            "--save_epoch",
            type=int,
            default=25,
            help="",
        )

    def parse(self):
        self.opt = self.parser.parse_args()
        print(self.opt)
        return self.opt
