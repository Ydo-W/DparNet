import argparse


class Parameter:
    def __init__(self):
        self.args = self.set_args()

    def set_args(self):
        self.parser = argparse.ArgumentParser(description='Imaging parameter assisted image restoration Networks')

        # Global parameters
        self.parser.add_argument('--seed', type=int, default=3407)
        self.parser.add_argument('--batch_size', type=int, default=2)

        # Data parameters
        self.parser.add_argument('--frame_length', type=int, default=15)
        self.parser.add_argument('--save_dir', type=str, default='checkpoints/')
        self.parser.add_argument('--results_dir', type=str, default='results/')
        self.parser.add_argument('--data_root', type=str, default='data/')

        # Model parameters
        self.parser.add_argument('--n_feats', type=int, default=16)
        self.parser.add_argument('--neighboring_frames', type=int, default=2)

        args, _ = self.parser.parse_known_args()

        return args
