#!/usr/bin/env python

import startup

from run import train_to
from run.predict_eval import compute_eval


def main():
    train_to.train()
    compute_eval()


if __name__ == '__main__':
    main()
