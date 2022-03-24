#!/usr/bin/env python3

"""Makes a slim checkpoint (without optimizer states) from a training checkpoint."""

import argparse
import pickle


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=str,
                   help='the input training checkpoint')
    p.add_argument('output', type=str,
                   help='the output slim checkpoint')
    args = p.parse_args()

    ckpt = pickle.load(open(args.input, 'rb'))
    del ckpt['opt_state']
    pickle.dump(ckpt, open(args.output, 'wb'))


if __name__ == '__main__':
    main()
