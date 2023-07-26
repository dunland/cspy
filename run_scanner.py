#!/usr/bin/python

import sys

from cityscopy import Cityscopy

if __name__ == '__main__':

    settings = sys.argv[1] if len(sys.argv) > 1 else "settings/settings.json"

    # init cityscopy class
    cityscopy = Cityscopy(settings)
    cityscopy.scan()
