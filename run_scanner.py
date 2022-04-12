#!/usr/bin/python

from cityscopy import Cityscopy

if __name__ == '__main__':
    # init cityscopy class
    cityscopy = Cityscopy("settings/cityscopy.json")
    cityscopy.scan()
