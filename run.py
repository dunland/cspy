#!/usr/bin/python

from cityscopy import Cityscopy

if __name__ == '__main__':
    # init cityscopy class
    cityscopy = Cityscopy("settings/cityscopy.json")

    # keystone the scanned area
    key = input("Want to use keystone? y/n: ")
    if key == 'y':
        # set keystones
        cityscopy.keystone()
    else:
        # scan the grid
        cityscopy.scan()
