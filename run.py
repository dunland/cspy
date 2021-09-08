#!/usr/bin/python

from cityscopy import Cityscopy
import json

if __name__ == '__main__':
    CITYSCOPY_SETTINGS_PATH = "settings/cityscopy.json"
    # init cityscopy class
    cityscopy = Cityscopy(CITYSCOPY_SETTINGS_PATH)

    # select camera:
    key = input("Want to search for cameras? y/n/r(ealsense): ")
    if key == 'y':
        cityscopy.print_cams()
        cam = input("pick camera: ")

        with open(CITYSCOPY_SETTINGS_PATH) as f:
            data = json.load(f)
            data['cam_id'] = cam
            data['realsense']['active'] = False
            json.dump(data, f)

        with open('settings/cityscopy.json', 'w') as output_file:
            json.dump(data, output_file)

    elif key == 'r':
        with open(CITYSCOPY_SETTINGS_PATH) as f:
            data = json.load(f)
            data['realsense']['active'] = True
        with open('settings/export.json', 'w') as output_file:
            json.dump(data, output_file)

    # run CityScopy main methods

    # keystone the scanned area
    key = input("Want to use keystone? y/n: ")
    if key == 'y':
        cityscopy.keystone()

    # scan the grid and send to cityIO
    cityscopy.scan()
    # start local UDP comms
    # # cityscopy.udp_listener()
