'''
>>>>>>>>>>>>> Starting CityScope Scanner >>>>>>>>>>>>


                    |||||||||||
                    |||||||||||
                            |||
                            |||
                            |||
                    |||      ||||||||||||
                    |||      ||||||||||||
                    |||               |||
                    |||               |||
                    |||               |||
                    ||||||||||||      |||
                    ||||||||||||      |||


>>>>>>>>>>>>> Starting CityScope Scanner >>>>>>>>>>>>

Copyright (C) {{ 2018 }}  {{ Ariel Noyman }}

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>

"@context": "https://github.com/CityScope/", "@type": "Person", "address": {
"@type": "75 Amherst St, Cambridge, MA 02139", "addressLocality":
"Cambridge", "addressRegion": "MA",},
"jobTitle": "Research Scientist", "name": "Ariel Noyman",
"alumniOf": "MIT", "url": "http://arielnoyman.com",
"https://www.linkedin.com/", "http://twitter.com/relno",
https://github.com/RELNO]


##################################################
CityScope Python Scanner
Keystone, decode and send over UDP/HTTTP a 2d array
of uniquely tagged LEGO array
##################################################
'''

import cv2
import numpy as np
from datetime import timedelta
from datetime import datetime
import time
import json
import os
import socket
from multiprocessing import Process, Manager
import pyrealsense2 as rs


class Cityscopy:
    '''scanner for CityScope'''

    def __init__(self, path):
        '''init function '''
        # load info from json file
        self.SETTINGS_PATH = path
        # get the table settings. This is used bu many metohds
        self.table_settings = self.parse_json_file('cityscopy')
        print('getting settings for CityScopy...')

        # init corners variables
        self.selected_corner = None
        self.magnitude = 1

        # realsense camera parameters
        self.exposure = 100
        self.gain = 1

        # color conversion threshold
        self.color_conversion_threshold = 125

        # init keystone variables
        self.FRAME = None
        self.POINT_INDEX = None
        self.POINTS = None
        self.MOUSE_POSITION = None

    def print_cams(self):
        print("reading first 100 cameras...")
        arr = []
        for index in range(0, 100):
            cap = cv2.VideoCapture(index)
            # returns video frames; False if no frames have been grabbed
            if cap.read()[0]:
                arr.append(index)
            cap.release()
        print("found cameras: ", arr)

    def realsense_init(self):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        self.device = pipeline_profile.get_device()
        device_product_line = str(self.device.get_info(rs.camera_info.product_line))
        print("using Intel Realsense", device_product_line)

        self.device = pipeline_profile.get_device().first_color_sensor()
        self.device.set_option(rs.option.exposure, 166)
        self.device.set_option(rs.option.gain, 64)

        # Start streaming
        try:
            # setup for USB 3
            try:
                config.enable_stream(rs.stream.color, 1920,
                                     1080, rs.format.bgr8, 30)
                print("trying to stream 1920x1080...", end=" ")
                self.pipeline.start(config)
            except Exception:
                print("no success.")
                config.enable_stream(rs.stream.color, 1280,
                                     720, rs.format.bgr8, 30)
                print("trying to stream 1280x720...", end=" ")
                self.pipeline.start(config)
        except Exception:
            # setup for USB 2
            print("no success.")
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            print("streaming in 640x480...", end=" ")
            self.pipeline.start(config)

        print("success!")
        print("Realsense initialization complete.")

    def scan(self):
        # define global list manager
        MANAGER = Manager()
        # create shared global list to work with both processes
        self.multiprocess_shared_dict = MANAGER.dict()

        # init a dict to be shared among procceses
        self.multiprocess_shared_dict['scan'] = None

        # defines a multiprocess for sending the data
        self.process_send_packet = Process(target=self.create_data_json,
                                           args=([self.multiprocess_shared_dict]))
        # start porcess
        self.process_send_packet.start()
        # start camera on main thread due to multiprocces issue
        self.scanner_function(self.multiprocess_shared_dict)
        # join the two processes
        self.process_send_packet.join()

    def get_init_keystone(self):
        # load the initial keystone data from file
        keystone_points_array = np.loadtxt(
            self.get_folder_path() + 'keystone.txt', dtype=np.float32)

        return keystone_points_array

    def scanner_function(self, multiprocess_shared_dict):
        # get init keystones
        self.init_keystone = self.get_init_keystone()

        # define the table params
        grid_dimensions_x = self.table_settings['ncols']
        grid_dimensions_y = self.table_settings['nrows']
        array_of_tags_from_json = [[int(ch) for ch in i] for i in self.table_settings['tags']]

        cell_gap = self.table_settings['cell_gap']

        # init type list array
        TYPES_LIST = []

        # holder of old cell colors array to check for new scan
        OLD_CELL_COLORS_ARRAY = []

        # serial num of camera, to switch between cameras
        camPos = self.table_settings['camId']

        # check if using intel realsense camera or not
        self.using_realsense = self.table_settings['realsense']

        # try from a device 1 in list, not default webcam
        if self.using_realsense:
            self.realsense_init()
            # video_capture = self.pipeline
        else:
            video_capture = cv2.VideoCapture(camPos)
        time.sleep(1)

        if self.using_realsense:
            if grid_dimensions_y < grid_dimensions_x:
                # get the smaller of two grid ratio x/y or y/x
                grid_ratio = grid_dimensions_y / grid_dimensions_x
                video_resolution_x = self.pipeline.wait_for_frames()\
                    .get_color_frame().get_width()
                video_resolution_y = self.pipeline.wait_for_frames()\
                    .get_color_frame().get_height() * grid_ratio
            else:
                # get the smaller of two grid ratio x/y or y/x
                grid_ratio = grid_dimensions_x / grid_dimensions_y
                video_resolution_x = self.pipeline.wait_for_frames()\
                    .get_color_frame().get_width() * grid_ratio
                video_resolution_y = self.pipeline.wait_for_frames()\
                    .get_color_frame().get_height()
        else:
            if grid_dimensions_y < grid_dimensions_x:
                # get the smaller of two grid ratio x/y or y/x
                grid_ratio = grid_dimensions_y / grid_dimensions_x
                video_resolution_x = int(video_capture.get(3))
                video_resolution_y = int(video_capture.get(3) * grid_ratio)
            else:
                # get the smaller of two grid ratio x/y or y/x
                grid_ratio = grid_dimensions_x / grid_dimensions_y
                video_resolution_x = int(video_capture.get(3) * grid_ratio)
                video_resolution_y = int(video_capture.get(3))

        # define the video window
        cv2.namedWindow('scanner_gui_window', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('scanner_gui_window', 400, 400)
        cv2.moveWindow('scanner_gui_window', 10, 100)

        # define the size for each scanner
        x_ratio = int(video_resolution_x / (grid_dimensions_x * 4))
        y_ratio = int(video_resolution_y / (grid_dimensions_y * 4))
        scanner_square_size = np.minimum(x_ratio, y_ratio)

        # create the location  array of scanners
        array_of_scanner_points = self.get_scanner_pixel_coordinates(
            grid_dimensions_x, grid_dimensions_y, cell_gap, video_resolution_x,
            video_resolution_y, scanner_square_size)

        # resize video resolution if cell gaps are used:
        video_resolution_x = int(video_resolution_x + grid_dimensions_x * cell_gap)
        video_resolution_y = int(video_resolution_y + grid_dimensions_y * cell_gap)

        # run the video loop forever
        while True:
            if self.using_realsense:
                frames = self.pipeline.wait_for_frames()  # returns composite_frame
                color_frame = frames.get_color_frame()  # returns video_frame
                color_frame = np.asanyarray(color_frame.get_data())
            else:
                # read video frames
                RET, color_frame = video_capture.read()

            # zero an array to collect the scanners
            CELL_COLORS_ARRAY = []

            # get a new matrix transformation every frame
            keystone_data = self.transform_matrix(
                video_resolution_x, video_resolution_y, self.listen_to_UI_interaction())

            # mirror camera (webcam)
            if self.table_settings['mirror_cam']:
                if self.using_realsense:
                    color_frame = np.flip(color_frame, 1)
                elif RET:
                    color_frame = cv2.flip(color_frame, 1)

            # warp the video based on keystone info
            keystoned_video = cv2.warpPerspective(
                color_frame, keystone_data, (int(video_resolution_x), int(video_resolution_y)))

            # run through locations list and make scanners
            for x, y in array_of_scanner_points:
                # use this to control reduction of scanner size
                this_scanner_max_dimension = int(scanner_square_size)
                scanner_reduction = int(scanner_square_size / 4)
                # short vars
                x_red = x + this_scanner_max_dimension - scanner_reduction
                y_red = y + this_scanner_max_dimension - scanner_reduction

                # set scanner crop box size and position
                # at x,y + crop box size
                this_scanner_size = keystoned_video[y + scanner_reduction:y_red,
                                                    x + scanner_reduction:x_red]

                # draw rects with mean value of color
                mean_color = cv2.mean(this_scanner_size)

                # convert colors to rgb
                mean_color_RGB = np.uint8([[np.uint8(mean_color)[:3]]])

                # select the right color based on sample
                scannerCol = self.select_color_by_mean_value(mean_color_RGB)

                # add colors to array for type analysis
                CELL_COLORS_ARRAY.append(scannerCol)

                # get color from dict
                thisColor = [(0, 0, 0), (255, 255, 255)][scannerCol]

                # only draw vis if settings has 1 in gui
                if self.table_settings['gui']:
                    # draw rects with frame colored by range result
                    cv2.rectangle(keystoned_video,
                                  (x + scanner_reduction, y + scanner_reduction),
                                  (x_red, y_red), thisColor, 1)

            # reduce unnecessary scan analysis and sending by comparing
            # the list of scanned cells to an old one
            if CELL_COLORS_ARRAY != OLD_CELL_COLORS_ARRAY:
                # send array to method for checking types
                TYPES_LIST = self.find_type_in_tags_array(
                    CELL_COLORS_ARRAY, array_of_tags_from_json,
                    grid_dimensions_x, grid_dimensions_y)

                # match the two
                OLD_CELL_COLORS_ARRAY = CELL_COLORS_ARRAY

                # [!] Store the type list results in the multiprocess_shared_dict
                multiprocess_shared_dict['scan'] = TYPES_LIST

            if self.table_settings['gui']:
                # draw arrow to interaction area
                self.ui_selected_corner(
                    video_resolution_x, video_resolution_y, keystoned_video)
                cv2.putText(keystoned_video, "magnitude: " + str(self.magnitude) + " [SPACE]", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 1, cv2.LINE_AA)
                cv2.putText(keystoned_video, "exposure: " + str(self.exposure) + " [e/r]", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 1, cv2.LINE_AA)
                cv2.putText(keystoned_video, "gain: " + str(self.gain) + " [g/h]", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 1, cv2.LINE_AA)
                cv2.putText(keystoned_video, "color_conversion_threshold: " + str(self.color_conversion_threshold) + " [+/-]", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 1, cv2.LINE_AA)

                # draw the video to screen
                cv2.imshow("scanner_gui_window", keystoned_video)

        # close opencv
        video_capture.release()
        cv2.destroyAllWindows()

    def ui_selected_corner(self, x, y, vid):
        """prints text on video window"""

        mid = (int(x / 2), int(y / 2))
        if self.selected_corner is not None:
            case = {
                '1': [(0, 0), mid],
                '2': [(x, 0), mid],
                '3': [(0, y), mid],
                '4': [(x, y), mid],
            }

            # print(type(self.selected_corner))
            col = (0, 0, 255) if self.magnitude == 1 else (255, 0, 0)
            cv2.arrowedLine(
                vid, case[self.selected_corner][0],
                case[self.selected_corner][1],
                col, 2)

    def get_scanner_pixel_coordinates(self, grid_dimensions_x, grid_dimensions_y, cell_gap,
                                      video_res_x, video_res_y, scanner_square_size):
        """Creates list of pixel coordinates for scanner.

        Steps:
            - Determine virtual points on the grid that
            will be the centers of blocks.
            - Transforms virtual[x, y] coordinate pairs to
            pixel representations for scanner.
            - Transform those virtual points pixel coordinates
            and expand them into 3x3 clusters of pixel points

        Args:

        Returns list of[x, y] pixel coordinates for scanner to read.
        """
        # calc the half of the ratio between
        # screen size and the x dim of the grid
        # to center the grid in both Y & X
        grid_x_offset = int(0.5 * (video_res_x - grid_dimensions_x * scanner_square_size * 4))
        grid_y_offset = int(0.5 * (video_res_y - grid_dimensions_y * scanner_square_size * 4))

        # create the list of points
        pixel_coordinates_list = []

        # create the 4x4 sub grid cells
        for y in range(grid_dimensions_y):
            for x in range(grid_dimensions_x):
                x_positions = x * (scanner_square_size * 4 + cell_gap)
                y_positions = y * (scanner_square_size * 4 + cell_gap)

                # make the actual location for the 4x4 scanner points
                for i in range(4):
                    for j in range(4):
                        # add them to list
                        pixel_coordinates_list.append(
                            # x value of this scanner location
                            [grid_x_offset + x_positions + i * scanner_square_size,
                             # y value of this scanner location
                             grid_y_offset + y_positions + j * scanner_square_size])
        # print(pixel_coordinates_list)
        return pixel_coordinates_list

    # 2nd proccess to
    def create_data_json(self, multiprocess_shared_dict):
        SEND_INTERVAL = self.table_settings['interval']
        # initial dummy value for old grid
        old_scan_results = [-1]
        SEND_INTERVAL = timedelta(milliseconds=SEND_INTERVAL)
        last_sent = datetime.now()

        while True:
            scan_results = multiprocess_shared_dict['scan']
            from_last_sent = datetime.now() - last_sent

            if scan_results and scan_results != old_scan_results and \
                    from_last_sent > SEND_INTERVAL:
                try:
                    # send as string via UDP:
                    self.send_json_to_UDP(scan_results)
                except Exception as ERR:
                    print(ERR)
                # match the two grid after send
                old_scan_results = scan_results
                last_sent = datetime.now()

                # debug print
                print('\n', 'CityScopy grid sent at:', datetime.now())
                print(scan_results)

    def send_json_to_UDP(self, scan_results):
        # defining the udp endpoint
        UDP_IP = "127.0.0.1"
        UDP_PORT = 5000
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.sendto(str(scan_results).encode('utf-8'), (UDP_IP, UDP_PORT))
        except Exception as e:
            print(e)

    def parse_json_file(self, field):
        """
        get data from JSON settings files.

        Steps:
        - opens file
        - checks if field has objects longer than one char [such as the 'tags' field]
        - if so, converts them to numpy arrays

        Args:

        Returns the desired filed
        """
        # init array for json fields
        settings_file = self.get_folder_path() + self.SETTINGS_PATH
        # open json file
        with open(settings_file) as d:
            data = json.load(d)
        return data[field]

    def get_folder_path(self):
        """
        gets the local folder
        return is as a string with '/' at the ednd
        """
        return str(os.path.realpath(
            os.path.join(os.getcwd(), os.path.dirname(__file__)))) + '/'

    def listen_to_UI_interaction(self):
        """
        listens to user interaction.

        Steps:
        listen to UI

        Args:

        Returns 4x2 array of points location for key-stoning
        """
        # INTERACTION
        corner_keys = ['1', '2', '3', '4']
        move_keys = ['w', 'a', 's', 'd']
        realsense_keys = ['e', 'r', 'g', 'h']
        bgr_threshold_keys = ['+', '-']

        KEY_STROKE = cv2.waitKey(1)
        key = chr(KEY_STROKE & 255)

        if key == ' ':
            self.magnitude = 10 if self.magnitude == 1 else 1
            print("MAGNITUDE", self.magnitude)

        elif key in bgr_threshold_keys:
            if key == '+':
                self.color_conversion_threshold += self.magnitude
                print("color to greyscale at ", self.color_conversion_threshold)
            elif key == '-':
                self.color_conversion_threshold -= self.magnitude
                print("color to greyscale at ", self.color_conversion_threshold)

        elif key in corner_keys:
            self.selected_corner = key

        elif self.selected_corner is not None and key in move_keys:
            if self.selected_corner == '1':
                if key == 'd':
                    self.init_keystone[0][0] -= self.magnitude
                elif key == 'a':
                    self.init_keystone[0][0] += self.magnitude
                elif key == 'w':
                    self.init_keystone[0][1] += self.magnitude
                elif key == 's':
                    self.init_keystone[0][1] -= self.magnitude

            elif self.selected_corner == '2':
                if key == 'd':
                    self.init_keystone[1][0] -= self.magnitude
                elif key == 'a':
                    self.init_keystone[1][0] += self.magnitude
                elif key == 'w':
                    self.init_keystone[1][1] += self.magnitude
                elif key == 's':
                    self.init_keystone[1][1] -= self.magnitude

            elif self.selected_corner == '3':
                if key == 'd':
                    self.init_keystone[2][0] -= self.magnitude
                elif key == 'a':
                    self.init_keystone[2][0] += self.magnitude
                elif key == 'w':
                    self.init_keystone[2][1] += self.magnitude
                elif key == 's':
                    self.init_keystone[2][1] -= self.magnitude

            elif self.selected_corner == '4':
                if key == 'd':
                    self.init_keystone[3][0] -= self.magnitude
                elif key == 'a':
                    self.init_keystone[3][0] += self.magnitude
                elif key == 'w':
                    self.init_keystone[3][1] += self.magnitude
                elif key == 's':
                    self.init_keystone[3][1] -= self.magnitude

        #  saves to file
        elif key == 'k':
            # reset selected corner
            self.selected_corner = None
            self.save_keystone_to_file(self.init_keystone)

        # realsense exposure control
        if self.using_realsense:
            if key in realsense_keys:
                # increase exposure
                if key == 'e':
                    self.exposure = self.device.get_option(rs.option.exposure)
                    self.device.set_option(rs.option.exposure, self.exposure + int(self.exposure/10))

                # decrease exposure
                elif key == 'r':
                    self.exposure = self.device.get_option(rs.option.exposure)
                    if self.exposure > 1:
                        self.device.set_option(rs.option.exposure, self.exposure - self.exposure/10)

                # increase gain
                elif key == 'g':
                    self.gain = self.device.get_option(rs.option.self.gain)
                    self.device.set_option(rs.option.gain, self.gain + int(self.gain/10))

                # decrease gain
                elif key == 'h':
                    self.gain = self.device.get_option(rs.option.gain)
                    if gain > 1:
                        self.device.set_option(rs.option.gain, self.gain - self.gain/10)

                print("exposure:", self.device.get_option(rs.option.exposure),
                      "gain:", self.device.get_option(rs.option.gain))

            # reset realsense values to standard:
            if key == ' ':
                self.device.set_option(rs.option.gain, 64)
                self.device.set_option(rs.option.exposure, 166)
                print("exposure:", self.device.get_option(rs.option.exposure),
                      "gain:", self.device.get_option(rs.option.gain))

        return self.init_keystone

    def save_keystone_to_file(self, keystone_data_from_user_interaction):
        """
        saves keystone data from user interaction.

        Steps:
        saves an array of points to file
        """
        filePath = self.get_folder_path() + "keystone.txt"
        np.savetxt(filePath, keystone_data_from_user_interaction)
        print("[!] keystone points were saved in", filePath)

    def transform_matrix(self, video_resolution_x, video_resolution_y, keyStonePts):
        '''
        NOTE: Aspect ratio must be flipped
        so that aspectRat[0,1] will be aspectRat[1,0]
        '''
        # inverted screen ratio for np source array
        video_aspect_ratio = (video_resolution_y, video_resolution_x)
        # np source points array
        keystone_origin_points_array = np.float32([
            [0, 0],
            [video_aspect_ratio[1], 0],
            [0, video_aspect_ratio[0]],
            [video_aspect_ratio[1], video_aspect_ratio[0]]
        ])
        # make the 4 pnts matrix perspective transformation
        return cv2.getPerspectiveTransform(
            keyStonePts, keystone_origin_points_array)

    def select_color_by_mean_value(self, mean_color_RGB):
        self.color_conversion_threshold
        '''
        convert color to hsv for oclidian distance
        '''
        bgr_to_grayscale = cv2.cvtColor(mean_color_RGB, cv2.COLOR_BGR2GRAY)
        return 0 if int(bgr_to_grayscale) < self.color_conversion_threshold else 1

    def find_type_in_tags_array(self, cellColorsArray, tagsArray,
                                grid_dimensions_x, grid_dimensions_y):
        """Get the right brick type out of the list of JSON types.

        Steps:
            - get the colors array from the scanners
            - get the JSON lists of type tags, mapping, rotations
            - parse the color data into an NP array of the table shape

        Args:
        Returns an array of found types
        """
        scan_results_array = []
        # create np colors array with table struct
        np_array_of_scanned_colors = np.reshape(
            cellColorsArray, (grid_dimensions_x * grid_dimensions_y, 16))

        # go through the results
        for this_16_bits in np_array_of_scanned_colors:
            result_tag = self.brick_rotation_check(
                this_16_bits, tagsArray)
            # if no results were found
            if result_tag is None:
                result_tag = [-1, -1]
            # add a list of results to the array
            scan_results_array.append(result_tag)

        # finally, return this list to main program for UDP
        return scan_results_array

    def brick_rotation_check(self, this_16_bits, tagsArray):
        tags_array_counter = 0

        for this_tag in tagsArray:
            # if this 16 bits equal the tag as is
            if np.array_equal(this_16_bits, this_tag):
                return [tags_array_counter, 0]
            # convert list of 16 bits to 4x4 matrix for rotation
            brk_4x4 = np.reshape(this_16_bits, (4, 4))
            # rotate once
            brk_4x4_270 = np.reshape(np.rot90(brk_4x4), 16)
            if np.array_equal(brk_4x4_270, this_tag):
                return [tags_array_counter, 1]
            # rotate once
            brk_4x4_180 = np.reshape(np.rot90(np.rot90(brk_4x4)), 16)
            if np.array_equal(brk_4x4_180, this_tag):
                return [tags_array_counter, 2]
            # rotate once
            brk_4x4_90 = np.reshape(np.rot90(np.rot90(np.rot90(brk_4x4))), 16)
            if np.array_equal(brk_4x4_90, this_tag):
                return [tags_array_counter, 3]
            else:
                # if no rotation was found go to next tag in tag list
                tags_array_counter += 1

    def keystone(self):
        # file path to save
        self.KEYSTONE_PATH = self.get_folder_path() + 'keystone.txt'
        print('keystone path:', self.KEYSTONE_PATH)

        # serial num of camera, to switch between cameras
        camPos = self.table_settings['camId']
        self.using_realsense = self.table_settings['realsense']

        # try from a device 1 in list, not default webcam
        if not self.using_realsense:
            WEBCAM = cv2.VideoCapture(camPos)
        else:
            self.realsense_init()

        time.sleep(1)

        # video winodw
        cv2.namedWindow('canvas', cv2.WINDOW_NORMAL)

        # top left, top right, bottom left, bottom right
        self.POINTS = 4 * [(0, 0)]
        self.POINT_INDEX = 0
        self.MOUSE_POSITION = (0, 0)

        def select_four_points():
            # let users select 4 points on WEBCAM GUI
            print("select 4 points, by double clicking on each of them in the order: \n\
            up right, up left, bottom right, bottom left.")

            # loop until 4 clicks
            while self.POINT_INDEX != 4:
                key = cv2.waitKey(20) & 0xFF
                if key == 27:
                    return False
                # wait for clicks
                cv2.setMouseCallback('canvas', save_this_point)
                # read the WEBCAM frames
                if not self.using_realsense:
                    _, self.FRAME = WEBCAM.read()
                else:
                    self.FRAME = self.pipeline.wait_for_frames().get_color_frame()

                if self.using_realsense:
                    self.FRAME = np.asanyarray(self.FRAME.get_data())

                # mirror cam:
                if self.table_settings['mirror_cam']:
                    if not self.using_realsense:
                        self.FRAME = cv2.flip(self.FRAME, 1)
                    # else:
                    #     self.FRAME = np.flip(self.FRAME, 1)

                # draw mouse pos
                cv2.circle(self.FRAME, self.MOUSE_POSITION, 10, (0, 0, 255), 1)
                cv2.circle(self.FRAME, self.MOUSE_POSITION, 1, (0, 0, 255), 2)
                # draw clicked points
                for thisPnt in self.POINTS:
                    cv2.circle(self.FRAME, thisPnt, 10, (255, 0, 0), 1)
                # show the video
                cv2.imshow('canvas', self.FRAME)
            # when done selecting 4 pnts return
            return True

        def save_this_point(event, x, y, flags, param):
            # mouse callback function
            if event == cv2.EVENT_MOUSEMOVE:
                self.MOUSE_POSITION = (x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                # draw a ref. circle
                print('point  # ', self.POINT_INDEX, (x, y))
                # save this point to the array pts
                self.POINTS[self.POINT_INDEX] = (x, y)
                self.POINT_INDEX = self.POINT_INDEX + 1

        # checks if finished selecting the 4 corners
        if select_four_points():
            np.savetxt(self.KEYSTONE_PATH, self.POINTS)
            print("keystone initial points were saved")

        if not self.using_realsense:
            WEBCAM.release()
        else:
            self.pipeline.stop()

        cv2.destroyAllWindows()
