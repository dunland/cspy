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

import math
import decimal
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

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


class Cityscopy:
    def __init__(self, path):
        # load info from json file
        with open(path) as settings:
            self.table_settings = json.load(settings)

        # communication
        self.UDP_IP = "127.0.0.1"
        self.UDP_PORT = 5000

        # init corners variables
        self.selected_corner = None
        self.magnitude = 1

        # realsense camera parameters
        self.exposure = self.table_settings['realsense']['exposure']
        self.gain = self.table_settings['realsense']['gain']
        self.using_realsense = self.table_settings['realsense']['active']

        # setup camera
        if self.using_realsense:
            self.realsense_init()

        # tags
        self.tag_length = self.table_settings.get('tag_length', 4)
        self.width = int(math.sqrt(self.tag_length))
        self.tags = self.table_settings['tags']
        self.tags_np = np.int8([[int(b) for b in tag] for tag in self.tags])

        for tag in self.tags:
            assert len(tag) == self.tag_length

        # color conversion thresholds
        self.max_l = self.table_settings.get('max_l', 127)
        self.max_a = self.table_settings.get('max_a', 255)
        self.max_b = self.table_settings.get('max_b', 255)
        self.slider_l = self.table_settings.get('slider_l', 127)
        self.slider_a = self.table_settings.get('slider_a', 255)
        self.slider_b = self.table_settings.get('slider_b', 255)
        self.quantile = self.table_settings.get('quantile', 0.5)

        self.sliders = [
            Slider(options) for options in self.table_settings.get('sliders', [])
        ]

        # init keystone variables
        self.FRAME = None
        self.POINT_INDEX = None
        self.POINTS = None
        self.MOUSE_POSITION = None

    def realsense_init(self):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()
        realsense_ctx = rs.context()
        connected_devices = []

        for i in range(len(realsense_ctx.devices)):
            detected_camera = realsense_ctx.devices[i].get_info(
                rs.camera_info.serial_number)
            connected_devices.append(detected_camera)

        # choose device if more than 1 connected:
        if len(connected_devices) > 1:
            print("choose device by pressing the number:")
            for i in range(len(realsense_ctx.devices)):
                print("[%s]: %s @ %s" % (i, realsense_ctx.devices[i].get_info(rs.camera_info.name), realsense_ctx.devices[i].get_info(rs.camera_info.physical_port)))
            idx = int(input(">>> "))
            device_product_line = connected_devices[idx]
            self.UDP_PORT = 5000 + idx
            print("sending at UDP %s:%s" % (self.UDP_IP, self.UDP_PORT))
        else:
            device_product_line = connected_devices[0]
            self.UDP_PORT = 5000
        config.enable_device(device_product_line)

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)

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

        # set sensitivity parameters:
        self.device = pipeline_profile.get_device().first_color_sensor()
        self.device.set_option(rs.option.exposure, self.exposure)
        self.device.set_option(rs.option.gain, self.gain)

        print("success!")
        print("Realsense initialization complete.")

    def scan(self):
        # define global list manager
        MANAGER = Manager()
        # create shared global list to work with both processes
        self.mp_shared_dict = MANAGER.dict()

        # init a dict to be shared among procceses
        self.mp_shared_dict['scan'] = None
        self.mp_shared_dict['sliders'] = None

        # defines a multiprocess for sending the data
        self.process_send_packet = Process(target=self.create_data_json,
                                           args=([self.mp_shared_dict]))

        self.process_send_packet.start()

        # start camera on main thread due to multiprocces issue
        self.scanner_function(self.mp_shared_dict)
        # join the two processes
        self.process_send_packet.join()

    def scanner_function(self, mp_shared_dict):
        # get init keystones
        self.init_keystone = np.loadtxt(
            self.get_folder_path() + 'keystone.txt', dtype=np.float32)

        # define the table params
        grid_dim = (int(self.table_settings['ncols']),
                    int(self.table_settings['nrows']))

        # serial num of camera, to switch between cameras
        camPos = self.table_settings['cam_id']

        if not self.using_realsense:
            video_capture = cv2.VideoCapture(camPos)

        if self.using_realsense:
            video_res = (int(self.pipeline.wait_for_frames().get_color_frame().get_width()),
                         int(self.pipeline.wait_for_frames().get_color_frame().get_height()))
        else:
            video_res = (int(video_capture.get(3)), int(video_capture.get(4)))

        # define the video window
        cv2.namedWindow('scanner_gui_window', cv2.WINDOW_NORMAL)

        # define the size for each scanner
        block_size = (video_res[0] / grid_dim[0], video_res[1] / grid_dim[1])
        codepoint_size = (block_size[0] / self.width, block_size[1] / self.width)

        # get coordinates for scanners (top left corner of each area)
        scanner_points = []
        for y in range(grid_dim[1]):
            for x in range(grid_dim[0]):
                scanner_points.extend([
                    (int(x * block_size[0] + i * codepoint_size[0]),
                     int(y * block_size[1] + j * codepoint_size[1]))
                    for i in range(self.width) for j in range(self.width)
                ])

        previous_colors = []

        # run the video loop forever
        while True:
            if self.using_realsense:
                frames = self.pipeline.wait_for_frames()  # returns composite_frame
                color_frame = frames.get_color_frame()  # returns video_frame
                color_frame = np.asanyarray(color_frame.get_data())
            else:
                # read video frames
                _, color_frame = video_capture.read()

            current_colors = []

            # get a new matrix transformation every frame
            keystone_data = self.transform_matrix(video_res, self.listen_to_UI_interaction())

            # mirror camera (webcam)
            if self.table_settings['mirror_cam']:
                if self.using_realsense:
                    color_frame = np.flip(color_frame, 1)
                else:
                    color_frame = cv2.flip(color_frame, 1)

            # rotate image
            if self.table_settings['rotate_image']:
                if self.using_realsense:
                    color_frame = np.rot90(color_frame, 2)
                else:
                    color_frame = cv2.rotate(color_frame, rotateCode=cv2.ROTATE_180)

            # warp the video based on keystone info
            keystoned_video = cv2.warpPerspective(color_frame, keystone_data, video_res)

            # convert input to LAB colour space
            lab_image = cv2.cvtColor(keystoned_video, cv2.COLOR_BGR2LAB)

            # uncomment this to show intermediate image
            # cv2.imshow("lab_image", lab_image)

            # get L/a/b channels
            ch_l, ch_a, ch_b = cv2.split(lab_image)

            # reduce the colors based on slider threshold
            binary_image_slider = np.where(
                (ch_l <= self.slider_l) & (ch_a <= self.slider_a) & (ch_b <= self.slider_b), 255, 0
                ).astype(np.uint8)

            # uncomment this to show intermediate image
            # cv2.imshow("binary_image_slider", binary_image_slider)

            # get slider values
            mp_shared_dict['sliders'] = {
                slider.id: slider.evaluate(binary_image_slider, video_res, block_size)
                for slider in self.sliders
            }

            # reduce the colors based on a threshold
            binary_image = np.where(
                (ch_l <= self.max_l) & (ch_a <= self.max_a) & (ch_b <= self.max_b), 255, 0
                ).astype(np.uint8)

            # uncomment this to show intermediate image
            # cv2.imshow("binary_image", binary_image)

            # run through coordinates and analyse each image
            for x, y in scanner_points:
                # get image slice for scanning
                scan_pixels = binary_image[y:int(y + codepoint_size[1]),
                                           x:int(x + codepoint_size[0])]

                # determine color based on the distribution of B/W values in the image
                current_colors.append(0 if np.quantile(scan_pixels, self.quantile) == 0 else 1)

            # reduce unnecessary scan analysis and sending by comparing
            # the list of scanned cells to the previous one
            if current_colors != previous_colors:
                # identify tags and and store result in mp_shared_dict
                mp_shared_dict['scan'] = [
                    self.brick_rotation_check(block) or [-1, -1]
                    for block in np.reshape(
                        current_colors, (grid_dim[0] * grid_dim[1], self.tag_length))
                ]
                previous_colors = current_colors

            if self.table_settings['gui']:
                # visualize grid
                for x in range(grid_dim[0]):
                    for y in range(grid_dim[1]):
                        cv2.rectangle(
                            keystoned_video,
                            (int(x * block_size[0]), int(y * block_size[1])),
                            (int((x + 1) * block_size[0]), int((y + 1) * block_size[1])),
                            WHITE, 1)

                # draw dots with detected color
                for (x, y), value in zip(scanner_points, current_colors):
                    center = (int(x + codepoint_size[0] / 2), int(y + codepoint_size[1] / 2))
                    cv2.circle(keystoned_video, center, 2, BLACK if value else WHITE, -1)

                # draw sliders
                for slider in self.sliders:
                    slider.draw(keystoned_video)

                # draw arrow to interaction area
                self.ui_selected_corner(video_res[0], video_res[1], keystoned_video)
                cv2.putText(keystoned_video, "magnitude: " + str(self.magnitude) + " [SPACE]",
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 1, cv2.LINE_AA)
                cv2.putText(keystoned_video, "exposure: " + str(self.exposure) + " [e/r]",
                            (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 1, cv2.LINE_AA)
                cv2.putText(keystoned_video, "gain: " + str(self.gain) + " [g/h]",
                            (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 1, cv2.LINE_AA)
                cv2.putText(keystoned_video, "max_l: " + str(self.max_l) + " [+/-]",
                            (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 1, cv2.LINE_AA)

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
            col = (0, 0, 255) if self.magnitude == 1 else (255, 0, 0)
            cv2.arrowedLine(
                vid, case[self.selected_corner][0],
                case[self.selected_corner][1],
                col, 2)

    def create_data_json(self, mp_shared_dict):
        SEND_INTERVAL = self.table_settings['interval']
        # initial dummy value for old grid
        old_scan_results = [-1]
        SEND_INTERVAL = timedelta(milliseconds=SEND_INTERVAL)
        last_sent = datetime.now()

        while True:
            scan_results = mp_shared_dict['scan']
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
                print('CityScopy grid sent at:', datetime.now(), "via %s:%s" % (self.UDP_IP, self.UDP_PORT))

    def send_json_to_UDP(self, scan_results):
        slider_val = self.mp_shared_dict['sliders']
        json_dict = {'grid': scan_results, 'sliders': slider_val}
        json_string = json.dumps(json_dict)

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.sendto(json_string.encode('utf-8'), (self.UDP_IP, self.UDP_PORT))
        except Exception as e:
            print(e)

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
        slider_threshold_keys = ['<','>']
        key = chr(cv2.waitKey(1) & 255)

        if key == ' ':
            self.magnitude = 10 if self.magnitude == 1 else 1
            print("MAGNITUDE", self.magnitude)

        elif key in bgr_threshold_keys:
            if key == '+':
                self.max_l += self.magnitude
                print("luminance threshold at ", self.max_l)
            elif key == '-':
                self.max_l -= self.magnitude
                print("luminance threshold at ", self.max_l)

        elif key in slider_threshold_keys:
            if key == '<':
                self.slider_l += self.magnitude
                print("slider luminance threshold at ", self.slider_l)
            elif key == '>':
                self.slider_l -= self.magnitude
                print("slider luminance threshold at ", self.slider_l)

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

        # save to file
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
                    self.device.set_option(rs.option.exposure, self.exposure + self.magnitude)

                # decrease exposure
                elif key == 'r':
                    self.exposure = self.device.get_option(rs.option.exposure)
                    if self.exposure > 1:
                        self.device.set_option(rs.option.exposure, self.exposure - self.magnitude)

                # increase gain
                elif key == 'g':
                    self.gain = self.device.get_option(rs.option.gain)
                    self.device.set_option(rs.option.gain, self.gain + self.magnitude)

                # decrease gain
                elif key == 'h':
                    self.gain = self.device.get_option(rs.option.gain)
                    if self.gain > 1:
                        self.device.set_option(rs.option.gain, self.gain - self.magnitude)

                print("exposure:", self.device.get_option(rs.option.exposure),
                      "gain:", self.device.get_option(rs.option.gain))

            elif key == 'u':
                self.table_settings['gui'] = not self.table_settings['gui']

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

    def transform_matrix(self, video_res, keyStonePts):
        '''
        NOTE: Aspect ratio must be flipped
        so that aspectRat[0,1] will be aspectRat[1,0]
        '''
        # np source points array
        keystone_origin_points_array = np.float32([
            [0, 0],
            [video_res[0], 0],
            [0, video_res[1]],
            video_res
        ])
        # make the 4 pnts matrix perspective transformation
        return cv2.getPerspectiveTransform(keyStonePts, keystone_origin_points_array)

    def brick_rotation_check(self, block):
        # convert block to square representation for rotation checks
        block = np.reshape(block, (self.width, self.width))

        for tag_count, tag in enumerate(self.tags_np):
            # test all four rotations
            for i in range(4):
                if np.array_equal(np.reshape(block, self.tag_length), tag):
                    return [tag_count, i]
                block = np.rot90(block)

    def keystone(self):
        # file path to save
        self.KEYSTONE_PATH = self.get_folder_path() + 'keystone.txt'
        print('keystone path:', self.KEYSTONE_PATH)

        # serial num of camera, to switch between cameras
        camPos = self.table_settings['cam_id']
        self.using_realsense = self.table_settings['realsense']['active']

        # try from a device 1 in list, not default webcam
        if not self.using_realsense:
            WEBCAM = cv2.VideoCapture(camPos)

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


class Slider:
    def __init__(self, config):
        '''Set up a slider instance'''
        self.id = config['id']
        self.step_size = decimal.Decimal(str(config['step_size']))
        self.y = config['y']          # y location (center)
        self.x_min = config['x_min']  # x location of minimum slider position (centroid)
        self.x_max = config['x_max']  # x location of maximum slider position (centroid)

    def evaluate(self, frame, video_res, block_size):
        '''Extract slider value from the original image.

        The slider tag should be as large as block_size.'''
        self.y0 = int(max(self.y - block_size[1] / 2, 0))
        self.y1 = int(min(self.y + block_size[1] / 2, video_res[1] - 1))
        self.x0 = int(max(self.x_min - block_size[0] / 2, 0))
        self.x1 = int(min(self.x_max + block_size[0] / 2, video_res[0] - 1))

        slider_row = frame[self.y0:self.y1, self.x0:self.x1]

        self.slider_coord = self.get_slider_coord(slider_row)

        if self.slider_coord:
            slider_x_max = self.x1 - self.x0 - block_size[0]
            slider_value = min(max(
                (self.slider_coord[0] - block_size[0] / 2) / slider_x_max, 0), 1)
            # round according to step_size
            return float(
                decimal.Decimal(slider_value).quantize(self.step_size, decimal.ROUND_HALF_UP))

    def draw(self, frame):
        '''Draw slider range and current location onto image'''
        cv2.line(frame, (self.x_min, self.y), (self.x_max, self.y), WHITE, 2)

        if self.slider_coord:
            cv2.line(frame,
                     (self.x0 + self.slider_coord[0], self.y0 + self.slider_coord[1] - 20),
                     (self.x0 + self.slider_coord[0], self.y0 + self.slider_coord[1] + 20),
                     WHITE, 8)

    def get_slider_coord(self, frame):
        '''Get x,y of slider position in frame. Any black blob is considered a slider'''
        # find contours
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # get centroid of first contour
        centroid = None
        if len(contours) > 0:
            moments = cv2.moments(contours[0])
            if moments['m00'] > 0:
                centroid = (int(moments['m10'] / moments['m00']),
                            int(moments['m01'] / moments['m00']))

        return centroid
