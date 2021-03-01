
# import pygame
# import serial
# import openmv.tools.rpc.rpc as rpc
# import io, serial.tools.list_ports,socket,sys
#!/usr/bin/env python2
# This file is part of the OpenMV project.
#
# Copyright (c) 2013-2019 Ibrahim Abdelkader <iabdalkader@openmv.io>
# Copyright (c) 2013-2019 Kwabena W. Agyeman <kwagyeman@openmv.io>
#
# This work is licensed under the MIT license, see the file LICENSE for details.
#
# An example script using pyopenmv to grab the framebuffer.
import sys
import time
import pygame
import pyopenmv
import math
import threading
from time import sleep
from getobjectblob import blob_script
from getobjectblob import h_angle_key
from uarmAPI import UarmEnv

# target information to be set by camera
h_angle = 0.0
v_angle = 0.0

# constants
height_obj=20.0

# set true by robot when wants information from camera
camera_event = threading.Event()
# set true by camera once data has been set to be read by robot
# set false by robot once data has been read
data_ready = threading.Event()
# set true by camera once it has connected
camera_started = threading.Event()

def main() :
    '''
    Main thread:
        - starts UArm thread
        - executes camera functionality
    '''
    camera_event.clear()
    data_ready.clear()
    camera_started.clear()
    uarm_thread = threading.Thread(target=uarm_exec)
    uarm_thread.start()
    camera_exec()
    uarm_thread.join()
    print("uarm done searching")


def uarm_exec():
    '''
    Main UArm thread
    Execution Order:
        1. Robot waits for camera to connect
        2. Robot seeks space for target object
        3. Once camera identifies target object location of object is determined
        4. Return - this step should instead trigger DDPG algorithm
    '''
    cords = []
    uarm_controller = UarmEnv()
    uarm_controller.waiting_ready() # wait for uarm to connect
    camera_started.wait() # wait for camera to boot
    time.sleep(2)
    while data_ready.is_set() is False:
        # camera has not found object
        camera_event.clear()
        print("about to move")
        uarm_controller.seek()
        print("Setting camera event")
        camera_event.set()
        time.sleep(0.5)
    camera_event.clear()
    uarm_controller.calc_object_cords(h_angle, v_angle)
    data_ready.clear()

    # TODO: trigger DDPG algorithm
    return cords


def camera_connect():
    '''
    Camera connection loop
    '''

    portname = "/dev/cu.usbmodem3172345631381"

    connected = False

    pyopenmv.disconnect()
    # try and connect
    for i in range(10):

        try:
            # opens CDC port.
            # Set small timeout when connecting
            pyopenmv.init(portname, baudrate=921600, timeout=0.050)
            connected = True
            break
        except Exception as e:
            connected = False
            sleep(0.100)
    if not connected:
        print("Failed to connect to OpenMV's serial port.\n"
              "Please install OpenMV's udev rules first:\n"
              "sudo cp openmv/udev/50-openmv.rules /etc/udev/rules.d/\n"
              "sudo udevadm control --reload-rules\n\n")
        sys.exit(1)
    # Set higher timeout after connecting for lengthy transfers.
    pyopenmv.set_timeout(1 * 2)  # SD Cards can cause big hicups.
    pyopenmv.stop_script()
    pyopenmv.enable_fb(True)
    pyopenmv.exec_script(blob_script)

    # init screen
    running = True
    Clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 15)

    return running, Clock, font

def camera_exec():
    '''
    Main camera execution
    1. Camera tries to connect continuously until successful
    2. Camera outputs frame buffer -> video stream
    3. Camera outputs text buffer -> data about target object
        - Data is passed to robot for evalution
    '''
    pygame.init()
    # if len(sys.argv)!= 2:
    #     print ('usage: pyopenmv_fb.py <serial port>')
    #     sys.exit(1)
    # portname = sys.argv[1]la
    locals()

    running, Clock, font = camera_connect()
    while running:
        Clock.tick(100)

        # read framebuffer
        fb = None
        while (True) :
            try:
                fb = pyopenmv.fb_dump()
                break
            except Exception as e:
                # try and reconnect on failure
                camera_connect()
        camera_started.set()
        if fb != None:
            # create image from RGB888
            image = pygame.image.frombuffer(fb[2].flat[0:], (fb[0], fb[1]), 'RGB')
            # TODO check if res changed
            screen = pygame.display.set_mode((fb[0], fb[1]), pygame.DOUBLEBUF, 32)

            fps = Clock.get_fps()
            # blit stuff
            screen.blit(image, (0, 0))
            screen.blit(font.render("FPS %.2f"%(fps), 1, (255, 0, 0)), (0, 0))

            # update display
            pygame.display.flip()
        tx_len = pyopenmv.tx_buf_len()
        # sleep(0.250)
        if tx_len:
            # object was found by camera
            if camera_event.is_set() and (data_ready.is_set() is False):
                # robot wants information and global values have not been updated yet
                buff = pyopenmv.tx_buf(tx_len).decode()
                split_buff = str(buff).splitlines()
                if h_angle_key in split_buff[0]:
                    # Most recent line in buff contains needed information
                    global h_angle, v_angle
                    tok = split_buff[0].split()
                    print("tok: ", tok)
                    # set angles to corresponding values determined by camera
                    h_angle, v_angle = float(tok[1]), float(tok[3])
                    # signal that global variables have been set
                    data_ready.set()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                 running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_c:
                    pygame.image.save(image, "capture.png")

    pygame.quit()
    pyopenmv.stop_script()

main()
