
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
import threading
from time import sleep
from getobjectblob import blob_script
from getobjectblob import distance_key
from getobjectblob import h_angle_key
from getobjectblob import v_angle_key
from uarmAPI import UarmEnv

distance = ""
h_angle = 0
v_angle = 0
camera_event = threading.Event()
data_ready = threading.Event()
camera_started = threading.Event()

lock = threading.Lock()

def main() :
    camera_event.clear()
    data_ready.clear()
    camera_started.clear()
    uarm_thread = threading.Thread(target=uarm_exec)
    uarm_thread.start()
    camera_exec()
    uarm_thread.join()
    print("uarm done searching")

def uarm_exec():
    cords = []
    uarm_controller = UarmEnv()
    uarm_controller.waiting_ready() # wait for uarm to connect
    camera_started.wait() # wait for camera to boot
    time.sleep(2)
    camera_event.set()  # want data from camera
    time.sleep(1)
    if data_ready.is_set():
        camera_event.clear()
        distance_ = distance
        print("Will send over cord to DDPG: ", distance_)
        data_ready.clear()
        return cords
    while data_ready.is_set() is False: #camera has not found object
        # search observation space
        camera_event.clear()
        print("about to move")
        uarm_controller.ENV_reset(should_wait=True)
        uarm_controller.flush_cmd(wait_stop=True)
        print("Setting camera event")
        camera_event.set()
        time.sleep(0.5)
    camera_event.clear()
    distance_ = distance
    print("Arm received distance: ", distance_)
    data_ready.clear()
    return cords

def camera_connect():
    portname = "/dev/cu.usbmodem3172345631381"

    connected = False

    pyopenmv.disconnect()
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
                buff = pyopenmv.tx_buf(tx_len).decode()
                split_buff = str(buff).splitlines()
                if distance_key in split_buff[0]:
                    global distance
                    distance = split_buff[0]
                    print("Camera sending distance: ", distance)
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