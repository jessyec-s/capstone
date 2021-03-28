
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
from uarmEnv import UarmEnv, convertToCartesianRobot, convertToPolar
from uarmController import UarmController
from ddpgHer import DDPG_HER
from gym.wrappers import TimeLimit
import matplotlib.pyplot as plt
from uarmTests import UarmTests

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
# set true by ddpg algo for plotting
plot_ready = threading.Event()
success_history = []
distance_history = []
time_history = []

def main(run_tests=True) :
    '''
    Main thread:
        - starts UArm thread
        - executes camera functionality
    '''
    if run_tests:
        print("Running Tests")
        tests = UarmTests()
        object_locations = tests.run_tests()
        print("Object locations polar: ", object_locations)
        ddpg_loop_no_camera(object_locations)
    else:
        camera_event.clear()
        data_ready.clear()
        camera_started.clear()
        ddpg_thread = threading.Thread(target=ddpg_loop_with_seek)
        ddpg_thread.start()
        camera_exec()
        ddpg_thread.join()
        print("uarm done searching")

def uarm_seek(uarm_controller):
    '''
       Uarm object seek
           1. Robot waits for camera to connect
           2. Robot seeks space for target object
           3. Once camera identifies target object location of object is determined
    '''
    while data_ready.is_set() is False:
        # camera has not found object
        camera_event.clear()
        print("about to move")
        uarm_controller.seek()
        print("Setting camera event")
        camera_event.set()
        time.sleep(0.5)
    camera_event.clear()
    data_ready.clear()
    return uarm_controller.calc_object_cords(h_angle, v_angle)

def plot_distance(distance_history, plot_num):
    plt.plot(distance_history)
    plt.title("Arm Distance from Object")
    plt.ylabel("Distance")
    plt.xlabel("Number iterations")
    plt.savefig("./plots/distance/distance_history_{}.png".format(plot_num))
    plt.clf()

def plot_success(success_rate, plot_num):
    average = []
    for i, point in enumerate(success_rate):
        average.append(success_rate[:i+1].count(True) / (i+1))
    plt.plot(success_rate, color= 'blue', label="Epoch Success Rate")
    plt.plot(average, color = 'red', label= "Average Success Rate", zorder = 3)
    plt.legend()
    plt.title("Success Rate for FetchReach")
    plt.ylabel("Success Rate")
    plt.xlabel("Number iterations")
    plt.savefig("./plots/success/success_rate_{}.png".format(plot_num))
    plt.clf()

def plot_time(time_to_complete, plot_num):
    average = []
    for i, point in enumerate(time_to_complete):
        average.append(sum(time_to_complete[:i+1])/ (i+1))
    plt.plot(time_to_complete, color= 'blue', label="Epoch Time")
    plt.plot(average, color = 'red', label= "Average Time", zorder = 3)
    plt.legend()
    plt.title("Time to complete FetchReach")
    plt.ylabel("Time (seconds)")
    plt.xlabel("Number iterations")
    plt.savefig("./plots/time/time_to_complete_{}.png".format(plot_num))
    plt.clf()


def ddpg_loop_no_camera(obj_locations):
    print("inside ddpg_loop_no_camera")
    uarm_controller = UarmController()
    uarm_env = UarmEnv(uarm_controller)
    uarm_env = TimeLimit(uarm_env, max_episode_steps=50)
    uarm_controller.waiting_ready()  # wait for uarm to connect
    uarm_controller.UArm_reset(should_wait=True)

    # instantiate DDPG_HER class
    ddpg_her = DDPG_HER(env=uarm_env)

    for count, obj in enumerate(obj_locations):
        print("AT TOP OF WHILE LOOP")
        uarm_controller.reset()
        cartesian_coords = convertToCartesianRobot(obj[0], obj[1], obj[2])
        print("obj_coords polar: ", obj)
        print("obj_coords cartesian: ", cartesian_coords)
        uarm_env.set_object_pos(cartesian_coords)
        # call ddpg -- should exit when object is found
        suc_history, dis_history, time_history = ddpg_her.run(train=False)
        print("Finished ddpg_her")
        # plot success_rate
        plot_success(suc_history, count)
        plot_distance(dis_history, count)
        plot_time(time_history, count)
        time.sleep(2)


def ddpg_loop_with_seek():
    uarm_controller = UarmController()
    uarm_env = UarmEnv(uarm_controller)
    uarm_env = TimeLimit(uarm_env, max_episode_steps=50)
    uarm_controller.waiting_ready() # wait for uarm to connect
    camera_started.wait() # wait for camera to boot
    time.sleep(2)
    uarm_controller.UArm_reset(should_wait=True)

    #instantiate DDPG_HER class
    ddpg_her = DDPG_HER(env=uarm_env)

    while True:
        print("AT TOP OF WHILE LOOP")
        uarm_controller.reset()
        uarm_env.set_object_pos(uarm_seek(uarm_controller))
        # call ddpg -- should exit when object is found
        print("Found block")
        global distance_history, success_history, time_history
        success_history, distance_history, time_history = ddpg_her.run(train=False)
        print("Finished ddpg_her")
        # Send signal to plot success_rate
        plot_ready.set()

        time.sleep(5)

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

    plot_num = 0
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
                    # print("tok: ", tok)
                    # set angles to corresponding values determined by camera
                    h_angle, v_angle = float(tok[1]), float(tok[3])
                    # signal that global variables have been set
                    print("Setting data_ready")
                    data_ready.set()

        if plot_ready.is_set():
            print("success_rate: ", success_history)
            plot_distance(distance_history, plot_num)
            plot_success(success_history, plot_num)
            plot_num += 1
            plot_ready.clear()
            print("success rate for ", len(success_history), " tests: ",
                  success_history.count(True) / len(success_history))

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
