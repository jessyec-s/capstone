
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
from uarmEnv import UarmEnv, convertToCartesianRobot, convertToPolar, convertToCartesian
from uarmController import UarmController
from ddpgHer import DDPG_HER
from gym.wrappers import TimeLimit
import matplotlib.pyplot as plt
from uarmTests import UarmTests

# target information to be set by camera
h_angle = 0.0
v_angle = 0.0
is_centered = False

# constants
height_obj=0.0
port_name = "/dev/cu.usbmodem3172345631381"

# set true by robot when wants information from camera
camera_event = threading.Event()
# set true by camera once data has been set to be read by robot
# set false by robot once data has been read
data_ready = threading.Event()
# set true by camera once it has connected
camera_started = threading.Event()
# set true by ddpg algo for plotting
plot_ready = threading.Event()

# history of data for plotting
success_history = []
distance_history = []
time_history = []


def main(run_tests=False):
    """
    Main "find and touch" task thread that starts UArm thread and executes camera functionality

    Parameters:
        run_tests (bool): True if should run tests
    """
    if run_tests:
        print("Running Tests")
        tests = UarmTests()
        object_locations = tests.run_tests()
        print("Object locations polar: ", object_locations)
        find_and_touch_test_loop(object_locations)
    else:
        # set mutex flags to false
        camera_event.clear()
        data_ready.clear()
        camera_started.clear()

        # create and call thread for UArm to execute find and touch task
        ddpg_thread = threading.Thread(target=execute_find_and_touch_task)
        ddpg_thread.start()

        # use main thread for camera execution
        camera_exec()

        ddpg_thread.join()
        print("uarm done searching")

def uarm_seek(uarm_controller):
    """
    UArm seeks for target object until it is found and its coordinates have been calculated

    Parameters:
        uarm_controller (UarmController): controls the UArm's movement and calculations related to its position

    Returns:
        [x, y, z] - numpy array of calculated target object cartesian coordinates
    """

    global is_centered

    '''
    seek for object while camera has not found the target object or the object
    is not centered in the camera's frame
    '''
    while data_ready.is_set() is False or is_centered is False:

        # if camera has found object but it is not centered in frame
        if data_ready.is_set():
            data_ready.clear()

        # prevent camera from identifying object while the UArm is moving
        camera_event.clear()

        # move UArm
        uarm_controller.seek()

        # Notify camera that it should try and identify object
        camera_event.set()
        time.sleep(1)

    # target object is found so reset all flags to false
    is_centered = False
    camera_event.clear()
    data_ready.clear()

    # calculate and then return coordinates of target object
    return uarm_controller.calc_object_cords(h_angle, v_angle)


def plot_distance(distance_history, plot_num):
    """
    Plot the distance of the UArm to the target object over time

    Parameters:
        distance_history (list): UArm distance to object per epoch
        plot_num (int): plot number to identify plot
    """

    plt.plot(distance_history)
    plt.title("Arm Distance from Object")
    plt.ylabel("Distance")
    plt.xlabel("Number iterations")
    plt.savefig("./plots/distance/distance_history_{}.png".format(plot_num))
    plt.clf()


def plot_success(success_rate, plot_num):
    """
    Plot the rolling average success rate and epoch success rate for completing the
    find and touch task.

    Parameters:
        success_rate (list): successes (either 0 or 1) of find and touch task
        plot_num (int): plot number to identify plot
    """

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
    """
    Plot the rolling average for the time to complete the find and touch task as well as the
    completion time for each individual task.

    Parameters:
        time_to_complete (list): time taken to complete the task each epoch
        plot_num (int): plot number to identify plot
    """
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


def find_and_touch_test_loop(obj_locations):
    """
    Test loop for find and touch task.  UArm tries to find and touch the locations specified.

    Parameters:
        obj_locations (list): target object locations (in polar coordinates) used to perform find and touch task.
    """

    # Create the uarm controller that uses UArm swift api functionality to physically control the robot
    uarm_controller = UarmController()

    # Set up the UArm physical environment to be used by the DDPG algorithm
    uarm_env = UarmEnv(uarm_controller)
    uarm_env = TimeLimit(uarm_env, max_episode_steps=50)

    # wait for uarm to connect
    uarm_controller.waiting_ready()

    # reset the UArm to its default position
    uarm_controller.UArm_reset(should_wait=True)

    # instantiate DDPG_HER class
    ddpg_her = DDPG_HER(env=uarm_env)

    for count, obj in enumerate(obj_locations):
        # reset the UArm to its default position
        uarm_controller.reset()

        # convert given polar coordinates to cartesian
        cartesian_coords = convertToCartesianRobot(obj[0], obj[1], obj[2])

        # set target object location in UArm environment class to be used by DDPG alogorithm
        uarm_env.set_object_pos(cartesian_coords)

        # run DDPG algorithm to touch object -- should exit when object is found
        suc_history, dis_history, time_history = ddpg_her.run(train=False)

        # plot information from completed epoch
        plot_success(suc_history, count)
        plot_distance(dis_history, count)
        plot_time(time_history, count)
        time.sleep(2)


def execute_find_and_touch_task():

    '''
    UArm main loop for the find and touch task.  UArm seeks for object until in view and then tries to
    touch it.
    '''

    # Create the uarm controller that uses UArm swift api functionality to physically control the robot
    uarm_controller = UarmController()

    # Set up the UArm physical environment to be used by the DDPG algorithm
    uarm_env = UarmEnv(uarm_controller)
    uarm_env = TimeLimit(uarm_env, max_episode_steps=50)

    # wait for uarm to connect
    uarm_controller.waiting_ready()

    # wait for camera to boot
    camera_started.wait()
    time.sleep(2)

    # reset the UArm to its default position
    uarm_controller.UArm_reset(should_wait=True)

    # instantiate DDPG_HER class
    ddpg_her = DDPG_HER(env=uarm_env)

    # Execute the find and touch task an infinite amount of times
    while True:

        # reset the UArm to its default position
        uarm_controller.reset()

        # UArm seeks until the target object is in view and its coordinates are determined
        obj_pos = uarm_seek(uarm_controller)

        # calculated target object location is set in the UArm environment class
        uarm_env.set_object_pos(obj_pos)

        # run DDPG algorithm to touch object
        global distance_history, success_history, time_history
        success_history, distance_history, time_history = ddpg_her.run(train=False)

        # Send signal to plot success_rate
        plot_ready.set()

        # pause so object can be moved to a new location manually
        time.sleep(5)


def camera_connect():
    """
    Camera connection loop.
    """

    connected = False
    pyopenmv.disconnect()
    # try and connect
    for i in range(10):

        try:
            # opens CDC port.
            # Set small timeout when connecting
            pyopenmv.init(port_name, baudrate=921600, timeout=0.050)
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
    """
    Main camera execution loop that runs until program is aborted.
    1. Camera tries to connect continuously until successful
    2. Camera outputs frame buffer -> video stream
    3. Camera outputs text buffer -> data about target object
        - Data is passed to robot for evaluation
    """
    pygame.init()
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

        # signal to UArm that camera has connected
        camera_started.set()
        if fb is not None:
            # create image from RGB888
            image = pygame.image.frombuffer(fb[2].flat[0:], (fb[0], fb[1]), 'RGB')
            screen = pygame.display.set_mode((fb[0], fb[1]), pygame.DOUBLEBUF, 32)

            fps = Clock.get_fps()
            # blit stuff
            screen.blit(image, (0, 0))
            screen.blit(font.render("FPS %.2f"%(fps), 1, (255, 0, 0)), (0, 0))

            # update display
            pygame.display.flip()

        # get output from text buffer
        tx_len = pyopenmv.tx_buf_len()

        # object was found by camera if there is outputted text
        if tx_len:

            '''
            if UArm has signaled to the camera to identify the object and the camera has not already
            assigned values to the global variables associated with the object's location
            '''
            if camera_event.is_set() and (data_ready.is_set() is False):

                # read the most recent data at index 0 from the text buffer
                buff = pyopenmv.tx_buf(tx_len).decode()
                split_buff = str(buff).splitlines()
                if h_angle_key in split_buff[0]:

                    # Most recent line in buff contains needed information
                    global h_angle, v_angle, is_centered
                    tok = split_buff[0].split()

                    # set angles to corresponding values determined by camera
                    h_angle, v_angle = float(tok[1]), float(tok[3])
                    if tok[5] == "True":
                        is_centered = True
                    else:
                        is_centered = False
                    # signal that global variables have been set
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
