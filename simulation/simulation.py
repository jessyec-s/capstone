
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
import numpy as np
import gym
from gym.envs.robotics.fetch.reach import FetchReachEnv
from gym.wrappers import TimeLimit
import time
#import pygame
#import pyopenmv
import math
#import threading
from time import sleep
#from getobjectblob import blob_script
#from getobjectblob import h_angle_key
from ddpgHer import DDPG_HER
import matplotlib.pyplot as mpl

def main() :
    '''
    Main thread:
    '''
    train=True
    environment=FetchReachEnv()
    param_file="./her_bit_env"
    ddpg_loop(train=train,environment=environment)

def ddpg_loop(train=None, environment=None, param_file="./her_bit_env"):
    #time.sleep(2)
    environment = TimeLimit(environment, max_episode_steps=50)
    #instantiate DDPG_HER class
    ddpg_her = DDPG_HER(env=environment,name=param_file)
    local_success=[]
        
    
    print("AT TOP OF WHILE LOOP")
    # call ddpg -- should exit when object is found
    print("Found block")
    ddpg_her.run(train=train)
    print("Finished ddpg_her")


    

main()
