# Simulation to Robot

Extra background for converting between simulated and physical coordinates.

## Simulation 

The basic idea: 
1. Robots range of motion is limited to a 150mm to 300mm radius
2. simulation and physical system each have a cylindrical state space that can be mapped 1 to 1 between systems. 
3. Simulated origin is at left corner, while physical origin is at robots base
4. When recieving an action from the trained parameters must convert into phsyical coordinates
5. When inputting state space of robot to the parameters, must convert into simulated coordinates


### Conversions
The [branch](https://github.com/kobinau/gym/tree/capstone) has the following features:
* Updated FetchReach Environment
* get_observation and set_observation functions for reading and changing the position using simulated coordinates
* get_obs_wm and set_obs_wm functions as getters and setters using the robots coordinates

From the simulated system we first add an offset to set the simulated origin at the base of the robot 
x=x-.8
y=y-.75

1 to 1 mapping between simulated and physical system:
simulation range (after offset):
radius=.2 to .72
theta=-90 to 90
z = .32 to 1.0

Physical range:
radius=150 to 300
theta=0 to 180
z = 0 to 150


### object location

Object is located via camera. Location of the arm is found relative to the end effector. One must add the distance vector between the camera to the object, the camera to the end effector and the position of the end effector.

#### camera object identification:
* object is found in downward facing camera. Given the cameras horizontal and vertical fields of view (FOV), and the centroid of the object, the horizontal and vertical angles the object makes with the centre of the camera are known. 
* Since the height of the camera is known ( position of the end effector + vertical offset of the camera), we can use angle formulas to find the position of the object relative to the centre of the camera. 
* The camera may be at an angle so the determined xy coordinates of the object must be converted to the global x-y coordinates of the robot found [here](https://github.com/jessyec-s/capstone/blob/d68466537a0ac814c729c744323ef835718a5110/physicalSystem/uarmController.py#L92)
#### camera to end effector:
* hardcoded vector from physical measurements, you will need to measure when assembling and modify the code [here](https://github.com/jessyec-s/capstone/blob/d68466537a0ac814c729c744323ef835718a5110/physicalSystem/uarmController.py#L11)
#### position of the end effector:
* can be found using the [Uarm API](https://github.com/uArm-Developer/uArm-Python-SDK/tree/2.0/uarm/swift). The subclass which inherits the Uarm commands is found in physicalSystem/uarmController.py.
