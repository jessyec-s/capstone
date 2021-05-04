Files used for executing the physical *find and touch* task.

## File Descriptions

### [ddpgHer](https://github.com/jessyec-s/capstone/blob/master/physicalSystem/ddpgHer.py)
DDPG + HER algorithm class that contains the ``run`` function, which is used to execute this RL algorithm on the physical system.

### [getobjectblob](https://github.com/jessyec-s/capstone/blob/master/physicalSystem/getobjectblob.py)
OpenMV camera image recognition script that is ran on the OpenMV camera.  Text output from this script
is printed to a text buffer that can be read.  This script can also be executed directly using the [OpenMV IDE](https://openmv.io/pages/download).

### [threaded.py](https://github.com/jessyec-s/capstone/blob/master/physicalSystem/threaded.py)
File containing functionality to control the UArm and OpenMV camera threads.  This is the main file that should be run to execute the "find and touch" task.

### [uarmController](https://github.com/jessyec-s/capstone/blob/master/physicalSystem/uarmController.py)
A class extending the UArm Swift pro API that is used to control the physical movements of the UArm as well as perform calculations that require data about the position
of the UArm at a given time.

### [uarmEnv](https://github.com/jessyec-s/capstone/blob/master/physicalSystem/uarmEnv.py)
A custom environment that describes the physical environment of the UArm and is used by the DDPG algorithm.  The structure of this environment has been made
to mirror the [simulation environment](https://github.com/kobinau/gym/blob/capstone/gym/envs/robotics/fetch_env.py) as closely as possible.

### [uarmTests](https://github.com/jessyec-s/capstone/blob/master/physicalSystem/uarmTests.py)
Tests for the *find and touch* task.
