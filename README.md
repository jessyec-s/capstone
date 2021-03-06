# Capstone

U of T Capstone project exploring the use of Reinforcement Learning on a physical system.

## Set Up

1. Mount Camera on Uarm 
2. Connect OpenMV camera to computer's USB port
3. Connect UArm to computer's USB port and plug in UArm power cord

## Run

### Tasks
To run the "find and touch task"
Make sure both camera and robot are connected to computer and then put red object in the robot's field of view:

```
cd physicalSystem
python3 threaded.py
```

### Tests
To run success rate testing for the "find and touch" task:

* Inside `threaded.py`:
   * Inside the `main` function, change `run_tests` parameter to True
   * The test is run from inside the `find_and_touch_test_loop` function
   * Plots for the distance, success rate, and time for each iteration will be produced in your current directory in a folder named `/plots`
* Inside `uarmTests.py`:
   * Creates an array of random target locations within the robot's bounds  
   * You can change the number of iterations that you want to run

```
cd physicalSystem
python3 threaded.py
```
### Simulations

To run simulations:
* install 'mujoco_py'
    * Register for a year long student license [on the mujoco website](https://www.roboti.us/license.html)
    * Follow the installation instructions on the [openai github](https://github.com/openai/mujoco-py)
* Customize and build simulations: 
    * The modified FetchReach model matching the robot [on this branch](https://github.com/kobinau/gym/tree/capstone)
    * To import a custom gym branch, move into the 'gym/' directory and run: 
``` 
	git clone https://github.com/kobinau/gym.git
	cd gym
	git checkout capstone
	pip install -e .
```	
    * in the 'simulations/' directory, run 'simulations.py' 
    * can adjust train boolean for train/run, training environment and file name of trained parameters in simulation.py
    * output file is then used for the robot
* Additional Details on relationship between simulation and robots are found in 'simulation/README.md' 
## Tips

1. If robot is not connecting
    * Unplug camera
    * Run ``threaded.py`` with only robot connected
    * You will know robot has connected once robot beeps
    * Retry running ``threaded.py`` with both camera and robot connected

    * If the robot is getting hot this can also cause problems.  Let it cool off then retry the steps above
2. If OpenMV camera cannot connect because the port cannot be found:
    * Print out information about your computer's ports and then identify the port in use by the camera
    ```
	cd /dev
	ls
	```
	* update the constant ``port_name`` in ``threaded.py`` with your found port name
3. If object cannot be recognized by the camera, meaning there is no white box surrounding the object when it is in view:
   * Open the OpenMV IDE
   * Go to Tools -> Machine Vision -> Threshold Editor
   * Click "Frame Buffer"
   * Adjust levels until the object is perfectly recognized and copy the modified threshold values (should be 6 numbers)
   * Replace the threshold values for the target object colour [here](https://github.com/jessyec-s/capstone/blob/master/physicalSystem/getobjectblob.py#L56)

4. If the camera window freezes while the robot is still moving, or is green and cannot identify the object:
   * Quit the program and try again

## Other relevant documentation 

Uarm and OpenMV:
* [Uarm Swift Pro User Manual](http://download.ufactory.cc/docs/en/uArm%20pro%20User%20Manual%20v1.1.0.pdf)

Algorithm:
* [Stable Baselines3 HER documentation](https://stable-baselines3.readthedocs.io/en/master/modules/her.html)
* [Stable Baselines3 DDPG documentation](https://stable-baselines3.readthedocs.io/en/master/modules/ddpg.html)

APIs:
* [UArm API](https://github.com/uArm-Developer/uArm-Python-SDK/blob/2.0/doc/api/swift_api.md)
* [OpenMV API](https://docs.openmv.io/openmvcam/quickref.html)

Tools:
* [OpenMV IDE](https://openmv.io/pages/download)
* [UArm Studio](https://www.ufactory.cc/pages/download-uarm)
