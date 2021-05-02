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
cd openMV
python3 threaded.py
```

### Tests
To run success rate testing for the "find and touch" task:

* Inside `threaded.py`:
   * Inside the `main` function, change `run_tests` parameter to True
   * The test is run from inside the `ddpg_loop_no_camera` function
   * Plots for the distance, success rate, and time for each iteration will be produced inside `openMV/plots/`
* Inside `uarmTests.py`:
   * Creates an array of random target locations within the robot's bounds  
   * You can change the number of iterations that you want to run

```
cd openMV
python3 threaded.py
```

## Tips

1. If robot is not connecting
    * Unplug camera
    * Run ``threaded.py`` with only robot connected
    * You will know robot has connected once robot beeps
    * Retry running ``threaded.py`` with both camera and robot connected

    * If the robot is getting hot this can also cause problems.  Let it cool off then retry the steps above
2. If UArm port cannot be found:
   * **TODO**
4. If object cannot be recognized by the robot:
   * **TODO**
