# Franka Panda: 3D End-Effector Trajectory Tracking

Classical Inverse Kinematics vs Proximal Policy Optimisation (PPO)

## Project Overview 
This project investigates trajectory tracking for a 7Dof Franka Panda robotic arm using MuJoCo simulation.

I implemented two approaches to control the robot's end-effector:
1. Classical Inverse Kinematics (IK): A Jacobian-based controller using damped least squares.
2. Reinforcement Learning (RL): A PPO agent trained using Gymnasium and Stable-Baseline3.

The objective was to follow a time-varying circular trajectory while minimising tracking error and maintaining smooth movement. 

I evaluated both controllers using positional tracking metrics and tested their sensitivity to measurement noise. 

## Technologies
- Python
- MuJoCo
- Gymnasium
- Stable-Baseline3
- Pytorch
- Numpy
- Pandas 
- MatplotLib
- Git and GitHub

## 1. Simulation Setup
The project uses the Franka Panda model from the MuJoCo Menagerie repository. 
The robot has seven actuated arm joints
The desired trajectory is a horizontal circle starting at the robot's initial end-effector position. 

## 2. Classical Inverse Kinematics Controller
The classical controller uses the end-effector's position error and desired velocity to calculate corrective Cartesian velocities. 

A damped least-squares Jacobian inverse converts these velocities into joint commands. The controller also applies joint-velocity limits and respects the robot's joint position limits. This approach provides a conventional model-based baseline against which the reinforcement learning controller can be evaluated. 

## 3 Reinforced Learning Using PPO
I developed a custom Gymnasium environment around the MuJoCo simulation and trained a PPO agent using Stable-Baseline3

Unlike the classical controller, PPO learns a control policy from interactons with the simulated robot rather than calculating joint movements directly from the Jacobian. 

### Observation Space
The original environment contains 36 observations, including:
- Seven joint positions
- Seven joint velocities
- 3D end-effector position and velocity
- 3D target position and velocity
- 3D tracking error
- Seven previous actions

### Action Space
The agent produces seven continuous actions, each in the range[-1, 1]
These actions are scaled into incremental joint-position commands.

### Reward Function
The reward penalties:
- End-effector tracking error
- Changes between consecutive actions
- Joint velocity magnitude
The objective is to encourage accurate tracking while discouraging unnecessary movements. 

### PPO Tracking Results
The PPO controller was evaluated over a 10 seconds circular trajectory. The trained PPO agent learned to follow the approximate circular path, but its movement contained repeated oscillations, particularly in the vertical direction. 

## 4. Robustness Testing
I evaluated both controllers under simulated end-effector position measurement noise. Independent Gaussian noise was added to the three position coordinates, using standard deviations of 0, 1, and 3mm. Each condition was tested using three random seeds. Tracking error was evaluated against the actual simulated end-effector position rather than the noisy measurement. 

Measurement noise	Classical IK mean error	PPO mean error
0 mm	0.782 mm	6.543 mm
1 mm	0.825 mm	6.529 mm
3 mm	1.123 mm	6.496 mm

### Findings
Classical IK achieved lower absolute tracking error across all tested noise levels. Its mean tracking error increased as measurement noise increased. 
PPO's mean tracking error changed very little under the same conditions, although its baseline error remained substantially higher. This apparent insensitivity to noise should not be interpreted as evidence of superior overall control performance. 

### Conclusion
The classical Jacobian-based controller substantially outperformed PPO in this experiment. One possible explanation is that the classical controller directly uses the robot's kinematic model, whereas PPO must learn an effective control policy through repeated simulation. The PPO controller learned approximate circular motion but continued to exhibit oscillations after training. 

These results demonstrate that reinforcement learning does not necessarily outperform classical control for a predictable trajectory with an available mathematical model.

## 8. Installation and Reproduction
### Installation
Clone this project and create a Python virtual environment:

*git clone https://github.com/YOUR_USERNAME/franka-rl-trajectory-tracking.git*
*cd franka-rl-trajectory-tracking*
*python3 -m venv .venv*
*source .venv/bin/activate*

Install the required packages:

*python -m pip install mujoco numpy pandas matplotlib gymnasium "stable-baselines3[extra]"*

The project also requires the MuJoCo Menagerie Franka Panda model.
The Python scripts currently expect the model at:

*~/robotics-projects/mujoco_menagerie/franka_emika_panda/scene.xml*

Clone the official MuJoCo Menagerie repository to that location, or update the model paths in the scripts to match your installation.

**Run the Classical Controller**

*python track_circle.py*

**Evaluate Classical Tracking**

*python evaluate_tracking.py*

**Test the Gymnasium Environment**

*python test_rl_env.py*

**Train PPO**

*python train_ppo.py --steps 200000*

This creates a new PPO model. Training results may vary with random initialisation and software versions.

**Evaluate PPO**

*python evaluate_ppo.py*

The evaluation script expects a trained model at:

models/panda_ppo.zip

The primary PPO model requires the original 36-dimensional observation environment.

**Run Robustness Tests**

*python robustness_test.py*

**Visualise Robot Movement**

Classical IK:

*python visualize_controllers.py --controller ik*

**PPO:**

*python visualize_controllers.py --controller ppo*

The PPO visualisation requires the saved model file.