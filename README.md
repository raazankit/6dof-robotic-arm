# 6-DOF Robotic Arm: Teleoperation & Kinematic Simulation

Hardware control loops, computer vision pipelines, and mathematical simulation models for a 6-Degree-of-Freedom (DOF) robotic manipulator. This project bridges physical teleoperation via real-time hand-tracking with spatial kinematic modeling in CoppeliaSim.

---

### 🧠 System Architecture

#### 1. Real-Time Teleoperation Pipeline (Physical)
* **Perception:** Real-time 21-point hand landmark tracking using **Python**, **OpenCV**, and **MediaPipe**.
* **Signal Conditioning:** Spatial coordinate smoothing via Exponential Moving Average (EMA) filters to suppress jitter before mapping to joint space.
* **Actuation Interface:** Serial UART packet transmission to an **ESP32**, interfacing over I2C with a **PCA9685** 16-channel PWM driver to control 6 servo actuators under an external 5V/10A power supply.

#### 2. Digital Kinematic Simulation
* **Simulation Engine:** Modeled in **CoppeliaSim** using a Universal Robots UR5 manipulator.
* **Mobility Verification:** Degrees of freedom validated via Grübler's formula for spatial mechanisms:
  $$m = 6(n - 1 - j) + \sum_{i=1}^{j} f_i$$
* **Kinematics:** Forward and inverse kinematics formulated using screw axes, matrix exponentials in $SE(3)$, and the Product of Exponentials (PoE) formula:
  $$T(\theta) = e^{[\mathcal{S}_1]\theta_1} e^{[\mathcal{S}_2]\theta_2} \cdots e^{[\mathcal{S}_n]\theta_n} M$$

---

### 🛠️ Tech Stack & Dependencies
* **Languages:** C++, Python 3.8+
* **Libraries:** OpenCV, MediaPipe, NumPy, PySerial
* **Hardware:** ESP32 DevKit, PCA9685 PWM Driver, 6x Servos, External 5V/10A SMPS
* **Simulation:** CoppeliaSim

---

### ⚙️ Getting Started

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/raazankit/6dof-robotic-arm.git](https://github.com/raazankit/6dof-robotic-arm.git)
   cd 6dof-robotic-arm