# Project README

## 1. Setting Up the Environment Using Anaconda
To create and activate a new Anaconda environment for your project, follow these steps:

```bash
# Create a new conda environment named 'ml-agent-env' with Python 3.8
conda create -n SDSLCS python=3.8

# Activate the newly created environment
conda activate SDSLCS

# Install pip list
pip install -r requirements.txt
```

## 2. Connecting to Arduino
This software communicates with two Arduinos:

- Motor_arduino: Controls motor movement.
- Current_arduino: Reads current sensor data.

Configure Arduino Ports
Modify the following lines in the script to match the actual port names on your system:

```bash
Motor_arduino = serial.Serial("COM3", 115200, timeout=1)  # Change to the correct port
Current_arduino = serial.Serial("COM4", 9600, timeout=1)   # Change to the correct port
```

Ensure the correct ports are assigned before running the program.

## 3. Running the Software
To start the software, simply run the script:

```bash
python main.py
```

The program will:
- Establish a serial connection with both Arduinos.
- Send predefined G-code commands to the Motor_arduino.
- Read and log current sensor data from the Current_arduino.
- Safely terminate the connection after execution.

## 4. Additional Notes
- Use Arduino IDE to upload firmware to Motor_arduino and Current_arduino(Program -> arduino IDE) before running the script.
- Ensure both Arduinos are correctly powered and connected to your system.
- Adjust the serial ports (COM3, COM4, etc.) based on your system configuration.

## Acknowledgement
This work is supported by Institute of information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No.RS-2022-00155911, Artificial Intelligence Convergence Innovation Human) Resources Development(Kyung Hee University)
