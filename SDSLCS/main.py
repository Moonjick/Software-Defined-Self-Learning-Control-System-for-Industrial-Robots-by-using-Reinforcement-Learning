import serial
import time
import csv

from datetime import datetime
from Tools.utils import find_arduino_port, Send_signal, Read_current_data
from threading import Thread

### Arduino 연결 ###
connected_arduino_port = find_arduino_port()
Motor_arduino = serial.Serial("COM3", 115200, timeout = 1)  #Arduino가 연결된 포트로 변경
Current_arduino = serial.Serial("COM4", 9600, timeout = 1)  #Arduino가 연결된 포트로 변경

# Arduino와 연결
try:
    time.sleep(2)  # 안정화 대기

    # G코드 리스트 정의
    gcode = [
        "$110=500",
        "G0 X5",
        "G0 X0"
    ]

    # 데이터 수집 스레드 실행 제어 변수
    stop_event = {"stop": False}

    # 데이터 수집 시작
    try:
        # 데이터 읽기 루프 실행
        print("Starting data collection and G-code transmission...")
        read_thread = Thread(target=Read_current_data, args=(Current_arduino, stop_event))
        read_thread.start()

        # G코드 전송
        Send_signal(Motor_arduino, gcode)

        # G코드 실행 완료 후 데이터 수집 중단
        stop_event['stop'] = True
        read_thread.join()  # 데이터 수집 스레드가 종료될 때까지 대기
        print("Data collection stopped.")

    except KeyboardInterrupt:
        stop_event['stop'] = True
        read_thread.join()

except serial.SerialException as e:
    print(f"Serial connection error: {e}")
finally:
    Motor_arduino.close()
    Current_arduino.close()