import serial
import serial.tools.list_ports
import time
import csv
from datetime import datetime


def find_arduino_port():
    ports = serial.tools.list_ports.comports()
    arduino_ports = []

    for port in ports:
        if "Arduino" in port.description or "CH340" in port.description or "USB-SERIAL" in port.description:
            arduino_ports.append(port.device)

    if arduino_ports:
        print("아두이노가 연결된 포트:", arduino_ports)
        return arduino_ports
    else:
        print("아두이노가 연결된 포트를 찾을 수 없습니다.")
        return None

def initialize_grbl(arduino):
    """
    GRBL 초기화
    """
    print("Initializing GRBL...")
    arduino.write("\r\n\r\n".encode())  # 초기화 명령 전송
    time.sleep(2)  # 초기화 대기
    arduino.flushInput()  # 입력 버퍼 비우기
    print("GRBL initialized.")


def Send_signal(arduino, gcode):
    for line in gcode:
        arduino.write(f'{line}\n'.encode())  # G 코드 라인을 Arduino로 전송
        time.sleep(2)
        print(f'명령 {line}을 실행하였습니다. 2초간 대기 합니다')
def Read_current_data(arduino, stop_event):
    try:
        print("Reading current data from ACS712 sensor...")
        while not stop_event['stop']:
            # 시리얼 데이터 읽기
            if arduino.in_waiting > 0:
                raw_data = arduino.readline()
                try:
                    # 디코딩 시 오류 발생 시 예외 처리
                    data = raw_data.decode("utf-8").strip()
                    print(f"Current: {data} A")
                except UnicodeDecodeError:
                    print(f"Decoding error: {raw_data}")
    except serial.SerialException as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        arduino.close()  # 시리얼 포트 닫기