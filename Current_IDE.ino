#define SENSOR_PIN A0
float sensitivity = 185.0; // For 5A module: 185 mV/A
float voltageOffset = 2.5; // 2.5V is the center voltage for ACS712
float vRef = 5.0; // Reference voltage of Arduino (5V for most)

void setup() {
    Serial.begin(9600);
}

void loop() {
    int sensorValue = analogRead(SENSOR_PIN);
    float voltage = (sensorValue / 1024.0) * vRef; // Convert to voltage
    float current = (voltage - voltageOffset) * 1000 / sensitivity; // Convert to current
    Serial.println(current);
    delay(1000); // 1-second delay
}