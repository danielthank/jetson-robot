#include <Arduino.h>
#include <Servo.h>

Servo servo[6];
const int calibrate[6][2] = {{90, 160}, {90, 20}, {155, 90}, {150, 80}, {90, 30}, {60, 0}};

void setDegree(int degrees[]) {
	for (int i=0; i<6; i++) {
		degrees[i] = (calibrate[i][1] - calibrate[i][0]) / 90.0 * degrees[i] + 2 * calibrate[i][0] - calibrate[i][1];
		servo[i].write(degrees[i]);
	}
}

void setup() {
	pinMode(2, OUTPUT);
	pinMode(3, OUTPUT);
	pinMode(4, OUTPUT);
	pinMode(5, OUTPUT);
	pinMode(6, OUTPUT);
	pinMode(7, OUTPUT);

	pinMode(8, OUTPUT);
	pinMode(9, OUTPUT);

	for (int i=0; i<6; i++) servo[i].attach(i+2);
	int init[6] = {90, 60, 90, 90, 90, 30};
	setDegree(init);

	Serial.begin(9200);
}

char serialData[32];
bool parseCommand(char* command, int* returnValues, char returnNumber) {
	int i = 1, j = 0, number, temp = 0;
	char sign = 0, ch = 0;
	while (i++) {
		switch (*(command + i)) {
			case '\0':
			case ',':
			if(ch != 0) {
				returnValues[j++] = sign?-temp:temp;
				sign = 0;
				temp = 0;
				ch = 0;
			}
			else return false;
			break;

			case '-':
			sign = 1;
			break;

			default:
			number = *(command + i) - '0';
			if(number < 0 || number > 9) return false;
			temp = temp * 10 + number;
			ch++;
		}
		if (j == returnNumber) return true;
		else if (*(command + i) == '\0') return false;
	}
}

void loop()
{
	bool available = false;
	if (Serial.available() > 0) {
		Serial.readBytesUntil('\n', serialData, 31);
		available = true;
	}
	if (available) {
		switch(serialData[0]) {
			case 'a':
			int degree[6];
			if (parseCommand(serialData, degree, 6)) {
				setDegree(degree);
				Serial.println("OK");
			}
			break;
			case 'l':
			int laser[2];
			if (parseCommand(serialData, laser, 2)) {
				digitalWrite(laser[0] + 8, laser[1]);
				Serial.println("OK");
			}
		}
	}
}
