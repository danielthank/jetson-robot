#include <Arduino.h>
#include <Servo.h>
#include <NewPing.h>
#define RIGHT_SERVO  3
#define LEFT_SERVO  5
#define RIGHT_IR 4
#define LEFT_IR 2
#define X_ACC 1
#define Y_ACC 2
#define Z_ACC 3
#define TRIGGER_1 8
#define TRIGGER_2 10
#define ECHO_1 9
#define ECHO_2 11
#define MAX_DISTANCE 500
Servo lServo;
Servo rServo;
NewPing sonar1(TRIGGER_1,ECHO_1,MAX_DISTANCE);
NewPing sonar2(TRIGGER_2,ECHO_2,MAX_DISTANCE);
void setup() {
    pinMode(RIGHT_SERVO, OUTPUT);
    pinMode(LEFT_SERVO, OUTPUT);

    pinMode(RIGHT_IR, INPUT);
    pinMode(LEFT_IR, INPUT);
    rServo.attach(RIGHT_SERVO);
    lServo.attach(LEFT_SERVO);
    Serial.begin(9600);
}

char serialData[32];
bool parseCommand(char* command, int* returnValues, char returnNumber) {
    int i = 1, j = 0, number, temp = 0;
    char sign = 0, ch = 0;
    while (i++) {
        switch(*(command + i)) {
            case '\0':
            case ',':
            if (ch != 0) {
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
            if (number < 0 || number > 9) return false;
            temp = temp * 10 + number;
            ch++;
        }
        if (j == returnNumber) return true;
        else if(*(command + i) == '\0') return false;
    }
}

bool ir;

void loop() {
    bool available = false;
    if (Serial.available() > 0) {
        int length = Serial.readBytesUntil('\n', serialData, 31);
        serialData[length] = '\0';
        available = true;
    }
    int lIR = digitalRead(LEFT_IR);
    int rIR = digitalRead(RIGHT_IR);
    if (lIR == 1 && rIR == 0) {
        ir = true;
        rServo.write(90 - 50);
        lServo.write(90 - 50 * 1.3);
        delay(1000);
        if (available) Serial.println("LEFT_IR");
        return;
    }
    else if (lIR == 0 && rIR == 1) {
        ir = true;
        rServo.write(90 + 50 * 1.3);
        lServo.write(90 + 50);
        delay(1000);
        if (available) Serial.println("RIGHT_IR");
        return;
    }
    else if (ir){
        ir = false;
        if (!available) {
            rServo.write(90);
            lServo.write(90);
            return;
        }
    }
    if (available) {
        switch(serialData[0]){
            case 'c':
            int speed[2];
            if(parseCommand(serialData, speed, 2)){
                if (speed[1] < 0) speed[1] *= 1.3;
                if (speed[0] > 0) speed[0] *= 1.3;
                int rSpeed = speed[0];
                int lSpeed = speed[1];
                rServo.write(90 + rSpeed);
                lServo.write(90 - lSpeed);
                Serial.println("OK");
            }
            else Serial.println("Bad");
            break;

            case 'i':
            int cm1 = sonar1.ping_cm();
            int cm2 = sonar2.ping_cm();
            Serial.println(String(cm1) + " " + String(cm2));
            /*
            int x = analogRead(X_ACC);
            int y = analogRead(Y_ACC);
            int z = analogRead(Z_ACC);
            Serial.println(x);
            Serial.println(y);
            Serial.println(z);
            */
            break;
        }
    }
}
