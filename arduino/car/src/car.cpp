#include <Arduino.h>
#include <Servo.h>

#define RIGHT_SERVO  3
#define LEFT_SERVO  5
#define RIGHT_IR 4
#define LEFT_IR 2

Servo lServo;
Servo rServo;

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
bool parseCommand(char* command, int* returnValues, char returnNumber)
{
    int i = 1, j = 0, number, temp = 0;
    char sign = 0, ch = 0;
    while(i++){
        switch(*(command + i)){
            case '\0':
            case ',':
            if(ch != 0){
                returnValues[j++] = sign?-temp:temp;
                sign = 0;
                temp = 0;
                ch = 0;
            }
            else{
                return false;
            }
            break;
            case '-':
            sign = 1;
            break;
            default:
            // convert string to int
            number = *(command + i) - '0';
            if(number < 0 || number > 9){
                return false;
            }
            temp = temp * 10 + number;
            ch++;
        }
        // enough return values have been set
        if(j == returnNumber){
            return true;
        }
        // end of command reached
        else if(*(command + i) == '\0'){
            return false;
        }
    }
}

void loop() {
    bool available = false;
    if (Serial.available() > 0) {
        int length = Serial.readBytesUntil('\n', serialData, 31);
        serialData[length] = '\0';
        available = true;
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
                rServo.write(rSpeed + 90);
                lServo.write(90 - lSpeed);
                Serial.println("OK");
            }
            else Serial.println("Bad");
            break;

            case 'i':
            int rIR = digitalRead(RIGHT_IR);
            int lIR = digitalRead(LEFT_IR);
            Serial.println(String(rIR) + String(lIR));
            break;
        }
    }
}
