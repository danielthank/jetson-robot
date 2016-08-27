#include <Arduino.h>
#include <Servo.h>

#define RIGHT_SERVO  3
#define LEFT_SERVO  5

#define RIGHT_IR 4
#define LEFT_IR 2

Servo lServo;
Servo rServo;
unsigned long timeStart=0;
unsigned long timePass;
char serialData[32];
int cnt=0;
int rSpeed, lSpeed, rNow, lNow;
int rIR, lIR;

void setup() {
    pinMode(RIGHT_SERVO, OUTPUT);
    pinMode(LEFT_SERVO, OUTPUT);

    pinMode(RIGHT_IR, INPUT);
    pinMode(LEFT_IR, INPUT);
    rServo.attach(RIGHT_SERVO);
    lServo.attach(LEFT_SERVO);
    Serial.begin(9600);
}

bool parseCommand(char* command, int* returnValues, byte returnNumber)
{
    // parsing state machine
    byte i = 1, j = 0, sign = 0, ch = 0, number;
    int temp = 0;
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
    if (Serial.available() > 0) {
        int length = Serial.readBytesUntil('\n', serialData, 31);
        serialData[length] = '\0';
        switch(serialData[0]){
        case 's':
            int speed[2];
            timePass = millis() - timeStart;
            timeStart = millis();
            //timeDelay=timePass*0.75+timeDelay*0.15;
            if(timePass > 100) timePass=100;
            //Serial.print("time : ");
            //Serial.println(timePass);
            if(parseCommand(serialData, speed, 2)){
                if (speed[1] > 0) speed[1] *= 0.8;
                if (speed[0] < 0) speed[0] *= 0.8;
                rSpeed = speed[0];
                lSpeed = speed[1];
                //delay(timePass/2 + timePass/4);
                //setSpeed(90,90);
            }
            break;
        case 'i':
            rIR = digitalRead(RIGHT_IR);
            lIR = digitalRead(LEFT_IR);
            Serial.println(String(rIR) + String(lIR));
            break;
        }
        //Serial.println("Finish");
    }
    if (cnt++ % 1000 == 0) {
        rServo.write(rNow + 90);
        lServo.write(90 - lNow);
        lNow += (lSpeed - lNow) * 0.1;
        rNow += (rSpeed - rNow) * 0.1;
    }
}
