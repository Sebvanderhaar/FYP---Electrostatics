#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>

const double pi = 3.1415;

typedef struct
{
    double position[2];
    double magnitude;
} Charge;

void getForce(Charge* chargeArray, int chargeCount, double r[2], double force[2])
{
    const double k_e = 8.99e9;

    for (int i = 0; i < chargeCount; ++i){
        double q = chargeArray[i].magnitude;
        double* pos = chargeArray[i].position;

        double r1x = r[0] - pos[0];
        double r1y = r[1] - pos[1];

        double r1MagCubed = pow(sqrt(r1x*r1x + r1y*r1y), 3);

        force[0] += k_e * q * r1x/r1MagCubed;
        force[1] += k_e * q * r1y/r1MagCubed;
    }
}

double(*getSolutionArray(Charge currentCharge, Charge* chargeArray, int chargeCount, int CurrentLineNum, int LinesPerUnitCharge, double initRadius, double dl, int windowX, int windowY))[2]{
    double initX = currentCharge.position[0] + cos((CurrentLineNum*pi*2)/(currentCharge.magnitude*LinesPerUnitCharge)) * initRadius;
    double initY = currentCharge.position[1] + sin((CurrentLineNum*pi*2)/(currentCharge.magnitude*LinesPerUnitCharge)) * initRadius;

    double initPos[2] = {initX, initY};
    double currentPos[2] = {initX, initY};

    double(*solutionArray)[2] = malloc(1000*sizeof(double[2]));
    int solutionIter = 0;

    solutionArray[solutionIter][0] = initPos[0];
    solutionArray[solutionIter][1] = initPos[1];

    bool ended = false;
    double prevForce[2] = {0,0};

    while (!ended){
        solutionIter++;
        double currentForce[2] = {0,0};
        getForce(chargeArray, chargeCount, currentPos, currentForce);
        double currentForceMag = sqrt(pow(currentForce[0], 2) + pow(currentForce[1], 2));
        double prevForceMag = sqrt(pow(prevForce[0], 2) + pow(prevForce[1], 2));

        double endPosX;
        double endPosY;

        if (solutionIter != 1) {
            double dotProduct = prevForce[0]*currentForce[0] + prevForce[1]*currentForce[1];
            double currentPrevAngle = acos(dotProduct/(currentForceMag*prevForceMag))/(pi/2);

            endPosX = currentPos[0] + (currentForce[0]/currentForceMag) * dl * (currentPrevAngle + 0.5);
            endPosY = currentPos[1] + (currentForce[1]/currentForceMag) * dl * (currentPrevAngle + 0.5);

        } else {
            endPosX = currentPos[0] + (currentForce[0]/currentForceMag) * dl;
            endPosY = currentPos[1] + (currentForce[1]/currentForceMag) * dl;
        }

        solutionArray[solutionIter][0] = endPosX;
        solutionArray[solutionIter][1] = endPosY;

        for (int i = 0; i < chargeCount; i++){
            Charge current = chargeArray[i];
            double r1x = endPosX - current.position[0];
            double r1y = endPosY - current.position[1];

            double r1Mag = sqrt(pow(r1x, 2) + pow(r1y, 2));
            double currentMag = sqrt(pow(endPosX - windowX/2, 2) + pow(endPosY - windowY/2, 2));

            if ((r1Mag < initRadius - 1) || (currentMag > 2000) || (solutionIter == 999)){
                ended = true;
            }
        }

        currentPos[0] = endPosX;
        currentPos[1] = endPosY;

        prevForce[0] = currentForce[0];
        prevForce[1] = currentForce[1];
    }
    
    solutionArray[solutionIter + 1][0] = -1;
    solutionArray[solutionIter + 1][1] = -1;

    return solutionArray;
}


