
#include<sstream>
#include<iostream>
#include<math.h>
#include<iostream>
#include<fstream>

using namespace std;

int main(){
const int num = 17;
double a[num][num][num];
double bond_price[100],cp[100],cpdis[100];
double p,delta,deltat,sigma,u[num],d[num];
double dataframe[16] = {
		0.078461538,
		0.156923077,
		0.235384615,
		0.313846154,
		0.351538462,
		0.368846154,
		0.386153846,
		0.403461538,
		0.420769231,
		0.438076923,
		0.455384615,
		0.472692308,
		0.49,
		0.5,
		0.51,
		0.52};

for (int i=0;i < 16; i++){
	a[0][i][0] = 1/exp(dataframe[i]* (i+1)/52/100);
}

p = .5;
sigma = 0.1;
deltat = 1.0 / 52.0;
delta = exp(- sigma * sqrt(deltat) / p / (1-p));

u[0] = d[0] = 1;
for (int i = 1; i < num; i++){
	d[i] = pow(delta, i) / (p + (1 - p) * pow(delta,i));
	u[i] = 1/ (p + (1-p) * pow(delta, i));

}

	for (int i = 1; i<num; i++){
		for (int j = i+1; j<num; j++){
			for (int k = 0; k<i+1; k++){
				a[i][j][k] = a[i-1][j][k] / a[i-1][i][k] * d[j-i];
			}
		a[i][j][i] = a[i-1][j][i-1] / a[i-1][i][i-1] * u[j-i];
		}
	}
for (int i=1; i<num; i++){
	for (int k=0;k<i+1;k++){
		int j = i+1;

	}

}

 for (int i = 0; i<13;i++){
 cpdis[i] = 1;
 }
for (int i=11;i>=0;i--){
	int j = i+1;
	for (int k = 0; k <i+1; k++){
		cpdis[k] = (cpdis[k] * (1-p) + cpdis[k+1] * p) * a[i][j][k];
	}

}
cout << cpdis[0]<<endl;
}
