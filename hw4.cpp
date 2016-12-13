
#include<iostream>
#include<string>
#include<math.h>
#include<iostream>
#include<fstream>
#include<sstream>
using namespace std;

double present_value(double r,double nptr)
{
	return (1-pow((1+r), -nptr))/r;
}
int main(){
const int num = 360;
double a[2][num+1][num+1];
double dis[num+1][num+1];
double data[num+1];
double p,delta,deltat,sigma,u[num+1],d[num+1];


ifstream in_file("data2.in");
int num = 0;
string readline;
double sk;
while (num <10 && getline(in_file, readline,'\n')){
	stringstream ss(readline);
	while (ss>>sk){
		num+=1;
		data[num]=sk;
	}

}



for (int i=0;i <= num; i++){
	a[0][i][0] = 1/exp(data[i]* i/12.0/100.0);

}
dis[0][0] = a[0][1][0];
p = .5;
sigma = 0.1;
deltat = 1.0 / 12.0;
delta = exp(- sigma * sqrt(deltat) / p / (1-p));


u[0] = d[0] = 1;
for (int i = 1; i <= num; i++){
	d[i] = pow(delta, i) / (p + (1 - p) * pow(delta,i));
	u[i] = 1/ (p + (1-p) * pow(delta, i));

}


	for (int i = 1; i<num; i++){
		for (int j = i+1; j<=num; j++){
			for (int k = 0; k<i; k++){
				a[i % 2][j][k] = a[(i-1) % 2][j][k] / a[(i-1) % 2][i][k] * d[j-i];
			}
		a[i % 2][j][i] = a[(i-1) % 2][j][i-1] / a[(i-1) % 2][i][i-1] * u[j-i];
		}
		for (int k = 0; k<=i; k++){
			dis[i][k] = a[i % 2][i+1][k];
			if (isnan(dis[i][k])){
				dis[i][k]=0;
			}
		}	
	}


const double loan = 100;
double maxr, midr, minr;
double calc[num+1], pmt[num+1], prepay[num+1],interest[num+1],tmpresent_valueal[num+1], bal[num+1];
maxr = 1.0;
minr = 0.0;


const double prepaymethod = 0.0374;
const double prec = 1e-5;
	midr = (maxr + minr) / 2;
	midr = 0.0474;
	bal[0] = loan;
	for (int i = 1; i < num; i++){
		pmt[i] = bal[i-1] / present_value(midr/12, num - i+1);
		interest[i] = bal[i-1] * midr/12;
		prepay[i] = bal[i-1] * prepaymethod;
		bal[i] = bal[i-1] - pmt[i] + interest[i] - prepay[i];
		calc[i] = pmt[i] + prepay[i];
}
	pmt[num] = bal[num-1] / present_value(midr/12, 1);
	interest[num] = bal[num-1] * midr/12;
	calc[num] = pmt[num];

	for (int i =0; i<=num; i++){
		tmpresent_valueal[i] = calc[num];
	}
	for (int i = num-1; i>=0; i--){
		for (int k = 0; k<=i; k++){
			tmpresent_valueal[k] = (tmpresent_valueal[k] * (1 - p) + tmpresent_valueal[k+1] * p) * dis[i][k] + calc[i];

		}

	}
	if (tmpresent_valueal[0] < loan){
		minr = midr;
	}
	else{
		maxr= midr;
	}


cout << tmpresent_valueal[0];

}



