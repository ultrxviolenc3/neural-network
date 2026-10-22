#include <iostream>
#include <cmath>
#include <fstream>
#include <vector>
#include <algorithm>
#include <random>
#include <cstdlib>
#include <ctime>

class Point {
	public:
	double x,y;
	int label;
	};
class Norm {
	public:
	double xmin,xmax,ymin,ymax;
	};
double Sigmoida(double x) {
    return 1.0/(1.0+exp(-x));
	}
double Sigmoida2(double x) {
    double s=Sigmoida(x);
    return s*(1.0-s);
	}

std::vector<Point> read(const std::string &file,int size) { ///чтение из файла
	std::vector<Point> vec;
	vec.reserve(size);
	std::ifstream f(file);
    if (!f) {
	    std::cout<<"ошибка открытия файла"<< std::endl;
        return vec;
		}
    Point p;
    while (f>>p.x>>p.y>>p.label) vec.push_back(p);
    return vec;
	}
Norm normp(const std::vector<Point> &traindata) {
    Norm obb;
    if (traindata.empty()) {
        obb={0,0,0,0};
        return obb;
		}
    obb.xmin=obb.xmax=traindata[0].x;
    obb.ymin=obb.ymax=traindata[0].y;
    for(const auto& p : traindata) {
        if (p.x<obb.xmin) obb.xmin=p.x;
        if (p.x>obb.xmax) obb.xmax=p.x;
        if (p.y<obb.ymin) obb.ymin=p.y;
        if (p.y>obb.ymax) obb.ymax=p.y;
		}
    return obb;
	}
void normirovka(double &x, double &y, const Norm &obb) {
    double dx=obb.xmax-obb.xmin;
    double dy=obb.ymax-obb.ymin;
    if (dx==0.0) dx=1.0;
    if (dy==0.0) dy=1.0;
    x=2.0*(x-obb.xmin)/dx-1.0;
    y=2.0*(y-obb.ymin)/dy-1.0;
	}
void split(std::vector<Point> &alldata, std::vector<Point> &traindata, std::vector<Point> &validdata) { ///делим на обучающую и валидационную
	std::random_device rd;
	std::mt19937 g(rd());
    shuffle(alldata.begin(), alldata.end(), g);
    int trainsize=(int)(alldata.size()*0.8);
    traindata.assign(alldata.begin(), alldata.begin()+trainsize);
    validdata.assign(alldata.begin()+trainsize, alldata.end());
	}
class Neuralnetwork {
    Norm params;
    std::vector<int> layers;
    std::vector<std::vector<std::vector<double>>> w;
    std::vector<std::vector<double>> b;
    std::vector<std::vector<double>> a;
    std::vector<std::vector<double>> f;
    std::vector<std::vector<double>> delta;
    double lr;
	public:
	void set_norm(const Norm &p) {params=p;}
    Neuralnetwork(const std::vector<int> &inner_size, double learning_rate) {
        lr=learning_rate;
		layers.reserve(inner_size.size()+2);

        layers.push_back(2);

        for(auto &h : inner_size)
        layers.push_back(h);
        layers.push_back(1);
        weights();
		}
    void weights() {
		int l=layers.size();
        w.resize(l);
        b.resize(l);
        for(int k=1;k<l;k++) {
            int prev=layers[k-1];
            int curr=layers[k];
            b[k].resize(curr);
            for (int i=0;i<curr; i++) b[k][i] = 0.0;
			w[k].resize(curr);
			for (int i=0;i<curr;i++) {
				w[k][i].resize(prev);
				for (int j=0;j<prev;j++) {
					w[k][i][j]=rand()/(double)RAND_MAX*2-1;
					}
				}
			}
		}
	double forward(double x, double y) {
		normirovka(x, y, params);
		int l = layers.size();
		a.resize(l);
		f.resize(l);
		a[0].resize(2);
		a[0][0] = x;
		a[0][1] = y;
		for(int k = 1; k < l; k++) {
			int prevn = layers[k-1];
			int currn = layers[k];
			f[k].resize(currn);
			a[k].resize(currn);
			for(int i = 0; i < currn; i++) {
				f[k][i] = b[k][i];
				for(int j = 0; j < prevn; j++) f[k][i] += w[k][i][j] * a[k-1][j];
				a[k][i] = Sigmoida(f[k][i]);
				}
			}
		return a[l-1][0];
		}
	void tofile(const std::vector<Point> &data, std::ofstream &file) {
		for(const auto &p : data) {
			Point temp = p;
			int pred=predict(temp);
			file<<p.x<<" "<<p.y<<" "<<pred<< std::endl;
			}

		}
	void backward (int truelabel) {
		int l=layers.size();
		delta.resize(l);
		delta[l-1].resize(1);
		double p=a[l-1][0];
		double dL_dp;
		if(truelabel==1) dL_dp=-1.0/p;
		else dL_dp = 1.0/(1.0-p);
		delta[l-1][0]=dL_dp*p*(1.0-p);
		for(int k=l-2;k>=1;k--) {
			int currn=layers[k];
			int nextn=layers[k+1];
			delta[k].resize(currn);
			for(int i=0;i<currn;i++) {
				delta[k][i]=0.0;
				for(int j=0;j<nextn;j++) delta[k][i]+=w[k+1][j][i]*delta[k+1][j];
				delta[k][i]*=Sigmoida2(f[k][i]);
				}
			}
		}
	void train_epoch(std::vector<Point> &traindata) {
		std::random_device rd;
		std::mt19937 g(rd());
		shuffle(traindata.begin(), traindata.end(), g);
		for(auto &p:traindata) {
			forward(p.x, p.y);
			backward(p.label);
			update();
			}
		}
	void update() {
		int l=layers.size();
		for(int k=1;k<l;k++) {
			int prevn=layers[k-1];
			int currn=layers[k];
			for(int i=0;i<currn;i++) {
				for(int j=0;j<prevn;j++) w[k][i][j]-=lr*delta[k][i]*a[k-1][j];
				b[k][i]-=lr*delta[k][i];
				}
			}
		}
	int predict(Point &p){
		 double prob=forward(p.x,p.y);
		 return (prob>=0.5) ? 1 : 0;
		 }
	double evaluate(const std::vector<Point> &data) {
		int count=0;
		for(const auto &p:data) {
			Point temp=p;
			int pred=predict(temp);
			if(pred==p.label) count++;
			}
		return (double)count/data.size();
		}
	};

void start(std::ofstream &f) {
	    f<<"set xlabel \"x\""<< std::endl
		<<"set ylabel \"y\""<< std::endl
		<<"set size ratio -1"<< std::endl
		<<"set palette defined(0 \"blue\", 1 \"red\")"<< std::endl
        <<"set key off"<< std::endl
        <<"plot \"pointnew.txt\" using 1:2:3 with points pt 7 ps 0.1 lc palette"<< std::endl;
        }

void gen(int size,double cx0,double cy0,double a0,double cx1,double cy1,double a1) {
	std::ofstream f("point.txt");
	int class_l;
	for(int i=0;i<size;i++) {
		class_l=rand()%2;
		if(class_l==0) {
			f<<cx0-a0/2+a0*((double)rand()/RAND_MAX)<<" "<<
			cy0-a0/2+a0*((double)rand()/RAND_MAX)<<" "<<class_l<< std::endl;
			}
		else {
			f<<cx1-a1/2+a1*((double)rand()/RAND_MAX)<<" "<<
			cy1-a1/2+a1*((double)rand()/RAND_MAX)<<" "<<class_l<<std::endl;
			}
		}
	}
int main() {
	int size;
	std::cout<<"введите количество точек:"<< std::endl;
	std::cin>>size;
	double cx0=0,cy0=1,a0=2,cx1=2.0,cy1=2.6,a1=2;
	gen(size,cx0,cy0,a0,cx1,cy1,a1);
	std::vector<Point> alldata=read("point.txt",size);

    if (alldata.empty()) {
	    std::cout<<"файл пуст"<< std::endl;
        return 1;
		}
	std::ofstream file("pointnew.txt");
	std::ofstream f("point.plt");
	start(f);
	std::vector<Point> traindata, validdata;
    split(alldata, traindata, validdata);
	std::vector<int> inner = {8,8};
    double lr=0.05;
    Neuralnetwork net(inner, lr);
	Norm params=normp(traindata);
	net.set_norm(params);
	int epochs=15;
	for (int e=0;e<epochs;e++) {
	net.train_epoch(traindata);
	double valid_acc=net.evaluate(validdata);
	std::cout<<e+1<< std::endl;
	std::cout<<"valid accuracy: "<<valid_acc<< std::endl;
	}
	net.tofile(alldata, file);
	return 0;
}
