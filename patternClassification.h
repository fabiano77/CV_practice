#pragma once
#ifndef PATTERNCLASSIFICATION_H 
#define PATTERNCLASSIFICATION_H


#include <iostream>
#include <fstream>
#include <io.h>
#include <assert.h>
#include <math.h>
#include "opencv2/opencv.hpp"
using namespace std;
using namespace cv;

#define d(a, b) abs(a - b)
#define min2(a, b) (a>b)b:a

void initial_setting();
Mat toRed(Mat src);
Mat toGray(Mat src);
Mat Otsu(Mat img);
Mat dilation(Mat img);
Mat erosion(Mat img);
Mat Contour_tracing_based_method(Mat img, vector<pair<int, int>>& point_list);
void calCoord(int i, int* y, int* x);
void read_neighbor8(int y, int x, unsigned char neighbor8[8], Mat img);
vector<pair<int, int>> BTracing8(int y, int x, int label, bool tag, Mat img, Mat& labImage, Mat& contourImage);
vector<double> getLCSarray(vector<pair<int, int>> contour_points);
vector<double> featureExtraction(Mat src);
void nomalization(vector<double>& pattern);
double DTW(vector<double> A, vector<double> B, vector<pair<int, int>>& path);
double SDTW(vector<double> A, vector<double> ref_avg, vector<double> ref_stdev);
int classification(Mat src);
void learningFeature(vector<Mat> img_list, string filename);
void learningAllData(vector<vector<Mat>> img_memory);
void loadRefPattern();
void write_csv(string name, vector<double> avg, vector<double> stdev);
void write_conf_mat_csv(int conf_mat[][4]);
void read_csv(string name, vector<double>& dst_avg, vector<double>& dst_stdev);
vector<string> csv_read_row(string& line, char delimiter);
vector<string> csv_read_row(istream& in, char delimiter);

double getAvg(vector<double> v);
double min3(double a, double b, double c);
int arg_min3(double a, double b, double c);
double error(double a, double b);

void outer_product(vector<double> row, vector<double> col, vector<vector<double>>& dst);
//computes row[i] - val for all i;
void subtract(vector<double> row, double val, vector<double>& dst);
//computes m[i][j] + m2[i][j]
void add(vector<vector<double>> m, vector<vector<double>> m2, vector<vector<double>>& dst);
double mean(vector<double>& data);
void scale(vector<vector<double>>& d, double alpha);
void compute_covariance_matrix(vector<vector<double>>& d, vector<vector<double>>& dst);
double compute_deviation(vector<double> d, double avg);
// 참고: https://stackoverflow.com/questions/23301451/c-pca-calculating-covariance-matrix/51403295

double getDeterminant(const vector<vector<double>> vect);
vector<vector<double>> getTranspose(const vector<vector<double>> matrix1);
vector<vector<double>> getCofactor(const vector<vector<double>> vect);
vector<vector<double>> getInverse(const vector<vector<double>> vect);
void printMatrix(const vector<vector<double>> vect);
// 참고: https://stackoverflow.com/questions/60300482/c-calculating-the-inverse-of-a-matrix
double statistical_distance(double a, double avg, double stdev);

vector<string> get_files_inDirectory(const string& _path, const string& _filter);

// string class_names[4] = { "circle","star","triangle","square" };
//string class_names[4] = { "star","circle","triangle","square" };
//vector<vector<double>> ref_avg(4);
//vector<vector<double>> ref_stdev(4);
//
//double rotation = 1;

extern vector<string> class_names;
extern vector<vector<double>> ref_avg;
extern vector<vector<double>> ref_stdev;
extern double rotation;
extern string train_dir;
extern string test_dir;
extern bool isLoaded;

#endif