#include "stdafx.h"
#include "patternClassification.h"
#include <cstdlib>

vector<string> class_names;
vector<vector<double>> ref_avg;
vector<vector<double>> ref_stdev;
string train_dir;
string test_dir;
double rotation;
bool isLoaded;

void initial_setting()
{
	isLoaded = false;
	train_dir = string("./shapes/train/");
	test_dir = string("./shapes/test/");
	class_names = vector<string>(4);
	class_names[0] = "star";
	class_names[1] = "circle";
	class_names[2] = "triangle";
	class_names[3] = "square";
	ref_avg = vector<vector<double>>(4);
	ref_stdev = vector<vector<double>>(4);
	rotation = double(1);
}
Mat toRed(Mat src)
{
	for (int y = 0; y < src.rows; y++)
	{
		unsigned char* ptr1 = src.data + 3 * (src.cols * y);
		unsigned char* resultptr = src.data + 3 * (src.cols * y);
		for (int x = 0; x < src.cols; x++)
		{
			// 이렇게 RGB값을 조정하여 그 범위 안에 있는 Rgb 픽셀값에 단색을 넣었다.
			//200 -> 160 -> 110
			//그림자
			//배경색을 초록으로 해도 결과가 바뀌므로 함부로 손대지 말자
			ptr1[3 * x + 0] = 0;
			ptr1[3 * x + 1] = 0;
			ptr1[3 * x + 2] = ptr1[3 * x + 2];
		}
	}
	return src;
}
Mat toGray(Mat src)
{
	Mat dst(src.size(), CV_8UC1);
	for (int y = 0; y < src.rows; y++)
	{
		unsigned char* ptr1 = src.data + 3 * (src.cols * y);
		unsigned char* dst_ptr = dst.data + (dst.cols * y);
		for (int x = 0; x < src.cols; x++)
		{
			unsigned char gray_value = (ptr1[3 * x + 0] + ptr1[3 * x + 1] + ptr1[3 * x + 2]) / 3;
			dst_ptr[x] = gray_value;
		}
	}
	return dst;
}
Mat Otsu(Mat img)
{
	int size = img.rows * img.cols;
	int histogram[256] = { 0, };
	for (int y = 0; y < img.rows; y++)
	{
		unsigned char* ptr1 = img.data + (img.cols * y);
		for (int x = 0; x < img.cols; x++)
		{
			unsigned char value = ptr1[x];
			histogram[value]++;
		}
	}
	double norm_histogram[256] = { 0, };
	for (int i = 0; i < 256; i++)
	{
		norm_histogram[i] = (double)histogram[i] / size;
	}
	double u(0);
	for (int i = 0; i < 256; i++)
	{
		u += i * norm_histogram[i];
	}

	double w[256];
	w[0] = norm_histogram[0];
	double u0[256], u1[256];
	u0[0] = 0.0;
	double v[256] = { 0, };
	for (int t = 1; t < 256; t++)
	{
		w[t] = w[t - 1] + norm_histogram[t];
		if (w[t] == 0)
			u0[t] = 0;
		else
			u0[t] = (w[t - 1] * u0[t - 1] + t * norm_histogram[t]) / w[t];
		u1[t] = (u - w[t] * u0[t]) / (1 - w[t]);

		v[t] = w[t] * (1 - w[t]) * (u0[t] - u1[t]) * (u0[t] - u1[t]);
	}
	int T(0);
	double v_max(0);
	for (int t = 0; t < 256; t++)
	{
		if (v_max < v[t])
		{
			v_max = v[t];
			T = t;
		}
	}
	for (int y = 0; y < img.rows; y++)
	{
		unsigned char* ptr1 = img.data + (img.cols * y);
		for (int x = 0; x < img.cols; x++)
		{
			if (ptr1[x] >= T)
			{
				ptr1[x] = 255;
			}
			else
			{
				ptr1[x] = 0;
			}
		}
	}
	return img;
}
Mat dilation(Mat img)
{
	Mat dst_img = img.clone();

	int dx[9] = { -1, -1, -1, 0, 0, 0, 1, 1, 1 };
	int dy[9] = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };

	for (int y = 0; y < dst_img.rows; y++)
	{
		unsigned char* ptr1 = img.data + (img.cols * y);
		for (int x = 0; x < dst_img.cols; x++)
		{
			if (ptr1[x] == 0)
				continue;

			for (int i = 0; i < 9; i++)
			{
				int nx = x + dx[i];
				int ny = y + dy[i];
				if (nx < 0 || ny < 0 || nx >= img.cols || ny >= img.rows)
					continue;
				unsigned char* dst_ptr = dst_img.data + (dst_img.cols * ny);
				dst_ptr[nx] = 255;
			}
		}
	}
	return dst_img;
}
Mat erosion(Mat img)
{
	Mat dst_img(img.size(), CV_8UC1);
	//img.clone();

	int dx[9] = { -1, -1, -1, 0, 0, 0, 1, 1, 1 };
	int dy[9] = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };

	for (int y = 0; y < dst_img.rows; y++)
	{
		unsigned char* ptr1 = img.data + (img.cols * y);
		unsigned char* dst_ptr = dst_img.data + (dst_img.cols * y);
		for (int x = 0; x < dst_img.cols; x++)
		{
			bool all = true;
			if (ptr1[x] == 0)
			{
				dst_ptr[x] = 0;
				all = false;
			}
			else
			{
				for (int i = 0; i < 9; i++)
				{
					int nx = x + dx[i];
					int ny = y + dy[i];
					if (nx < 0 || ny < 0 || nx >= img.cols || ny >= img.rows)
						continue;
					unsigned char* temp_ptr = img.data + (img.cols * ny);
					if (temp_ptr[nx] == 0)
					{
						all = false;
						break;
					}
				}
			}
			if (all == true)
				dst_ptr[x] = 255;
			else
				dst_ptr[x] = 0;
		}
	}
	return dst_img;
}
int LUT_BLabeling8[8][8] = {
	//   0, 1, 2, 3, 4, 5, 6, 7
		{0, 0, 0, 0, 0, 0, 0, 0 }, // 0
		{0, 0, 0, 0, 0, 1, 0, 0 }, // 1
		{0, 0, 0, 0, 0, 1, 1, 0 }, // 2
		{0, 0, 0, 0, 0, 1, 1, 1 }, // 4
		{1, 0, 0, 0, 0, 1, 1, 1 }, // 3
		{1, 1, 0, 0, 0, 1, 1, 1 }, // 5
		{1, 1, 1, 0, 0, 1, 1, 1 }, // 6
		{1, 1, 1, 1, 0, 1, 1, 1 }, // 7
};
int num_region[100000];
// int labelnumber = 1;
Mat Contour_tracing_based_method(Mat binimg, vector<pair<int, int>>& contour_points)
{
	int labelnumber = 0;
	Mat labImage(binimg.size(), CV_16UC1, Scalar(0));
	Mat contourImage(binimg.size(), CV_8UC1, Scalar(0));
	for (int i = 0; i < 100000; i++) num_region[i] = 0;
	for (int i = 0; i < binimg.rows; i++)
	{
		for (int j = 0; j < binimg.cols; j++)
		{


			unsigned char cur_p = *(binimg.data + (binimg.cols * (i)) + (j));
			int cur_l = *(labImage.data + (labImage.cols * (i)) + (j));
			if (cur_l > 0)
				continue;
			if (cur_p > 127)	//object
			{
				unsigned char ref_p1 = *(labImage.data + (labImage.cols * (i)) + (j - 1));
				unsigned char ref_p2 = *(labImage.data + (labImage.cols * (i - 1)) + (j - 1));
				if (ref_p1 > 0) // propagation
				{
					num_region[ref_p1]++;
					*(labImage.data + (labImage.cols * (i)) + (j)) = ref_p1;
				}
				else if ((ref_p1 == 0) && (ref_p2 > 0)) // hole
				{
					num_region[ref_p2]++;
					*(labImage.data + (labImage.cols * (i)) + (j)) = ref_p2;
					vector<pair<int, int> > points_v = BTracing8(i, j, ref_p2, 0, binimg, labImage, contourImage);
					if (contour_points.size() < points_v.size()) contour_points = points_v;
				}
				else if ((ref_p1 == 0) && (ref_p2 == 0)) // region start
				{
					labelnumber++;
					num_region[labelnumber]++;
					*(labImage.data + (labImage.cols * (i)) + (j)) = labelnumber;
					vector<pair<int, int> > points_v = BTracing8(i, j, labelnumber, 1, binimg, labImage, contourImage);
					if (contour_points.size() < points_v.size()) contour_points = points_v;
				}
			}
			else *(labImage.data + (labImage.cols * (i)) + (j)) = 0;
		}
	}
	// cout << "label number " << labelnumber << '\n';
	return contourImage;
}
void calCoord(int i, int* y, int* x)
{
	switch (i)
	{
	case 0: *x = *x + 1; break;
	case 1: *y = *y + 1; *x = *x + 1; break;
	case 2: *y = *y + 1; break;
	case 3: *y = *y + 1; *x = *x - 1; break;
	case 4: *x = *x - 1; break;
	case 5: *y = *y - 1; *x = *x - 1; break;
	case 6: *y = *y - 1; break;
	case 7: *y = *y - 1; *x = *x + 1; break;
	}
}
void read_neighbor8(int y, int x, unsigned char neighbor8[8], Mat img)
{
	int dy[8] = { 0, 1, 1, 1, 0, -1, -1, -1 };
	int dx[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };

	int nx;
	int ny;
	for (int i = 0; i < 8; i++)
	{
		ny = y + dy[i];
		nx = x + dx[i];

		if (nx < 1 || ny < 1 || nx >= img.cols - 1 || ny >= img.rows - 1)
		{
			neighbor8[i] = 0;
		}
		else
		{
			neighbor8[i] = *(img.data + (img.cols * (ny)) + (nx));
		}
	}
	//neighbor8[0] = *(img.data + (img.cols * (y)) + (x + 1));
	//neighbor8[1] = *(img.data + (img.cols * (y + 1)) + (x + 1));
	//neighbor8[2] = *(img.data + (img.cols * (y + 1)) + (x));
	//neighbor8[3] = *(img.data + (img.cols * (y + 1)) + (x - 1));
	//neighbor8[4] = *(img.data + (img.cols * (y)) + (x - 1));
	//neighbor8[5] = *(img.data + (img.cols * (y - 1)) + (x - 1));
	//neighbor8[6] = *(img.data + (img.cols * (y - 1)) + (x));
	//neighbor8[7] = *(img.data + (img.cols * (y - 1)) + (x + 1));
}
vector<pair<int, int>> BTracing8(int y, int x, int label, bool tag, Mat img, Mat& labImage, Mat& contourImage)
{
	int cur_orient = 0;
	int pre_orient = 0;
	int dy[8] = { 0, 1, 1, 1, 0, -1, -1, -1 };
	int dx[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };
	int ny, nx;
	//unsigned char neighbor8[8];
	if (tag == 1)
	{
		int pre_orient = 0;
		int cur_orient = pre_orient;
	}
	else
	{
		int pre_orient = 6;
		int cur_orient = pre_orient;
	}
	int pre_x = x;
	int end_x = pre_x;
	int pre_y = y;
	int end_y = pre_y;
	vector<pair<int, int> > points_v;
	do
	{


		//read_neighbor8(y, x, neighbor8, img);
		int start_o = (8 + cur_orient - 2) % 8;
		int add_o, okay(false);
		int i;
		for (i = 0; i < 8; i++)
		{
			add_o = (start_o + i) % 8;
			//if (neighbor8[add_o] != 0) break;
			ny = y + dy[add_o];
			nx = x + dx[add_o];
			if (nx < 0 || ny < 0 || nx >= img.cols || ny >= img.rows)
				continue;
			uchar next_val = *(img.data + (img.cols * (ny)) + (nx));
			if (next_val != 0)
			{
				okay = true;
				break;
			}
		}
		if (!okay)
		{
			//cout << "이웃 없음\n";
			break;
		}
		if (i < 8)
		{
			calCoord(add_o, &y, &x);
			cur_orient = add_o;
		}
		if (LUT_BLabeling8[pre_orient][cur_orient])
		{
			num_region[label]++;
			*(labImage.data + (labImage.cols * (pre_y)) + (pre_x)) = label;
		}
		*(contourImage.data + (contourImage.cols * (pre_y)) + (pre_x)) = 255;

		points_v.push_back({ x, y });

		pre_x = x;
		pre_y = y;
		pre_orient = cur_orient;
	} while ((y != end_y) || (x != end_x));
	return points_v;
}
vector<double> getLCSarray(vector<pair<int, int>> contour_points)
{
	vector<double> LCSarray;
	//int window_size = 9;	// 1, 2만 나옴
	//int window_size = 11;	// 3만 나옴
	//int window_size = 13; // 3졸래강세
	int window_size = 19; // 1, 2, (3)강세
	//int window_size = 21; // 1, 2, 3	강세
	//int window_size = 23; // 1, 2, 3 강세 0빼고 완벽.
	//int window_size = 25; // 1, 2, 3 강세 0빼고 완벽.
	//int window_size = 27; // 1, 2		강세시작
	//int window_size = 35;	// 1, 2 강세
	//int window_size = 23;


	//int window_size = (contour_points.size() / 35)*2 +1;
	//cout << "window_size : " << window_size << '\n';

	for (int i = 0; i < contour_points.size(); i++)
	{
		pair<int, int> point = contour_points[i];
		int f_index = i - (window_size - 1) / 2;
		int e_index = i + (window_size - 1) / 2;
		if (f_index < 0)
			f_index += (int)contour_points.size(); // 0 1 2 3 4 5 6
		if (e_index >= contour_points.size())
			e_index -= (int)contour_points.size();
		pair<int, int> front = contour_points[f_index];
		pair<int, int> end = contour_points[e_index];
		//y = mx + b 형태의 직선의 방정식
		double m = ((double)(end.second - front.second) / (double)(end.first - front.first));
		double b = (double)front.second - m * (double)front.first;

		//d = | mx + b - y | / root( m^2 + 1 )
		double d;
		if (isfinite(m))
			d = abs(m * point.first + b - point.second) / sqrt(m * m + 1);
		else
			d = abs((double)point.first - end.first);
		LCSarray.push_back(d);
	}

	// 가장 큰 점의 index를 찾는다.
	double max_point = 0;
	int shift_index = 0;
	for (int i = 0; i < LCSarray.size(); i++)
	{
		if (LCSarray[i] > max_point)
		{
			max_point = LCSarray[i];
			shift_index = i;
		}
	}
	vector<double> shiftedLCSarray(LCSarray.size());

	// 가장 큰 점이 처음에 오도록 shift
	for (int i = 0; i < shiftedLCSarray.size(); i++)
	{
		if ((i + shift_index) >= LCSarray.size())
			shiftedLCSarray[i] = LCSarray[i + shift_index - LCSarray.size()];
		else
			shiftedLCSarray[i] = LCSarray[i + shift_index];
	}

	nomalization(shiftedLCSarray);
	return shiftedLCSarray;
}
vector<double> featureExtraction(Mat src)
{
	Mat img_bin = Otsu(toGray(src));
	Mat opening = dilation(erosion(img_bin));
	vector<pair<int, int>> contour_points;
	Mat opening_contour = Contour_tracing_based_method(opening, contour_points);
	/*Mat closing = erosion(dilation(img_bin));
	vector<pair<int, int>> contour_points;
	Mat closing_contour = Contour_tracing_based_method(closing, contour_points);*/
	vector<double> h_array = getLCSarray(contour_points);
	return h_array;
}
void nomalization(vector<double>& pattern)
{
	double stdev = compute_deviation(pattern, getAvg(pattern));
	for (int i = 0; i < pattern.size(); i++)
		pattern[i] = pattern[i] / stdev;
}
double DTW(vector<double> A, vector<double> B, vector<pair<int, int>>& path)
{
	unsigned int A_size((unsigned int)A.size()), B_size((unsigned int)B.size());

	double S(INFINITY);
	// 2차원 배열 동적할당
	double** D; int** G;
	D = new double* [A_size]; G = new int* [A_size];
	for (unsigned int i = 0; i < A_size; i++)
	{
		D[i] = new double[B_size]; G[i] = new int[B_size];
	}

	// initialization
	D[0][0] = d(A[0], B[0]); G[0][0] = 0;
	for (unsigned int j = 1; j < B_size; j++)
	{
		D[0][j] = D[0][j - 1] + d(A[0], B[j]);
		G[0][j] = 2;
	}
	for (unsigned int i = 1; i < A_size; i++)
		//D[i][0] = INFINITY;
		D[i][0] = D[i - 1][0] + d(A[i], B[0]);

	// Forward
	for (uint i = 1; i < A_size; i++)
		for (uint j = 1; j < B_size; j++)
		{
			D[i][j] = d(A[i], B[j]) + min3(D[i - 1][j], D[i][j - 1], D[i - 1][j - 1]);
			G[i][j] = arg_min3(D[i - 1][j], D[i][j - 1], D[i - 1][j - 1]);
		}
	// Backward
	int i(A_size - 1), j(B_size - 1);
	int k = 1;
	//vector<pair<int, int>> temp_path;
	while ((i != 0) && (j != 0))
	{
		// p[k] = G[i][j];
		path.push_back({ i, j });
		switch (G[i][j])
		{
		case 1: i--; k++; break;
		case 2: j--; k++; break;
		case 3: i--; j--; k++; break;
		}
	}
	path.push_back({ 0, 0 });
	// Termination
	S = D[A_size - 1][B_size - 1] / k;
	//cout << "k : " << k << '\n';

	// 2차원 배열 동적할당 해제
	for (unsigned int i = 0; i < A_size; i++)
	{
		delete[] D[i];	delete[] G[i];
	}
	delete[] D; delete[] G;

	return S;
}
double SDTW(vector<double> A, vector<double> ref_avg, vector<double> ref_stdev)
{
	unsigned int A_size((unsigned int)A.size()), ref_size((unsigned int)ref_avg.size());

	double S(INFINITY);
	// 2차원 배열 동적할당
	double** D; int** G;
	D = new double* [A_size]; G = new int* [A_size];
	for (unsigned int i = 0; i < A_size; i++)
	{
		D[i] = new double[ref_size]; G[i] = new int[ref_size];
	}

	// initialization
	D[0][0] = statistical_distance(A[0], ref_avg[0], ref_stdev[0]); G[0][0] = 0;
	for (unsigned int j = 1; j < ref_size; j++)
	{
		// D[0][j] = D[0][j - 1] + d(A[0], A_rotation[j]);
		D[0][j] = D[0][j - 1] + statistical_distance(A[0], ref_avg[j], ref_stdev[j]);
		G[0][j] = 2;
	}
	for (unsigned int i = 1; i < A_size; i++)
		D[i][0] = INFINITY;

	// Forward
	for (uint i = 1; i < A_size; i++)
		for (uint j = 1; j < ref_size; j++)
		{
			// D[i][j] = d(A[i], A_rotation[j]) + min3(D[i - 1][j], D[i][j - 1], D[i - 1][j - 1]);
			D[i][j] = statistical_distance(A[i], ref_avg[j], ref_stdev[j]) + min3(D[i - 1][j], D[i][j - 1], D[i - 1][j - 1]);
			G[i][j] = arg_min3(D[i - 1][j], D[i][j - 1], D[i - 1][j - 1]);
		}
	// Backward
	int i(A_size - 1), j(ref_size - 1);
	int k = 1;
	while ((i != 0) && (j != 0))
	{
		switch (G[i][j])
		{
		case 1: i--; k++; break;
		case 2: j--; k++; break;
		case 3: i--; j--; k++; break;
		}
	}
	// Termination
	S = D[A_size - 1][ref_size - 1]/ k;

	// 2차원 배열 동적할당 해제
	for (unsigned int i = 0; i < A_size; i++)
	{
		delete[] D[i];	delete[] G[i];
	}
	delete[] D; delete[] G;


	return S;
}

int classification(Mat src)
{
	if (!isLoaded)
	{
		loadRefPattern();
		isLoaded = true;
	}
	int estimated_label;
	vector<double> pattern = featureExtraction(src);
	double min_dissimilarity(INFINITY);
	for (int i = 0; i < 4; i++)
	{
		double dissimilarity = SDTW(pattern, ref_avg[i], ref_stdev[i]);
		cout << class_names[i] << "      \tscore : " << dissimilarity << '\n';
		if (min_dissimilarity > dissimilarity)
		{
			min_dissimilarity = dissimilarity;
			estimated_label = i;
		}
	}
	return estimated_label;
}

void learningAllData(vector<vector<Mat>> img_memory)
{
	for (int i = 0; i < 4; i++)
		learningFeature(img_memory[i], class_names[i]);
}
void learningFeature(vector<Mat> img_list, string filename)
{
	cout << "learning start " << '\n';
	assert(img_list.size());

	// 임의의 패턴을 평균 패턴으로 초기화.
	vector<double> randPattern = featureExtraction(img_list[0]);
	vector<double> avgPattern(randPattern.size());
	for (int i = 0; i < avgPattern.size(); i++)
		avgPattern[i] = randPattern[i] ? randPattern[i] : (rand()/(double)RAND_MAX)*10;

	// img_list로 부터 LCS 패턴 리스트 추출
	vector<vector<double>> patternList;
	for (int i = 0; i < img_list.size(); i++)
		patternList.push_back(featureExtraction(img_list[i]));


	// 평균 패턴과 매치되는 값들을 저장할 리스트
	vector<vector<double>> avgPattern_matching_values;
	vector<double> pattern_deviation(avgPattern.size());

	int iteration = 1;
	int changed_cnt = 5;
	// 평균 패턴이 변하지 않을 때 까지 iteration을 증가시키며 평균 패턴 학습.
	while (changed_cnt)
	{
		if (changed_cnt <= 3 || iteration > 200)
			break;
		if (iteration > 70 && changed_cnt < 13)
			break;
		if (iteration > 30 && changed_cnt < 6)
			break;

		// DTW로 평균 패턴과 그들 패턴각각의 매칭 위치
		avgPattern_matching_values = vector<vector<double>>(avgPattern.size());
		for (int i = 0; i < patternList.size(); i++)
		{
			vector<pair<int, int>> matchingPath;
			DTW(avgPattern, patternList[i], matchingPath);
			for (int j = 0; j < matchingPath.size(); j++)
				avgPattern_matching_values[matchingPath[j].first].push_back(patternList[i][matchingPath[j].second]);
		}


		// 평균 패턴 갱신
		changed_cnt = 0;
		for (int i = 0; i < avgPattern.size(); i++)
		{
			double new_avg_val = getAvg(avgPattern_matching_values[i]);
			//if (error(avgPattern[i], new_avg_val) > 0.5) // 백분율 오차 0.8% 이내
			if (error(avgPattern[i], new_avg_val)) // 백분율 오차 0.8% 이내
			{
				avgPattern[i] = new_avg_val;
				changed_cnt++;
			}
		}
		//pattern_deviation = vector<double>(avgPattern.size());
		//for (int i = 0; i < avgPattern.size(); i++)
		//	pattern_deviation[i] = compute_deviation(avgPattern_matching_values[i], avgPattern[i]);

		cout << filename << "\t iteration : " << iteration++ << ",  \tchanged : " << changed_cnt << '\n';// << ", \tavg of stdev : " << avgOfDeviation << '\n';
	}
	for (int i = 0; i < avgPattern.size(); i++)
		pattern_deviation[i] = compute_deviation(avgPattern_matching_values[i], avgPattern[i]);
	// compute_covariance_matrix(avgPattern_matching_values, covar_mat);
	// csv 파일로 학습한 결과 저장.

	write_csv(filename, avgPattern, pattern_deviation);
}

void loadRefPattern()
{
	for (int i = 0; i < 4; i++)
		read_csv(class_names[i], ref_avg[i], ref_stdev[i]);
}
void write_csv(string name, vector<double> avg, vector<double> stdev)
{
	ofstream myfile;
	myfile.open(name + ".csv");
	for (double val : avg)
		myfile << val << ",";
	myfile << "\n";
	for (double val : stdev)
		myfile << val << ",";
	myfile << "\n";
	myfile.close();
}
void write_conf_mat_csv(int conf_mat[][4])
{
	ofstream myfile;
	myfile.open("confusion_matrix.csv");
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			myfile << conf_mat[i][j] << ",";
		}
		myfile << "\n";
	}
	myfile.close();
}
void read_csv(string name, vector<double>& dst_avg, vector<double>& dst_stdev)
{
	ifstream myfile;
	myfile.open(name + ".csv");
	if (myfile.fail())
	{
		cout << "해당 경로에 학습된 패턴 데이터가 존재하지 않습니다." << '\n';
		return;
	}
	vector<vector<string>> data;
	string line;
	double weight = 1.0;
	if (name == "star")
		weight = 1.7;
	else if (name == "circle")
		weight = 1.0;
	else if (name == "triangle")
		weight = 1.04;
	else if (name == "square")
		weight = 0.98;
	cout << "가중치 : " << weight << '\n';
	while (myfile.good())
	{
		vector<string> row = csv_read_row(myfile, ',');
		data.push_back(row);
	}
	for (int i = 0; i < data[0].size() - 1; i++)
		dst_avg.push_back(stod(data[0][i]));
	for (int i = 0; i < data[1].size() - 1; i++)
		dst_stdev.push_back(stod(data[1][i])* weight);
}
vector<string> csv_read_row(string& line, char delimiter)
{
	stringstream ss(line);
	return csv_read_row(ss, delimiter);
}
vector<string> csv_read_row(istream& in, char delimiter)
{
	stringstream ss;
	bool inquotes = false;
	vector<string> row;//relying on RVO
	while (in.good())
	{
		char c = in.get();
		if (!inquotes && c == '"') //beginquotechar
		{
			inquotes = true;
		}
		else if (inquotes && c == '"') //quotechar
		{
			if (in.peek() == '"')//2 consecutive quotes resolve to 1
			{
				ss << (char)in.get();
			}
			else //endquotechar
			{
				inquotes = false;
			}
		}
		else if (!inquotes && c == delimiter) //end of field
		{
			row.push_back(ss.str());
			ss.str("");
		}
		else if (!inquotes && (c == '\r' || c == '\n'))
		{
			if (in.peek() == '\n') { in.get(); }
			row.push_back(ss.str());
			return row;
		}
		else
		{
			ss << c;
		}
	}
}
double getAvg(vector<double> v)
{
	double total(0);
	for (int i = 0; i < v.size(); i++)
		total += v[i];
	return total / (double)v.size();
}
double min3(double a, double b, double c)
{
	double retVal = (a > b) ? b : a;
	retVal = (retVal > c) ? c : retVal;
	return retVal;
}
int arg_min3(double a, double b, double c)
{
	double minVal = min3(a, b, c);
	if (minVal == c)
		return 3;
	else if (minVal == b)
		return 2;
	return 1;
}
double error(double a, double b)
{
	return (abs(a - b) / a) * 100;
}
void outer_product(vector<double> row, vector<double> col, vector<vector<double>>& dst)
{
	for (unsigned i = 0; i < row.size(); i++)
	{
		for (unsigned j = 0; j < col.size(); j++)
		{
			dst[i][j] = row[i] * col[j];
		}
	}
}
void subtract(vector<double> row, double val, vector<double>& dst) {
	for (unsigned i = 0; i < row.size(); i++) {
		dst[i] = row[i] - val;
	}
}
void add(vector<vector<double>> m, vector<vector<double>> m2, vector<vector<double>>& dst)
{
	for (unsigned i = 0; i < m.size(); i++)
	{
		for (unsigned j = 0; j < m[i].size(); j++)
		{
			dst[i][j] = m[i][j] + m2[i][j];
		}
	}
}
double mean(vector<double>& data) {
	double mean = 0.0;

	for (unsigned i = 0; (i < data.size()); i++) {
		mean += data[i];
	}

	mean /= data.size();
	return mean;
}
void scale(vector<vector<double>>& d, double alpha) {
	for (unsigned i = 0; i < d.size(); i++) {
		for (unsigned j = 0; j < d[i].size(); j++) {
			d[i][j] *= alpha;
		}
	}
}
void compute_covariance_matrix(vector<vector<double>>& d, vector<vector<double>>& dst)
{
	for (unsigned i = 0; i < d.size(); i++)
	{
		double y_bar = mean(d[i]);
		vector<double> d_d_bar(d[i].size());
		subtract(d[i], y_bar, d_d_bar);
		vector<vector<double>> t(d.size());
		for (int i = 0; i < t.size(); i++)
			t[i] = vector<double>(d.size());
		outer_product(d_d_bar, d_d_bar, t);
		add(dst, t, dst);
	}
	scale(dst, 1 / (d.size() - 1));
}
double compute_deviation(vector<double> d, double avg)
{
	double square_total = 0.0;
	for (double val : d)
		square_total += pow((val - avg), 2);
	return sqrt(square_total / (double)d.size());
}
double getDeterminant(const vector<vector<double>> vect) {
	if (vect.size() != vect[0].size()) {
		throw runtime_error("Matrix is not quadratic");
	}
	int dimension = vect.size();

	if (dimension == 0) {
		return 1;
	}

	if (dimension == 1) {
		return vect[0][0];
	}

	//Formula for 2x2-matrix
	if (dimension == 2) {
		return vect[0][0] * vect[1][1] - vect[0][1] * vect[1][0];
	}

	double result = 0;
	int sign = 1;
	for (int i = 0; i < dimension; i++) {

		//Submatrix
		vector<vector<double>> subVect(dimension - 1, vector<double>(dimension - 1));
		for (int m = 1; m < dimension; m++) {
			int z = 0;
			for (int n = 0; n < dimension; n++) {
				if (n != i) {
					subVect[m - 1][z] = vect[m][n];
					z++;
				}
			}
		}

		//recursive call
		result = result + sign * vect[0][i] * getDeterminant(subVect);
		sign = -sign;
	}

	return result;
}
vector<vector<double>> getTranspose(const vector<vector<double>> matrix1)
{

	//Transpose-matrix: height = width(matrix), width = height(matrix)
	vector<vector<double>> solution(matrix1[0].size(), vector<double>(matrix1.size()));

	//Filling solution-matrix
	for (size_t i = 0; i < matrix1.size(); i++) {
		for (size_t j = 0; j < matrix1[0].size(); j++) {
			solution[j][i] = matrix1[i][j];
		}
	}
	return solution;
}
vector<vector<double>> getCofactor(const vector<vector<double>> vect) {
	if (vect.size() != vect[0].size()) {
		throw runtime_error("Matrix is not quadratic");
	}

	vector<vector<double>> solution(vect.size(), vector<double>(vect.size()));
	vector<vector<double>> subVect(vect.size() - 1, vector<double>(vect.size() - 1));

	for (size_t i = 0; i < vect.size(); i++) {
		for (size_t j = 0; j < vect[0].size(); j++) {

			int p = 0;
			for (size_t x = 0; x < vect.size(); x++) {
				if (x == i) {
					continue;
				}
				int q = 0;

				for (size_t y = 0; y < vect.size(); y++) {
					if (y == j) {
						continue;
					}

					subVect[p][q] = vect[x][y];
					q++;
				}
				p++;
			}
			solution[i][j] = pow(-1, i + j) * getDeterminant(subVect);
		}
	}
	return solution;
}
vector<vector<double>> getInverse(const vector<vector<double>> vect) {
	if (getDeterminant(vect) == 0) {
		throw runtime_error("Determinant is 0");
	}

	double d = 1.0 / getDeterminant(vect);
	vector<vector<double>> solution(vect.size(), vector<double>(vect.size()));

	for (size_t i = 0; i < vect.size(); i++) {
		for (size_t j = 0; j < vect.size(); j++) {
			solution[i][j] = vect[i][j];
		}
	}

	solution = getTranspose(getCofactor(solution));

	for (size_t i = 0; i < vect.size(); i++) {
		for (size_t j = 0; j < vect.size(); j++) {
			solution[i][j] *= d;
		}
	}

	return solution;
}
void printMatrix(const vector<vector<double>> vect) {
	for (size_t i = 0; i < vect.size(); i++) {
		//cout << vect[i].size() << " " ;
		for (size_t j = 0; j < vect[i].size(); j++) {
			cout << setw(8) << vect[i][j] << " ";
		}
		cout << "\n";
	}
	cout << "\n";
}
double statistical_distance(double a, double avg, double stdev)
{
	if (stdev == 0.0)
		stdev = 0.0005;
	//cout << "stdev : " << stdev << ", ";
	//cout << " a = " << a << " 일 때, stdev" << stdev << ", 거리 : " << (abs(a - avg) / stdev) << "\n";
	return (abs(a - avg) / stdev);
}

vector<string> get_files_inDirectory(const string& _path, const string& _filter)
{
	string searching = _path + _filter;
	cout << "searching" << searching << '\n';
	vector<string> returnVal;

	_finddata_t fd;
	intptr_t handle = _findfirst(searching.c_str(), &fd); //현재 폴더 내 모든 파일 찾기

	if (handle == -1) return returnVal;

	int result = 0;
	do
	{
		returnVal.push_back(fd.name);
		result = _findnext(handle, &fd);
	} while (result != -1);
	_findclose(handle);
	return returnVal;
}