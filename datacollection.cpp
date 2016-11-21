#include <opencv.hpp>
#include<core/core.hpp>
#include <features2d/features2d.hpp>
#include <highgui/highgui.hpp>
#include <ctime>
#include <Windows.h>
#include <vector>
#include <numeric>
#include <iostream>
#include <fstream>
#include <cvaux.h>


using namespace cv;
using namespace std;



float ratio = 1;   //resize ratio
RNG rng(12345);
Mat img;
int flag = 0;


void getaccAvr(Mat &src, Mat &dst) {
	double dHeight = src.rows;
	double dWidth = src.cols;
	int avrpage = 10;      //영향을 줄 프레임 장 수
	double alpha = 1.0 / avrpage;   //1.0초를 프레임 개수로 나눔
	double beta = 1.0 - alpha;

	Mat sum = Mat::zeros(dHeight, dWidth, CV_32FC3);

	addWeighted(src, alpha, dst, beta, 0, dst);
	dst.convertTo(dst, CV_8UC3);
	//sum.copyTo(dst);
	return;
}


class dis_feature {
	Point pt;
public:
	float point_dist(Point x_, Point y_);
	float point_dist(Point2f x_, Point2f y_);
	float point_dist(Point2f x_, Point y_);
	float mean_point(vector<float> a);
	float mean_point(float *arr);
	float std_dev_point(vector<float> a);
	float dev_from_mat(Mat src);
	Point mean_point(Mat &img);
	Point getmean();
};

Point dis_feature::getmean() {
	return pt;
}

float dis_feature::point_dist(Point x_, Point y_) {
	return sqrt((x_.x - y_.x)*(x_.x - y_.x) + (x_.y - y_.y)*(x_.y - y_.y));
}
float dis_feature::point_dist(Point2f x_, Point2f y_) {
	return sqrt((x_.x - y_.x)*(x_.x - y_.x) + (x_.y - y_.y)*(x_.y - y_.y));
}

float dis_feature::point_dist(Point2f x_, Point y_) {
	return sqrt((x_.x - y_.x)*(x_.x - y_.x) + (x_.y - y_.y)*(x_.y - y_.y));
}

float dis_feature::mean_point(vector<float> a) {
	return std::accumulate(a.begin(), a.end(), 0.0f) / a.size();
}

float dis_feature::mean_point(float *arr) {
	float mean = 0;
	int size = sizeof(arr) / sizeof(float);
	for (int i = 0; i < size; i++) {
		mean += arr[i];
	}
	return mean / size;

}

float dis_feature::std_dev_point(vector<float> a) {
	float mean = mean_point(a);
	vector<float> diff(a.size());
	std::transform(a.begin(), a.end(), diff.begin(), [mean](double x) { return x - mean; });
	float sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
	return std::sqrt(sq_sum / a.size());

}

float dis_feature::dev_from_mat(Mat src) {
	vector<vector<Point> > cont;
	vector<Vec4i> hier;
	vector<float> dist;
	Mat img;
	int cnt = 0;
	src.copyTo(img);
	float rst;
	Point mean = mean_point(img);

	cv::Canny(img, img, 50, 200);
	cv::findContours(img, cont, hier, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	for (int i = 0; i < cont.size(); i++) {
		for (int j = 0; j < cont[i].size(); j++) {
			dist.push_back(point_dist(mean, cont.at(i).at(j)));
			cnt++;
		}
	}
	//cout << cnt << endl;

	rst = std_dev_point(dist);
	return rst;
}

Point dis_feature::mean_point(Mat &img) {
	vector<Point> nonZero;
	Point p;


	findNonZero(~img, nonZero);
	Point sum = std::accumulate(nonZero.begin(), nonZero.end(), p);
	Point mean(sum*(1.0f / nonZero.size()));
	pt = mean;
	return mean;
}

class CircleFeature {
	Mat Cir[5];
public:
	void makeCircle();
	int rtCirnum(Mat src, int num);
	void rtFullCirnum(Mat src, vector<int> &num);
	void getCir(int num, Mat &dst);
	void CircleByMat(Mat &src);
};

void CircleFeature::CircleByMat(Mat &src) {
	dis_feature dis;
	Point mean = dis.mean_point(src);

	for (int i = 0; i < 5; i++) {
		Cir[i] = Mat::zeros(200, 200, CV_8UC1);
		Cir[i] = ~Cir[i];
		circle(Cir[i], mean, 90 - 20 * i, Scalar(0, 0, 0), 13);
	}
	return;
}

void CircleFeature::makeCircle() {
	for (int i = 0; i < 5; i++) {
		Cir[i] = Mat::ones(200, 200, CV_8UC1);
		circle(Cir[i], Point(100, 100), 90 - 20 * i, Scalar(0, 0, 0), 13);
	}
	return;
}
int CircleFeature::rtCirnum(Mat src, int num) {
	Mat temp;
	cv::resize(src, src, Size(200, 200));
	cv::bitwise_or(src, Cir[num], temp);
	return 200 * 200 - countNonZero(temp);
}
void CircleFeature::rtFullCirnum(Mat src, vector<int> &num) {
	for (int i = 0; i < 5; i++) {
		num.push_back(rtCirnum(src, i));
	}
	return;
}
void CircleFeature::getCir(int num, Mat &dst) {
	Cir[num].copyTo(dst);
	return;
}
////////////////////////////////////////////////////////////////
class location {
	char data;
	Point2f pt;//(mean value)
	int rnk;
public:
	char get_data();
	Point get_pt();
	int get_rnk();
	void set_data(char c);
	void set_pt(Rect rt);
	void set_pt(Point2f pnt);
	void set_rnk(vector<Point2f> vpt);
};

char location::get_data() {
	return data;
}
Point location::get_pt() {
	return pt;
}
int location::get_rnk() {
	return rnk;
}
void location::set_data(char c) {
	data = c;
	return;
}
void location::set_pt(Rect rt) {
	pt.x = rt.x + 0.5*rt.width;
	pt.y = rt.y + 0.5*rt.height;
	return;
}

void location::set_pt(Point2f pnt) {
	pt = pnt;
	return;
}

void location::set_rnk(vector<Point2f> vpt) {
	for (int i = 0; i < vpt.size(); i++) {
		if (vpt[i] == pt) {
			rnk = i;
			return;
		}
	}
	return;
}

/*void rect2pt(vector<Point2f> &vpt, Rect rt) {
Point2f pt;
pt.x = rt.x + 0.5*rt.width;
pt.y = rt.y + 0.5*rt.height;
vpt.push_back(pt);
return;
}*/

vector<Point2f> get_rnk_pt(vector<Point2f> vpt) {
	vector<double> vdouble;
	double a = -0.2;
	vector<pair<double, int>> item;
	vector<Point2f> dst;
	for (int i = 0; i < vpt.size(); i++) {
		item.push_back(std::make_pair(vpt[i].y - a*vpt[i].x, vpt[i].x));
	}
	std::sort(item.begin(), item.end());
	for (int i = 0; i < item.size(); i++) {
		dst.push_back(Point2f(item[i].second, item[i].first + a*item[i].second));
	}
	return dst;
}

void print_loc(Mat &img, vector<Point2f> pt) {
	for (int i = 0; i < pt.size(); i++) {
		putText(img, to_string(i), pt[i], FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 0, 0));
	}
	return;
}

////////////////////////////////////////////////////////////////

void CallBackFunc(int event, int x, int y, int flags, void* userdata) {
	if (event == EVENT_LBUTTONDOWN) {
		cout << "position :" << x << "," << y << endl;
		//cout << "RGB :" << img.at<Vec3b>(y, x)[0]<<","<< img.at<Vec3b>(y, x)[1]<< "," << img.at<Vec3b>(y, x)[2]<<endl;
	}
}

class sixnineFeature {
	bool isNine(Mat src);
};

bool sixnineFeature::isNine(Mat src) {
	Rect roi(0, 0, src.rows, src.cols / 2);
	Mat sub_img = src(roi);

	if (countNonZero(~sub_img) / countNonZero(~src) > 0.5)
		return true;
	return false;
}


void padding(Mat &src) {
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	Mat standard;


	double max = 0;
	int j = 0;
	float rows = src.size().height;
	float cols = src.size().width;
	if (rows > cols) {
		float ratio = 200 / rows;
		resize(src, src, Size(cols*ratio, ratio*rows));
		src = ~src;
		copyMakeBorder(src, src, 1, 1, ((200 - ratio*cols) / 2) + 1, ((200 - ratio*cols) / 2) + 1, BORDER_CONSTANT);
		src = ~src;
	}
	else if (rows < cols) {
		float ratio = 200 / cols;
		resize(src, src, Size(ratio*cols, ratio*rows));
		src = ~src;
		copyMakeBorder(src, src, ((200 - ratio*rows) / 2) + 1, ((200 - ratio*rows) / 2) + 1, 1, 1, BORDER_CONSTANT);
		src = ~src;
	}
	else if (rows == cols) {
		resize(src, src, Size(200, 200));
	}

	//imshow("before", src);
	src.copyTo(standard);
	Canny(standard, standard, 30, 100);
	standard = ~standard;
	erode(standard, standard, Mat());
	findContours(standard, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	for (int i = 0; i < contours.size(); i++) {
		if (max < contourArea(contours[i], false)) {
			max = contourArea(contours[i], false);
			j = i;
		}
	}
	drawContours(standard, contours, j, Scalar(255), CV_FILLED);
	bitwise_or(standard, src, standard);
	//imshow("andyeonsan", standard);

	return;
}
double stdDev(int* array, int size) {
	int i, sum = 0;
	double avg = 0, total = 0;

	for (i = 0; i < size; i++)
		sum += array[i];
	avg = (double)sum / size;

	for (i = 0; i < size; i++)
		total += pow(avg - array[i], 2);
	total /= size;

	return sqrt(total);
}



int* radial_line_detection(Mat src) {
	
	int result[4];
	for (int i = 0; i < 4; i++) {
		result[i] = 0;
	}
	resize(src, src, Size(200, 200));
	imshow("radialimg", src);
	/*   if (linenum < 0 || (linenum&(linenum - 1)) != 0) {
	cout << "wrong numofline input" << endl;
	return;
	}
	*/
	//세로선으로 확인
	int blackpixel1 = 0;
	int j = src.size().width / 2;
	bool* data = (bool*)src.data;
	for (int i = 0; i < src.size().height; i++) {
		if (data[i*src.cols + j] == 0) {
			blackpixel1++;
		}
	}
	//cout << "horizontal : " << blackpixel1 << endl;

	//가로선으로 확인
	int blackpixel2 = 0;
	int i = src.size().height / 2;
	for (j = 0; j < src.size().width; j++) {
		if (data[i*src.cols + j] == 0) {
			blackpixel2++;
		}
	}
	//   cout << "vertical : " << blackpixel2 << endl;

	//y=x 직선으로 확인

	int blackpixel3 = 0;
	for (int i = 0; i < src.size().height; i++) {
		if (data[i*src.cols + i] == 0) {
			blackpixel3++;
		}
	}

	//y=-x+200 직선으로 확인
	int blackpixel4 = 0;
	for (int i = 0; i < src.size().height; i++) {
		if (data[i*src.cols + (src.size().width - i)] == 0) {
			blackpixel4++;
		}
	}

	result[0] = blackpixel1;
	result[1] = blackpixel2;
	result[2] = blackpixel3;
	result[3] = blackpixel4;

	//cout << "count : " << result[0] << "\t" << result[1] << "\t" << result[2] << "\t" << result[3] << endl;
	//myCSVfile << result[0] << "," << result[1] << "," << result[2] << "," << result[3] << ",";

	return result;
}



char decision_making(char c,Mat standard) {
	ofstream myCSVfile;
	myCSVfile.open("data.csv", ios::ate | ios::app);

	dis_feature dis_calcul;
	CircleFeature cir_calcul;
	cir_calcul.makeCircle();
	vector<int> vcir;

	double decision[11];

	for (int i = 0; i < 11; i++) {
		decision[i] = 0;
	}

	float rotatedrect = float(standard.cols) / float(standard.rows);
	if (rotatedrect < 0.3)
		decision[0] = 1;
	if (rotatedrect >= 0.8)
		decision[1] = 1;

	myCSVfile << c << "," << rotatedrect;
	
	padding(standard);
	cout << "running" << endl;
	float dev = dis_calcul.dev_from_mat(standard);
	if (dev >= 29)
		decision[2] = 1;
	cir_calcul.CircleByMat(standard);
	cir_calcul.rtFullCirnum(standard, vcir);
	
	myCSVfile  << ","<<dev << "," << vcir[0] << "," << vcir[1] << "," << vcir[2] << "," << vcir[3] << "," << vcir[4] <<",";

	if (vcir[1] - vcir[0] >= 1800)
		decision[3] = 1;
	if (vcir[0] - vcir[4] >= 1300)
		decision[4] = 1;
	if (vcir[0] - vcir[4] < 200)
		decision[5] = 1;
	if (vcir[0] - vcir[3] >= 700)
		decision[6] = 1;
	if (vcir[2] - vcir[4] >= 3200)
		decision[7] = 1;
	if (vcir[4] < 10)
		decision[8] = 1;

	int* radial = radial_line_detection(standard);
	double std_dev = stdDev(radial, 4);

	if (std_dev > 30)
		decision[9] = 1;
	if (std_dev < 14)
		decision[10] = 1;

	//cout << "circle:" << vcir[0] << " " << vcir[1] << " " << vcir[2] << " " << vcir[3] << " " << vcir[4] << endl;
	myCSVfile<<std_dev << endl;
	double id =
		decision[0] * pow(2, 10) +
		decision[1] * pow(2, 9) +
		decision[2] * pow(2, 8) +
		decision[3] * pow(2, 7) +
		decision[4] * pow(2, 6) +
		decision[5] * pow(2, 5) +
		decision[6] * pow(2, 4) +
		decision[7] * pow(2, 3) +
		decision[8] * pow(2, 2) +
		decision[9] * pow(2, 1) +
		decision[10];
	for (int i = 0; i < 11; i++) {
		//cout << decision[i];
	}
	/*
	int erosion_size = 3;
	Mat element = getStructuringElement(cv::MORPH_ELLIPSE,
		cv::Size(3 * erosion_size + 1, 3 * erosion_size + 1),
		cv::Point(erosion_size, erosion_size));

	cout << "here"<< endl;
	dilate(standard,standard,element);
	erode(standard, standard, element);
	dilate(standard, standard, element);
	erode(standard, standard, element);

	vector<KeyPoint> keyPoints;
	static Ptr<FastFeatureDetector>fast = FastFeatureDetector::create(13f0);
	fast->detect(standard, keyPoints);
	drawKeypoints(standard, keyPoints, standard, Scalar::all(255), DrawMatchesFlags::DRAW_OVER_OUTIMG);
	imshow("FAST feature", standard);
	waitKey(0);
	*/




	/*
	if (decision[0] == 1) return '1';
	else if (decision[2] == 1 && (decision[7] != 1 || decision[8] != 0) && (decision[6] != 1 || decision[10] != 0)) return'7';
	else if (decision[3] == 1 && decision[8] != 1 && decision[6] != 1 && decision[10] != 1) return '6';
	else if (decision[4] == 1 && decision[9] == 1) return '2';
	else if (decision[8] == 1) return '0';
	else if (decision[6] == 1 && decision[10] == 1 && vcir[3] - vcir[4] >500) return'8';
	else if (decision[6] == 1 && decision[10] == 0 && vcir[0]<2600) return '3';
	else if (decision[4] == 1 && decision[9] == 0) return '5';
	else if (decision[5] == 1) return '/';


	else if (decision[7] == 1 && decision[8] == 0) return '4';
	else if (decision[1] == 1 && (decision[7] != 1 || decision[8] != 1)) return'+';

	else return'n';

	*/

	return 'n';
}






int main() {
	cout << "input block : ";
	char name;
	cin >> name;
	VideoCapture cap(0);
	if (!cap.isOpened())return 0;

	vector<Mat> RoiSet;
	Mat sub, draw_sub;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	Rect rect;
	Mat diff;
	uchar *myData;
	dis_feature dis_calcul;
	CircleFeature cir_calcul;
	cir_calcul.makeCircle();
	vector<int> vcir;
	Mat whiteBalance;
	int avr = 100;

	////////////////////////////////
	/*int hlow = 0, hhigh = 255;
	int slow = 0, shigh = 255;
	int vlow = 0, vhigh = 255;
	int canlow = 0, canhigh = 255;*/
	////////////////////////////////


	while (1) {
		while (1)//차영상 이용해서 break
		{
			Mat avrimg;

			cap >> avrimg;

			for (int i = 0; i < avr; i++) {
				cap >> img;
				//if (img.empty())goto out;
				resize(avrimg, avrimg, Size(), ratio, ratio);//이미지 리사이징
				getaccAvr(img, avrimg);//중첩영상 만들기
			}
			if (!flag) {
				whiteBalance = avrimg;
				flag = !flag;
				avr = 10;
			}//처음 한번만 밸런스 매트릭스를 만든다



			int height = 0, width = 0;
			int _stride = 0;
			float cnt1 = 0, wide = 0;
			Mat imgry, avrgry;


			imshow("balance_mat", whiteBalance);
			cvtColor(img, imgry, CV_BGR2GRAY);
			cvtColor(avrimg, avrgry, CV_BGR2GRAY);
			absdiff(imgry, avrgry, diff);

			threshold(diff, diff, 20, 255, THRESH_BINARY);

			//imshow("diff", diff);
			height = diff.rows;
			width = diff.cols;
			wide = height*width;
			_stride = diff.step;
			myData = diff.data;
			for (int i = 0; i < height; i++)
			{
				for (int j = 0; j < width; j++)
				{
					uchar val = myData[i * _stride + j];
					if (val)cnt1++;
				}
			}
			//cout << cnt1 << " " << wide << " " << cnt1 / wide << endl;
			if (cnt1 / wide < 0.01) {
				avrimg.copyTo(img);
				break;
			}
			if (waitKey(1) == 27)goto out;
		}
		Mat temp;
		cv::scaleAdd(whiteBalance, -0.1, img, img);
		absdiff(whiteBalance, img, img);
		imshow("balance img", img);

		////////////////////////trackbar//////////////////////
		/*namedWindow("track", 1);
		createTrackbar("Hlow_value", "track", &canlow, 255);
		createTrackbar("Hhigh_value", "track", &canhigh, 255);
		createTrackbar("Slow_value", "track", &slow, 255);
		createTrackbar("Shigh_value", "track", &shigh, 255);
		createTrackbar("Vlow_value", "track", &vlow, 255);
		createTrackbar("Vhigh_value", "track", &vhigh, 255);*/
		//////////////////////////////////////////////////////

		img.copyTo(sub);

		GaussianBlur(img, img, Size(3, 3), 0.8);
		//cvtColor(img, img, COLOR_BGR2HSV);
		//inRange(img, Scalar(hlow, slow, vlow), Scalar(hhigh, shigh, vhigh), img);
		//cvtColor(img, img, COLOR_BGR2GRAY);
		Canny(img, temp, 30, 200);

		//threshold(~img, img, 100, 255, THRESH_BINARY);

		findContours(temp, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

		draw_sub = Mat::zeros(temp.size(), CV_8UC3);
		vector<Point> approxShape;
		for (size_t i = 0; i < contours.size(); i++) {
			approxPolyDP(contours[i], approxShape, arcLength(Mat(contours[i]), true)*0.04, true);
			drawContours(draw_sub, contours, i, Scalar(255, 255, 255), CV_FILLED);
		}
		imshow("draw_sub", draw_sub);
		Canny(draw_sub, temp, 0, 255);
		imshow("temp2", temp);
		findContours(temp, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
		/*Mat temp_;
		Mat element = getStructuringElement(cv::MORPH_GRADIENT,
		cv::Size(3, 3),
		cv::Point(3, 3));
		draw_sub.copyTo(temp_);
		erode(temp_, temp_, element);
		absdiff(draw_sub, temp_, draw_sub);
		findContours(draw_sub, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
		*/

		vector<RotatedRect> minRect(contours.size());
		vector<RotatedRect> minEllipse(contours.size());

		setMouseCallback("src", CallBackFunc, NULL);

		RoiSet.clear();
		for (int i = 0; i < contours.size(); i++) {
			minRect[i] = minAreaRect(Mat(contours[i]));
			if (contours[i].size()>5) {
				minEllipse[i] = fitEllipse(Mat(contours[i]));
			}
		}
		int j = 0;
		vector<float> width_length;
		vector<location> loc;
		vector<Point2f> cen_for_loc;
		system("cls");
		for (int i = 0; i < contours.size(); i++) {

			Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			width_length.clear();
			float div = 0;
			Point2f rect_point[4];
			Point2f center;
			minRect[i].points(rect_point);
			Mat M, rotated;
			location tmp;
			float angle = minRect[i].angle;
			Size Rrect_size = minRect[i].size;

			if (minRect[i].angle < -45.) {
				angle += 90.0;
				swap(Rrect_size.width, Rrect_size.height);
			}

			M = getRotationMatrix2D(minRect[i].center, angle, 1.0);
			warpAffine(draw_sub, rotated, M, draw_sub.size(), INTER_CUBIC);
			getRectSubPix(rotated, Rrect_size, minRect[i].center, rotated);
			char num[10];
			sprintf(num, "#%d img", j);

			for (int j = 0; j < 4; j++) {
				width_length.push_back(dis_calcul.point_dist(rect_point[j], rect_point[(j + 1) % 4]));
			}
			div = width_length[0] < width_length[1] ? width_length[0] / width_length[1] : width_length[1] / width_length[0];

			if ((rotated.cols * rotated.rows>2000))//&& (div>0.5))// && (div < 0.9))
			{
				for (int j = 0; j < 4; j++) {
					line(sub, rect_point[j], rect_point[(j + 1) % 4], Scalar(255, 0, 0));
					center.x += rect_point[j].x;
					center.y += rect_point[j].y;
				}
				center.x /= 4;
				cvtColor(rotated, rotated, COLOR_BGR2HSV);
				inRange(rotated, Scalar(0, 0, 0), Scalar(255, 255, 125), rotated);
				//imshow("rotated", rotated);
				//rotated = ~rotated;

				//cout << num << " : " << dis_calcul.dev_from_mat(rotated) << endl;
				cir_calcul.CircleByMat(rotated);
				cir_calcul.rtFullCirnum(rotated, vcir);
				vcir.clear();

				tmp.set_pt(center);
				cen_for_loc.push_back(center);
				///////
				//class에 값 집어넣는 과정
				char decided = decision_making(name,rotated);
				
				tmp.set_data(decided);
				///////
				loc.push_back(tmp);

				imshow(num, rotated);

				cout << num << "::" << decided << endl;
				j++;
			}

		}

		for (int k = 0; k < loc.size(); k++) {
			loc[k].set_rnk(get_rnk_pt(cen_for_loc));

		}
		print_loc(sub, get_rnk_pt(cen_for_loc));
		//imshow("img", img);
		imshow("src", sub);

		if (waitKey(5) == 27)break;
	}
out:
	destroyAllWindows();
	return 0;
}
