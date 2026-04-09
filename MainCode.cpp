#include <opencv2/opencv.hpp>
#include <vector>
using namespace cv;
using namespace std;

// ===== Drawing Data =====
vector<vector<Point>> strokes;
vector<Point> currentStroke;
bool isDrawing = false;

// ===== Simple Finger Tracking (Basic) =====
Point detectFinger(Mat &frame) {
    Mat gray, blurImg, thresh;
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, blurImg, Size(7,7), 0);
    threshold(blurImg, thresh, 60, 255, THRESH_BINARY_INV);

    vector<vector<Point>> contours;
    findContours(thresh, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    if (contours.empty()) return Point(-1, -1);

    int largest = 0;
    for (int i = 1; i < contours.size(); i++) {
        if (contourArea(contours[i]) > contourArea(contours[largest]))
            largest = i;
    }

    Rect box = boundingRect(contours[largest]);
    return Point(box.x + box.width/2, box.y); // fingertip approx
}

// ===== Main =====
int main() {
    VideoCapture cap(0);
    if (!cap.isOpened()) return -1;

    Mat frame;

    while (true) {
        cap >> frame;
        flip(frame, frame, 1);

        Point finger = detectFinger(frame);

        // ===== Drawing Logic =====
        if (finger.x != -1) {
            circle(frame, finger, 10, Scalar(0,255,0), -1);

            if (!isDrawing) {
                currentStroke.clear();
                isDrawing = true;
            }

            currentStroke.push_back(finger);
        } else {
            if (isDrawing) {
                strokes.push_back(currentStroke);
                isDrawing = false;
            }
        }

        // ===== Render Strokes =====
        for (auto &stroke : strokes) {
            for (int i = 1; i < stroke.size(); i++) {
                line(frame, stroke[i-1], stroke[i], Scalar(255,0,0), 2);
            }
        }

        imshow("Air Doodle", frame);

        char key = waitKey(1);
        if (key == 27) break; // ESC to exit
        if (key == 'c') strokes.clear(); // clear
    }

    return 0;
}