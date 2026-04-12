#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

// ===== Drawing Data =====
vector<vector<Point>> strokes;
vector<Point> currentStroke;
bool isDrawing = false;

// ===== Better Finger Detection =====
Point detectFinger(Mat &frame) {
    Mat hsv, mask, blurImg;

    // Convert to HSV (better than grayscale for segmentation)
    cvtColor(frame, hsv, COLOR_BGR2HSV);

    // Simple skin color range (basic but better than thresholding gray)
    Scalar lower(0, 30, 60);
    Scalar upper(20, 150, 255);
    inRange(hsv, lower, upper, mask);

    // Reduce noise
    GaussianBlur(mask, blurImg, Size(7,7), 0);
    erode(blurImg, blurImg, Mat(), Point(-1,-1), 2);
    dilate(blurImg, blurImg, Mat(), Point(-1,-1), 2);

    vector<vector<Point>> contours;
    findContours(blurImg, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    if (contours.empty()) return Point(-1, -1);

    int largest = 0;
    double maxArea = contourArea(contours[0]);

    for (int i = 1; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        if (area > maxArea) {
            maxArea = area;
            largest = i;
        }
    }

    if (maxArea < 1000) return Point(-1, -1); // filter noise

    Rect box = boundingRect(contours[largest]);

    // fingertip approximation (top-middle of hand blob)
    return Point(box.x + box.width / 2, box.y);
}

// ===== Main =====
int main() {
    VideoCapture cap(0);
    if (!cap.isOpened()) return -1;

    Mat frame;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        flip(frame, frame, 1);

        Point finger = detectFinger(frame);

        // ===== Drawing Logic =====
        if (finger.x != -1 && finger.y != -1) {
            circle(frame, finger, 8, Scalar(0,255,0), FILLED);

            if (!isDrawing) {
                currentStroke.clear();
                isDrawing = true;
            }

            currentStroke.push_back(finger);

        } else {
            if (isDrawing && !currentStroke.empty()) {
                strokes.push_back(currentStroke);
                currentStroke.clear();
                isDrawing = false;
            }
        }

        // ===== Render Strokes =====
        for (const auto &stroke : strokes) {
            for (size_t i = 1; i < stroke.size(); i++) {
                line(frame, stroke[i-1], stroke[i], Scalar(255,0,0), 2);
            }
        }

        imshow("Air Doodle", frame);

        char key = (char)waitKey(1);
        if (key == 27) break;       // ESC
        if (key == 'c') strokes.clear(); // clear screen
    }

    return 0;
}
