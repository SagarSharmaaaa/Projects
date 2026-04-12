#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

// ===== Drawing Data =====
vector<vector<Point>> strokes;
vector<Point> currentStroke;
bool isDrawing = false;

// Pre-created kernel (avoids realloc every frame)
Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));

// Optional: reduce resolution for speed
const double SCALE = 0.6;

// ===== Fast Finger Detection =====
Point detectFinger(Mat &frame) {
    Mat hsv, mask;

    // Resize for speed (VERY IMPORTANT)
    resize(frame, frame, Size(), SCALE, SCALE);

    cvtColor(frame, hsv, COLOR_BGR2HSV);

    // Skin mask
    Scalar lower(0, 30, 60);
    Scalar upper(20, 150, 255);
    inRange(hsv, lower, upper, mask);

    // Faster than GaussianBlur chain
    morphologyEx(mask, mask, MORPH_OPEN, kernel);

    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    if (contours.empty()) return Point(-1, -1);

    // FAST: single pass + early rejection
    int bestIdx = -1;
    double bestArea = 0;

    for (int i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        if (area > bestArea) {
            bestArea = area;
            bestIdx = i;
        }
    }

    if (bestArea < 1200) return Point(-1, -1);

    Rect box = boundingRect(contours[bestIdx]);

    // fingertip estimate
    return Point(box.x + box.width / 2, box.y);
}

// ===== Main =====
int main() {
    VideoCapture cap(0);

    // Try lowering camera buffer lag (important for FPS feel)
    cap.set(CAP_PROP_BUFFERSIZE, 1);

    if (!cap.isOpened()) return -1;

    Mat frame;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        flip(frame, frame, 1);

        Point finger = detectFinger(frame);

        // scale correction (because we resized inside function)
        finger.x = (int)(finger.x / SCALE);
        finger.y = (int)(finger.y / SCALE);

        if (finger.x >= 0 && finger.y >= 0) {
            circle(frame, finger, 8, Scalar(0,255,0), FILLED);

            if (!isDrawing) {
                currentStroke.clear();
                isDrawing = true;
            }

            currentStroke.push_back(finger);
        }
        else {
            if (isDrawing && !currentStroke.empty()) {
                strokes.push_back(currentStroke);
                currentStroke.clear();
                isDrawing = false;
            }
        }

        // Draw strokes
        for (const auto &stroke : strokes) {
            for (size_t i = 1; i < stroke.size(); i++) {
                line(frame, stroke[i-1], stroke[i], Scalar(255,0,0), 2);
            }
        }

        imshow("Air Doodle", frame);

        char key = (char)waitKey(1);
        if (key == 27) break;
        if (key == 'c') strokes.clear();
    }

    return 0;
}
