#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

// ===== Drawing Data =====
vector<vector<Point>> strokes;
vector<Point> currentStroke;
bool isDrawing = false;

Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
const double SCALE = 0.6;

// ===== Fast Finger Detection (NOW RETURNS AREA) =====
Point detectFinger(Mat &frame, double &areaOut) {
    Mat hsv, mask, small;

    resize(frame, small, Size(), SCALE, SCALE);

    cvtColor(small, hsv, COLOR_BGR2HSV);

    Scalar lower(0, 30, 60);
    Scalar upper(20, 150, 255);
    inRange(hsv, lower, upper, mask);

    morphologyEx(mask, mask, MORPH_OPEN, kernel);

    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    if (contours.empty()) {
        areaOut = 0;
        return Point(-1, -1);
    }

    int bestIdx = -1;
    double bestArea = 0;

    for (int i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        if (area > bestArea) {
            bestArea = area;
            bestIdx = i;
        }
    }

    if (bestArea < 1200) {
        areaOut = 0;
        return Point(-1, -1);
    }

    areaOut = bestArea;

    Rect box = boundingRect(contours[bestIdx]);

    // fingertip estimate
    return Point(box.x + box.width / 2, box.y);
}

// ===== Main =====
int main() {
    VideoCapture cap(0);
    cap.set(CAP_PROP_BUFFERSIZE, 1);

    if (!cap.isOpened()) return -1;

    Mat frame;

    double smoothArea = 0;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        flip(frame, frame, 1);

        double handArea = 0;
        Point finger = detectFinger(frame, handArea);

        // scale correction
        finger.x = (int)(finger.x / SCALE);
        finger.y = (int)(finger.y / SCALE);

        // ===== Smooth area (IMPORTANT) =====
        smoothArea = 0.8 * smoothArea + 0.2 * handArea;

        // ===== THICKNESS BASED ON DISTANCE =====
        int thickness;

        if (smoothArea > 6000) thickness = 6;       // very close
        else if (smoothArea > 3500) thickness = 4;  // close
        else if (smoothArea > 2000) thickness = 2;  // medium
        else thickness = 1;                         // far

        if (finger.x >= 0 && finger.y >= 0) {
            circle(frame, finger, 8, Scalar(0,255,0), FILLED);

            if (!isDrawing) {
                currentStroke.clear();
                isDrawing = true;
            }

            // small filtering to reduce noise
            if (currentStroke.empty() ||
                norm(finger - currentStroke.back()) > 5) {
                currentStroke.push_back(finger);
            }
        }
        else {
            if (isDrawing && !currentStroke.empty()) {
                strokes.push_back(currentStroke);
                currentStroke.clear();
                isDrawing = false;
            }
        }

        // ===== DRAW STROKES =====
        for (const auto &stroke : strokes) {
            for (size_t i = 1; i < stroke.size(); i++) {
                line(frame, stroke[i-1], stroke[i], Scalar(255,0,0), thickness);
            }
        }

        putText(frame,
                "Thickness: " + to_string(thickness),
                Point(10,30),
                FONT_HERSHEY_SIMPLEX,
                0.7,
                Scalar(255,255,255),
                2);

        imshow("Air Doodle", frame);

        char key = (char)waitKey(1);
        if (key == 27) break;
        if (key == 'c') strokes.clear();
    }

    return 0;
}
