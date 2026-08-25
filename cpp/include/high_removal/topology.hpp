// MediaPipe FaceMesh 478 点拓扑索引，与 highlight_removal/face_landmarks.py 完全一致。
#pragma once

#include <vector>

namespace hr::topo {

inline const std::vector<int> FACE_OVAL = {
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
};
inline const std::vector<int> LEFT_EYE = {33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7};
inline const std::vector<int> RIGHT_EYE = {362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382};
inline const std::vector<int> LEFT_BROW = {70, 63, 105, 66, 107, 55, 65, 52, 53, 46};
inline const std::vector<int> RIGHT_BROW = {336, 296, 334, 293, 300, 285, 295, 282, 283, 276};
inline const std::vector<int> OUTER_LIPS = {
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    409, 270, 269, 267, 0, 37, 39, 40, 185,
};
inline const std::vector<int> INNER_LIPS = {78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191};
inline const std::vector<int> NOSE_BRIDGE = {168, 6, 197, 195, 5, 4};
inline const std::vector<int> NOSE_TIP = {1, 4, 5, 19, 94, 195, 197, 2};
inline const std::vector<int> NOSE_WING = {49, 98, 97, 2, 326, 327, 279};
inline const std::vector<int> MOUTH_CORNERS = {61, 291};
inline const std::vector<int> CHIN_AREA = {152, 148, 176, 149, 150, 136, 172, 58, 172, 378, 400, 377, 152, 365, 379};
inline const std::vector<int> LEFT_LOWER_EYE = {33, 7, 163, 144, 145, 153, 154, 155, 133};
inline const std::vector<int> RIGHT_LOWER_EYE = {362, 382, 381, 380, 374, 373, 390, 249, 263};

}  // namespace hr::topo
