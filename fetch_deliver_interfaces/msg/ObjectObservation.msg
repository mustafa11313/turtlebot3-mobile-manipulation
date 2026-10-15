# A single detected object, localized in a fixed frame.
std_msgs/Header header            # frame of `pose` (normally "map" or "odom")
string label                      # class label, e.g. "red_cube"
float32 confidence                # detector confidence in [0, 1]
geometry_msgs/Pose pose           # estimated object center
float32 range                     # distance from the camera [m]
float32 bearing                   # horizontal angle in the camera frame [rad], + = left
uint32[4] bbox                    # x, y, width, height in pixels
bool is_target                    # true if label matches the mission target
