# Periodic status of the fetch & deliver mission.
std_msgs/Header header
string phase                      # EXPLORE, APPROACH, ALIGN, PICK, DELIVER, PLACE, RETURN, DONE, FAILED
string active_behavior            # leaf of the behavior tree currently RUNNING
float64 elapsed                   # seconds since mission start
uint32 pick_attempts
uint32 nav_failures
uint32 recoveries
bool object_attached
