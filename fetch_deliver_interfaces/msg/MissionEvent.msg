# Discrete mission event, consumed by the metrics recorder.
std_msgs/Header header
string event                      # e.g. "object_detected", "attach", "attach_rejected", "delivered"
string phase
string detail                     # free text / JSON payload
float64 value                     # optional numeric payload (e.g. an error in metres)
