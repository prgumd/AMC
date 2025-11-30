
import cv2
import numpy as np
import pupil_apriltags
from pupil_apriltags import Detection

def get_pos_with_april_tag(detection: Detection) -> tuple[np.ndarray, np.ndarray]:
    """
    Processes an AprilTag detection to compute the camera-to-world translation (T_wc)
    and rotation (R_wc).
    Assumes that the AprilTag is vertically mounted in the world, and returns poses
    with z upwards.

    This function uses a fixed transformation (R_wa) from the AprilTag (or "a")
    coordinate frame to a world (or "w") coordinate frame. The transformation is
    derived by inverting the detected pose.

    Args:
        detection (Detection): An AprilTag detection object that contains the pose
                               information (translation and rotation).

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - T_wc: The translation vector from the world frame to the camera frame.
            - R_wc: The rotation matrix from the world frame to the camera frame.
    """
    # Assume the tag is mounted vertically in the world frame
    # In tag frame, Z comes into the tag, X goes to the right, Y goes down
    # In world frame, Z goes up, X goes into the tag, Y goes left
    R_wa = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    t_ca_c = np.array(detection.pose_t).flatten()
    R_ca = np.array(detection.pose_R)

    R_wc = R_wa @ R_ca.T

    T_ac = -R_ca.T @ t_ca_c

    t_wc = R_wa @ T_ac

    return t_wc, R_wc

class AprilPose(object):
    def __init__(self, K, family="tagStandard41h12", marker_size_m=(4.5 * 2.54 / 100)):
        # Hard code a bunch of defaults for now, they are not important
        family = family
        threads = 8
        max_hamming = 0
        decimate = 1
        blur = 0.8
        refine_edges = True
        debug = False

        # Camera intrinsics
        self._K = K

        # self._detector = apriltag(family, threads, max_hamming, decimate, blur, refine_edges, debug) #max_hamming debug
        # self._detector = apriltag(family, threads)#, max_hamming, decimate, blur, refine_edges, debug)
        self._detector = pupil_apriltags.Detector(family, threads)  # , decimate, blur, refine_edges)
        self.tag_size = marker_size_m

    def find_tags(self, frame_gray) -> list[Detection]:

        camera_params = (self._K[0, 0], self._K[1, 1], self._K[0, 2], self._K[1, 2])

        detections = self._detector.detect(
            frame_gray,
            estimate_tag_pose=True,
            camera_params=camera_params,
            tag_size=self.tag_size,
        )
        return detections # type: ignore

    def find_detection(self, detections, id):
        for i, detection in enumerate(detections):
            if detection["id"] == id:
                return detection
        return None

    def draw_detections(self, frame, detections: list[pupil_apriltags.Detection], scale=1.0):
        for detection in detections:
            pts = detection.corners.reshape((-1, 1, 2)).astype(np.int32) # type: ignore
            center = tuple(detection.center.astype(np.int32)) # type: ignore

            pts = (pts * scale).astype(np.int32)
            frame = cv2.polylines(
                frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2
            )
            # cv2.circle(
            #     frame, tuple(detection.center.astype(np.int32)), 7, (0, 0, 255), -1
            # )
            top_left = tuple(pts[0][0])  # First corner
            top_right = tuple(pts[1][0])  # Second corner
            bottom_right = tuple(pts[2][0])  # Third corner
            bottom_left = tuple(pts[3][0])  # Fourth corner
            cv2.line(frame, top_left, bottom_right, color=(0, 0, 255), thickness=2)
            cv2.line(frame, top_right, bottom_left, color=(0, 0, 255), thickness=2)
