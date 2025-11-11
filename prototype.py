import cv2 as cv
import mediapipe as mp
import time

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    refine_landmarks=True,  # penting agar hasil akurat
    max_num_faces=2,
    min_detection_confidence=0.5
)
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


def estimasi_pandangan(face_landmarks, image_w, image_h):
    """Estimasi arah pandangan berdasarkan perbedaan posisi hidung dan mata"""
    nose = face_landmarks.landmark[1]
    left_eye = face_landmarks.landmark[33]
    right_eye = face_landmarks.landmark[263]

    nose_x = int(nose.x * image_w)
    left_x = int(left_eye.x * image_w)
    right_x = int(right_eye.x * image_w)

    center_face_x = (left_x + right_x) // 2
    diff_x = nose_x - center_face_x
    threshold = 13  # semakin besar → lebih toleran

    if diff_x > threshold:
        return "Menoleh Kiri", diff_x
    elif diff_x < -threshold:
        return "Menoleh Kanan", diff_x
    else:
        return "Menghadap Depan", diff_x


cap = cv.VideoCapture(0)

face_miss_start = None
gaze_away_since = None
frame_count = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    h, w, _ = frame.shape
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    fd_results = face_detection.process(rgb)
    fm_results = face_mesh.process(rgb)
    pose_results = pose.process(rgb)


    alerts = []
    num_faces = len(fd_results.detections) if fd_results.detections else 0
    face_present = num_faces > 0

    # Wajah hilang
    if not face_present:
        if face_miss_start is None:
            face_miss_start = time.monotonic()
        elif time.monotonic() - face_miss_start > 2.0:
            alerts.append("Face Missing >2s")
            gaze_away_since = None
    else:
        face_miss_start = None

    # Arah pandangan 
    gaze_status = "Unknown"
    if fm_results.multi_face_landmarks:
        fm = fm_results.multi_face_landmarks[0]
        gaze_status, diff_x = estimasi_pandangan(fm, w, h)

        # logic anti-cheating: jika menoleh kiri/kanan >3s
        if gaze_status != "Menghadap Depan":
            if gaze_away_since is None:
                gaze_away_since = time.monotonic()
            elif time.monotonic() - gaze_away_since > 3.0:
                alerts.append("Looking Away >3s")
        else:
            gaze_away_since = None

    pose_status = "Unknown"
    if pose_results.pose_landmarks:
        lm = pose_results.pose_landmarks.landmark
        left_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = lm[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP]

        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_y = (left_hip.y + right_hip.y) / 2

        
        if shoulder_y < 0.5 and hip_y >= 0.5:
            pose_status = "Standing"
        elif shoulder_y >= 0.5:
            pose_status = "Sitting"
        else:
            pose_status = "Unknown"
    elif not face_present:
        pose_status = "Away"



    #Multi-face alert
    if num_faces > 1:
        alerts.append("Multiple Faces Detected")

    # Tampilkan hasil
    cv.putText(frame, f"Faces: {num_faces}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv.putText(frame, f"Gaze: {gaze_status}", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv.putText(frame, f"Pose: {pose_status}", (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    if alerts:
        cv.putText(frame, " | ".join(alerts), (10, 130), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    cv.imshow("AntiCheat", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
