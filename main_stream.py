import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run SUTrack on webcam stream.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of config file.')
    parser.add_argument('--device', type=int, default=0, help='Webcam device id.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level for tracker.')
    args = parser.parse_args()

    from lib.test.evaluation.tracker import Tracker
    import cv2

    # Initialize tracker
    tracker = Tracker(args.tracker_name, args.tracker_param, dataset_name='webcam')
    params = tracker.get_parameters()
    params.debug = args.debug
    tracker_instance = tracker.create_tracker(params)

    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        print(f'Unable to open webcam {args.device}')
        return

    ret, frame = cap.read()
    if not ret:
        print('Failed to read frame from webcam')
        return

    display_name = 'Webcam'
    cv2.namedWindow(display_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(display_name, 960, 720)

    # Select initial ROI
    frame_disp = frame.copy()
    cv2.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.2, (0, 0, 0), 1)
    cv2.imshow(display_name, frame_disp)
    x, y, w, h = cv2.selectROI(display_name, frame_disp, fromCenter=False)
    init_state = [x, y, w, h]
    tracker_instance.initialize(frame, {'init_bbox': init_state})

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        out = tracker_instance.track(frame)
        state = [int(s) for s in out['target_bbox']]
        cv2.rectangle(frame, (state[0], state[1]),
                      (state[0] + state[2], state[1] + state[3]),
                      (0, 255, 0), 2)
        cv2.putText(frame, 'Tracking - press q to quit, r to reset', (20, 30),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1)
        cv2.imshow(display_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            frame_disp = frame.copy()
            cv2.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.2, (0, 0, 0), 1)
            cv2.imshow(display_name, frame_disp)
            x, y, w, h = cv2.selectROI(display_name, frame_disp, fromCenter=False)
            init_state = [x, y, w, h]
            tracker_instance.initialize(frame, {'init_bbox': init_state})

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
