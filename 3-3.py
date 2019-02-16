import os
import sys
import cv2
import math
import numpy as np

def distance_delta(x, y, px, py):
    return round(math.sqrt((px - x) ** 2 + (py - y) ** 2), 2)

def detect(filename, real_width):
    base_frame = []
    cap = cv2.VideoCapture(filename)
    ret, prev_frame = cap.read()

    # Calculate ratio
    cm_per_dot = float(real_width) / float(prev_frame.shape[1])

    x, y, w, h = 0, 0, 0, 0

    prev_x_center = 0
    prev_y_center = 0
    x_center = 0
    y_center = 0

    dist = []
    cnt = []
    center_rec = []
    center_avg_x = []
    center_avg_y = []

    while cap.isOpened():
        ret, frame = cap.read()

        # End of video reading
        if not ret:
            break

        # First frame for route plotting
        if len(base_frame) == 0:
            img_height, img_width = frame.shape[0], frame.shape[1]
            n_channels = 4
            base_frame = np.zeros((img_height, img_width, n_channels), dtype=np.uint8)

        # output = cv2.bitwise_and(frame, frame, mask = mask)
        output = cv2.absdiff(prev_frame, frame)

        gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        bilateral = cv2.bilateralFilter(gray, 7, 100, 100)

        _, thresh = cv2.threshold(bilateral, 8, 255, cv2.THRESH_BINARY)

        _, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        target_contours = []

        max_area = 0
        max_con = None

        for c in contours:
            if cv2.contourArea(c) > 1 and cv2.contourArea(c) < 1000:
                target_contours.append(c)
                if cv2.contourArea(c) > max_area:
                    max_con = c

        prev_frame = frame

        # cv2.drawContours(display, target_contours, -1, (0, 255, 0), 2)

        targets = len(target_contours)

        """
        if targets > 0 and targets < 10:
            x, y, w, h = cv2.boundingRect(max_con)
            x_center = x + w // 2
            y_center = y + h // 2
        elif targets == 0:
            # No move
            pass
        else:
            # Too much movements, treat as video jitter
            pass

        mite_movement = distance_delta(x_center, y_center, prev_x_center, prev_y_center)
        # if prev_x_center != 0 and mite_movement != 0:
            # print(mite_movement)

        display = np.copy(frame)

        cv2.rectangle(display, (x, y), (x+w, y+h), (0, 0, 255), 2)
        # cv2.circle(frame, (x + w // 2, y + h // 2), 20, (0, 255, 0))

        cv2.imshow('frame', display)
        cv2.waitKey(1)
        """

        cur_time = int(cap.get(cv2.CAP_PROP_POS_MSEC)) // 1000

        # Uncomment to test time-limited videos
        # if cur_time > 60: break

        if len(cnt) <= cur_time:
            # appeng counting array
            cnt.append(0)

            # calculate previous center
            if cur_time > 0:
                # Print running
                print('.', end = '')
                sys.stdout.flush()

                if len(center_avg_x) == 0 or len(center_avg_y) == 0:
                    location = center_rec[-1] if len(center_rec) > 0 else (0, 0)
                else:
                    location = (int(np.average(center_avg_x)), int(np.average(center_avg_y)))

                center_rec.append(location)
                center_avg_x.clear()
                center_avg_y.clear()

        if targets > 0 and targets <= 10:
            x, y, w, h = cv2.boundingRect(max_con)
            x_center = x + w // 2
            y_center = y + h // 2
            # print(cur_time, targets)
            center_avg_x.append(x_center)
            center_avg_y.append(y_center)
            cnt[cur_time] += 1
        elif targets > 10:
            # Camera fails
            cnt[cur_time] = -1000

    print()

    for index, point in enumerate(center_rec):
        dis = distance_delta(point[0], point[1], center_rec[index - 1][0], center_rec[index - 1][1])

        if dis < 80:
            # draw route on result image
            ratio = 170 * index / len(center_rec)
            cv2.circle(base_frame, point, 5, (0, 255, 170 - ratio, 255), -1)
            if index > 0:
                cv2.line(base_frame, center_rec[index], center_rec[index - 1], (0, 255, 170 - ratio, 255), 2)

            # record distance
            dist.append(round(cm_per_dot * dis, 2))
        else:
            dist.append(-1)

    # final frame has no valid speed data
    dist.append(-1)
    cap.release()
    cv2.destroyAllWindows()

    return cnt, dist, base_frame

def main():
    if len(sys.argv) < 3:
        print('Please use \'python trace.py <wideo width> <video folder name>\'')
        os._exit(1)
    else:
        width_cm = int(sys.argv[1])
        videos = os.listdir(sys.argv[2])
        if not os.path.exists('result'):
            os.mkdir('result')
        print('Will processing followong videos:\n' + '\n'.join(videos))
        for vid in videos:
            print('\nProcessing', vid)
            result, dist, img = detect(os.path.join(sys.argv[2], vid), width_cm)
            # print(result)
            total_dist = np.sum([np.array(dist) != -1] * np.array(dist))
            result_arr = np.array(result)
            result_arr[result_arr < 0] = -1
            moving_time = np.sum(result_arr >= 10)
            broken_frame = np.sum(result_arr < 0)
            eating_time = len(result) - moving_time - broken_frame
            print('Moving:', moving_time)
            print('Eating:', len(result) - moving_time - broken_frame)

            csv_name = vid.replace('.avi', '.csv')
            img_name = vid.replace('.avi', '.png')
            cv2.imwrite(os.path.join('result', img_name), img)
            with open(os.path.join('result', csv_name), 'w') as csv:
                csv.write('Total distance,' + str(total_dist) + '\n')
                csv.write('Moving time,' + str(moving_time) + '\n')
                csv.write('Eating time,' + str(eating_time) + '\n')
                csv.write('Broken data,' + str(broken_frame) + '\n')
                csv.writelines('\n'.join([str(i) + ',' + str(dist[i]) + ',' + str(d) + (',x' if d < 0 else '') + (',e' if d < 10 and d >= 0 else '') for i, d in enumerate(result_arr)]))

if __name__ == "__main__":
    main()

