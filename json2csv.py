import os
import json
import csv

def main():
    base = './output/'
    path_list = []

    labels = ['nose-x', 'nose-y', 'nose-conf',
            'left-eye-x', 'left-eye-y', 'left-eye-conf',
            'right-eye-x', 'right-eye-y', 'right-eye-conf',
            'left-ear-x', 'left-ear-y', 'left-ear-conf',
            'right-ear-x', 'right-ear-y', 'right-ear-conf',
            'left-shoulder-x', 'left-shoulder-y', 'left-shoulder-conf',
            'right-shoulder-x', 'right-shoulder-y', 'right-shoulder-conf',
            'left-elbow-x', 'left-elbow-y', 'left-elbow-conf',
            'right-elbow-x', 'right-elbow-y', 'right-elbow-conf',
            'left-hand-x', 'left-hand-y', 'left-hand-conf',
            'right-hand-x', 'right-hand-y', 'right-hand-conf',
            'left-hip-x', 'left-hip-y', 'left-hip-conf',
            'right-hip-x', 'right-hip-y', 'right-hip-conf',
            'left-knee-x', 'left-knee-y', 'left-knee-conf',
            'right-knee-x', 'right-knee-y', 'right-knee-conf',
            'left-foot-x', 'left-foot-y', 'left-foot-conf',
            'right-foot-x', 'right-foot-y', 'right-foot-conf']

    for f in os.listdir(base):
        path_list.append(base + f)

    for path in path_list:
        files = [file for file in os.listdir(path) if file.endswith('.json')]
        for file in files:
            name = file.split('.')[0]
            with open(f'{path}/{file}', 'r') as j:
                data = json.load(j)
                frame = data['tag']['num of frame']
                for key in data.keys():
                    if len(data[key]) < (frame*0.5):
                        continue
                    with open(f'{path}/{name}-{key}.csv', 'w', newline='') as c:
                        w = csv.writer(c)
                        w.writerow(labels)
                        for kpts in data[key]:
                            w.writerow(kpts)
                with open(f'{path}/{name}-tag.txt', 'w', encoding='utf-8') as t:
                    t.write('트래킹(모델 생성 이후)부터 종료까지의 소요시간 \n')
                    t.write(f'time: {data["tag"]["time"]} \n\n')
                    t.write('영상의 총 프레임 수 \n')
                    t.write(f'num of frame: {data["tag"]["num of frame"]} \n\n')
                    t.write('frame을 time으로 나눈 값 \n')
                    t.write(f'frame/time: {data["tag"]["frame/time"]} \n\n')


if __name__ == '__main__':
    main()