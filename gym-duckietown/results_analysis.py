import numpy as np
import math
import json 

import pandas as pd
import matplotlib.pyplot as plt

# [{"relative_angle": 0.0, "relative_pos": 0.4401252662595048, "aruco_angle": 0.18298988335183175, "aruco_distance": 0.41479102224720443, "yolo_angle": -0.34906585039886595, "yolo_distance": 0.55}, {"relative_angle": -0.02952048255494155, "relative_pos": 0.4462896435843828, "aruco_angle": 0.10453118249130133, "aruco_distance": 0.4185831425643824, "yolo_angle": -0.34906585039886595, "yolo_distance": 0.55}, {"relative_angle": -0.05473236018538241, "relative_pos": 0.4524326363736537, "aruco_angle": 0.18817570857857313, "aruco_distance": 0.4377930536923209, "yolo_angle": -0.34906585039886595, "yolo_distance": 0.55}, {"relative_angle": -0.0760128649417755, "relative_pos": 0.45855492949286164, "aruco_angle": 0.12602623400623925, "aruco_distance": 0.4486365473141045, "yolo_angle": -0.34906585039886595, "yolo_distance": 0.55}, {"relative_angle": -0.09368948213253958, "relative_pos": 0.4646584493387439, "aruco_angle": 0.12609242908094065, "aruco_distance": 0.44855763606236115, "yolo_angle": -0.34906585039886595, "yolo_distance": 0.55}, {"relative_angle": -0.10841696911563581, "relative_pos": 0.4697257134837156, "aruco_angle": 0.17849754390303585, "aruco_distance": 0.4783202941215661, "yolo_angle": -0.34906585039886595, "yolo_distance": 0.55}, {"relative_angle": -0.12040001591912652, "relative_pos": 0.47392875682549956, "aruco_angle": 0.12097580199814861, "aruco_distance": 0.46209189317432575, "yolo_angle": -0.34906585039886595, "yolo_distance": 0.55}, {"relative_angle": -0.13037822322742976, "relative_pos": 0.47741135405684515, "aruco_angle": 0.1396853308940036, "aruco_distance": 0.47748436491340773, "yolo_angle": -0.34906585039886595, "yolo_distance": 0.55}, {"relative_angle": -0.1384309117072009, "relative_pos": 0.4802935842806932, "aruco_angle": 0.09963081260473339, "aruco_distance": 0.49103923400572136, "yolo_angle": -0.34906585039886595, "yolo_distance": 0.55}, {"relative_angle": -0.1446302852022292, "relative_pos": 0.48267606206356894, "aruco_angle": 0.12005550804331899, "aruco_distance": 0.4922459611200922, "yolo_angle": -0.34906585039886595, "yolo_distance": 0.55}, {"relative_angle": -0.1497955566335989, "relative_pos": 0.4846425844320349, "aruco_angle": 0.5167967775359363, "aruco_distance": 0.5025347647157118, "yolo_angle": -0.34906585039886595, "yolo_distance": 0.55}, {"relative_angle": -0.15330898180646635, "relative_pos": 0.48626327528776575, "aruco_angle": 0.5740673225663429, "aruco_distance": 0.5008083005387235, "yolo_angle": -0.34906585039886595, "yolo_distance": 0.55}, {"relative_angle": -0.15596756337612838, "relative_pos": 0.4875968195513641, "aruco_angle": 0.5687297165842189, "aruco_distance": 0.5009140293459748, "yolo_angle": -0.34906585039886595, "yolo_distance": 0.55}, {"relative_angle": -0.15770074717306937, "relative_pos": 0.48869192599945455, "aruco_angle": 0.5633907933285194, "aruco_distance": 0.5010174430877771, "yolo_angle": -0.34906585039886595, "yolo_distance": 0.55}, {"relative_angle": -0.15845656924360974, "relative_pos": 0.48958950133089163, "aruco_angle": 0.5491635786345852, "aruco_distance": 0.5017821191601005, "yolo_angle": -0.34906585039886595, "yolo_distance": 0.55}, {"relative_angle": -0.1589249740983263, "relative_pos": 0.49032339557111776, "aruco_angle": 0.543739944703852, "aruco_distance": 0.5018817387924914, "yolo_angle": -0.34906585039886595, "yolo_distance": 0.55}]


def main():
    # read the info_record.json file
    with open('info_record.json', 'r') as f:
        info_record = json.load(f)

    results = pd.DataFrame(info_record)
    results['aruco_relative_angle_diff'] = results['aruco_angle'] - results['relative_angle']
    results['yolo_relative_angle_diff'] = results['yolo_angle'] - results['relative_angle']

    results['aruco_relative_pos_diff'] = results['aruco_distance'] - results['relative_pos']
    results['yolo_relative_pos_diff'] = results['yolo_distance'] - results['relative_pos']

    # plot the relative angle difference in aruco, with relative angle, scatter 
    plt.scatter(range(len(results['aruco_relative_angle_diff'])), results['aruco_relative_angle_diff'], label='aruco')
    plt.legend()
    plt.xlabel('timesteps')
    plt.ylabel('angle diff')
    plt.title('Difference between Aruco angle and Relative Angle')
    plt.show()


    plt.scatter(range(len(results['yolo_relative_angle_diff'])), results['yolo_relative_angle_diff'], label='yolo')
    plt.legend()
    plt.xlabel('timesteps')
    plt.ylabel('angle diff')
    plt.title('Difference between Yolo angle and Relative Angle')
    plt.show()

    # distance
    plt.scatter(range(len(results['aruco_relative_pos_diff'])), results['aruco_relative_pos_diff'], label='aruco')
    plt.legend()
    plt.xlabel('timesteps')
    plt.ylabel('distance diff')
    plt.title('Difference between Aruco distance and Relative distance')
    plt.show()

    plt.scatter(range(len(results['yolo_relative_pos_diff'])), results['yolo_relative_pos_diff'], label='yolo')
    plt.legend()
    plt.xlabel('timesteps')
    plt.ylabel('distance diff')
    plt.title('Difference between Yolo distance and Relative distance')
    plt.show()
    



if __name__ == "__main__":
    main()
