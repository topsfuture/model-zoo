
import os
import cv2
import numpy as np
from typing import List
from acuitylib.vsi_nn import VSInn
from scipy.ndimage import gaussian_filter
import logging
import math
import tensorflow as tf
logging.basicConfig(level=logging.INFO)

colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85], [255, 0, 0], [255, 0, 255], [255, 85, 255], \
          [255, 170, 255], [255, 255, 255], [170, 255, 255], [85, 255, 255], [0, 255, 255], [255, 255, 0], [255, 255, 85], [255, 255, 170]]


class OpenPose:
    def __init__(self, args):

        self.net_name = "OpenPose"
        self.model_path = args.model_path
        self.weights_path = args.model_weight
        self.quantize_type = args.quantize_type
        
        self.input_shape = args.input_shape
        self.batch_size = self.input_shape[0]
        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]
        self.thre1 = 0.1
        self.thre2 = 0.05
        self.point_num = 18
        self.create_acuity_net()
        self.POSE_PAIRS = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7], [1,8], [8,9], [9,10], [1,11], \
                               [11,12], [12,13], [1,0], [0,14], [14,16], [0,15], [15,17], [2, 16], [5, 17]]
        self.mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44], [19,20], [21,22], [23,24], \
                           [25,26], [27,28], [29,30], [47,48], [49,50], [53,54], [51,52], [55,56], [37,38], [45,46]]
        
    def create_acuity_net(self):
        
        self.nn = VSInn()
        if os.path.exists(f'./{self.net_name}.json'):
            logging.info('.json ready')
            self.acuity_net = self.nn.create_net()
            self.nn.load_model(self.acuity_net, f'./{self.net_name}.json')
            self.nn.load_model_data(self.acuity_net, f'./{self.net_name}.data')
        else:
            self.acuity_net = self.nn.load_caffe(self.model_path,
                self.weights_path,
                "caffe")
            self.nn.save_model(self.acuity_net, f'{self.net_name}.json')
            self.nn.save_model_data(self.acuity_net, f'{self.net_name}.data')
        
    def init_model_infer(self, args):
        if self.quantize_type != "float32":
            self.load_q_net(self.quantize_type)
        self.nn.build_inference_session(self.acuity_net)
        
    def load_q_net(self, quantize_type):
        logging.info(f"quantize tyep: {quantize_type}")
        if quantize_type not in ['int8', 'uint8', 'float16', 'bfloat16', 'int16']:
            logging.error("wrong quantize type.")
            raise ValueError("wrong quantize type.")
            
        if os.path.exists(f"./{self.net_name}_{quantize_type}.quantize") :
                    logging.info(f"Load {quantize_type} quantize file.")
                    self.nn.load_model_quantize(self.acuity_net,
                                                f"./{self.net_name}_{quantize_type}.quantize")
        else:
            logging.info(f"Quantize file not found. Please run quantize.py first.")
            raise FileNotFoundError(f"{self.net_name}_{quantize_type}.quantize does not exits.")
        
        
    def vsi_quantize_net(self, img_list:List,
                         quantize_type:str,
                         cali_batch_size:int,
                         hybrid:bool = False):
        q_er_table = {"uint8": "asymmetric_affine",
                "int8": "perchannel_symmetric_affine",
                "int16": "dynamic_fixed_point",
                "float16": "float16",
                "bfloat16": "bfloat16"}
        def get_input_for_quantize():
            for img in img_list[:cali_batch_size]:
                preprocessed_img, _, _ = self.preprocess(img)
                logging.info(f"type of img for np_to_tf: {type(preprocessed_img)}")

                single_input = []
                single_input.append(self.np_to_tf(preprocessed_img))
                
                yield single_input
                
        q_net = self.nn.quantize(self.acuity_net,
                                 qtype=quantize_type,
                                 quantizer=q_er_table[quantize_type],
                                 batch_size=1,
                                 iterations=cali_batch_size,
                                 input_generator_func=get_input_for_quantize,
                                 compute_entropy=hybrid)
        
        if hybrid:
            q_net = self.nn.quantize(q_net,
                                     qtype=quantize_type,
                                     quantizer=q_er_table[quantize_type],
                                     batch_size=1,
                                     iterations=cali_batch_size,
                                     input_generator_func=get_input_for_quantize,
                                     hybrid=True)
            
        self.nn.save_model_quantize(q_net, f"./{self.net_name}_{quantize_type}.quantize")
    
            
    def preprocess(self, ori_img):
        img = cv2.resize(ori_img, (self.net_h, self.net_w), cv2.INTER_LINEAR)
        img = img.astype("float32")
        img -= 128
        img /= 255
        img = np.transpose(img, (2, 0, 1))
        return img
    
    
    def postprocess(self, outputs, img_list):
        results = []
        for i, output in enumerate(outputs):
            ori_img = img_list[i]
            output = output[0]
            output = np.transpose(output, (1, 2, 0))
            output = cv2.resize(output, (ori_img.shape[1], ori_img.shape[0]), interpolation=cv2.INTER_CUBIC)
            
            all_peaks = []
            peak_counter = 0
            
            for part in range(self.point_num):
                map_ori = output[:, :, part]
                one_heatmap = gaussian_filter(map_ori, sigma=3)
                
                map_left = np.zeros(one_heatmap.shape)
                map_left[1:, :] = one_heatmap[:-1, :]
                map_right = np.zeros(one_heatmap.shape)
                map_right[:-1, :] = one_heatmap[1:, :]
                map_up = np.zeros(one_heatmap.shape)
                map_up[:, 1:] = one_heatmap[:, :-1]
                map_down = np.zeros(one_heatmap.shape)
                map_down[:, :-1] = one_heatmap[:, 1:]
                
                peaks_binary = np.logical_and.reduce(
                    (one_heatmap >= map_left, one_heatmap >= map_right, one_heatmap >= map_up, one_heatmap >= map_down, one_heatmap > self.thre1))
                peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
                peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
                peak_id = range(peak_counter, peak_counter + len(peaks))
                peaks_with_score_and_id = [peaks_with_score[i] + (peak_id[i],) for i in range(len(peak_id))]

                all_peaks.append(peaks_with_score_and_id)
                peak_counter += len(peaks)
            
            connection_all = []
            special_k = []
            mid_num = 10
            
            for k in range(len(self.mapIdx)):
                score_mid = output[:, :, [x for x in self.mapIdx[k]]]
                candA = all_peaks[self.POSE_PAIRS[k][0]]
                candB = all_peaks[self.POSE_PAIRS[k][1]]
                nA = len(candA)
                nB = len(candB)
                indexA, indexB = np.array(self.POSE_PAIRS[k]) + 1
                if (nA != 0 and nB != 0):
                    connection_candidate = []
                    for i in range(nA):
                        for j in range(nB):
                            vec = np.subtract(candB[j][:2], candA[i][:2])
                            norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                            norm = max(0.001, norm)
                            vec = np.divide(vec, norm)

                            startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                                                np.linspace(candA[i][1], candB[j][1], num=mid_num)))

                            vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                                            for I in range(len(startend))])
                            vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                                            for I in range(len(startend))])

                            score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                            score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
                                0.5 * ori_img.shape[0] / norm - 1, 0)
                            criterion1 = len(np.nonzero(score_midpts > self.thre2)[0]) > 0.8 * len(score_midpts)
                            criterion2 = score_with_dist_prior > 0
                            if criterion1 and criterion2:
                                connection_candidate.append(
                                    [i, j, score_with_dist_prior, score_with_dist_prior + candA[i][2] + candB[j][2]])

                    connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
                    connection = np.zeros((0, 5))
                    for c in range(len(connection_candidate)):
                        i, j, s = connection_candidate[c][0:3]
                        if (i not in connection[:, 3] and j not in connection[:, 4]):
                            connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                            if (len(connection) >= min(nA, nB)):
                                break

                    connection_all.append(connection)
                else:
                    special_k.append(k)
                    connection_all.append([])

            # last number in each row is the total parts number of that person
            # the second last number in each row is the score of the overall configuration
            subset = -1 * np.ones((0, self.point_num + 2))
            candidate = np.array([item for sublist in all_peaks for item in sublist])

            for k in range(len(self.mapIdx)):
                if k not in special_k:
                    partAs = connection_all[k][:, 0]
                    partBs = connection_all[k][:, 1]
                    indexA, indexB = self.POSE_PAIRS[k]

                    for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                        found = 0
                        subset_idx = [-1, -1]
                        for j in range(len(subset)):  # 1:size(subset,1):
                            if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                                subset_idx[found] = j
                                found += 1

                        if found == 1:
                            j = subset_idx[0]
                            if subset[j][indexB] != partBs[i]:
                                subset[j][indexB] = partBs[i]
                                subset[j][-1] += 1
                                subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                        elif found == 2:  # if found 2 and disjoint, merge them
                            j1, j2 = subset_idx
                            membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                            if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                                subset[j1][:-2] += (subset[j2][:-2] + 1)
                                subset[j1][-2:] += subset[j2][-2:]
                                subset[j1][-2] += connection_all[k][i][2]
                                subset = np.delete(subset, j2, 0)
                            else:  # as like found == 1
                                subset[j1][indexB] = partBs[i]
                                subset[j1][-1] += 1
                                subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                        # if find no partA in the subset, create a new subset
                        elif not found and k < self.point_num - 1:
                            row = -1 * np.ones(self.point_num + 2)
                            row[indexA] = partAs[i]
                            row[indexB] = partBs[i]
                            row[-1] = 2
                            row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                            subset = np.vstack([subset, row])
            # delete some rows of subset which has few parts occur
            deleteIdx = []
            for i in range(len(subset)):
                if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
                    deleteIdx.append(i)
            subset = np.delete(subset, deleteIdx, axis=0)

            # subset: n*20 array, 0-17 is the index in candidate, 18 is the total score, 19 is the total parts
            # candidate: x, y, score, id
            results.append((candidate, subset))    

        return results
    
    @staticmethod
    def np_to_tf(img):
        img = np.expand_dims(img, axis=0)
        logging.debug(f"type of img for np_to_tf: {type(img)}")
        logging.debug(f"tuple -> np.ndarray shape: {img.shape}")
        img_tf = tf.convert_to_tensor(img)
        return img_tf
    
    def vsinn_predict(self, preprocessed_img_list, img_num):
        def get_input_for_infer():
            for i, preprocessed_img in enumerate(preprocessed_img_list):
                # print(f">>>>>>>>>>>>>>>>>>>>>>>>>\ninfer load image:{i}\n>>>>>>>>>>>>>>>>>>>>>>>>>")
                single_input = []
                single_input.append(self.np_to_tf(preprocessed_img))
                yield single_input
        
        outputs, batch = [], []   
        
        for i, data in enumerate(get_input_for_infer()):
            ins, outs = self.nn.run_inference_session(data)
            batch = outs[0]
            outputs.append(batch)  
        return outputs
    
    def draw_pose(self, ori_img, candidate, subset):
        canvas = ori_img.copy()
        for i in range(self.point_num):
            for n in range(len(subset)):
                index = int(subset[n][i])
                if index == -1:
                    continue
                x, y = candidate[index][0:2]
                cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)
        for i in range(len(self.POSE_PAIRS)):
            for n in range(len(subset)):
                index = subset[n][np.array(self.POSE_PAIRS[i])]
                if -1 in index:
                    continue
                cur_canvas = canvas.copy()
                Y = candidate[index.astype(int), 0]
                X = candidate[index.astype(int), 1]
                mX = np.mean(X)
                mY = np.mean(Y)
                length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
                polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), 4), int(angle), 0, 360, 1)
                cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
                canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
        
        return canvas


    
    def __call__(self, img_list):
        img_num = len(img_list)
        ori_size_list = []
        preprocessed_img_list = []
        for ori_img in img_list:
            ori_h, ori_w = ori_img.shape[:2]
            ori_size_list.append((ori_w, ori_h))
            preprocessed_img = self.preprocess(ori_img)
            preprocessed_img_list.append(preprocessed_img)
        
        if img_num == self.batch_size:
            input_img = np.stack(preprocessed_img_list)
        else:
            input_img = np.zeros(self.input_shape, dtype='float32')
            input_img[:img_num] = np.stack(preprocessed_img_list)
        outs = self.vsinn_predict(input_img, img_num)    
        
        result_list = self.postprocess(outs, img_list)      
        return result_list
