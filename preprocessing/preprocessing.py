import os
import numpy as np
import scipy.io.wavfile as wav
from tqdm import tqdm
import random
from tools import calculate_fused_signal, calculate_total_correlation_energy, calculate_weights, CalculateStatisticalFeatuers
from scipy.fftpack import fft
import time


dataset_root_name = "data"
equipment_name = "fan"
noise_rates = ["0_dB"]
unmasked_length = 20000


## Split Dataset for MIMII
def split_dataset(train_type="", noise_rates=None):
    if train_type != "":
        train_type = f"_{train_type}"
        
    for noise_rate in noise_rates:
        root_dir = os.path.join(equipment_name, noise_rate, "Raw")
        train_list = []
        valid_list = []
        test_list = []
        for device in os.listdir(root_dir):        
            device_dir = os.path.join(root_dir, device)
            for state in os.listdir(device_dir):
                state_dir = os.path.join(device_dir, state)

                files_list = os.listdir(state_dir)
                random.shuffle(files_list)
                for index, file in enumerate(files_list):
                    file_dir = os.path.join(state_dir, file)  
                    state_indicator = 0 if state == "normal" else 1
                    sample_infos = os.path.join(dataset_root_name, file_dir)+"\t"+str(state_indicator)+"\n"
                    
                    if device in ["id_00", "id_02", "id_04"]:
                        if index % 8 != 0:
                            train_list.append(sample_infos)
                        else:
                            valid_list.append(sample_infos)
                    else:
                        test_list.append(sample_infos)
                        
        with open(os.path.join(equipment_name, noise_rate, f"train{train_type}.txt"), "w") as f:
            f.writelines(train_list)
        with open(os.path.join(equipment_name, noise_rate, f"valid{train_type}.txt"), "w") as f:
            f.writelines(valid_list)
        with open(os.path.join(equipment_name, noise_rate, f"test{train_type}.txt"), "w") as f:
            f.writelines(test_list)
        print(len(train_list),len(valid_list),len(test_list))


## generate data and calculate the statistical features
def generate_data(noise_rates=["_6_dB",], equipment_name="None",
                  goal_dir="FusedSignal", process_method=None):
    
    for noise_rate in noise_rates:
        root_dir = os.path.join(equipment_name, noise_rate, "Raw")
        for device in os.listdir(root_dir):
            device_dir = os.path.join(root_dir, device)
            for state in os.listdir(device_dir):
                state_dir = os.path.join(device_dir, state)
                if not os.path.exists(state_dir.replace("Raw", goal_dir)):
                    os.makedirs(state_dir.replace("Raw", goal_dir))
                bar = tqdm(os.listdir(state_dir))
                for file in bar:
                    source_file = os.path.join(state_dir, file)

                    # process method
                    output_file = process_method(source_file, goal_dir)

                    bar.set_description(f"Processed file: {output_file}")
                bar.close()
    
    print("DONE")


def statistic(noise_rates=["_6_dB",], equipment_name="None",
              goal_dir="FusedSignal", process_method=None, train_type="",
              output_file="feature_statistic", axis=None):
    
    for noise_rate in noise_rates:
        root_dir = os.path.join(equipment_name, noise_rate, "Raw")
        # load training list
        with open(os.path.join(equipment_name, noise_rate, f"train{train_type}.txt"), "r") as f:
            train_list = f.readlines()
        
        signals = None
        
        for device in os.listdir(root_dir):
            device_dir = os.path.join(root_dir, device)
            for state in os.listdir(device_dir):
                state_dir = os.path.join(device_dir, state)
                # convert state into 0/1, 0 indicates normal and 1 indicates abnormal
                state_indicator = 0 if state == "normal" else 1
                
                bar = tqdm(os.listdir(state_dir), desc=f"Processing {state_dir} ...")
                for file in bar:
                    source_file = os.path.join(state_dir, file)
                    # if not in training list, it should be skipped
                    if not os.path.join(dataset_root_name, source_file)+"\t"+str(state_indicator)+"\n" in train_list:
                        time.sleep(0.001)
                        continue

                    # process method
                    signals = process_method(source_file, goal_dir, signals)

                    bar.set_description(f"Processed file: {source_file}")
                bar.close()
        if "Feature" not in goal_dir:
            assert signals.shape[0] == len(train_list)
        elif "Cluster" not in goal_dir:
            assert int(signals.shape[0]//200) == len(train_list)
        else:
            assert signals.shape[0] == len(train_list)
        
        data_mean = signals.mean(axis=axis)    
        data_std = signals.std(axis=axis)
        data_max = signals.max(axis=axis)
        data_min = signals.min(axis=axis)
        
        saved_file = os.path.join(root_dir.replace("Raw", goal_dir), output_file+f'{train_type}.npz')
        np.savez(saved_file,
                 DataMean=data_mean,
                 DataStd=data_std,
                 DataMax=data_max,
                 DataMin=data_min
                )
        print(f"Saved as {saved_file}")
    
    print("DONE")


## Reconstruct Signal for MIMII
## 1 Fused Signal

def fused_signal(source_file, goal_dir):
    sr, signal = wav.read(source_file)
    signal = signal.transpose(1, 0)
    total_energy = calculate_total_correlation_energy(signal)
    weights = calculate_weights(total_energy)
    fused_signal = calculate_fused_signal(signal, weights)

    output_file = source_file.replace("Raw", goal_dir)
    wav.write(output_file, sr, fused_signal)
    return output_file

# fuse signal
generate_data(noise_rates=noise_rates, 
              goal_dir="FusedSignal", 
              process_method=fused_signal, 
              equipment_name=equipment_name)

def fused_signal_feature(source_file, goal_dir):
    sr, signal = wav.read(source_file.replace("Raw", goal_dir.split("Feature")[0]))
    signal_fft = np.abs(fft(signal))[:80000]
    features = CalculateStatisticalFeatuers(signal_fft, window_size=400)
    output_file = source_file.replace("Raw", goal_dir).replace("wav", "npz")
    np.savez(output_file, Feature=features)
    return output_file

# extract statistic features
generate_data(noise_rates=noise_rates,
              goal_dir="FusedSignalFeature",
              process_method=fused_signal_feature, 
              equipment_name=equipment_name)

def fused_fft_statistic(source_file, goal_dir, features):
    file = source_file.replace("Raw", goal_dir)
    _, signal = wav.read(file)
    feature = np.abs(fft(signal))[:unmasked_length]
    if features is None:
        features = feature.reshape(1, -1)
    else:
        features = np.vstack((features, feature.reshape(1, -1)))

    return features

# calculate statistic of masked fft
statistic(noise_rates=noise_rates,
          goal_dir="FusedSignal",
          process_method=fused_fft_statistic,
          output_file="fused_signal_fft_statistic", 
          equipment_name=equipment_name)


def fused_signal_feature_statistic(source_file, goal_dir, features):
    file = source_file.replace("Raw", goal_dir).replace("wav", "npz")
    feature = np.load(file, allow_pickle=True)["Feature"]
    if features is None:
        features = feature
    else:
        features = np.concatenate((features, feature), axis=0)

    return features

# calculate statistic of fused signal
statistic(noise_rates=noise_rates,
          goal_dir="FusedSignalFeature",
          process_method=fused_signal_feature_statistic,
          output_file="fused_signal_feature_statistic",
          axis=0, 
          equipment_name=equipment_name)

# # Add clustering
from sklearn.cluster import KMeans

def kmeans_feature(noise_rate="0_dB", equipment_name="None", 
                   goal_dir="FusedSignal", process_method=None, train_type=""):
    features = None
    with open(os.path.join(equipment_name, noise_rate, f"train{train_type}.txt"), "r") as f:
        train_list = f.readlines()

    root_dir = os.path.join(equipment_name, noise_rate, "Raw")
    for device in os.listdir(root_dir):
        device_dir = os.path.join(root_dir, device)
        for state in os.listdir(device_dir):
            state_dir = os.path.join(device_dir, state)
            state_indicator = 0 if state == "normal" else 1

            bar = tqdm(os.listdir(state_dir))
            for file in bar:
                source_file = os.path.join(state_dir, file)
                if not os.path.join(dataset_root_name, source_file)+"\t"+str(state_indicator)+"\n" in train_list:
                    continue

                # process method
                feature = process_method(source_file, goal_dir)

                if features is None:
                    features = feature
                else:
                    features = np.concatenate((features, feature), axis=0)
                bar.set_description(f"Processed file: {source_file}")
            bar.close()
    
    print("DONE")
    return features


def fused_signal_kmeans_feature(source_file, goal_dir):
    file = source_file.replace("Raw", goal_dir).replace("wav", "npz")
    feature = np.load(file, allow_pickle=True)["Feature"]
    return feature

def generate_kmeans_feature(source_file, goal_dir):
    file = source_file.replace("Raw", "FusedSignalFeature").replace("wav", "npz")
    feature = np.load(file, allow_pickle=True)["Feature"]
    hist_types=np.zeros(num_types_all)
    y_pred = kmeans.predict(feature)
    indexs, counts = np.unique(y_pred, return_counts=True)
    hist_types[indexs] = counts

    heights = {i:[] for i in range(num_types_all)}
    RMSEs = {i:[] for i in range(num_types_all)}
    for item in feature:
        index = kmeans.predict(item.reshape(1,-1))[0]
        heights[index].append(item)
        center = kmeans.cluster_centers_[index]
        rmse = np.sqrt(np.mean(np.power(center - item, 2)))
        RMSEs[index].append(rmse)

    mean_height=np.zeros(num_types_all)
    std_height=np.zeros(num_types_all)
    mean_RMSE=np.zeros(num_types_all)

    for j in range(num_types_all):
        if len(heights[j]) > 0:
            mean_height[j] = np.mean(heights[j])
            std_height[j] = np.std(heights[j])
            mean_RMSE[j] = np.mean(RMSEs[j])
    ClusterFeatures = np.hstack((hist_types.reshape(-1,1),
                                 mean_height.reshape(-1,1),
                                 std_height.reshape(-1,1),
                                 mean_RMSE.reshape(-1,1)))

    output_file = source_file.replace("Raw", goal_dir).replace("wav", "npz")
    np.savez(output_file, Feature=ClusterFeatures)
                    
    return output_file

num_types_all = 9
# extract kmeans features
for noise_rate in noise_rates:
    # calculate kmeans feature of fused signal
    features = kmeans_feature(noise_rate=noise_rate,
                              goal_dir="FusedSignalFeature",
                              process_method=fused_signal_kmeans_feature, 
                              equipment_name=equipment_name)
    
    kmeans = KMeans(n_clusters=num_types_all, random_state=9, init='k-means++').fit(features)
    
    generate_data(noise_rates=[noise_rate,],
                  goal_dir="ClusterFusedFeature",
                  process_method=generate_kmeans_feature, 
                  equipment_name=equipment_name)
    
    del features, kmeans

def kmeans_feature_statistic(source_file, goal_dir, features):
    file = source_file.replace("Raw", goal_dir).replace("wav", "npz")
    feature = np.load(file, allow_pickle=True)["Feature"]
    if features is None:
        features = np.expand_dims(feature, axis=0)
    else:
        features = np.concatenate((features, np.expand_dims(feature, axis=0)), axis=0)

    return features

# calculate statistic of fused signal
statistic(noise_rates=noise_rates,
          goal_dir="ClusterFusedFeature",
          process_method=kmeans_feature_statistic,
          output_file="cluster_fused_feature_statistic",
          axis=0, 
          equipment_name=equipment_name)





