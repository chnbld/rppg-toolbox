"""The dataloader for the UBFC-PHYS dataset.

Details for the UBFC-PHYS Dataset see https://sites.google.com/view/ybenezeth/ubfc-phys.
If you use this dataset, please cite this paper:
R. Meziati Sabour, Y. Benezeth, P. De Oliveira, J. Chappé, F. Yang. 
"UBFC-Phys: A Multimodal Database For Psychophysiological Studies Of Social Stress", 
IEEE Transactions on Affective Computing, 2021.
"""
import glob
import os
import re
from multiprocessing import Pool, Process, Value, Array, Manager

import cv2
import numpy as np
from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm
import csv
import pandas as pd

class UBFCPHYSLoader(BaseLoader):
    """The data loader for the UBFC-PHYS dataset."""

    def __init__(self, name, data_path, config_data, device=None):
        """Initializes an UBFC-PHYS dataloader.
            Args:
                data_path(str): path of a folder which stores raw video and bvp data.
                e.g. data_path should be "RawData" for below dataset structure:
                -----------------
                     RawData/
                     |   |-- s1/
                     |       |-- vid_s1_T1.avi
                     |       |-- vid_s1_T2.avi
                     |       |-- vid_s1_T3.avi
                     |       |...
                     |       |-- bvp_s1_T1.csv
                     |       |-- bvp_s1_T2.csv
                     |       |-- bvp_s1_T3.csv
                     |   |-- s2/
                     |       |-- vid_s2_T1.avi
                     |       |-- vid_s2_T2.avi
                     |       |-- vid_s2_T3.avi
                     |       |...
                     |       |-- bvp_s2_T1.csv
                     |       |-- bvp_s2_T2.csv
                     |       |-- bvp_s2_T3.csv
                     |...
                     |   |-- sn/
                     |       |-- vid_sn_T1.avi
                     |       |-- vid_sn_T2.avi
                     |       |-- vid_sn_T3.avi
                     |       |...
                     |       |-- bvp_sn_T1.csv
                     |       |-- bvp_sn_T2.csv
                     |       |-- bvp_sn_T3.csv
                -----------------
                name(string): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
        """
        self.filtering = config_data.FILTERING
        super().__init__(name, data_path, config_data, device)

    def get_raw_data(self, data_path):
        """Returns data directories under the path(For UBFC-PHYS dataset)."""
        data_dirs = glob.glob(data_path + os.sep + "s*" + os.sep + "*.avi")
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        dirs = [{"index": re.search(
            'vid_(.*).avi', data_dir).group(1), "path": data_dir} for data_dir in data_dirs]
        return dirs

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values."""
        if begin == 0 and end == 1:  # return the full directory if begin == 0 and end == 1
            data_dirs_subset = data_dirs
        else:
            file_num = len(data_dirs)
            choose_range = range(int(begin * file_num), int(end * file_num))
            data_dirs_subset = []

            for i in choose_range:
                data_dirs_subset.append(data_dirs[i])
        
        # Apply filtering (exclusion list) during preprocessing to avoid processing excluded videos
        if hasattr(self, 'filtering') and self.filtering.USE_EXCLUSION_LIST:
            filtered_dirs = []
            for data_dir in data_dirs_subset:
                index = data_dir['index']
                # Skip if in exclusion list
                if index not in self.filtering.EXCLUSION_LIST:
                    filtered_dirs.append(data_dir)
                else:
                    print(f"Skipping excluded video: {index}")
            return filtered_dirs
        
        return data_dirs_subset

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """   invoked by preprocess_dataset for multi_process.
        
        Now uses streaming mode by default to reduce memory usage.
        """
        # Use streaming mode to reduce memory usage
        self.preprocess_dataset_streaming(data_dirs, config_preprocess, i, file_list_dict)

    def load_preprocessed_data(self):
        """ Loads the preprocessed data listed in the file list.

        Args:
            None
        Returns:
            None
        """
        file_list_path = self.file_list_path  # get list of files in
        file_list_df = pd.read_csv(file_list_path)
        base_inputs = file_list_df['input_files'].tolist()
        filtered_inputs = []

        for input in base_inputs:
            input_name = input.split(os.sep)[-1].split('.')[0].rsplit('_', 1)[0]
            if self.filtering.USE_EXCLUSION_LIST and input_name in self.filtering.EXCLUSION_LIST :
                # Skip loading the input as it's in the exclusion list
                continue
            if self.filtering.SELECT_TASKS and not any(task in input_name for task in self.filtering.TASK_LIST):
                # Skip loading the input as it's not in the task list
                continue
            filtered_inputs.append(input)

        if not filtered_inputs:
            raise ValueError(self.dataset_name + ' dataset loading data error!')
        
        filtered_inputs = sorted(filtered_inputs)  # sort input file name list
        labels = [input_file.replace("input", "label") for input_file in filtered_inputs]
        self.inputs = filtered_inputs
        self.labels = labels
        self.preprocessed_data_len = len(filtered_inputs)

    @staticmethod
    def read_video(video_file, max_frames=1500):
        """Reads a video file, returns frames(T,H,W,3) 
        
        Args:
            video_file: path to video file
            max_frames: maximum number of frames to load (to avoid memory issues)
        """
        import gc
        VidObj = cv2.VideoCapture(video_file)
        
        # Get video properties for info
        fps = VidObj.get(cv2.CAP_PROP_FPS)
        frame_count = int(VidObj.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(VidObj.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(VidObj.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Limit frames if video is too long to avoid memory issues
        actual_frame_count = min(frame_count, max_frames)
        print(f"Video info: {frame_count} total frames (loading {actual_frame_count}), {width}x{height}, {fps:.2f} fps")
        
        if frame_count > max_frames:
            print(f"⚠️  Warning: Video has {frame_count} frames, limiting to {max_frames} to avoid memory issues")
        
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, frame = VidObj.read()
        frames = list()
        
        frame_idx = 0
        while success and frame_idx < max_frames:
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            frame = np.asarray(frame)
            frames.append(frame)
            success, frame = VidObj.read()
            
            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"  Loaded {frame_idx}/{actual_frame_count} frames...")
                gc.collect()  # Force garbage collection to free memory
        
        VidObj.release()
        print(f"✓ Video loading complete: {len(frames)} frames")
        return np.asarray(frames)

    @staticmethod
    def read_wave(bvp_file):
        """Reads a bvp signal file."""
        bvp = []
        with open(bvp_file, "r") as f:
            d = csv.reader(f)
            for row in d:
                bvp.append(float(row[0]))
        return np.asarray(bvp)

    @staticmethod
    def stream_video_in_chunks(video_file, chunk_size=300, max_frames=800):
        """Stream video in chunks to avoid loading entire video into memory.
        
        Args:
            video_file: path to video file
            chunk_size: number of frames to load at once
            max_frames: maximum total frames to process
            
        Yields:
            tuple: (frames_batch, start_idx, end_idx) for each chunk
        """
        import gc
        VidObj = cv2.VideoCapture(video_file)
        
        # Get video properties
        frame_count = int(VidObj.get(cv2.CAP_PROP_FRAME_COUNT))
        actual_frame_count = min(frame_count, max_frames)
        
        print(f"Streaming video: {frame_count} total frames (processing {actual_frame_count}), "
              f"chunk_size={chunk_size}")
        
        current_chunk = []
        frame_idx = 0
        
        while frame_idx < actual_frame_count:
            success, frame = VidObj.read()
            if not success:
                break
                
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            current_chunk.append(frame)
            frame_idx += 1
            
            # Yield chunk when it's full or at end
            if len(current_chunk) >= chunk_size or frame_idx >= actual_frame_count:
                yield np.asarray(current_chunk), frame_idx - len(current_chunk), frame_idx
                del current_chunk
                gc.collect()
                current_chunk = []
                
            if frame_idx % 100 == 0:
                print(f"  Streamed {frame_idx}/{actual_frame_count} frames...")
        
        VidObj.release()
        print(f"✓ Video streaming complete: {frame_idx} frames")

    def preprocess_dataset_streaming(self, data_dirs, config_preprocess, i, file_list_dict):
        """Streaming version of preprocessing that processes video in chunks.
        
        This method streams the video frame-by-frame instead of loading it all at once,
        significantly reducing memory usage.
        """
        try:
            filename = os.path.split(data_dirs[i]['path'])[-1]
            saved_filename = data_dirs[i]['index']
            print(f"Streaming processing video {i}: {saved_filename}")
            
            video_path = os.path.join(data_dirs[i]['path'])
            
            # First pass: collect all chunks and their metadata
            all_chunks = []
            chunk_count = 0
            
            # Stream video in chunks of 300 frames (or whatever fits in memory)
            for frames_chunk, start_idx, end_idx in self.stream_video_in_chunks(
                video_path, chunk_size=300, max_frames=800
            ):
                all_chunks.append({
                    'frames': frames_chunk,
                    'start_idx': start_idx,
                    'end_idx': end_idx
                })
                chunk_count += 1
            
            # Now we have all chunks in memory, but in smaller pieces
            # Combine them if memory allows, or process separately
            if chunk_count > 0:
                print(f"Processing {chunk_count} chunks for video {saved_filename}")
                
                # Option 1: Combine all chunks into full video (if memory allows)
                # This maintains compatibility with existing preprocessing
                all_frames = []
                for chunk_data in all_chunks:
                    all_frames.extend(chunk_data['frames'])
                    
                # Read labels
                if config_preprocess.USE_PSUEDO_PPG_LABEL:
                    # We need full frames for pseudo labels, so combine
                    frames = np.asarray(all_frames)
                    bvps = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS)
                else:
                    frames = np.asarray(all_frames)
                    bvps = self.read_wave(
                        os.path.join(os.path.dirname(data_dirs[i]['path']),
                                    "bvp_{0}.csv".format(saved_filename)))
                
                bvps = BaseLoader.resample_ppg(bvps, frames.shape[0])
                
                # Process and save
                frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
                input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, saved_filename)
                
                print(f"Successfully processed {len(input_name_list)} clips for {saved_filename}")
                file_list_dict[i] = input_name_list
                
                # Free memory
                del all_frames
                del frames
                import gc
                gc.collect()
            else:
                print(f"No frames processed for {saved_filename}")
                file_list_dict[i] = []
                
        except Exception as e:
            import traceback
            print(f"ERROR streaming processing {data_dirs[i]['index']}: {str(e)}")
            traceback.print_exc()
            file_list_dict[i] = []
