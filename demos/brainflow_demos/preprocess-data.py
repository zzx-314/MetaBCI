from demos.Utils import *
from scipy import signal
from demos.brainflow_demos.Preprocess import *
from metabci.brainda.utils.io import loadmat   #使用更鲁棒的mat读取函数

class Preprocess_OSA(Preprocess):

    def __init__(self):
        super().__init__()
        u = Utils()
        self.dataset_config = u.ReadConfig('Dataset_OSA')
        self.preprocess_config = u.ReadConfig('Preprocess_OSA')
        self.data_file_list = u.GetFileList(self.dataset_config['original_data_path'], '.mat',
                                            self.preprocess_config['exclude_subjects_data'])
        self.label_file_list = u.GetFileList(self.dataset_config['label_path'], '.txt',
                                             self.preprocess_config['exclude_subjects_label'])
        self.data_file_list.sort()
        self.label_file_list.sort()

    def Read1DataFile(self, file_name):   #读取单个数据文件
        file_path = os.path.join(self.dataset_config['original_data_path'], file_name)
        mat_data = loadmat(file_path)  # 使用 io.py 中更通用的 loadmat 读取函数
        resample = 3000
        psg_use = list()
        for each_channel in self.preprocess_config['channels_to_use']:
            psg_use.append(np.expand_dims(signal.resample(mat_data[each_channel], resample, axis=-1).astype(np.float32), 1))
        psg_use = np.concatenate(psg_use, axis=1)
        return psg_use

    def Read1LabelFile(self, file_name):
        file_path = os.path.join(self.dataset_config['label_path'], file_name)
        original_label = list()

        with open(file_path, "r") as f:
            for line in f.readlines():
                if line.strip():  # 检查是否为空行
                    label = int(line.strip('\n'))
                    original_label.append(label)

        return np.array(original_label)

    def count_class_statistics(self, label):   # 类别统计相关函数
        class_statistics = {
            'W': np.sum(label == 0),
            'N1': np.sum(label == 1),
            'N2': np.sum(label == 2),
            'N3': np.sum(label == 3),
            'REM': np.sum(label == 4),
        }
        return class_statistics

    def save_statistics_to_txt(self, patient_class_statistics, total_class_statistics, save_path):  # 保存每类患者样本的数量
        with open(os.path.join(save_path, 'osa_class_statistics.txt'), 'w') as file:
            file.write("每个患者的每个类别数量：\n")
            for patient_id, class_statistics in patient_class_statistics.items():
                file.write(f"患者 {patient_id}: {class_statistics}\n")

            file.write("\n所有患者样本的各个类的数量：\n")
            for class_key, class_count in total_class_statistics.items():
                file.write(f"类别 {class_key}: {class_count}\n")

if __name__ == '__main__':
    osa_process = Preprocess_OSA()

    fold_label = []
    fold_data = []
    fold_len = []
    patient_class_statistics = {}   # 用于保存每个患者的类别统计信息

    data_dir = osa_process.dataset_config['original_data_path']
    label_dir = osa_process.dataset_config['label_path']

    for i in range(0, len(osa_process.data_file_list)):   # 遍历每个病人的数据和标签文件
        print('Read data file:', osa_process.data_file_list[i], ' label file:', osa_process.label_file_list[i])
        data_path = os.path.join(data_dir, osa_process.data_file_list[i])
        data = osa_process.Read1DataFile(data_path)
        label_path = os.path.join(label_dir, osa_process.label_file_list[i])
        label = osa_process.Read1LabelFile(label_path)

        # 修正标签，将大于等于5的值映射为合适的范围（0到4）
        label[label >= 5] = 4

        print('data shape:', data.shape, ', label shape', label.shape)
        assert len(label) == len(data)#检查样本数是否一致

        fold_label.append(np.eye(5)[label])
        fold_data.append(data.astype(np.float32))
        fold_len.append(len(label))

        # 统计每个患者的类别数量
        patient_id = osa_process.label_file_list[i].split('.')[0]
        if patient_id not in patient_class_statistics:
            patient_class_statistics[patient_id] = osa_process.count_class_statistics(label)
        else:
            for class_key, class_count in osa_process.count_class_statistics(label).items():
                patient_class_statistics[patient_id][class_key] += class_count

    print('Preprocess over.')

    # 打印每个患者的类别数量
    print("\n每个患者的每个类别数量：")
    for patient_id, class_statistics in patient_class_statistics.items():
        print(f"患者 {patient_id}: {class_statistics}")

    # 统计所有患者样本的各个类的数量
    total_class_statistics = {}
    for class_key in ['W', 'N1', 'N2', 'N3', 'REM']:
        total_class_statistics[class_key] = sum(
            [class_statistics[class_key] for class_statistics in patient_class_statistics.values()]
        )
    print("\n所有患者样本的各个类的数量：", total_class_statistics)

    osa_process.save_statistics_to_txt(patient_class_statistics, total_class_statistics,
                                        osa_process.preprocess_config['save_path'])

    np.savez(os.path.join(osa_process.preprocess_config['save_path'], '1.npz'),
             Fold_data=fold_data,
             Fold_label=fold_label,
             Fold_len=fold_len
    )
    print('Saved to', os.path.join(osa_process.preprocess_config['save_path'], '1.npz'))
