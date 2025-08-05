from Utils import *
from metabci.brainda.algorithms.deep_learning.STGCN import build_STGCN
from keras import optimizers
from keras.callbacks import ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import classification_report
import seaborn as sns
import shap
import pandas as pd
import keras

# 读取配置
utils = Utils()
para_train = utils.ReadConfig('Train')
para_model = utils.ReadConfig('STGCN')

# 训练参数
channels = int(para_train["channels"])
fold = int(para_train["fold"])
context = int(para_train["context"])

num_epochs = int(para_train["epoch"])
batch_size = int(para_train["batch_size"])
learn_rate = float(para_train["learn_rate"])
lr_decay = float(para_train["lr_decay"])
GLalpha = float(para_train["GLalpha"])
dense_size = np.array(para_model["Globaldense"])
num_of_time_filters = int(para_model["time_filters"])
time_conv_strides = int(para_model["time_conv_strides"])
time_conv_kernel = int(para_model["time_conv_kernel"])
cheb_k = int(para_model["cheb_k"])
l1 = float(para_model["l1"])
l2 = float(para_model["l2"])
dropout = float(para_model["dropout"])

# 路径设置
Path = {
    "feature": para_train["path_feature"],
    "data": para_train["path_preprocessed_data"],
    "output": para_train["path_output"]
}
if not os.path.exists(Path['feature']):
    os.makedirs(Path['feature'])

# 数据读取
ReadList = np.load(Path['data'], allow_pickle=True)
Fold_Num = ReadList['Fold_len']
Fold_Num_c = Fold_Num + 1 - context
print("读取数据成功，共有样本：", np.sum(Fold_Num), "(加上下文后：", np.sum(Fold_Num_c), ")")

# 优化器与正则项
opt = keras.optimizers.Adam(learning_rate=learn_rate, decay=lr_decay)
if l1 != 0 and l2 != 0:
    regularizer = keras.regularizers.l1_l2(l1=l1, l2=l2)
elif l1 != 0:
    regularizer = keras.regularizers.l1(l1)
elif l2 != 0:
    regularizer = keras.regularizers.l2(l2)
else:
    regularizer = None

# 保存结果
all_scores = []
all_mf1_scores = []
all_acc_scores = []

for i in range(fold):
    print(128 * "_")
    print(f"Fold #{i}")

    # 加载数据
    Features = np.load(Path['feature'] + f'Feature_{i}.npz', allow_pickle=True)
    train_feature = Features['train_feature']
    val_feature = Features['val_feature']
    train_targets = Features['train_targets']
    val_targets = Features['val_targets']

    train_feature, train_targets = AddContext_MultiSub(train_feature, train_targets,
                                                       np.delete(Fold_Num.copy(), i), context, i)
    val_feature, val_targets = AddContext_SingleSub(val_feature, val_targets, context)
    input_shape = val_feature.shape[1:]
    print("特征维度：", train_feature.shape, val_feature.shape)

    model = build_STGCN(input_shape, dense_size, opt, GLalpha, regularizer, dropout,
                        num_of_time_filters, time_conv_strides, time_conv_kernel)

    if i == 0:
        model.summary()

    history = model.fit(
        x=train_feature,
        y=train_targets,
        epochs=num_epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(val_feature, val_targets),
        verbose=2,
        callbacks=[ModelCheckpoint(
            Path['output'] + f'Best_model_{i}.h5',
            monitor='val_accuracy',
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode='auto',
            save_freq='epoch')]
    )

    # 预测与评估
    predictions = model.predict(val_feature)
    y_true = np.argmax(val_targets, axis=1)
    y_pred = np.argmax(predictions, axis=1)

    if np.any(np.isnan(predictions)) or np.any(np.isnan(y_true)):
        print(f"Fold #{i} 出现 NaN，跳过")
        continue

    cm = confusion_matrix(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    if np.isnan(f1) or np.isnan(precision) or np.isnan(recall):
        print(f"Fold #{i} 指标 NaN，跳过")
        continue

    acc = np.trace(cm) / np.sum(cm)
    all_mf1_scores.append(f1)
    all_acc_scores.append(acc)
    all_scores.append(cm)
    print(f"Fold #{i} Acc: {acc:.4f}, MF1: {f1:.4f}")

    # === SHAP解释 ===
    print("解释模型 SHAP...")
    explainer = shap.GradientExplainer(model, val_feature)
    shap_values = explainer.shap_values(val_feature[50:150])
    val_feature_reshaped = np.mean(val_feature[50:150], axis=(1, 3))  # (50, 11)

    mean_shap = np.mean(np.array(shap_values), axis=2)
    mean_shap = np.mean(mean_shap, axis=3)  # (5, 50, 11)
    class_names = ['W', 'N1', 'N2', 'N3', 'REM']
    feature_names = ["F3", "C3", "O1", "F4", "C4", "O2", "E1", "E2", "Chin1", "Chin2", "Chin3"]

    # === BAR 图 ===
    bar_data = pd.DataFrame(index=feature_names)
    for c in range(5):
        bar_data[class_names[c]] = np.abs(mean_shap[c]).mean(axis=0)
    bar_data["Total"] = bar_data.sum(axis=1)
    bar_data = bar_data.sort_values(by="Total", ascending=False).drop(columns="Total")
    bar_data = bar_data[::-1]  # 反转顺序

    plt.figure(figsize=(10, 8))
    bar_data.plot(kind='barh', stacked=True, color=sns.color_palette('Set2'))
    plt.legend(title="Class", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title("Feature Importance by Class")
    plt.xlabel("Mean SHAP Value")
    plt.tight_layout()
    plt.savefig(Path["output"] + f"/SHAP_Bar_Fold_{i}.png", dpi=1200, bbox_inches='tight')
    plt.close()

    # === 蜂群图 ===
    def plot_beeswarm(shap_vals, cls_idx, title):
        shap.summary_plot(shap_vals, val_feature_reshaped,
                          feature_names=feature_names,
                          plot_type="dot",
                          show=False)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(Path["output"] + f"/SHAP_Beeswarm_Fold_{i}_{title}.png", dpi=1200, bbox_inches='tight')
        plt.close()

    for c in range(5):
        plot_beeswarm(mean_shap[c], c, f"{class_names[c]}")



    # === 增强混淆矩阵图 ===
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    precision_per_class = [report[cls]["precision"] for cls in class_names]
    recall_per_class = [report[cls]["recall"] for cls in class_names]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)

    # 标注数字与百分比
    for ii in range(cm.shape[0]):
        for jj in range(cm.shape[1]):
            count = cm[ii, jj]
            percent = cm[ii, jj] / np.sum(cm[ii]) if np.sum(cm[ii]) > 0 else 0
            ax.text(jj + 0.5, ii + 0.3, f"{count}", ha='center', va='center', fontsize=10, color='black')
            ax.text(jj + 0.5, ii + 0.7, f"{percent:.1%}", ha='center', va='center', fontsize=8, color='gray')

    # 整体标签
    ax.set_xlabel('Predicted Labels', fontsize=12)
    ax.set_ylabel('True Labels', fontsize=12)
    ax.set_title(f'Confusion Matrix with Accuracy - Fold #{i}', fontsize=14)

    # 显示整体准确率
    ax.text(0, -0.7, f"Overall Accuracy: {acc:.2%}", fontsize=12, color='green')

    # 右侧标注 Precision / Recall
    for idx, cls in enumerate(class_names):
        ax.text(len(class_names) + 0.2, idx + 0.3, f"P: {precision_per_class[idx]:.2f}", fontsize=9, color='blue')
        ax.text(len(class_names) + 0.2, idx + 0.7, f"R: {recall_per_class[idx]:.2f}", fontsize=9, color='red')

    plt.tight_layout()
    plt.savefig(Path['output'] + f'/Confusion_Matrix_Enhanced_Fold_{i}.png', dpi=300, bbox_inches='tight')
    plt.close()

# 平均得分输出
if len(all_mf1_scores) > 0:
    avg_mf1 = np.mean(all_mf1_scores)
    avg_acc = np.mean(all_acc_scores)
    print("Average MF1:", avg_mf1)
    print("Average ACC:", avg_acc)

    with open(Path['output'] + "Average_Scores.txt", 'w') as f:
        f.write(f"Average MF1: {avg_mf1}\n")
        f.write(f"Average ACC: {avg_acc}\n")

    with open(Path['output'] + "Result_STGCN.txt", 'a+') as f:
        print(history.history, file=f)
else:
    print("所有 fold 被跳过，未生成有效结果")

print(128 * "_")
print("训练结束 ")

