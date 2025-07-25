import gc
import tensorflow as tf
from keras import backend as KTF
from metabci.brainda.algorithms.deep_learning.FeatureNet import build_FeatureNet
from metabci.brainda.algorithms.deep_learning.DG import kFoldGenerator
from Utils import *
from keras import optimizers
from keras.callbacks import ModelCheckpoint

print(128 * '#')
print('Start to train FeatureNet.')

# # 1. Get configuration

# ## 1.1. Read .config file

utils = Utils()
preprocess_config = utils.ReadConfig('Preprocess_OSA')
featurenet_train_config = utils.ReadConfig('FeatureNet_Train')

# set GPU number or use CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = featurenet_train_config['GPU']
if featurenet_train_config['GPU'] != "-1":
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    KTF.set_session(sess)
    print("Use GPU #" + featurenet_train_config['GPU'])
else:
    print("Use CPU only")

# ## 1.2. Analytic parameters

channels = int(featurenet_train_config["channels"])
fold = int(featurenet_train_config["fold"])
num_epochs_f = int(featurenet_train_config["epoch_f"])
batch_size_f = int(featurenet_train_config["batch_size_f"])
optimizer_f = featurenet_train_config["optimizer_f"]
learn_rate_f = float(featurenet_train_config["learn_rate_f"])
preprocessed_data_path = featurenet_train_config["path_preprocessed_data"]
output_path = featurenet_train_config["path_feature"]

# ## 1.3. Parameter check and enable

# Create save pathand copy .config to it0
if not os.path.exists(output_path):
    os.makedirs(output_path)

# # 2. Read data and process data

# ## 2.1. Read data
# Each fold corresponds to one subject's data (OSA dataset)

# OSA load
ReadList = np.load(preprocessed_data_path, allow_pickle=True)
Fold_Num = ReadList['Fold_len']  # Num of samples of each fold
Fold_Data = ReadList['Fold_data']  # Data of each fold
Fold_Label = ReadList['Fold_label']  # Labels of each fold

print("Read data successfully")
print('Number of samples: ', np.sum(Fold_Num))

# ## 2.2. Build kFoldGenerator or DominGenerator
DataGenerator = kFoldGenerator(Fold_Data, Fold_Label, fold_size=3)

# # 3. Model training (cross validation)

# k-fold cross validation
all_scores = []
for i in range(0, fold):
    print(128 * '_')
    print('Fold #', i)

    # optimizer of FeatureNet
    opt_f = keras.optimizers.Adam(learning_rate=learn_rate_f)
    # get i th-fold data
    train_data, train_targets, val_data, val_targets = DataGenerator.getFold(i)
    print('train_data.shape', train_data.shape)
    print('train_targets', train_targets.shape)
    print('val_data', val_data.shape)

    ## build FeatureNet & train
    featureNet, featureNet_p = build_FeatureNet(opt_f, channels,
                                                train_targets)  # '_p' model is without the softmax layer
    # if i == 0:
    #     featureNet.summary()
    # ...

    history_fea = featureNet.fit(
        x=train_data,
        y=train_targets,
        epochs=num_epochs_f,
        batch_size=batch_size_f,
        shuffle=True,
        validation_data=(val_data, val_targets),
        verbose=2,
        callbacks=[keras.callbacks.ModelCheckpoint(output_path + 'FeatureNet_Best_' + str(i) + '.h5',
                                                   monitor='acc',
                                                   verbose=0,
                                                   save_best_only=True,
                                                   save_weights_only=False,
                                                   mode='auto',
                                                   save_freq='epoch')])  # 使用save_freq替代period

    # Save training information
    if i == 0:
        fit_loss = np.array(history_fea.history['loss']) * Fold_Num[i]
        fit_acc = np.array(history_fea.history['acc']) * Fold_Num[i]
        fit_val_loss = np.array(history_fea.history['val_loss']) * Fold_Num[i]
        fit_val_acc = np.array(history_fea.history['val_acc']) * Fold_Num[i]
    else:
        fit_loss = fit_loss + np.array(history_fea.history['loss']) * Fold_Num[i]
        fit_acc = fit_acc + np.array(history_fea.history['acc']) * Fold_Num[i]
        fit_val_loss = fit_val_loss + np.array(history_fea.history['val_loss']) * Fold_Num[i]
        fit_val_acc = fit_val_acc + np.array(history_fea.history['val_acc']) * Fold_Num[i]

    # load the weights of best performance
    featureNet.load_weights(output_path + 'FeatureNet_Best_' + str(i) + '.h5')

    # get and save the learned feature
    train_feature = featureNet_p.predict(train_data)
    val_feature = featureNet_p.predict(val_data)
    print('Save feature of Fold #' + str(i) + ' to' + output_path + 'Feature_' + str(i) + '.npz')
    np.savez(output_path + 'Feature_' + str(i) + '.npz',
             train_feature=train_feature,
             val_feature=val_feature,
             train_targets=train_targets,
             val_targets=val_targets
             )

    saveFile = open(output_path + "Result_FeatureNet.txt", 'a+')
    print('Fold #' + str(i), file=saveFile)
    print(history_fea.history, file=saveFile)
    saveFile.close()

    # Fold finish
    KTF.clear_session()
    del featureNet, featureNet_p, train_data, train_targets, val_data, val_targets, train_feature, val_feature
    gc.collect()

print(128 * '_')
print('End of training FeatureNet.')
print(128 * '#')
