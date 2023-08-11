import json
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from datasets import Hemophilia
from models.gat_classifier import GraphAttentionNetwork
from sklearn.metrics import f1_score
from loguru import logger as log

seed = 123456789
tf.keras.utils.set_random_seed(seed)


def trainning_gat(hemo, x_train, y_train, x_val, y_val, x_test, y_test, fold, filename, gnumber):
    # Define hyper-parameters
    HIDDEN_UNITS = 100
    NUM_HEADS = 10
    NUM_LAYERS = 4
    OUTPUT_DIM = 2

    NUM_EPOCHS = 100
    BATCH_SIZE = 12
    VALIDATION_SPLIT = 0.1
    LEARNING_RATE = 3e-1
    MOMENTUM = 0.9

    # Build model
    gat_model = GraphAttentionNetwork(
        hemo.node_features, hemo.edges, HIDDEN_UNITS, NUM_HEADS, NUM_LAYERS, OUTPUT_DIM
    )

    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = keras.optimizers.SGD(LEARNING_RATE, momentum=MOMENTUM)
    accuracy_fn = keras.metrics.SparseCategoricalAccuracy(name="acc")
    
    #
    # Callbacks
    #
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_acc", min_delta=1e-5, patience=5, restore_best_weights=True
    )
    # callback to save log train
    history_logger=tf.keras.callbacks.CSVLogger(f'history/{fold}_graph_{gnumber}_{filename}', separator=",", append=True)
    # callback to save weigths 
    fname_model =  f"{fold}_{filename.split('.')[0]}_graph_{gnumber}.h5"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=f'checkpoints/{fname_model}', 
                                                     save_best_only=True, 
                                                     save_weights_only=True, 
                                                     verbose=0)
    # Compile model
    gat_model.compile(loss=loss_fn, optimizer=optimizer, metrics=[accuracy_fn])
    
    # fit the model
    gat_model.fit(
        x=x_train,
        y=y_train,
        validation_data=(x_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        callbacks=[early_stopping, history_logger, cp_callback],
        verbose=0,
    )

    _, test_accuracy = gat_model.evaluate(x=x_test, y=y_test, verbose=0)
    y_pred = gat_model.predict(x_test)
    f1 = f1_score(y_test, np.argmax(y_pred, axis=1), average='macro')
    #print("--" * 38)
    #print(f"Test Accuracy {test_accuracy*100:.1f}%")
    #print(f"Test F1 {f1:.2f}%")
    #print("--" * 38)

    return test_accuracy, f1

def run_experiment(folds, hemo, filename, gnumber):

    scores = {'acc': [], 'f1': []}

    for i, fold in enumerate(folds):
        train, val, test = fold['train'], fold['val'], fold['test']
        x_train = np.array(train)
        y_train = hemo.y[train]
        x_val = np.array(val)
        y_val = hemo.y[val]
        x_test = np.array(test)
        y_test = hemo.y[test]

        acc, f1 = trainning_gat(hemo, x_train, y_train, x_val, y_val, x_test, y_test, i, filename, gnumber)

        scores['acc'].append(round(acc, 2))
        scores['f1'].append(round(f1, 2))

        log.info(f'Fold {i+1} acc: {acc:.2f} f1: {f1:.2f}')

    log.info('-'*30)
    log.info(f"Summary: mean acc: {np.mean(scores['acc']):.2f} f1: {np.mean(scores['f1']):.2f}")

    return scores


if __name__ == '__main__':
    
    #gnumber = 4
    
    for gnumber in [2,3,4]:

        fname = 'classfication_severity_variant_2'
        root = 'data/'
        
        gf = f'{root}base_graphs/v2/AF_FVIII_structure_variant_{gnumber}_v2.csv'
        log.add(f'logs/gat_{fname}.log')

        with open(f'6splits/{fname}.json', 'r') as fp:
            data = json.load(fp)    

        ff = f'{root}Input_datasets_FVIII/v2/{fname}.csv' #AF_FVIII_structure_variant_1

        fn = ['areaSAS', 'consurf_old']
        #fn = ['relSESA', 'consurf_old'] #
        ab = 'leg_pos'
        at = 'severity'
        filename = f'gat_{fname}.csv'

        hemo = Hemophilia(gf, ff, fn, ab, at, k=6)

        scores = run_experiment(data, hemo, filename, gnumber)
        data = []
        for key in scores.keys():
            log.debug(f"{key} = {np.mean(scores[key]):.2f}: {scores[key]}")
            row = {f'fold_{i+1}': scores[key][i] for i in range(len(scores[key]))}
            data.append(row)

        df = pd.DataFrame(data)
        df.to_csv(f'results/gat_{fname}_graph_{gnumber}.csv', index=False)

