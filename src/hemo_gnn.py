import json
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from datasets import Hemophilia
from models.gnn_classifier import GnnNodeClassifier
from sklearn.metrics import f1_score
from loguru import logger as log

np.random.seed(seed=1234)
tf.random.set_seed(1234)


def trainning(hemo, x_train, y_train, x_val, y_val, x_test, y_test, fold, filename, gnumber):
    
    node_features, edges, edge_weights = hemo.meta['graph_info']
    lr=0.001
    num_epochs=300
    batch=32

    dnn_model = GnnNodeClassifier(graph_info=[node_features, edges.T, edge_weights],
                                num_classes=hemo.num_classes,
                                aggregation_type='mean',
                                combination_type='gru',
                                normalize=True,
                                hidden_units=[8,8],
                                dropout_rate=0.2,
                                name="gnn_model")

    # Compile the model.
    dnn_model.compile(optimizer=keras.optimizers.Adam(lr),
                    #optimizer=keras.optimizers.SGD(hiper['learning_rate'], momentum=0.9),
                    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    #loss=keras.losses.BinaryCrossentropy(),
                    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
                    #metrics=[keras.metrics.BinaryAccuracy(name="acc")],
    )

    # Create an early stopping callback.
    early_stopping = keras.callbacks.EarlyStopping(monitor="acc", patience=50, restore_best_weights=True)
    # callback to save log train
    history_logger=tf.keras.callbacks.CSVLogger(f'history/{fold}_graph_{gnumber}_{filename}', separator=",", append=True)
    # callback to save weigths 
    fname_model =  f"{fold}_{filename.split('.')[0]}_graph_{gnumber}.h5"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=f'checkpoints/{fname_model}', 
                                                     save_best_only=True, 
                                                     save_weights_only=True, 
                                                     verbose=0)

    history = dnn_model.fit(
            x=x_train,
            y=y_train,
            epochs=num_epochs,
            batch_size=batch,
            validation_data=(x_val, y_val),
            callbacks=[early_stopping, history_logger, cp_callback],     
            verbose=0)

    _, test_accuracy = dnn_model.evaluate(x=x_test, y=y_test, verbose=0)
    logits = dnn_model.predict(tf.convert_to_tensor(x_test))
    probabilities = keras.activations.softmax(tf.convert_to_tensor(logits)).numpy()
    y_pred = np.argmax(probabilities, axis=1)
    #y_pred = np.argmax(probabilities, axis=1)
    #y_pred = np.where(logits[:,0] < 0.5, 1, 0)

    f1 = f1_score(y_test, y_pred, average='macro')

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

        acc, f1 = trainning(hemo, x_train, y_train, x_val, y_val, x_test, y_test, i, filename, gnumber)

        scores['acc'].append(round(acc, 2))
        scores['f1'].append(round(f1, 2))

        log.info(f'Fold {i+1} acc: {acc:.2f} f1: {f1:.2f}')

    log.info('-'*30)
    log.info(f"Summary: mean acc: {np.mean(scores['acc']):.2f} f1: {np.mean(scores['f1']):.2f}")

    return scores


if __name__ == '__main__':

    # classfication_severity_variant_1 - leg_pos	Residue	Domain	relSESA	consurf_old	severity
    
    gnumber = 4
    fname = 'classification_antigen_FVIII_variant_2'#classification_antigen_FVIII_variant_2
    root = 'data/'
    gf = f'{root}base_graphs/v2/AF_FVIII_structure_variant_{gnumber}_v2.csv'
    log.add(f'logs/gnn_{fname}.log')

    with open(f'6splits/{fname}.json', 'r') as fp:
        data = json.load(fp)    

    ff = f'{root}Input_datasets_FVIII/v2/{fname}.csv' #AF_FVIII_structure_variant_1

    #fn = ['areaSAS', 'consurf_old']
    fn = ['relSESA', 'consurf_old'] #
    ab = 'leg_pos'
    at = 'severity'
    filename = f'gnn_{fname}.csv'

    hemo = Hemophilia(gf, ff, fn, ab, at, k=6, seed=1)

    scores = run_experiment(data, hemo, filename, gnumber)
    data = []
    for key in scores.keys():
        print(f"{key} = {np.mean(scores[key])}: {scores[key]}")
        row = {f'fold_{i+1}': scores[key][i] for i in range(len(scores[key]))}
        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(f'results/gnn_{fname}_graph_{gnumber}.csv', index=False)