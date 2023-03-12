import glob
import json
import os
import dill
import pandas as pd

path = os.environ.get('PROJECT_PATH', '..')

def predict():
    def prediction(df_predict):
        y = model.predict(df)
        body = {'id': df.id, 'predict': y}
        df_body = pd.DataFrame(body)
        df_predict = pd.concat([df_predict, df_body], axis=0)

        return df_predict

    with open(max(glob.glob(f'{path}/data/models/*.pkl'), key=os.path.getctime), 'rb') as file_pkl:
        model = dill.load(file_pkl)

    df_predicted = pd.DataFrame(columns=['id', 'predict'])

    for file in os.listdir(f'{path}/data/test'):
        with open(os.path.join(f'{path}/data/test', file), 'r') as f:
            data = json.load(f)
            df = pd.DataFrame([data])
            df_predicted = prediction(df_predicted)

    name = file_pkl.name.split("/")[-1].split("\\")[-1].split(".")[0]
    df_predicted.to_csv(f'{path}/data/predictions/{name}.csv', index=False)


if __name__ == '__main__':
    predict()
