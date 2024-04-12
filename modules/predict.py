import json
import dill as pickle
import pandas as pd
import os

from datetime import datetime

path = os.environ.get('PROJECT_PATH', '.')


def gen_files_list() -> list:
    """
    Генерируем листы с путями и названиями файлов
    :return list:
    """

    path_to_test = f'{path}/data/test/'
    all_files = os.listdir(path_to_test)
    model_file = os.listdir(f'{path}/data/models/')[0]

    return model_file, [f'{path_to_test}{i}' for i in all_files]


def predict() -> None:
    """
    Предсказываем фичу на натренерованных данных model
    :return None:
    """
    new_list = list()
    pk_file = f'{path}/data/models/{gen_files_list()[0]}'
    print('Отчет 321!!!!')
    with open(pk_file, 'rb') as file:
        model = pickle.load(file)

    for file_json in gen_files_list()[1]:
        with open(file_json, 'r') as f:
            data = json.load(f)

        df = pd.DataFrame.from_dict([data])
        y = model.predict(df)

        new_list.append(
            {'car_id': df.id[0], 'pred': y[0]}
        )

    new_data = pd.DataFrame(new_list)
    new_data.to_csv(f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv',
                    encoding='utf-8',
                    index=False)


if __name__ == '__main__':
    predict()

