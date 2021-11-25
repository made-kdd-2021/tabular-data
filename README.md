# Первая часть проекта: работа с табличными данными
1. [**EDA_and_Preprocessing**](https://github.com/made-kdd-2021/tabular-data/tree/develop/EDA_and_Preprocessing):
  * [*Heart_Failure_EDA_and_Train_Test_Split.ipynb*](https://github.com/made-kdd-2021/tabular-data/blob/develop/EDA_and_Preprocessing/Heart_Failure_EDA_and_Train_Test_Split.ipynb) - Разведочный анализ данных + базовая подготовка данных БЕЗ масштабирования и кодирования категориальных признаков. 
  Разбиение данных на трейн и тест. 
  На выходе: полученные выборки данных - в папке **data**.

  * [*Heart_Failure_Preprocessing.ipynb*](https://github.com/made-kdd-2021/tabular-data/blob/develop/EDA_and_Preprocessing/Heart_Failure_Preprocessing.ipynb) - Подготовка пайплайнов для обработки данных. 
  На выходе: два сохраненных пайплайна - **prep_with_cat.dill** (с обработкой категориальных признаков) и **prep_without_cat.dill** (без обработки категориальных признаков)
  
  * **prep_with_cat.dill** и **prep_without_cat.dill** - Пайплайны для предобработки данных

2. [**Examples**](https://github.com/made-kdd-2021/tabular-data/tree/develop/Examples):
  * [*Get_RF_feature_importances_from_pipeline.ipynb*](https://github.com/made-kdd-2021/tabular-data/blob/develop/Examples/Get_RF_feature_importances_from_pipeline.ipynb) - Пример, как доставать названия категориальных признаков из пайплайна. 
  * [*Get_cat_features_for LIME+Pipeline_using_example.ipynb*](https://github.com/made-kdd-2021/tabular-data/blob/develop/Examples/Get_cat_features_for%20LIME%2BPipeline_using_example.ipynb) - Примеры, как использовать препроцессинг, как доставать названия категориальных признаков и как использовать все это в модели. 
На выходе: **model_LR_example.dill** - сохраненная обученная тестовая модель(логистическая регрессия без подбора гиперпараметров просто для примера)

3. [**data**](https://github.com/made-kdd-2021/tabular-data/tree/develop/data) - Итоговые выборки, с которыми работаем

4. [**models**](https://github.com/made-kdd-2021/tabular-data/tree/develop/models) - Итоговые модели, результаты которых будем интерпретировать

5. [**Heart_Failure_Discovering_causal_dependencies.ipynb**](https://github.com/made-kdd-2021/tabular-data/blob/develop/Heart_Failure_Discovering_causal_dependencies.ipynb) - Поиск причинно-следственных связей

6. [**Heart_Failure_SHAP_LIME.ipynb**](https://github.com/made-kdd-2021/tabular-data/blob/develop/Heart_Failure_SHAP_LIME.ipynb) - SHAP и LIME анализ

7. **requirements.txt** - Необходимые версии библиотек
