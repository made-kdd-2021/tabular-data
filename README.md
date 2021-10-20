# tabular-data
1. EDA&Preprocessing:
  * **Heart_Failure_EDA_and_Train_Test_Split.ipynb** - Разведочный анализ данных + базовая подготовка данных БЕЗ масштабирования и кодирования категориальных признаков. 
  Разбиение данных на трейн, тест и валидацию. 
  На выходе: полученные выборки данных - в папке **data**.

  * **Heart_Failure_Preprocessing.ipynb** - Подготовка пайплайнов для обработки данных. 
  На выходе: два сохраненных пайплайна - **prep_with_cat.dill** (с обработкой категориальных признаков) и **prep_without_cat.dill** (без обработки категориальных признаков)

2. **Examples.ipynb** - Примеры, как использовать препроцессинг, как доставать названия категориальных признаков и как использовать все это в модели. 
На выходе: **model_LR_example.dill** - сохраненная обученная тестовая модель(логистическая регрессия без подбора гиперпараметров просто для примера)

3. **data** - Итоговые выборки, с которыми работаем
4. **prep_with_cat.dill** и **prep_without_cat.dill** - Пайплайны для предобработки данных
5. **model_LR_example.dill** - Пример сохраненной обученной модели
