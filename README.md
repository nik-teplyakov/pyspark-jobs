# pyspark-jobs
Some Spark and SparkML jobs examples (preprocess-train-predict)

### [Ru]
Небольшой процесс, состоящий из трех pyspark-скриптов, выполняющих следующее:
1. **spark-preprocess-split.py**. Загрузка и обработка локальных данных. Запись в три блока данных для обучения: train, test и validate.
2. **sparkml-fit.py**. Обучение модели Random Forest с подбором гиперпараметров на кросс-валидации. Сохранение модели как пайплайн-объекта.
3. **sparkml-predict.py**. Применяем наш готовый пайплайн для предсказания на новых данных.

### [En]
A small pipeline consisting of three pyspark jobs doing the following:
1. **spark-preprocess-split.py**. Loading and processing of the local data. Writing data into three blocks: train, test and validate.
2. **sparkml-fit.py**. Training the Random Forest regressor with hyperparameter tuning on cross-validation. Saving the whole model as a pipeline object.
3. **sparkml-predict.py**. Usage of the ready-made pipeline for prediction on the unseen data.

**P.S.** Intra-code comments are in Russian.
