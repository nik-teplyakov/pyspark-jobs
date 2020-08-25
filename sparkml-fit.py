import io
import sys

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.sql.functions import mean, col, split, col, regexp_extract, when, lit, avg
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.feature import QuantileDiscretizer

from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator


# Путь, куда сохранить модель
MODEL_PATH = 'spark_ml_model'


def process(spark, train_data, test_data):
    
    ads_train = spark.read.parquet(train_data)
    ads_test = spark.read.parquet(test_data)
    
    ### Создаем пайплайн
    # Первая часть пайплайна - генерация фичей
    discretize_audience = QuantileDiscretizer(numBuckets=5, inputCol='target_audience_count', outputCol='aud_bins')
    discretize_days = QuantileDiscretizer(numBuckets=3, inputCol='day_count', outputCol='days_bins')
    discretize_cost = QuantileDiscretizer(numBuckets=4, inputCol='ad_cost', outputCol='cost_bins')
    featureGenerator = [discretize_audience, discretize_days, discretize_cost]

    # Вторая часть пайплайна - векторизация фичей
    featurize_list = ads_train.columns
    for elem in ['aud_bins', 'days_bins', 'cost_bins']:
        featurize_list.append(elem)

    featurize_list = [col for col in featurize_list if col not in 'ctr']
    featurizer = VectorAssembler(inputCols=featurize_list, outputCol='raw_features')

    # Третья часть пайплайна - индексация фичей
    featureIndexer = VectorIndexer(inputCol='raw_features', outputCol='features', maxCategories=5)

    # Четвертая часть пайплайна - модель
    rf = RandomForestRegressor(labelCol='ctr', featuresCol='features')
    rfEval = RegressionEvaluator(predictionCol='prediction', labelCol='ctr', metricName='rmse')
    
    # Не будем перебирать все еще раз. Оставим лучшую версию модели
    paramGrid = ParamGridBuilder() \
        .addGrid(rf.maxDepth, [10]) \
        .addGrid(rf.numTrees, [75]) \
        .addGrid(rf.maxBins, [64]) \
        .addGrid(rf.featureSubsetStrategy, ['auto']) \
        .build()
        
        #.addGrid(rf.maxDepth, [6, 8, 10, 12]) \
        #.addGrid(rf.numTrees, [50, 75, 150, 300]) \
        #.addGrid(rf.maxBins, [32, 64, 128]) \
        #.addGrid(rf.featureSubsetStrategy, ['auto', 'sqrt', 'log2']) \

    # Полный ML-пайплайн
    pipeline = Pipeline(stages=[discretize_audience, discretize_days, discretize_cost, 
                                featurizer, featureIndexer, rf])

    cv = CrossValidator(estimator=pipeline,
                        estimatorParamMaps=paramGrid,
                        evaluator=rfEval,
                        numFolds=3)

    rfModel = cv.fit(ads_train)
    rfModel.save(MODEL_PATH)
    
    trainPreds = rfModel.transform(ads_train)   
    testPreds = rfModel.transform(ads_test)

    train_rmse = round(rfEval.evaluate(trainPreds), 5)
    test_rmse = round(rfEval.evaluate(testPreds), 5)
    
    # Параметры лучшей модели
    maxDepth = rfModel.bestModel.stages[-1]._java_obj.getMaxDepth()
    numTrees = rfModel.bestModel.stages[-1]._java_obj.getNumTrees()
    maxBins = rfModel.bestModel.stages[-1]._java_obj.getMaxBins()
    featureSubsetStrategy = rfModel.bestModel.stages[-1]._java_obj.getFeatureSubsetStrategy()
  
    return train_rmse, test_rmse, maxDepth, numTrees, maxBins, featureSubsetStrategy

# Очень общительная консоль
def main(argv):
    train_data = argv[0]
    test_data = argv[1]
    print('')
    print('---------------------------------------')
    print("Input path to train data: " + train_data)
    print("Input path to test data: " + test_data)
    print('---------------------------------------')
    print('')
    spark = _spark_session()
    train_rmse, test_rmse, maxDepth, numTrees, \
    maxBins, featureSubsetStrategy = process(spark, train_data, test_data)
    print('')
    print('-------------------')
    print('Train RMSE:', train_rmse)
    print('Test RMSE:', test_rmse)
    print('-------------------')
    print('')
    print('-----------------------------')
    print('Parameters of the best model:')
    print('Max Depth: {}'.format(maxDepth))
    print('Num Trees: {}'.format(numTrees))
    print('Max Bins: {}'.format(maxBins))
    print('Feature Subset Strategy: {}'.format(featureSubsetStrategy))
    print('-----------------------------')


def _spark_session():
    return SparkSession.builder.appName('PySparkMLFitJob').getOrCreate()


if __name__ == "__main__":
    arg = sys.argv[1:]
    if len(arg) != 2:
        sys.exit("Train and test data required.")
    else:
        main(arg)