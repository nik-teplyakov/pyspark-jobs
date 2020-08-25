import io
import sys

from pyspark.ml import Pipeline
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark.sql.functions import mean, col, split, col, regexp_extract, when, lit, avg
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.feature import QuantileDiscretizer
from pyspark.ml.tuning import CrossValidatorModel

# Путь для загрузки модели
MODEL_PATH = 'spark_ml_model'


def process(spark, input_path, output_file):
    
    ads_test = spark.read.parquet(input_path)
        
    # Загрузка модели, скоринг и запись
    output = CrossValidatorModel.load('spark_ml_model').transform(ads_test)
    cols = ['ad_id', 'prediction']
    output = output.select(*cols)
    output.write.option('header', 'true').csv(str(output_file))

def main(argv):
    input_path = argv[0]
    output_file = argv[1]
    print('')
    print('--------------------------------')
    print('Input path to file: ' + input_path)
    print('Output path to file: ' + output_file)
    print('--------------------------------')
    print('')
    spark = _spark_session()
    process(spark, input_path, output_file)

def _spark_session():
    return SparkSession.builder.appName('PySparkMLPredict').getOrCreate()


if __name__ == "__main__":
    arg = sys.argv[1:]
    if len(arg) != 2:
        sys.exit("Input and Target path are require.")
    else:
        main(arg)