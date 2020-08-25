import io
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import min, max, sum, count, col, round, when, datediff
from pyspark.sql import functions as F

def process(spark, input_path, target_path):

    # Читаем локальный parquet-файл
    data = spark.read.parquet(input_path)

    # Ф-я для подсчета количества по условию
    count_cond = lambda cond: sum(when(cond, 1).otherwise(0))

    # Делаем все и сразу по обработке
    # Расчет -> обработка 0/1 -> присвоение типа -> присвоение лейбла
    data = data.groupby('ad_id') \
               .agg(sum('target_audience_count').cast('decimal').alias('target_audience_count'),
                    when(count_cond(F.col('has_video') == 1) > 0, 1).otherwise(0).cast('integer').alias('has_video'),
                    when(count_cond(F.col('ad_cost_type') == 'CPM') > 0, 1).otherwise(0).cast('integer').alias('is_cpm'),
                    when(count_cond(F.col('ad_cost_type') == 'CPC') > 0, 1).otherwise(0).cast('integer').alias('is_cpc'),
                    round(sum('ad_cost'), 2).alias('ad_cost'),
                    datediff(max(F.col('date')), min(F.col('date'))).alias('day_count'),
                    round((count_cond(F.col('event') == 'click') / count_cond(F.col('event') == 'view')), 4).alias('CTR'))
                    #.sort(col('has_video').desc())
    
    # Устранили пропуски
    data = data.fillna({'target_audience_count': 0, 'CTR': 0})

    for column in data.columns:
        print('Пропуски в', column, ':', data.select(column).withColumn('isNull_c',F.col(column).isNull()).where('isNull_c = True').count())

    # Сплитим на train, test и validate
    train, test, validate = data.randomSplit([0.5, 0.25, 0.25])

    for item in [(train, 'train'), (test, 'test'), (validate, 'validate')]:
        item[0].coalesce(4).write.option('header', 'true').parquet(str(target_path) + '/' + str(item[1]))


def main(argv):
    input_path = argv[0]
    print("Input path to file: " + input_path)
    target_path = argv[1]
    print("Target path: " + target_path)
    spark = _spark_session()
    process(spark, input_path, target_path)


def _spark_session():
    return SparkSession.builder.appName('PySparkJob').getOrCreate()


if __name__ == "__main__":
    arg = sys.argv[1:]
    if len(arg) != 2:
        sys.exit("Input and Target path are require.")
    else:
        main(arg)
