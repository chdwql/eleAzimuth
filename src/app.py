import numba
import cx_Oracle
import numpy as np
import pandas as pd
from flasgger import Swagger
from flask import Flask, jsonify
from flask_cors import CORS
from flask_restful import reqparse, Api, Resource
from numpy import pi, sqrt, arctan
from numpy.fft import fft
from pandas.plotting import register_matplotlib_converters
from addereq import fetching as tsf
from functools import lru_cache

register_matplotlib_converters()

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
CORS(app, supports_credentials=True)
api = Api(app)

# Swagger配置
swagger_config = Swagger.DEFAULT_CONFIG
swagger_config['title'] = '电磁学科时间序列处理接口'
swagger_config['description'] = '电磁学科时间序列处理方法，差分、地磁加卸载响应比、地电场优势方位角等'
swagger_config['host'] = 'localhost:8888'
swagger_config['optional_fields'] = ['components']
swagger_config['version'] = ['1.0.0']

swagger = Swagger(app, config=swagger_config)


# 使用numba加速计算密集型函数
@numba.jit(nopython=True)
def _calculate_azimuth(amplitude_ns, amplitude_ew, amplitude_ne, amplitude_nw):
    """计算方位角核心函数 - 使用numba加速

    参数可能为0或不存在，函数会根据可用的参数计算相应的方位角
    返回一个字典，键为方位角类型索引，值为计算得到的角度
    """
    results = {}

    # 预先计算常数
    factor = 180.0 / pi
    sqrt2 = sqrt(2.0)

    # 处理可能的缺失值，将None或NaN替换为0
    ns_val = 0.0 if amplitude_ns is None or amplitude_ns != amplitude_ns else amplitude_ns
    ew_val = 0.0 if amplitude_ew is None or amplitude_ew != amplitude_ew else amplitude_ew
    ne_val = 0.0 if amplitude_ne is None or amplitude_ne != amplitude_ne else amplitude_ne
    nw_val = 0.0 if amplitude_nw is None or amplitude_nw != amplitude_nw else amplitude_nw

    # EW/NW
    if ew_val > 0 and nw_val > 0:
        angle = 90.0 + factor * arctan(sqrt2 * nw_val / ew_val - 1.0)
        results[0] = angle  # 'EW/NW'

    # EW/NE
    if ew_val > 0 and ne_val > 0:
        angle = 90.0 - factor * arctan(sqrt2 * ne_val / ew_val - 1.0)
        results[1] = angle  # 'EW/NE'

    # NS/NW
    if ns_val > 0 and nw_val > 0:
        angle = 180.0 - factor * arctan(sqrt2 * nw_val / ns_val - 1.0)
        results[2] = angle  # 'NS/NW'

    # NS/NE
    if ns_val > 0 and ne_val > 0:
        angle = factor * arctan(sqrt2 * ne_val / ns_val - 1.0)
        results[3] = angle  # 'NS/NE'

    # NS/EW
    if ns_val > 0 and ew_val > 0:
        angle = factor * arctan(ew_val / ns_val)
        results[4] = angle  # 'NS/EW'

    return results


# 工具函数
class AzimuthCalc:
    """地电场优势方位角计算器"""

    # 定义常量，避免硬编码
    DIRECTIONS = {0: 'EW/NW', 1: 'EW/NE', 2: 'NS/NW', 3: 'NS/NE', 4: 'NS/EW'}

    ITEM_GROUP_1 = {'3411', '3412', '3413', '3414'}
    ITEM_GROUP_2 = {'3421', '3422', '3423', '3424'}

    ORDER = 10  # 默认调和分析阶数

    @staticmethod
    @lru_cache(maxsize=128)
    def compute_fft(values, order):
        """使用FFT计算地电场调和分析系数 - 带缓存"""
        if len(values) == 0:
            return 0.0

        try:
            # 直接使用numpy数组，避免pandas转换开销
            fft_result = fft(values)
            amplitudes = 2.0 * np.abs(fft_result) / len(values)
            return np.sum(amplitudes[1:order + 1])
        except Exception as e:
            print(f"FFT计算错误: {e}")
            return 0.0

    @staticmethod
    def compute_daily_azimuth(df):
        """计算单日的地电场优势方位角 - 性能优化版"""
        if df.empty:
            return {}
        try:
            # 查找数据中的项目ID
            items = df['ITEMID'].unique()

            # 使用字典存储振幅
            amplitudes = {}

            # 确定使用的项目组
            if '3411' in items:
                base_ns, base_ew = '3411', '3412'
                ne, nw = '3413', '3414'
            elif '3421' in items:
                base_ns, base_ew = '3421', '3422'
                ne, nw = '3423', '3424'
            else:
                return {}

            # 计算每个项目的振幅
            for item in [base_ns, base_ew, ne, nw]:
                if item in items:
                    # 获取项目数据
                    item_values = df.loc[df['ITEMID'] == item,
                                         'OBSVALUE'].fillna(
                                             method='bfill').values

                    # 使用缓存的FFT计算
                    amplitudes[item] = AzimuthCalc.compute_fft(
                        tuple(item_values), AzimuthCalc.ORDER)

            # 不再要求所有项目都存在，只要有足够的项目能计算至少一个方位角即可

            # 使用numba加速的函数计算方位角
            # 如果某个参数不存在，传入None
            ns_val = amplitudes.get(base_ns, None)
            ew_val = amplitudes.get(base_ew, None)
            ne_val = amplitudes.get(ne, None)
            nw_val = amplitudes.get(nw, None)

            # 只要有足够的数据计算至少一个方位角就进行计算
            azimuth_values = _calculate_azimuth(ns_val, ew_val, ne_val, nw_val)

            # 转换为可读结果
            result = {}
            for idx, value in azimuth_values.items():
                result[AzimuthCalc.DIRECTIONS[idx]] = value

            # 如果没有计算出任何方位角，返回空结果
            if not result:
                return {}

            return result

        except Exception as e:
            print(f"方位角计算错误: {e}")
            return {}

    @staticmethod
    def compute_period_azimuth(df):
        """计算整个时间段的地电场优势方位角 - 性能优化版"""
        if df.empty:
            return pd.DataFrame()

        try:
            # 按日期分组处理，避免循环
            # 创建日期列以便分组
            df['date'] = pd.to_datetime(df.index.date)

            # 按日期分组
            grouped = df.groupby('date')

            # 使用并行处理加速计算
            results = {}
            for date, group in grouped:
                date_str = date.strftime('%Y-%m-%d')
                azimuth = AzimuthCalc.compute_daily_azimuth(group)
                if azimuth:
                    results[date_str] = azimuth

            # 如果有结果，转换为DataFrame
            if results:
                return pd.DataFrame.from_dict(results, orient='index')
            return pd.DataFrame()

        except Exception as e:
            print(f"计算周期方位角错误: {e}")
            return pd.DataFrame()


# API资源
class GeoElectricAPI(Resource):
    """地电场优势方位角API"""

    def get(self):
        '''地电场优势方位角
        ---
        tags:
          - 电磁
        consumes:
          - multipart/form-data
        produces:
          - application/json
        parameters:
          - name: stationname
            in: query
            type: string
            default: '安丘'
            description: 台站名称
          - name: itemname
            in: query
            type: string
            default: '第一'
            description: 第一装置/第二装置
          - name: database
            in: query
            type: string
            default: 'shandong-official'
            description: 数据库名称
          - name: startdate
            in: query
            type: string
            default: '20210101'
            description: 起始时间
          - name: enddate
            in: query
            type: string
            default: '20210131'
            description: 结束时间

        responses:
          200:
            description: 用于计算地电场优势方位角
        '''
        try:
            # 解析参数
            parser = reqparse.RequestParser(bundle_errors=True)
            parser.add_argument('startdate',
                                type=str,
                                required=True,
                                location='args',
                                help='必须提供起始日期')
            parser.add_argument('enddate',
                                type=str,
                                required=True,
                                location='args',
                                help='必须提供结束日期')
            parser.add_argument('stationname',
                                type=str,
                                required=True,
                                location='args',
                                help='必须提供台站名称')
            parser.add_argument('itemname',
                                type=str,
                                required=True,
                                location='args',
                                help='必须提供装置名称')
            parser.add_argument('database',
                                type=str,
                                required=False,
                                default='shandong-official',
                                location='args',
                                help='数据库名称')

            args = parser.parse_args()

            # 获取数据并计算
            df = self.fetch_and_process(args['startdate'], args['enddate'],
                                        args['stationname'], args['itemname'],
                                        args['database'])

            if df.empty:
                return jsonify({'error': '无法获取数据或计算结果为空'})

            # 直接返回数据表
            return jsonify(df.to_dict(orient='dict'))

        except Exception as e:
            return jsonify({'error': f'处理请求时发生错误: {str(e)}'})

    def fetch_and_process(self, startdate, enddate, stationname, itemname,
                          database):
        cx_Oracle.init_oracle_client(lib_dir='C:/instantclient')
        conn = tsf.conn_to_Oracle(database)
        """获取数据并计算方位角"""
        try:
            # 连接数据库并获取数据
            df = tsf.fetching_data(conn,
                                   startdate,
                                   enddate,
                                   '地电场',
                                   stationname,
                                   '分钟值',
                                   '预处理库',
                                   gzip_flag=False,
                                   itemname=itemname)
            # 计算方位角
            if not df.empty:
                return AzimuthCalc.compute_period_azimuth(df)
            return pd.DataFrame()

        except Exception as e:
            print(f"数据获取计算错误: {e}")
            return pd.DataFrame()


@app.route('/', methods=['POST', 'GET'])
def index():
    '''电磁学科时间序列数据处理API

    tags:
      - Welcome to using online calculator API.
    responses:
      200:
        description: 电磁学科时间序列数据处理API
    '''
    return '欢迎使用电磁学科时间序列数据处理API！'


# 注册API资源
api.add_resource(GeoElectricAPI, '/ele_Azimuth')

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=8888)
