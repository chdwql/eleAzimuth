import os
import numba
import numpy as np
import pandas as pd
import plotly.express as px
from flasgger import Swagger
from flask import Flask, jsonify
from flask_cors import CORS
from flask_restful import reqparse, Api, Resource
from numpy import pi, sqrt, arctan, sin, cos
from numpy.fft import fft
from pandas.plotting import register_matplotlib_converters
from addereq import fetching as tsf
from functools import lru_cache

register_matplotlib_converters()

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
CORS(app, supports_credentials=True)
api = Api(app)

# 移除未使用的变量
# UPLOAD_PATH = os.path.abspath('.')

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
    """计算方位角核心函数 - 使用numba加速"""
    results = {}

    # 预先计算常数
    factor = 180.0 / pi
    sqrt2 = sqrt(2.0)

    # EW/NW
    if amplitude_ew > 0 and amplitude_nw > 0:
        angle = 90.0 + factor * arctan(sqrt2 * amplitude_nw / amplitude_ew -
                                       1.0)
        results[0] = angle  # 'EW/NW'

    # EW/NE
    if amplitude_ew > 0 and amplitude_ne > 0:
        angle = 90.0 - factor * arctan(sqrt2 * amplitude_ne / amplitude_ew -
                                       1.0)
        results[1] = angle  # 'EW/NE'

    # NS/NW
    if amplitude_ns > 0 and amplitude_nw > 0:
        angle = 180.0 - factor * arctan(sqrt2 * amplitude_nw / amplitude_ns -
                                        1.0)
        results[2] = angle  # 'NS/NW'

    # NS/NE
    if amplitude_ns > 0 and amplitude_ne > 0:
        angle = factor * arctan(sqrt2 * amplitude_ne / amplitude_ns - 1.0)
        results[3] = angle  # 'NS/NE'

    # NS/EW
    if amplitude_ns > 0 and amplitude_ew > 0:
        angle = factor * arctan(amplitude_ew / amplitude_ns)
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
                has_valid_items = all(item in items
                                      for item in AzimuthCalc.ITEM_GROUP_1)
            elif '3421' in items:
                base_ns, base_ew = '3421', '3422'
                ne, nw = '3423', '3424'
                has_valid_items = all(item in items
                                      for item in AzimuthCalc.ITEM_GROUP_2)
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

            # 如果缺少任何必需的项目，返回空结果
            if not all(item in amplitudes
                       for item in [base_ns, base_ew, ne, nw]):
                return {}

            # 使用numba加速的函数计算方位角
            azimuth_values = _calculate_azimuth(amplitudes[base_ns],
                                                amplitudes[base_ew],
                                                amplitudes[ne], amplitudes[nw])

            # 转换为可读结果
            result = {}
            for idx, value in azimuth_values.items():
                result[AzimuthCalc.DIRECTIONS[idx]] = value

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
        """获取数据并计算方位角"""
        try:
            # 连接数据库并获取数据
            with tsf.conn_to_Oracle(database) as conn:
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
