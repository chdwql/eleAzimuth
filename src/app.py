import os
import sys
import glob
import cx_Oracle
import configparser
from urllib import parse
import datetime as dt
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import razorback as rb
from scipy import signal
from scipy.signal import argrelmax, argrelmin
from flasgger import Swagger
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from flask_restful import reqparse, Api, Resource
from matplotlib import rcParams
from matplotlib.pyplot import cm
from numpy import pi, sin, cos, sqrt, arctan
from numpy.fft import fft
from pandas.plotting import register_matplotlib_converters
import sqlalchemy
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import MetaData, create_engine
from addereq import fetching as tsf

register_matplotlib_converters()
if sys.platform == 'darwin':
    cx_Oracle.init_oracle_client(
        lib_dir='/Users/wangqinglin/Work.localized/Python/instantclient')

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
CORS(app, supports_credentials=True)
api = Api(app)

UPLOAD_PATH = os.path.abspath('.')

swagger_config = Swagger.DEFAULT_CONFIG
swagger_config['title'] = '电磁学科时间序列处理接口'
swagger_config['description'] = '电磁学科时间序列处理方法，差分、地磁加卸载响应比、地电场优势方位角等'
swagger_config['host'] = '0.0.0.0:8888'
swagger_config['optional_fields'] = ['components']
swagger_config['version'] = ['1.0.0']

swagger = Swagger(app, config=swagger_config)


@app.route('/', methods=['POST', 'GET'])
def hello():
    '''电磁学科时间序列数据处理API
    
    tags:
      - Welcome to using online calculator API.
    responses:
      200:
        description:  电磁学科时间序列数据处理API
    '''

    return '欢迎使用电磁学科时间序列数据处理API！'


class eleAzimuth(Resource):

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
          - name: returntype
            in: query
            type: boolean
            default: false
            description:  false:JSON数据表/true:JSON图片

        responses:
          200:
            description: 用于计算地电场优势方位角
        '''
        parser = reqparse.RequestParser(bundle_errors=True)
        parser.add_argument('startdate',
                            type=str,
                            required=False,
                            location='args',
                            help='startdate')
        parser.add_argument('enddate',
                            type=str,
                            required=False,
                            location='args',
                            help='enddate')
        parser.add_argument('stationname',
                            type=str,
                            required=False,
                            location='args',
                            help='stationname')
        parser.add_argument('itemname',
                            type=str,
                            required=False,
                            location='args',
                            help='itemname')
        parser.add_argument('database',
                            type=str,
                            required=False,
                            location='args',
                            help='database')
        parser.add_argument('returntype',
                            type=str,
                            required=False,
                            location='args',
                            help='return type')
        args = parser.parse_args()
        stationname = args['stationname']
        itemname = args['itemname']
        database = args['database']
        startdate = args['startdate']
        enddate = args['enddate']
        returntype = args['returntype']
        if returntype == 'true':
            df = self.geo_ele_Azimuth(startdate,
                                      enddate,
                                      stationname,
                                      itemname,
                                      figure_or_not=True,
                                      database=database)
            return jsonify({'figure': df})
        else:
            df = self.geo_ele_Azimuth(startdate,
                                      enddate,
                                      stationname,
                                      itemname,
                                      figure_or_not=False,
                                      database=database)
            return jsonify(df.to_dict(orient='dict'))

    def geo_ele_Azimuth(self,
                        startdate,
                        enddate,
                        stationname,
                        itemname,
                        figure_or_not=True,
                        database='shandong-official'):

        conn = tsf.conn_to_Oracle(database)
        df = tsf.fetching_data(conn,
                               startdate,
                               enddate,
                               '地电场',
                               stationname,
                               '分钟值',
                               '预处理库',
                               gzip_flag=False,
                               itemname=itemname)
        df_Azimuth = self.calculate_Azimuth(df)
        if figure_or_not:
            trace = px.scatter(df_Azimuth,
                               labels={
                                   'index': 'Date',
                                   'value': 'Azimuth',
                                   'variable': '测向'
                               },
                               title='地电场优势方位角')
            trace.show()
            df_Azimuth = trace.to_json()
        return df_Azimuth

    def calculate_Azimuth(self, df):
        days = pd.date_range(df.index.min(), df.index.max())
        df_Azimuth = pd.DataFrame()
        for i in days:
            day = i.strftime('%Y-%m-%d')
            df_oneday = df.loc[day]
            df_Azimuth_oneday = eleAzimuth.calculate_Azimuth_oneday(df_oneday)
            df_Azimuth = df_Azimuth.append(df_Azimuth_oneday,
                                           ignore_index=True)
        df_Azimuth.index = days.strftime('%Y-%m-%d')
        return df_Azimuth

    @staticmethod
    def calculate_Azimuth_oneday(df):
        amplitude = dict()
        azimuth = dict()
        items = df['ITEMID'].drop_duplicates().sort_values().values
        for item in items:
            amplitude[item] = eleAzimuth.fft_ck(df[df['ITEMID'] == item], 10)
        if '3411' in items:
            try:
                azimuth['EW/NW'] = 90 + 180 / pi * arctan(
                    sqrt(2) * amplitude['3414'] / amplitude['3412'] - 1)
            except:
                pass
            try:
                azimuth['EW/NE'] = 90 - 180 / pi * arctan(
                    sqrt(2) * amplitude['3413'] / amplitude['3412'] - 1)
            except:
                pass
            try:
                azimuth['NS/NW'] = 180 - 180 / pi * arctan(
                    sqrt(2) * amplitude['3414'] / amplitude['3411'] - 1)
            except:
                pass
            try:
                azimuth['NS/NE'] = 180 / pi * arctan(
                    sqrt(2) * amplitude['3413'] / amplitude['3411'] - 1)
            except:
                pass
            try:
                azimuth['NS/EW'] = 180 / pi * arctan(
                    amplitude['3412'] / amplitude['3411'])
            except:
                pass
        elif '3421' in items:
            try:
                azimuth['EW/NW'] = 90 + 180 / pi * arctan(
                    sqrt(2) * amplitude['3424'] / amplitude['3422'] - 1)
            except:
                pass
            try:
                azimuth['EW/NE'] = 90 - 180 / pi * arctan(
                    sqrt(2) * amplitude['3423'] / amplitude['3422'] - 1)
            except:
                pass
            try:
                azimuth['NS/NW'] = 180 - 180 / pi * arctan(
                    sqrt(2) * amplitude['3424'] / amplitude['3421'] - 1)
            except:
                pass
            try:
                azimuth['NS/NE'] = 180 / pi * arctan(
                    sqrt(2) * amplitude['3423'] / amplitude['3421'] - 1)
            except:
                pass
            try:
                azimuth['NS/EW'] = 180 / pi * arctan(
                    amplitude['3422'] / amplitude['3421'])
            except:
                pass
        return azimuth

    @staticmethod
    def harmonic_analysis_ck(df, order):
        df.fillna(method='bfill', inplace=True)
        n = df.shape[0]
        df['IND'] = range(1, n + 1)
        ck = 0
        for i in range(1, order + 1):
            ak_bk = 2.0 * pi * df['IND'] / n * i
            ak = sum(df['OBSVALUE'] * cos(ak_bk)) / n * 2.0
            bk = sum(df['OBSVALUE'] * sin(ak_bk)) / n * 2.0
            ck = sqrt(ak * ak + bk * bk) + ck
        return ck

    @staticmethod
    def fft_ck(df, order):
        df.fillna(method='bfill', inplace=True)
        df_fft = fft(df['OBSVALUE'])
        tmp = 2.0 * abs(df_fft) / 1440
        ck = sum(tmp[1:order + 1])
        return ck


api.add_resource(eleAzimuth, '/ele_Azimuth')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8888)
