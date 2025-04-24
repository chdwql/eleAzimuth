# 地电场优势方位角计算系统

## 项目简介

本项目是一个基于Flask的REST API服务，用于计算地电场优势方位角。该系统提供了从数据库获取地电场数据并进行方位角计算的功能，主要面向地球电磁学科研人员。

## 主要功能

- 从数据库中获取地电场时间序列数据
- 计算地电场优势方位角
- 提供REST API接口
- 支持多种数据格式输出

## 技术栈

- Python 3.6+
- Flask 和 Flask-RESTful
- NumPy, SciPy, Pandas用于数据处理和分析
- Numba用于计算性能优化
- Plotly用于数据可视化
- Flasgger用于API文档

## 安装说明

### 系统要求

- Python 3.6+
- Oracle客户端
- addereq

### 安装步骤

1. 克隆仓库

```bash
git clone https://github.com/chdwql/eleAzimuth.git
cd eleAzimuth
```

2. 安装依赖

```bash
pip install -r requirements.txt
```

3. 配置数据库连接

参考[addereq官方页面](https://pypi.org/project/addereq/)

## 使用方法

### 启动服务

```bash
cd src
python app.py
```

服务将在 http://localhost:8888 上运行。

### API文档

API文档可通过访问 http://localhost:8888/apidocs 获取。

### API端点

#### 获取地电场优势方位角

```
GET /ele_Azimuth
```

**参数:**

- `stationname`: 台站名称（例如：安丘）
- `itemname`: 装置名称（例如：第一）
- `database`: 数据库名称（默认：shandong-official）
- `startdate`: 起始日期（格式：YYYYMMDD，例如：20210101）
- `enddate`: 结束日期（格式：YYYYMMDD，例如：20210131）

**示例请求:**

```
GET /ele_Azimuth?stationname=安丘&itemname=第一&startdate=20210101&enddate=20210131&database=shandong-official
```

**响应格式:**

```json
{
  "2021-01-01": {
    "EW/NW": 135.2,
    "EW/NE": 45.8,
    "NS/NW": 172.1,
    "NS/NE": 8.9,
    "NS/EW": 82.5
  },
  "2021-01-02": {
    // 同上格式
  }
  // 更多日期数据...
}
```

## 项目结构

```
eleAzimuth/
│
├── src/                    # 源代码
│   ├── app.py              # 主应用程序
│   └── app.log             # 应用日志
│
├── requirements.txt        # 项目依赖
└── README.md               # 项目说明文档
```

## 性能优化

本项目针对大数据量和计算密集型任务进行了多项优化：

1. 使用Numba加速计算密集型函数
2. 实现LRU缓存减少重复计算
3. 优化数据处理流程，减少内存占用
4. 使用向量化操作代替循环
5. 优化数据库连接和查询

## 注意事项

1. 该系统依赖`addereq`包进行数据库访问
2. 需要适当的数据库权限才能获取数据
3. 计算过程需要完整的地电场分量数据

## 许可证

License: MIT
