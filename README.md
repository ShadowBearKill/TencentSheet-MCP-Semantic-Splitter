# SheetMCP-Semantic-Splitter

## 项目介绍
SheetMCP-Semantic-Splitter 是一个基于多维度语义分析的智能表格文档切分工具，通过MCP（Model Context Protocol）服务器提供服务。该项目旨在将腾讯文档的在线文档智能分割为语义完整的片段，便于后续的文档处理、分析和检索。

## 功能特性
- 支持对在线表格进行语义分割，自动处理所有工作表
- 提供查询表格信息、获取分割质量报告等辅助功能
- 集成到MCP平台，可作为工具函数供其他应用调用

## 核心功能
- **智能语义分割**：基于OpenAI嵌入模型进行语义分析
- **表格数据获取**：支持在线表格API调用和数据解析
- **相似度计算**：使用余弦相似度算法进行片段关联分析
- **片段优化**：智能合并碎片和拆分超长片段
- **结构化输出**：生成包含元数据和质量报告的JSON格式结果

## 使用说明
```bash
# 安装依赖
pip install -r requirements.txt

# 设置环境变量
export OPENAI_API_KEY="your-openai-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_MODEL="text-embedding-3-small"
export SIMILARITY_THRESHOLD="0.7"
export MIN_SEGMENT_LENGTH="50"
export MAX_SEGMENT_LENGTH="1000"

# 启动MCP服务器
python mcp_server.py
```
或者
```json
{
  "mcpServers": {
    "sheet-semantic-splitter": {
      "command": "python",
      "args": [
        "/path/to/SheetMCP-Semantic-Splitter/mcp_server.py"
      ],
      "env": {
        "OPENAI_API_KEY": "your-openai-api-key",
        "OPENAI_BASE_URL": "https://api.openai.com/v1",
        "OPENAI_MODEL": "text-embedding-3-small",
        "SIMILARITY_THRESHOLD": "0.7",
        "MIN_SEGMENT_LENGTH": "50",
        "MAX_SEGMENT_LENGTH": "1000",
        "LOG_FILE_PATH": "logs/semantic_splitter.log",
        "TEMP_BASE_DIR": "temp/"
      }
    }
  }
}
```

## 详细流程

### 数据流转过程
```
输入参数验证
    ↓
表格信息查询 (query_sheet_info)
    ↓
表格数据获取 (query_sheet_data) - 支持大范围自动分块
    ↓
数据解析 (parse_response_data)
    ↓
预处理 (preprocess_data) - 转换为TextSegment
    ↓
语义分割决策 (decide_segments)
    ↓
片段优化 (optimize_segments)
    ↓
结果合并和输出生成 (merge_sheet_results)
```
### 核心步骤
1. agent调用mcp的`tools/call`接口，传入`semantic_split_sheet`工具函数，同时需要传入assess_token、client_id、open_id、user_open_id、参数，spilt_type默认为单元格分割
2. 构造url，获取表格的信息sheetinfo，返回为字典格式。处理sheetinfo，从中提取出每一个工作表sheet的sheetID、title、row_count、column_count。封装为一个SheetInfo。返回SheetInfo列表。
3. 处理每一个工作表sheet
4. 首先获取工作表信息，计算工作表的大小是否超出了查询范围限制，如果没有，那么计算一个最优范围并构造url，得到字典形式的sheet_data。如果表格太大，那么则进行多次处理，切块进行请求；如果列数没有超出50，则进行横向切片策略，保持所有列不变，将行数按照单元格限制进行切分。如果列数超出了限制，则进行网格切块策略，将行和列都进行切分，保证每次请求的单元格数量不超过限制。进行数据合并，将每个分块的数据填入对应的位置，包括空单元格。返回的数据为合并后的字典。
5. 解析工作表信息，首先将工作表的单元格数据封装成一个CellData对象，属性包括row、col、value、data_type、cell_format，转成字符串，然后将所有的CellData对象封装返回成一个列表。
6. 对CellData列表进行切分成segments，可选为行切分，列切分，单元格切分，根据spilt_type进行选择。之后过滤掉空的segment或者长度过小的segment（按单元格切分的只会过滤空的segment）。
7. 计算segments两两之间的相似度，得到相似度矩阵。构建segments之间的邻接表，如果两个segment的相似度大于阈值，则在邻接表中添加一条边。对邻接表进行连通分量分解，使用DFS求解得到所有连通分量，即最终的segment分组。
8. 进行分组优化，首先是合并过小的组：对于如果一个segmentsGroup的segments只有1个，则将这个segmentsGroup和所有segmentsGroup计算平均余弦相似度，将segmentsGroup合并到最相似的segmentsGroup中；如果segmentsGroup的segments不止一个，那么进行组内合并，如果segment长度小于长度阈值，则贪心地和下一个segment合并，直到大于等于长度阈值或者合并到组末尾为止。
9. 进行分组优化，其次是拆分过大的组：对于如果一个segmentsGroup的segments总长度大于最大长度阈值，则进行拆分，依次尝试句子、段落、二分法拆分，直到所有segments的长度都小于等于最大长度阈值为止。
10. 每个工作表处理后返回的内容为一个字典，包含sheetID，segmentsGroup的列表和解析后的sheetData，每个segmentsGroup包含group_id，segments的列表，每个segment包含segment_id，content，segment_type，cells的列表，start_pos，end_pos。
11. 合并所有工作表的处理结果并返回。

### 数据类

1. SheetInfo
**属性：** 
- sheet_id: 工作表ID: str
- row_count: 工作表行数: int
- column_count: 工作表列数: int
- title: 工作表标题: str

2. TextSegment
**属性：** 
- segment_id: 片段唯一标识: str
- content: 片段文本内容: str
- segment_type: 片段类型（row/column/cell）: str
- cells: 包含的单元格数据列表: List[CellData]
- start_pos: 起始位置 (row, col): Tuple[int, int]
- end_pos: 结束位置 (row, col): Tuple[int, int]

3. SegmentGroup
**属性：** 
- group_id: 分组唯一标识: str
- segments: TextSegment列表: List[TextSegment]

4. CellData
**属性：** 
- row: 行号: int
- col: 列号: int
- value: 单元格值: str
- data_type: 数据类型: str
- cell_format: 验证格式: str



### 核心函数

1. _query_sheet_info()
**功能：** 查询表格基本信息
**返回值类型：** `Dict[str, Any]`
**返回值结构：**
```json
{
  "code": 0,
  "message": "success",
  "properties": [
    {
      "sheetId": "sheet_id_string",
      "title": "工作表标题",
      "rowCount": 100,
      "columnCount": 26,
    } ...
  ]
}
```
2. _extract_sheet_range_info()
**功能：** 从表格信息中提取工作表信息
**返回值类型：** `List[SheetInfo]`
**返回值结构：**
```python
[SheetInfo对象列表]
```

3. _process_single_sheet()
**功能：** 处理单个工作表
**返回值类型：** `Dict[str, Any]`
**返回值结构：**
```json
{
  "sheet_id": "sheet_id_string",
  "segments": [
    {
      "group_id": "group_0",
      "segments": [TextSegment对象列表]
    }
  ],
  "parsed_data": {
    "cells": [CellData对象列表]
  }
}
```


### process_sheet函数执行流程及返回值结构

#### 1. _query_sheet_info() 函数
**功能：** 查询表格基本信息
**返回值类型：** `Dict[str, Any]`
**返回值结构：**
```json
{
  "code": 0,
  "message": "success",
  "properties": [
    {
      "sheetId": "sheet_id_string",
      "title": "工作表标题",
      "rowCount": 100,
      "columnCount": 26,
      "gridProperties": {...}
    }
  ]
}
```

#### 2. _process_all_sheets() 函数
**功能：** 处理所有工作表
**返回值类型：** `List[Dict[str, Any]]`
**返回值结构：**
```json
[
  {
    "sheet_id": "sheet_id_string",
    "segments": [SegmentGroup对象列表],
    "parsed_data": {"cells": [CellData对象列表]}
  },
  {
    "sheet_id": "another_sheet_id",
    "error": "错误信息",
    "segments": [],
    "parsed_data": {}
  }
]
```

#### 3. _process_single_sheet() 函数
**功能：** 处理单个工作表
**返回值类型：** `Dict[str, Any]`
**返回值结构：**
```json
{
  "sheet_id": "sheet_id_string",
  "segments": [
    {
      "group_id": "group_0",
      "segments": [TextSegment对象列表]
    }
  ],
  "parsed_data": {
    "cells": [CellData对象列表]
  }
}
```

#### 4. _query_sheet_data() 函数
**功能：** 查询表格数据（支持大范围自动分块查询）
**返回值类型：** `Dict[str, Any]`
**单次查询返回值结构：**
```json
{
  "code": 0,
  "msg": "success",
  "data": {
    "gridData": {
      "startRow": 0,
      "startColumn": 0,
      "rows": [
        {
          "cells": [
            {
              "cellValue": "单元格内容",
              "dataType": "DATA_TYPE_TEXT",
              "cellFormat": "格式信息"
            }, ...
          ]
        }, ...
      ]
    }
  }
}
```

#### 5. _parse_response_data() 函数
**功能：** 解析工作表响应数据
**返回值类型：** `Dict[str, Any]`

#### 6. _preprocess_data() 函数
**功能：** 数据预处理，将CellData列表转换为TextSegment。即将单元格通过行|列|单元格进行切分，并过滤掉空的segment或者长度过小的segment，得到TextSegment列表。
**返回值类型：** `List[TextSegment]`

#### 7. _make_segmentation_decisions() 函数
**功能：** 基于相似度矩阵进行分割决策，通过计算不同segment之间的相似度，得到相似度矩阵。构建segments之间的邻接表，如果两个segment的相似度大于阈值，则在邻接表中添加一条边。对邻接表进行连通分量分解，使用DFS求解得到所有连通分量，即最终的segment分组。
**返回值类型：** `List[SegmentGroup]`

#### 8. _optimize_segments() 函数
**功能：** 优化片段组，包括合并小片段和拆分大片段。
**返回值类型：** `List[SegmentGroup]`

#### 9. _merge_sheet_results() 函数
**功能：** 合并多个工作表的处理结果，也即MCP tool最后的返回结果
**返回值类型：** `Dict[str, Any]`
**最终输出结构：**
```json
{
  "segments": [
    {
      "sheet_id": "sheet_id_string",
      "segments":[
        {
        "segment_id": "group_{i}",
        "content": "合并后的片段内容",
        "content_length": 文本长度,
        "sub_segments": segments数量,
        "sub_segment_ids": ["row_1", "row_2", "row_3"],
        "segment_type": "row",
        "positions": [{"start_pos": [1, 1],"end_pos": [1, 5]}]
        }...
      ] 
    } ...
  ]
}
```