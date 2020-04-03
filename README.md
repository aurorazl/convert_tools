# 上传数据集文档
> 详细转换和上传流程请见[upload_detail_steps.md](https://github.com/apulis/dev_document/blob/master/label/upload_detail_steps.md)

### 配置config.yaml文件
```shell script
image_tar_name: simple.tar.gz                           # 打包图片的压缩包名字，可随意指定
json_tar_name: simple_json.tar.gz                       # 打包json文件的压缩包名字，可随意指定
nfs_base_path: /mntdlws/nfs                             # 数据平台服务器的数据集存放的根路径
user: dlwsadmin                                         # 连接到数据平台服务器的用户名
host: apulis-sz-dev-worker01.sigsus.cn                  # 连接到数据平台服务器的域名
identity_file: DataConvert绝对路径/id_rsa   # 连接到数据平台服务器的私钥，需绝对路径
```

### 如何上传(更新)模型预测的数据集
1. 转换并上传
    - list格式预测标注文件
        ```shell script
        python upload.py upload_model_predict_result_from_list [模型预测结果list文件路径] [项目uuid] [数据集的uuid]
        ```
    - coco格式预测标注文件
         ```shell script
        python upload.py upload_model_predict_result_from_coco [模型预测结果coco的json文件路径] [项目uuid] [数据集的uuid]
        ```
    - voc格式预测标注文件
         ```shell script
        python upload.py upload_model_predict_result_from_voc [模型预测结果voc标注文件路径] [模型预测结果voc图片路径] [项目uuid] [数据集的uuid]
        ```
    - ocr格式预测标注文件
       ```shell script
        python upload.py upload_model_predict_result_from_ocr [模型预测结果ocr标注文件路径] [模型预测结果ocr图片路径] [项目uuid] [数据集的uuid]
        ```

### 如何上传新的数据集

1. 打开标注项目的网页，选择or新建项目，点击新建一个数据集，填好创建信息

2. 通过编辑按钮获取项目的UUID和数据集的UUID，用于第三步
    ``` 
    uuid格式：04F3662A-C848-46D2-B21C-6A68D591AC12
    ```
3. 转换，并上传图片和json标注文件
    - coco格式数据集
        ```shell script
        python upload.py --use-category-mapping upload_dataset_from_coco coco [coco标注文件路径] [coco图片路径] [项目uuid] [数据集的uuid] [用户uid]
       如果忽略图片上传，只上传更新标注文件，执行如下命令
       python upload.py --use-category-mapping --ignore-image upload_dataset_from_coco [coco标注文件路径] [coco图片路径] [项目uuid] [数据集的uuid] [用户uid]
        ```
    - ocr格式数据集
        ```shell script
        python upload.py upload_dataset_from_ocr  [ocr标注文件路径] [ocr图片路径] [项目uuid] [数据集的uuid] [用户uid]
        ```
    - voc格式数据集
        ```shell script
        python upload.py upload_dataset_from_voc [voc数据集路径] [项目uuid] [数据集的uuid] [用户uid]
        ```
