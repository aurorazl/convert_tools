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

### 如何获取数据集或项目的UUID
1. 打开标注项目的网页，选择or新建一个项目，再点击新建一个数据集

2. 通过编辑按钮获取项目的UUID和数据集的UUID
    ``` 
    uuid格式：04F3662A-C848-46D2-B21C-6A68D591AC12
    ```

### 如何上传(更新)模型预测的数据集（会先清空之前的预测结果）
```shell script
python upload.py upload-map-file [模型预测结果文件路径] [人工标注文件路径] [标注对应的图片路径] [项目uuid] [数据集的uuid]
```

### 如何上传新的数据集
```shell script
python upload.py upload-dataset [数据集标注文件路径] [数据集图片路径] [项目uuid] [数据集的uuid] [用户uid]
```
 

