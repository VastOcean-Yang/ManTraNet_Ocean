# introduction
本项目是对
```python
  @inproceedings{Wu2019ManTraNet,
      title={ManTra-Net: Manipulation Tracing Network For Detection And Localization of Image ForgeriesWith Anomalous Features},
      author={Yue Wu, Wael AbdAlmageed, and Premkumar Natarajan},
      journal={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      year={2019}
  }
```
的复现，方便网友们快速复现。
# how to start?
快速开始，安装python3.6 ，并按照requirements的文本文件下载对应的python包版本。
因为项目是2018年的，在使用最新的keras等包的过程中会出现很多不兼容的错误，需要conda创建相应的虚拟环境并下载低版本的包

注意：如果运行失败请把tensorflow~=1.10.0改为1.13.0版本

运行main_ocean.py,其中

test_one_pair()函数是自定义一组图片进行效果展示

random_show()函数是源代码中自定义的随机展示数据集八张图片的效果

dirs_test_and_save_mask()函数是自定义的遍历指定数据集（文件夹）所有子文件夹的图片，并生成相应的mask图像并保存

### 工作量
工作量主要来自源代码过于古老并且没有相应的配置文件，运行过程中经常出现不兼容无法运行。


