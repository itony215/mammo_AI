# VGG Model Usage
Use this to build the VGG object
```
vgg = vgg19.Vgg19()
vgg.build(images)
```
or
```
vgg = vgg16.Vgg16()
vgg.build(images)
```
The `images` is a tensor with shape `[None, 224, 224, 3]`. 

>To use the VGG networks, the npy files for [VGG16 NPY](https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM) or [VGG19 NPY](https://mega.nz/#!xZ8glS6J!MAnE91ND_WyfZ_8mvkuSa2YcA7q-1ehfSm-Q1fxOvvs) has to be downloaded.


# Dicom Header Parser
### Install
```
pip install pydicom
```
### and needs to be imported with 
```
import dicom
```
### Usage
```
dicom_info =dicom.read_file(file_path)
```

# XML Parser
Python 提供兩種libary可以使用，一個是Python base的 xml.etree.ElementTree，另一個是C寫的 xml.etree.cElementTree，用C速度較快，而且記憶體使用上也要少很多。如果不確定有無安裝Python版本中的cElementTree libary，可以這樣import：
```
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
```

### Usage
read XML
```
tree = ET.ElementTree(file='SR1.2.840.113817.20140414.11957872.66425974.91.xml')
```
get root element
```
tree.getroot()
```
print all element
```
for elem in tree.iter():
    print(elem.tag, elem.attrib)
```
