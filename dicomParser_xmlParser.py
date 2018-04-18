import os
import cv2
import dicom
import xml.etree.cElementTree as ET
import SimpleITK as sitk

def read_dcm_image(path):
        ds = sitk.ReadImage(path)
        img_array = sitk.GetArrayFromImage(ds)
        return img_array[0]
def equalizeHist16bit(img):
    hist, bins = np.histogram(img.flatten(), 65536, [0, 65536])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 65535 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint16')
    return cdf[img]  
def equalizeHist16bit2(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(img)
    return cl1

dir = 'D:/kao/dicom/'
dir_xml = 'C:/tonywang/workspace/SR_XML/SR'
a=0
b=0

for file in os.listdir(dir):
    try:
        #print(file)
        img = equalizeHist16bit(read_dcm_image(dir + file))
        dicom_info = dicom.read_file(dir + file)
        
        if 'StudyInstanceUID' in dicom_info:
            #print(dicom_info.StudyInstanceUID)
            tree = ET.ElementTree(file= dir_xml + dicom_info.StudyInstanceUID+'.xml')
            tree.getroot()
            
            for elem in tree.iter(tag='cap'):
                if 'category b' in str(elem.attrib):
                    #print(elem.attrib)
                    cv2.imwrite('C1'+'_'+str(a)+'_'+str(file)+'.png', img)
                    a=a+1
                elif 'category c' in str(elem.attrib):
                    #print(elem.attrib)
                    cv2.imwrite('C2'+'_'+str(b)+'_'+str(file)+'.png', img)
                    b=b+1
    except:
        print( "Error: XML格式錯誤")
        continue