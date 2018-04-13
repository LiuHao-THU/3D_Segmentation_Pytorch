import os
import shutil
def Mkdir(dir):
    if os.path.isdir(dir):
        shutil.rmtree(dir, ignore_errors=True)
    else:
        os.makedirs(dir)
# def Get_DataName(DataSet_Dir, Txt_file):
# 	"""Make DataSet file Name to .txt file"""
# 	DataSet_List = os.listdir(DataSet_Dir)
# 	txt = open(Txt_file, 'w')
# 	for name in DataSet_Dir:
# 		Txt_file = 
	
