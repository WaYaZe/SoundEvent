import os
from os import getcwd

classes = ['V1', 'V2', 'V3']
sets = ['./audioclassification/dataset/test']

if __name__ == '__main__':
    wd = getcwd()
    for se in sets:
        list_file = open('audioclassification/dataset/test_list.txt', 'w')

        datasets_path = se
        types_name = os.listdir(datasets_path)  # os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
        for type_name in types_name:
            key_name = type_name
            cls_id = classes.index(key_name)  # 输出0-1
            last_path = datasets_path + '/' + type_name
            file_names = os.listdir(last_path)
            for file_name in file_names:
                _, postfix = os.path.splitext(file_name)  # 该函数用于分离文件名与拓展名
                if postfix not in ['.jpg', '.png', '.jpeg', '.wav']:
                    continue
                list_file.write('%s/%s' % (wd, os.path.join(last_path[2:], file_name)) + '\t' + str(cls_id))
                list_file.write('\n')
        list_file.close()
