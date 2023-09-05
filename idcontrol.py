import random
import os
import json


class IDManager():
    '''
    简易任务管理器
    '''
    def __init__(self) -> None:
        self.resultdir = os.listdir('results')
        self.result_list = []
        self.wait_list = []
        
        for dir in self.resultdir:
            if os.path.isdir('results/'+dir):
                dirlist = os.listdir('results/'+dir)
                if 'result.wav' in dirlist or 'output_video.mp4' in dirlist:
                    self.result_list.append(dir)
        
        # wait list一般以json格式储存
        if os.path.exists('results/wait_list.json'):
            f = open('results/wait_list.json', 'r')
            self.wait_list = json.loads(f.read())
            f.close()
        # else:
        #     self.wait_list = []

        # if os.path.exists(path+'/result_list.json'):
        #     f = open(path+'/result_list.json', 'r')
        #     self.result_list = json.loads(f.read())
        #     f.close()
        # else:
        #     self.result_list = []
        
    def generate_id(self, pre=""):
        '''
        生成唯一任务id, 前缀pre="ts" or "am"
        '''
        newid = pre + str(random.randint(100000, 999999))
        while newid in self.resultdir:
            newid = pre + str(random.randint(100000, 999999))

        self.resultdir.append(newid)

        return newid

    def add(self, item, index=None):
        '''
        添加或者插入任务
        '''
        if not index:
            self.wait_list.append(item)
        else:
            self.wait_list.insert(index, item)

    def pop(self, taskid=None):
        '''
        取消任务
        '''
        if not taskid:
            self.wait_list.pop()
            return

        if taskid in self.result_list:
            self.result_list.remove(taskid)
            return

        for i in range(len(self.wait_list)):
            if self.wait_list[i]['taskid'] == taskid:
                self.wait_list.pop(i)
                return
    
    def save(self, path='results'):
        '''
        保存wait list
        '''
        with open(path+'/wait_list.json', 'w') as f:
            f.write(json.dumps(self.wait_list))
        with open(path+'/result_list.json', 'w') as f:
            f.write(json.dumps(self.result_list))
            
    def search(self, taskid):
        '''
        检查任务是否在等待
        '''

        for i in range(len(self.wait_list)):
            if self.wait_list[i]['taskid'] == taskid:
                return i

        return -1

    def isdone(self, taskid):
        if taskid in self.result_list:
            return True
        
        return False