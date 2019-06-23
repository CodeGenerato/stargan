import os
import shutil

def copy_file(src_dir,tgt_dir,img_name):
        shutil.copy2(src_dir+"/"+img_name,tgt_dir)

def create_empty_dir(tgt_dir):
        if os.path.exists(tgt_dir):
            shutil.rmtree(tgt_dir)
        
        os.mkdir(tgt_dir)
