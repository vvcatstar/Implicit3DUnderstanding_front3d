'''
Preprocess pix3d data.
author: ynie
date: Sep, 2019
'''

import sys

sys.path.append('.')
from configs.front3d_config import Config
import os
import glob
import tqdm
from multiprocessing import Pool
from libs.tools import read_obj, write_obj, sample_pnts_from_obj, normalize_to_unit_square
from PIL import Image
import numpy as np
import json, gzip
from external.mesh_fusion import scale, fusion, simplify
import sys
import subprocess
from scipy.spatial import cKDTree
from IPython import embed 
import cv2 


# preprocess param
del_intermediate_result = False
skip_done = False
processes = 0

# path settings
config = Config('front3d')
mesh_folder = os.path.join(config.model_path, 'model')
output_root = config.save_path
os.makedirs(output_root, exist_ok=True)
gaps = './external/ldif/gaps/bin/x86_64'
mesh_fusion = 'external/mesh_fusion'
python_bin = sys.executable
skip = ['IKEA_JULES_1.model_-108.706406967_-139.417398691']

# ldif param
scale_norm = 0.25
bbox_half = 0.7
bbox = ' '.join([str(-bbox_half), ] * 3 + [str(bbox_half), ] * 3)
spacing = bbox_half * 2 / 32
print({'bbox_half': bbox_half, 'spacing': spacing})

# mgnet param
neighbors = 30


def normalize(input_path, output_folder):
    output_path = os.path.join(output_folder, 'mesh_normalized.obj')

    obj_data = read_obj(input_path, ['v', 'f'])
    obj_data['v'] = normalize_to_unit_square(obj_data['v'])[0]
    write_obj(output_path, obj_data)
    return output_path


def make_watertight(input_path, output_folder):
    output_path = os.path.join(output_folder, 'mesh_orig.obj')

    # convert mesh to off
    off_path = os.path.splitext(output_path)[0] + '.off'
    subprocess.check_output(f'xvfb-run -a -s "-screen 0 800x600x24" meshlabserver -i {input_path} -o {off_path}',
                            shell=True)

    # scale mesh
    # app = scale.Scale(
    #     f'--in_file {off_path} --out_dir {output_folder} --t_dir {output_folder} --overwrite'.split(' '))
    # app.run()
    subprocess.check_output(f'{python_bin} {mesh_fusion}/scale.py'
                            f' --in_file {off_path} --out_dir {output_folder} --t_dir {output_folder} --overwrite',
                            shell=True)

    # create depth maps
    # app = fusion.Fusion(
    #     f'--mode=render --in_file {off_path} --out_dir {output_folder} --overwrite'.split(' '))
    # app.run()
    subprocess.check_output(f'xvfb-run -a -s "-screen 0 800x600x24" {python_bin} {mesh_fusion}/fusion.py'
                            f' --mode=render --in_file {off_path} --out_dir {output_folder} --overwrite',
                            shell=True)

    # produce watertight mesh
    depth_path = off_path + '.h5'
    transform_path = os.path.splitext(output_path)[0] + '.npz'
    # app = fusion.Fusion(
    #     f'--mode=fuse --in_file {depth_path} --out_dir {output_folder} --t_dir {output_folder} --overwrite'.split(' '))
    # app.run()
    subprocess.check_output(f'{python_bin} {mesh_fusion}/fusion.py --mode=fuse'
                            f' --in_file {depth_path} --out_dir {output_folder} --t_dir {output_folder} --overwrite',
                            shell=True)

    # # simplify mesh
    # obj_path = os.path.splitext(output_path)[0] + '.obj'
    # app = simplify.Simplification(
    #     f'--in_file={obj_path} --out_dir {output_folder}'.split(' '))
    # app.run()
    # subprocess.check_output(f'xvfb-run -a -s "-screen 0 800x600x24" {python_bin} {mesh_fusion}/simplify.py'
    #                         f' --in_file={obj_path} --out_dir {output_folder}', shell=True)

    os.remove(off_path)
    os.remove(transform_path)
    os.remove(depth_path)
    return output_path


def remove_if_exists(f):
    if os.path.exists(f):
        os.remove(f)


def make_output_folder(mesh_path):
    rel_folder = os.path.relpath(mesh_path, mesh_folder).split('/')
    model_folder = '.'.join(os.path.splitext(mesh_path)[0].split('/')[-2:])
    rel_folder = os.path.join(*rel_folder[:-2], model_folder)
    output_folder = os.path.join(output_root, rel_folder)
    os.makedirs(output_folder, exist_ok=True)
    return output_folder


def process_mgnet(obj_path, output_folder, ext):
    obj_data = read_obj(obj_path, ['v', 'f'])
    sampled_points = sample_pnts_from_obj(obj_data, 10000, mode='random')
    sampled_points.tofile(os.path.join(output_folder, f'gt_3dpoints.{ext}'))

    tree = cKDTree(sampled_points)
    dists, indices = tree.query(sampled_points, k=neighbors)
    densities = np.array([max(dists[point_set, 1]) ** 2 for point_set in indices])
    densities.tofile(os.path.join(output_folder, f'densities.{ext}'))


def process_mesh(mesh_path):
    output_folder = make_output_folder(mesh_path)
    mesh_name = os.path.basename(output_folder)
    if mesh_name in skip:
        print(f"skipping {mesh_name}")
        return
    if skip_done and os.path.exists(f'{output_folder}/uniform_points.sdf'):
        return

    # Step 0) Normalize and watertight the mesh before applying all other operations.
    normalized_obj = normalize(mesh_path, output_folder)
    watertight_obj = make_watertight(normalized_obj, output_folder)

    # conver mesh to ply
    normalized_ply = os.path.splitext(normalized_obj)[0] + '.ply'
    subprocess.check_output(
        f'xvfb-run -a -s "-screen 0 800x600x24" meshlabserver -i {normalized_obj} -o {normalized_ply}',
        shell=True)
    watertight_ply = os.path.splitext(watertight_obj)[0] + '.ply'
    subprocess.check_output(
        f'xvfb-run -a -s "-screen 0 800x600x24" meshlabserver -i {watertight_obj} -o {watertight_ply}',
        shell=True)

    scaled_ply = os.path.join(output_folder, 'scaled_watertight.ply')
    os.system(f'{gaps}/msh2msh {watertight_ply} {scaled_ply} -scale_by_pca -translate_by_centroid'
              f' -scale {scale_norm} -debug_matrix {output_folder}/orig_to_gaps.txt')

    # Step 1) Generate the coarse inside/outside grid:
    os.system(f'{gaps}/msh2df {scaled_ply} {output_folder}/coarse_grid.grd'
              f' -bbox {bbox} -border 0 -spacing {spacing} -estimate_sign')

    # Step 2) Generate the near surface points:
    os.system(f'{gaps}/msh2pts {scaled_ply} {output_folder}/nss_points.sdf'
              f' -near_surface -max_distance {spacing} -num_points 100000 -binary_sdf')

    # Step 3) Generate the uniform points:
    os.system(f'{gaps}/msh2pts {scaled_ply} {output_folder}/uniform_points.sdf'
              f' -uniform_in_bbox -bbox {bbox} -npoints 100000 -binary_sdf')

    # Step 4) Generate surface points for MGNet:
    process_mgnet(watertight_obj, output_folder, 'mgn')
    process_mgnet(normalized_obj, output_folder, 'org')

    if del_intermediate_result:
        remove_if_exists(normalized_obj)
        remove_if_exists(watertight_obj)
        remove_if_exists(scaled_ply)


def process_front3d(sample):
    imgid, objid, cname, sample_id  = sample
    img_name = f'{imgid.split("/")[-1].split(".")[0]}_{objid}'
    img_path = os.path.join(config.root_path, imgid)
    post_fix = img_path.split('.')[-1]
    
    mask_path = img_path.replace('rgb', 'mask').replace(f'.{post_fix}', '.npy.gz')  
    anno_path = img_path.replace('rgb', 'annotation').replace(f'.{post_fix}', '.json')
    with open(anno_path, 'r') as f:
        annotation = json.load(f)   
        
    obj = annotation['obj_dict'][objid]
    bbox_2d = obj['bbox_2d']
    
    output_folder = os.path.join(config.save_path, img_name)
    os.makedirs(output_folder, exist_ok=True)
    # generate image 
    output_path = os.path.join(output_folder, img_name + '.npy')
    if not skip_done or not os.path.exists(output_path):
        img = np.array(Image.open(img_path).convert('RGB'))
        height, width = img.shape[:2]
        img = img[bbox_2d[1]:bbox_2d[3], bbox_2d[0]:bbox_2d[2], :]
        np.save(output_path, img)
        cv2.imwrite(output_path.replace('.npy', '.png'), img)
    # generate mask 
    output_path = os.path.join(output_folder, img_name + '_mask.npy')
    if not skip_done or not os.path.exists(output_path):
        with gzip.GzipFile(mask_path, 'r') as f:
            segm = np.load(f)
        segm = segm[100:100+height, 100:100+width, :]
        obj_id = obj['obj_id'][0]
        segm_index = np.argwhere(segm == obj_id)
        px = [index[0] for index in segm_index]                 # height    uv[1]
        py = [index[1] for index in segm_index]                 # width     uv[0]
        obj_map = np.zeros((height, width), dtype=np.uint8)
        obj_map[px, py] = 255
        img = np.array(Image.fromarray(obj_map).convert('L'))
        img = img[bbox_2d[1]:bbox_2d[3], bbox_2d[0]:bbox_2d[2]]
        np.save(output_path, img)
        cv2.imwrite(output_path.replace('.npy', '.png'), img)
        
    # generate model 
    jid = obj['model_file_name'][0]
    mesh_path = os.path.join(config.model_path, jid, 'raw_watertight.obj')
    # Step 0) Normalize and watertight the mesh before applying all other operations.
    normalized_obj = normalize(mesh_path, output_folder)
    watertight_obj = make_watertight(normalized_obj, output_folder)
    # conver mesh to ply
    normalized_ply = os.path.splitext(normalized_obj)[0] + '.ply'
    subprocess.check_output(
        f'xvfb-run -a -s "-screen 0 800x600x24" meshlabserver -i {normalized_obj} -o {normalized_ply}',
        shell=True)
    watertight_ply = os.path.splitext(watertight_obj)[0] + '.ply'
    subprocess.check_output(
        f'xvfb-run -a -s "-screen 0 800x600x24" meshlabserver -i {watertight_obj} -o {watertight_ply}',
        shell=True)

    scaled_ply = os.path.join(output_folder, 'scaled_watertight.ply')
    os.system(f'{gaps}/msh2msh {watertight_ply} {scaled_ply} -scale_by_pca -translate_by_centroid'
              f' -scale {scale_norm} -debug_matrix {output_folder}/orig_to_gaps.txt')

    # Step 1) Generate the coarse inside/outside grid:
    os.system(f'{gaps}/msh2df {scaled_ply} {output_folder}/coarse_grid.grd'
              f' -bbox {bbox} -border 0 -spacing {spacing} -estimate_sign')

    # Step 2) Generate the near surface points:
    os.system(f'{gaps}/msh2pts {scaled_ply} {output_folder}/nss_points.sdf'
              f' -near_surface -max_distance {spacing} -num_points 100000 -binary_sdf')

    # Step 3) Generate the uniform points:
    os.system(f'{gaps}/msh2pts {scaled_ply} {output_folder}/uniform_points.sdf'
              f' -uniform_in_bbox -bbox {bbox} -npoints 100000 -binary_sdf')

    # Step 4) Generate surface points for MGNet:
    process_mgnet(watertight_obj, output_folder, 'mgn')
    process_mgnet(normalized_obj, output_folder, 'org')

    if del_intermediate_result:
        remove_if_exists(normalized_obj)
        remove_if_exists(watertight_obj)
        remove_if_exists(scaled_ply)

    


if __name__ == '__main__':
    print('Processing imgs...')
    with open(config.train_split, 'r') as file:
        train_split = json.load(file)

    with open(config.test_split, 'r') as file:
        test_split = json.load(file)

    train_sample_info = [train_split[i] +[i] for i in range(len(train_split))]
    test_sample_info = [test_split[i] + [i] for i in range(len(test_split))]
    samples= train_sample_info + test_sample_info




    if processes:
        with Pool(processes=processes) as p:
            r = list(tqdm.tqdm(p.imap(process_front3d, samples), total=len(samples)))
    else:
        for sample in tqdm.tqdm(samples[: 10]):
            process_front3d(sample)

