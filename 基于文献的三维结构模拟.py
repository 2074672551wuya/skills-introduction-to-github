import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from skimage.filters import threshold_otsu
from torch.utils.data import Dataset, DataLoader
from stl import mesh
import trimesh
import json
from rdflib import Graph
import dgl
import dgl.nn as dglnn
from PIL import Image
import torchvision.transforms as transforms
import mcubes
import logging
from scipy.spatial import cKDTree
import time
import matplotlib.pyplot as plt
import matplotlib
import random
from scipy.ndimage import gaussian_filter
import open3d as o3d

# 设置 matplotlib 后端以避免显示问题
matplotlib.use('Agg')

# 固定随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 检查并导入 skimage 模块
try:
    from skimage import measure, morphology
except ImportError as e:
    logging.error(f"无法导入 skimage 模块: {str(e)}")
    logging.warning("连通孔隙分析和骨骼化功能将不可用")
    measure = None
    morphology = None

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

# 用户代码中定义的类和函数（部分摘录，假设已定义）
class MultiModal3DModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.geo_encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool3d(1)
        )
        self.meta_encoder = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 64))
        self.gnn = dglnn.GATConv(64, 64, num_heads=2)
        self.sem_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.fusion = nn.Linear(64 + 64 + 64 * 2 + 128, 256)
        self.classifier = nn.Linear(256, 3)
        self.voxel_generator = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 64 * 64 * 64), nn.Sigmoid())

    def forward(self, voxels, metadata, graph, sem_image):
        batch_size = voxels.size(0)
        geo_feat = self.geo_encoder(voxels).squeeze()
        if geo_feat.dim() == 1:
            geo_feat = geo_feat.unsqueeze(0)
        meta_feat = self.meta_encoder(metadata)
        if meta_feat.dim() == 1:
            meta_feat = meta_feat.unsqueeze(0)
        batch_num_nodes = graph.batch_num_nodes()
        if len(batch_num_nodes) != batch_size:
            raise ValueError(f"Graph batch size {len(batch_num_nodes)} != feature batch size {batch_size}")
        geo_feat_expanded = torch.cat([geo_feat[i].repeat(n, 1) for i, n in enumerate(batch_num_nodes)], dim=0)
        graph.ndata['h'] = geo_feat_expanded
        graph.ndata['h'] = self.gnn(graph, graph.ndata['h']).view(graph.ndata['h'].size(0), -1)
        graph_feat = dgl.mean_nodes(graph, 'h')
        if graph_feat.dim() == 1:
            graph_feat = graph_feat.unsqueeze(0)
        sem_feat = self.sem_encoder(sem_image).squeeze()
        if sem_feat.dim() == 1:
            sem_feat = sem_feat.unsqueeze(0)
        fused = torch.cat([geo_feat, meta_feat, graph_feat, sem_feat], dim=1)
        fused = self.fusion(fused)
        class_output = self.classifier(fused)
        voxel_output = self.voxel_generator(fused).view(-1, 64, 64, 64)
        return class_output, voxel_output

def preprocess_sem_image(image_path):
    """预处理 SEM 图像"""
    try:
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(image)
    except Exception as e:
        logging.error(f"预处理SEM图像 {image_path} 失败: {str(e)}")
        raise

def segment_pores(image_path):
    """分割 SEM 图像中的孔隙和固体部分"""
    raw_image = Image.open(image_path).convert('L')
    raw_array = np.array(raw_image, dtype=np.float32) / 255.0
    threshold_sem = threshold_otsu(raw_array)
    binary_sem = (raw_array > threshold_sem).astype(np.uint8)
    return binary_sem

def generate_skeleton_lines(skeleton, max_distance=3.0):
    """生成骨架线"""
    skeleton_coords = np.argwhere(skeleton > 0)
    if len(skeleton_coords) == 0:
        logging.warning("骨骼化结果为空，无法生成线阵网络结构")
        return [], skeleton_coords
    tree = cKDTree(skeleton_coords)
    pairs = tree.query_pairs(r=max_distance, p=2)
    lines = [[i, j] for i, j in pairs]
    logging.info(f"生成了 {len(lines)} 条骨架线段")
    return lines, skeleton_coords

def lines_to_mesh(vertices, lines, radius=0.1, max_lines=1000, timeout=300):
    """将骨架线转换为网格"""
    meshes = []
    invalid_lines = 0
    start_time = time.time()
    max_lines = min(max_lines, len(lines))
    logging.info(f"开始将 {max_lines} 条线段转换为三角网格（总共 {len(lines)} 条）")

    for idx, line in enumerate(lines[:max_lines]):
        if time.time() - start_time > timeout:
            logging.warning(f"处理线段超时（{timeout}秒），已处理 {idx} 条线段，跳过剩余部分")
            break
        start_idx, end_idx = line
        start = vertices[start_idx]
        end = vertices[end_idx]
        direction = end - start
        length = np.linalg.norm(direction)
        if length < 1e-6:
            invalid_lines += 1
            continue
        direction = direction / length
        try:
            cylinder = trimesh.creation.cylinder(radius=radius, height=length, sections=8)
            z_axis = np.array([0, 0, 1])
            axis = np.cross(z_axis, direction)
            if np.linalg.norm(axis) > 0:
                axis = axis / np.linalg.norm(axis)
                angle = np.arccos(np.dot(z_axis, direction))
                rotation = trimesh.transformations.rotation_matrix(angle, axis)
                cylinder.apply_transform(rotation)
            mid_point = (start + end) / 2
            translation = trimesh.transformations.translation_matrix(mid_point)
            cylinder.apply_transform(translation)
            cylinder.visual.vertex_colors = [0, 255, 0, 255]
            meshes.append(cylinder)
        except Exception as e:
            logging.warning(f"生成圆柱 {idx} 失败: {str(e)}")
            invalid_lines += 1

    logging.info(f"有效线段数: {len(lines[:max_lines]) - invalid_lines}, 无效线段数: {invalid_lines}")
    return trimesh.util.concatenate(meshes) if meshes else None

def generate_3d_model(sem_image_path, output_dir="/pytest/training/测试样本/output"):
    """根据 SEM 图像生成 3D 模型"""
    logging.info(f"开始生成3D模型: {sem_image_path}")
    if not os.path.exists(sem_image_path):
        logging.error(f"SEM图像文件不存在: {sem_image_path}")
        return

    # 加载模型
    model = MultiModal3DModel()
    model_path = 'model_cpu.pth'
    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
        logging.info("模型加载成功")
    except Exception as e:
        logging.error(f"加载模型失败: {str(e)}")
        return
    model.eval()

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(sem_image_path))[0]
    output_combined_path = os.path.join(output_dir, f"{base}_generated.ply")
    output_pores_path = os.path.join(output_dir, f"{base}_pores.ply")
    output_skeleton_path = os.path.join(output_dir, f"{base}_skeleton.ply")
    output_material_path = os.path.join(output_dir, f"{base}_material.ply")
    output_combined_plot_path = os.path.join(output_dir, f"{base}_combined_plot.png")

    try:
        with torch.no_grad():
            sem_image = preprocess_sem_image(sem_image_path).unsqueeze(0)
            # 使用占位输入调用模型
            _, voxel_output = model(
                torch.zeros(1, 1, 64, 64, 64), torch.zeros(1, 3), dgl.graph(([0], [0])), sem_image
            )
        voxel_data = voxel_output.numpy().squeeze()
        logging.info(f"体素数据形状: {voxel_data.shape}")
        logging.info(f"体素数据最小值: {voxel_data.min()}, 最大值: {voxel_data.max()}, 平均值: {voxel_data.mean()}")
        if voxel_data.max() == voxel_data.min():
            logging.error("体素数据无效，所有值相同，无法生成网格")
            return

        # 归一化体素数据
        voxel_data = (voxel_data - voxel_data.min()) / (voxel_data.max() - voxel_data.min())
        logging.info(f"归一化体素数据最小值: {voxel_data.min()}, 最大值: {voxel_data.max()}, 平均值: {voxel_data.mean()}")

        # 可视化切片
        z_slice = 50
        voxel_slice_data = voxel_data[z_slice, :, :]
        voxel_slice_smooth = gaussian_filter(voxel_slice_data, sigma=2)
        threshold_voxel = threshold_otsu(voxel_slice_smooth)
        voxel_slice = (voxel_data[z_slice, :, :] > threshold_voxel).astype(np.uint8)
        slice_pore_pixels = np.sum(voxel_slice == 0)
        slice_material_pixels = np.sum(voxel_slice == 1)
        logging.info(f"体视切片 (z={z_slice}): Otsu 阈值 {threshold_voxel:.3f}, 孔隙像素数: {slice_pore_pixels}, 材料像素数: {slice_material_pixels}")

        # 生成对比图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        raw_image = Image.open(sem_image_path).convert('L')
        raw_array = np.array(raw_image, dtype=np.float32) / 255.0
        binary_sem = segment_pores(sem_image_path)
        axes[0].imshow(raw_array, cmap='gray')
        axes[0].set_xlabel('X pixels')
        axes[0].set_ylabel('Y pixels')
        axes[0].set_title('Original SEM')
        axes[1].imshow(binary_sem, cmap='gray')
        axes[1].set_xlabel('X pixels')
        axes[1].set_ylabel('Y pixels')
        axes[1].set_title('Binary SEM')
        axes[2].imshow(voxel_slice, cmap='gray')
        axes[2].set_xlabel('X pixels')
        axes[2].set_ylabel('Y pixels')
        axes[2].set_title(f'Volume Slice (z={z_slice})')
        plt.tight_layout()
        plt.savefig(output_combined_plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        logging.info(f"对比图已保存: {output_combined_plot_path}")

        # 生成材料网格
        voxel_data_smooth = gaussian_filter(voxel_data, sigma=1)
        threshold_voxel_data = threshold_otsu(voxel_data_smooth.flatten()) * 0.65
        vertices_mat, triangles_mat = mcubes.marching_cubes(voxel_data, threshold_voxel_data)
        mesh_mat = trimesh.Trimesh(vertices=vertices_mat, faces=triangles_mat)
        if len(triangles_mat) > 0:
            try:
                mesh_o3d = o3d.geometry.TriangleMesh(
                    vertices=o3d.utility.Vector3dVector(mesh_mat.vertices),
                    triangles=o3d.utility.Vector3iVector(mesh_mat.faces)
                )
                mesh_o3d = mesh_o3d.simplify_quadric_decimation(target_number_of_triangles=len(triangles_mat) // 2)
                mesh_mat = trimesh.Trimesh(vertices=np.asarray(mesh_o3d.vertices), faces=np.asarray(mesh_o3d.triangles))
            except Exception as e:
                logging.warning(f"网格简化失败: {str(e)}，将使用原始网格")
        mesh_mat.visual.vertex_colors = [255, 0, 0, 255]
        mat_scene = trimesh.Scene(mesh_mat)
        mat_scene.export(output_material_path, file_type='ply')
        logging.info(f"材料网格已保存: {output_material_path}")

        # 生成孔隙网格
        pore_voxel = 1 - voxel_data
        vertices_por, triangles_por = mcubes.marching_cubes(pore_voxel, threshold_voxel_data)
        mesh_por = trimesh.Trimesh(vertices=vertices_por, faces=triangles_por)
        mesh_por.visual.vertex_colors = [0, 0, 255, 32]
        scene_por = trimesh.Scene(mesh_por)
        scene_por.export(output_pores_path, file_type='ply')
        logging.info(f"孔隙网格已保存: {output_pores_path}")

        # 骨架化处理
        if measure is not None and morphology is not None:
            try:
                pore_binary = (pore_voxel > threshold_voxel_data).astype(np.uint8)
                pore_binary = morphology.binary_opening(pore_binary, morphology.ball(3))
                skeleton = morphology.skeletonize(pore_binary)
                lines, skeleton_coords = generate_skeleton_lines(skeleton, max_distance=3.0)
                if lines:
                    skeleton_mesh = lines_to_mesh(skeleton_coords, lines, radius=0.1, max_lines=1000, timeout=300)
                    if skeleton_mesh:
                        skeleton_scene = trimesh.Scene(skeleton_mesh)
                        skeleton_scene.export(output_skeleton_path, file_type='ply')
                        logging.info(f"骨架网格已保存: {output_skeleton_path}")
            except Exception as e:
                logging.error(f"骨架化处理失败: {str(e)}")

        # 组合模型
        meshes_to_combine = [mesh_mat, mesh_por]
        scene_combined = trimesh.Scene(meshes_to_combine)
        scene_combined.export(output_combined_path, file_type='ply')
        logging.info(f"组合模型已保存: {output_combined_path}")

    except Exception as e:
        logging.error(f"生成模型失败: {str(e)}")
        raise

def main():
    """主函数，指定 SEM 图像路径并生成 3D 模型"""
    sem_image_path = "/pytest/training/测试样本/测试/实木板.tif"
    output_dir = "/pytest/training/测试样本/output"
    generate_3d_model(sem_image_path, output_dir)

if __name__ == "__main__":
    logging.info("程序启动")
    main()
    logging.info("程序已完成")