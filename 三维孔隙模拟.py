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

matplotlib.use('Agg')

# 固定随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

try:
    from skimage import measure, morphology
    from skimage.morphology import binary_opening, footprint_rectangle
except ImportError as e:
    logging.error(f"无法导入 skimage 模块: {str(e)}")
    logging.warning("连通孔隙分析和骨骼化功能将不可用")
    measure = None
    morphology = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)


def stl_to_voxel(stl_path, voxel_res=64):
    try:
        mesh_obj = trimesh.load(stl_path)
        voxels = mesh_obj.voxelized(pitch=mesh_obj.extents.max() / voxel_res)
        voxel_matrix = voxels.matrix.astype(np.float32)
        pad_width = [(0, max(0, voxel_res - s)) for s in voxel_matrix.shape]
        voxel_matrix = np.pad(voxel_matrix, pad_width, mode='constant', constant_values=0)
        voxel_matrix = voxel_matrix[:voxel_res, :voxel_res, :voxel_res]
        return voxel_matrix
    except Exception as e:
        logging.error(f"加载 {stl_path} 时出错: {str(e)}")
        return np.zeros((voxel_res, voxel_res, voxel_res), dtype=np.float32)


def load_json_metadata(json_path):
    with open(json_path) as f:
        data = json.load(f)
    features = {'material_density': 0.0, 'temperature_rating': 0.0, 'components': 0}
    try:
        project_info = data.get('projects', [{}])[0].get('project_information', {})
        features['material_density'] = project_info.get('material_density', 0.0)
        features['temperature_rating'] = project_info.get('temperature_rating', 0.0)
        features['components'] = len(data.get('projects', [{}])[0].get('meshes', []))
    except Exception as e:
        logging.error(f"加载 {json_path} 时出错: {str(e)}")
    return np.array([features['material_density'], features['temperature_rating'], features['components']],
                    dtype=np.float32)


def load_ttl_knowledge(ttl_path):
    try:
        g = Graph()
        g.parse(ttl_path, format="ttl")
        triples = [(str(s), str(p), str(o)) for s, p, o in g if len((s, p, o)) == 3]
        return triples
    except Exception as e:
        logging.error(f"加载 {ttl_path} 时出错: {str(e)}")
        return []


def build_graph_from_triples(triples):
    src, dst, edge_types = [], [], []
    node_map, edge_type_map = {}, {}
    for triple in triples:
        try:
            s, p, o = triple
            for node in [s, o]:
                if node not in node_map:
                    node_map[node] = len(node_map)
            if p not in edge_type_map:
                edge_type_map[p] = len(edge_type_map)
            src.append(node_map[s])
            dst.append(node_map[o])
            edge_types.append(edge_type_map[p])
        except ValueError:
            continue
    g = dgl.graph((src, dst)) if src else dgl.graph(([0], [0]))
    g.edata['type'] = torch.tensor(edge_types, dtype=torch.long) if edge_types else torch.tensor([0], dtype=torch.long)
    return g


def preprocess_sem_image(image_path):
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


class MultiModal3DDataset(Dataset):
    def __init__(self, data_root, voxel_res=64):
        self.data_root = data_root
        self.voxel_res = voxel_res
        self.samples = self._prepare_samples()
        self.labels = self._generate_labels()

    def _prepare_samples(self):
        stl_dir = os.path.join(self.data_root, "stl")
        json_dir = os.path.join(self.data_root, "json")
        ttl_dir = os.path.join(self.data_root, "ttl")
        sem_dir = os.path.join(self.data_root, "sem")
        samples = []
        for fname in sorted(os.listdir(stl_dir)):
            if fname.endswith('.stl'):
                base = os.path.splitext(fname)[0]
                paths = [
                    os.path.join(stl_dir, fname),
                    os.path.join(json_dir, f"{base}.json"),
                    os.path.join(ttl_dir, f"{base}.ttl"),
                    os.path.join(sem_dir, f"{base}.tif")
                ]
                if all(os.path.exists(p) for p in paths):
                    samples.append(paths)
        return samples

    def _generate_labels(self):
        labels = {}
        for stl_path, json_path, _, _ in self.samples:
            base = os.path.splitext(os.path.basename(stl_path))[0]
            try:
                metadata = load_json_metadata(json_path)
                density = metadata[0]
                labels[base] = 0 if density < 2000 else 1 if density < 5000 else 2
            except Exception as e:
                logging.error(f"生成 {base} 标签时出错: {str(e)}")
                labels[base] = -1
        return labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        stl_path, json_path, ttl_path, sem_path = self.samples[idx]
        base = os.path.splitext(os.path.basename(stl_path))[0]
        triples = load_ttl_knowledge(ttl_path)
        graph = build_graph_from_triples(triples)
        return {
            'voxels': torch.from_numpy(stl_to_voxel(stl_path, self.voxel_res)).unsqueeze(0),
            'metadata': torch.tensor(load_json_metadata(json_path)),
            'graph': graph,
            'sem_image': preprocess_sem_image(sem_path),
            'label': torch.tensor(self.labels[base])
        }


def validate_data_structure(data_root):
    required_dirs = ["stl", "json", "ttl", "sem"]
    missing_dirs = [d for d in required_dirs if not os.path.exists(os.path.join(data_root, d))]
    if missing_dirs:
        raise FileNotFoundError(f"缺失必要目录: {missing_dirs}")
    json_dir = os.path.join(data_root, "json")
    for fname in sorted(os.listdir(json_dir)):
        if fname.endswith('.json'):
            path = os.path.join(json_dir, fname)
            try:
                with open(path) as f:
                    data = json.load(f)
                if 'projects' not in data or 'project_information' not in data['projects'][0]:
                    logging.warning(f"警告: {fname} 缺少必要字段")
            except Exception as e:
                logging.error(f"解析 {fname} 失败: {str(e)}")


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


def custom_collate(batch):
    return {
        'voxels': torch.stack([item['voxels'] for item in batch]),
        'metadata': torch.stack([item['metadata'] for item in batch]),
        'graph': dgl.batch([dgl.add_self_loop(item['graph']) for item in batch]),
        'sem_image': torch.stack([item['sem_image'] for item in batch]),
        'label': torch.stack([item['label'] for item in batch])
    }


def train_model():
    data_root = "/pytest/training/测试样本"
    try:
        validate_data_structure(data_root)
    except FileNotFoundError as e:
        logging.error(f"数据验证失败: {e}")
        return

    dataset = MultiModal3DDataset(data_root=data_root, voxel_res=64)
    valid_samples = [i for i in range(len(dataset)) if
                     dataset.labels[os.path.splitext(os.path.basename(dataset.samples[i][0]))[0]] != -1]
    dataset = torch.utils.data.Subset(dataset, valid_samples)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=custom_collate)

    model = MultiModal3DModel()
    model_path = 'model_cpu.pth'

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
        logging.info(f"已加载现有模型: {model_path}")
        return

    criterion_class = nn.CrossEntropyLoss()
    criterion_voxel = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    for epoch in range(5):
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            voxels = batch['voxels']
            metadata = batch['metadata']
            graph = batch['graph']
            sem_images = batch['sem_image']
            labels = batch['label']

            optimizer.zero_grad()
            class_output, voxel_output = model(voxels, metadata, graph, sem_images)
            loss_class = criterion_class(class_output, labels)
            loss_voxel = criterion_voxel(voxel_output, voxels.squeeze())
            loss = loss_class + 0.5 * loss_voxel
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        logging.info(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), model_path)
    logging.info(f"模型已保存至 {model_path}")


def generate_skeleton_lines(skeleton, max_distance=3.0):  # 增大 max_distance 到 3.0
    skeleton_coords = np.argwhere(skeleton > 0)
    if len(skeleton_coords) == 0:
        logging.warning("骨骼化结果为空，无法生成线阵网络结构")
        return [], skeleton_coords
    tree = cKDTree(skeleton_coords)
    pairs = tree.query_pairs(r=max_distance, p=2)
    lines = [[i, j] for i, j in pairs]
    logging.info(f"生成了 {len(lines)} 条骨架线段")
    if len(lines) == 0:
        logging.warning(f"未生成线段，尝试增大 max_distance，目前为 {max_distance}")
    return lines, skeleton_coords


def save_lines_as_obj(vertices, lines, output_path, color=[0, 255, 0, 255]):
    with open(output_path, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]} {color[0] / 255} {color[1] / 255} {color[2] / 255}\n")
        for line in lines:
            start_idx, end_idx = line
            f.write(f"l {start_idx + 1} {end_idx + 1}\n")
    logging.info(f"手动保存 .obj 文件: {output_path}, 顶点数: {len(vertices)}, 边数: {len(lines)}")


def lines_to_mesh(vertices, lines, radius=0.1, max_lines=1000, timeout=300):
    meshes = []
    invalid_lines = 0
    start_time = time.time()
    max_lines = min(max_lines, len(lines))
    logging.info(f"开始将 {max_lines} 条线段转换为三角网格（总共 {len(lines)} 条）")

    for idx, line in enumerate(lines[:max_lines]):
        if time.time() - start_time > timeout:
            logging.warning(f"处理线段超时（{timeout}秒），已处理 {idx} 条线段，跳过剩余部分")
            break
        if idx % 1000 == 0 and idx > 0:
            logging.info(f"已处理 {idx} 条线段，耗时 {time.time() - start_time:.2f} 秒")

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
            cylinder = trimesh.creation.cylinder(
                radius=radius,
                height=length,
                sections=8
            )
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
    if meshes:
        return trimesh.util.concatenate(meshes)
    else:
        return None


def generate_3d_model(sem_image_path, output_dir="/pytest/training/测试样本/output"):
    logging.info(f"开始生成3D模型: {sem_image_path}")
    if not os.path.exists(sem_image_path):
        logging.error(f"SEM图像文件不存在: {sem_image_path}")
        return

    model = MultiModal3DModel()
    model_path = 'model_cpu.pth'
    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
        logging.info("模型加载成功")
    except Exception as e:
        logging.error(f"加载模型失败: {str(e)}")
        return
    model.eval()

    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(sem_image_path))[0]
    output_combined_path = os.path.join(output_dir, f"{base}_generated.ply")
    output_pores_path = os.path.join(output_dir, f"{base}_pores.ply")
    output_skeleton_path = os.path.join(output_dir, f"{base}_skeleton.ply")
    output_skeleton_obj_path = os.path.join(output_dir, f"{base}_skeleton.obj")
    output_skeleton_lines_path = os.path.join(output_dir, f"{base}_skeleton_lines.ply")
    output_skeleton_lines_obj_path = os.path.join(output_dir, f"{base}_skeleton_lines.obj")
    output_material_path = os.path.join(output_dir, f"{base}_material.ply")
    output_skeleton_points_path = os.path.join(output_dir, f"{base}_skeleton_points.ply")
    output_combined_plot_path = os.path.join(output_dir, f"{base}_combined_plot.png")

    try:
        with torch.no_grad():
            sem_image = preprocess_sem_image(sem_image_path).unsqueeze(0)

            raw_image = Image.open(sem_image_path).convert('L')
            raw_array = np.array(raw_image, dtype=np.float32) / 255.0

            threshold_sem = threshold_otsu(raw_array)
            binary_sem = (raw_array > threshold_sem).astype(np.uint8)
            pore_pixels = np.sum(binary_sem == 0)
            material_pixels = np.sum(binary_sem == 1)
            logging.info(
                f"二值化 SEM 图像: 使用 Otsu 阈值 {threshold_sem:.3f}, 孔隙像素数: {pore_pixels}, 材料像素数: {material_pixels}")

            _, voxel_output = model(
                torch.zeros(1, 1, 64, 64, 64), torch.zeros(1, 3), dgl.graph(([0], [0])), sem_image
            )

        voxel_data = voxel_output.numpy().squeeze()
        logging.info(f"Voxel data shape: {voxel_data.shape}")
        logging.info(f"Voxel data min: {voxel_data.min()}, max: {voxel_data.max()}, mean: {voxel_data.mean()}")
        logging.info(f"Voxel data non-zero count: {np.count_nonzero(voxel_data)}")
        if voxel_data.max() == voxel_data.min():
            logging.error("体素数据无效，所有值相同，无法生成网格")
            return

        voxel_data = (voxel_data - voxel_data.min()) / (voxel_data.max() - voxel_data.min())
        logging.info(
            f"Normalized voxel data min: {voxel_data.min()}, max: {voxel_data.max()}, mean: {voxel_data.mean()}")

        z_slice = 50
        voxel_slice_data = voxel_data[z_slice, :, :]
        voxel_slice_smooth = gaussian_filter(voxel_slice_data, sigma=2)  # 增大 sigma 到 2
        threshold_voxel = threshold_otsu(voxel_slice_smooth)
        voxel_slice = (voxel_data[z_slice, :, :] > threshold_voxel).astype(np.uint8)
        slice_pore_pixels = np.sum(voxel_slice == 0)
        slice_material_pixels = np.sum(voxel_slice == 1)
        logging.info(
            f"体视切片 (z={z_slice}): 使用调整后的 Otsu 阈值 {threshold_voxel:.3f}, 孔隙像素数: {slice_pore_pixels}, 材料像素数: {slice_material_pixels}")

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

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
        logging.info(f"Combined plot saved: {output_combined_plot_path}")

        projection_0 = np.sum(voxel_data, axis=0)
        logging.info(
            f"投影图 (0°): 形状: {projection_0.shape}, 最小值: {projection_0.min():.4f}, 最大值: {projection_0.max():.4f}, 平均值: {projection_0.mean():.4f}")

        center_slice = 32
        reconstructed_slice = voxel_data[center_slice, :, :]
        logging.info(
            f"重建切片: 中心切片 (z={center_slice}), 最小值: {reconstructed_slice.min():.4f}, 最大值: {reconstructed_slice.max():.4f}, 平均值: {reconstructed_slice.mean():.4f}")

        voxel_data_smooth = gaussian_filter(voxel_data, sigma=1)
        threshold_voxel_data = threshold_otsu(voxel_data_smooth.flatten()) * 0.8
        logging.info(f"体视数据 Otsu 阈值 (调整后): {threshold_voxel_data:.3f}")
        vertices_mat, triangles_mat = mcubes.marching_cubes(voxel_data, threshold_voxel_data)
        mesh_mat = trimesh.Trimesh(vertices=vertices_mat, faces=triangles_mat)
        if len(triangles_mat) > 0:
            try:
                # 使用 open3d 进行网格简化
                mesh_o3d = o3d.geometry.TriangleMesh(
                    vertices=o3d.utility.Vector3dVector(mesh_mat.vertices),
                    triangles=o3d.utility.Vector3iVector(mesh_mat.faces)
                )
                mesh_o3d = mesh_o3d.simplify_quadric_decimation(target_number_of_triangles=len(triangles_mat) // 2)
                mesh_mat = trimesh.Trimesh(vertices=np.asarray(mesh_o3d.vertices), faces=np.asarray(mesh_o3d.triangles))
            except Exception as e:
                logging.warning(f"网格简化失败: {str(e)}，将使用原始网格")
        mesh_mat.visual.vertex_colors = [255, 0, 0, 255]
        logging.info(f"Material mesh vertices (after simplification): {len(mesh_mat.vertices)}, triangles: {len(mesh_mat.faces)}")
        mat_scene = trimesh.Scene(mesh_mat)
        mat_scene.export(output_material_path, file_type='ply')
        if os.path.exists(output_material_path):
            logging.info(f"材料网格已保存: {output_material_path}")
        else:
            logging.error(f"材料网格保存失败: {output_material_path}")

        pore_voxel = 1 - voxel_data
        logging.info(f"Pore voxel min: {pore_voxel.min()}, max: {pore_voxel.max()}, mean: {pore_voxel.mean()}")
        logging.info(f"Pore voxel non-zero count: {np.count_nonzero(pore_voxel)}")
        vertices_por, triangles_por = mcubes.marching_cubes(pore_voxel, threshold_voxel_data)
        mesh_por = trimesh.Trimesh(vertices=vertices_por, faces=triangles_por)
        mesh_por.visual.vertex_colors = [0, 0, 255, 32]
        scene_por = trimesh.Scene(mesh_por)
        scene_por.export(output_pores_path, file_type='ply')
        if os.path.exists(output_pores_path):
            logging.info(f"成功生成孔隙模型: {output_pores_path}")
        else:
            logging.error(f"孔隙模型保存失败: {output_pores_path}")

        skeleton_generated = False
        if measure is not None and morphology is not None:
            try:
                pore_binary = (pore_voxel > threshold_voxel_data).astype(np.uint8)
                pore_binary = binary_opening(pore_binary, footprint=footprint_rectangle((3, 3, 3)))
                total_pore_voxels = np.sum(pore_binary)
                labeled_pores, num_labels = measure.label(pore_binary, connectivity=1, return_num=True)
                if num_labels > 0:
                    pore_sizes = np.bincount(labeled_pores.ravel())[1:]
                    largest_pore_size = np.max(pore_sizes) if len(pore_sizes) > 0 else 0
                    logging.info(f"Total pore voxels: {total_pore_voxels}")
                    logging.info(f"Largest connected pore voxels: {largest_pore_size}")
                    logging.info(f"Connected pore ratio: {largest_pore_size / total_pore_voxels:.4f}")

                skeleton = morphology.skeletonize(pore_binary, method='lee')
                logging.info(f"Skeleton non-zero count: {np.count_nonzero(skeleton)}")
                lines, skeleton_coords = generate_skeleton_lines(skeleton, max_distance=3.0)

                if len(skeleton_coords) > 0:
                    point_cloud = trimesh.Trimesh(vertices=skeleton_coords, vertex_colors=[0, 255, 0, 255])
                    point_scene = trimesh.Scene(point_cloud)
                    point_scene.export(output_skeleton_points_path, file_type='ply')
                    if os.path.exists(output_skeleton_points_path):
                        logging.info(f"成功生成骨架点云: {output_skeleton_points_path}")
                    else:
                        logging.error(f"骨架点云保存失败: {output_skeleton_points_path}")

                if lines:
                    skeleton_lines = trimesh.Trimesh(
                        vertices=skeleton_coords,
                        edges=np.array(lines, dtype=np.int32),
                        vertex_colors=[0, 255, 0, 255]
                    )
                    skeleton_lines_scene = trimesh.Scene(skeleton_lines)
                    skeleton_lines_scene.export(output_skeleton_lines_path, file_type='ply')
                    if os.path.exists(output_skeleton_lines_path):
                        logging.info(f"成功生成骨架线段 (PLY): {output_skeleton_lines_path}")
                    else:
                        logging.error(f"骨架线段 (PLY) 保存失败: {output_skeleton_lines_path}")

                    save_lines_as_obj(skeleton_coords, lines, output_skeleton_lines_obj_path)

                if lines:
                    skeleton_mesh = lines_to_mesh(skeleton_coords, lines, radius=0.1, max_lines=1000, timeout=300)
                    if skeleton_mesh:
                        skeleton_scene = trimesh.Scene(skeleton_mesh)
                        skeleton_scene.export(output_skeleton_path, file_type='ply')
                        skeleton_scene.export(output_skeleton_obj_path, file_type='obj')
                        if os.path.exists(output_skeleton_path):
                            logging.info(f"成功生成骨骼化网络结构 (PLY): {output_skeleton_path}")
                            skeleton_generated = True
                        else:
                            logging.error(f"骨骼化网络结构 (PLY) 保存失败: {output_skeleton_path}")
                        if os.path.exists(output_skeleton_obj_path):
                            logging.info(f"成功生成骨骼化网络结构 (OBJ): {output_skeleton_obj_path}")
                        else:
                            logging.error(f"骨骼化网络结构 (OBJ) 保存失败: {output_skeleton_obj_path}")
                    else:
                        logging.warning("未生成骨架网格（可能是线段无效）")
                else:
                    logging.warning("未生成骨架线段")
            except Exception as e:
                logging.error(f"骨架化处理失败: {str(e)}")

        meshes_to_combine = [mesh_mat]
        mesh_por_combined = trimesh.Trimesh(vertices=vertices_por, faces=triangles_por)
        mesh_por_combined.visual.vertex_colors = [0, 0, 0, 128]
        meshes_to_combine.append(mesh_por_combined)
        scene_combined = trimesh.Scene(meshes_to_combine)
        scene_combined.export(output_combined_path, file_type='ply')
        if os.path.exists(output_combined_path):
            logging.info(f"成功生成组合模型: {output_combined_path}")
        else:
            logging.error(f"组合模型保存失败: {output_combined_path}")

    except Exception as e:
        logging.error(f"生成模型失败: {str(e)}")
        raise


if __name__ == "__main__":
    logging.info("程序启动")
    train_model()
    generate_3d_model("/pytest/training/测试样本/测试/实木板.tif")
    logging.info("程序已全部完成")
    logging.getLogger().handlers[0].flush()