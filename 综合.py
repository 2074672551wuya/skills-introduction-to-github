import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import trimesh
import json
from rdflib import Graph
import dgl
import dgl.nn as dglnn
from PIL import Image
import torchvision.transforms as transforms
import mcubes
import scipy.ndimage as ndi
from skimage.morphology import skeletonize
from skimage import filters
import warnings
import cv2
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)


# 数据解析与预处理函数
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
        print(f"加载 {stl_path} 时出错: {str(e)}")
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
        print(f"加载 {json_path} 时出错: {str(e)}")
    return np.array([features['material_density'], features['temperature_rating'], features['components']],
                    dtype=np.float32)


def load_ttl_knowledge(ttl_path):
    try:
        g = Graph()
        g.parse(ttl_path, format="ttl")
        triples = [(str(s), str(p), str(o)) for s, p, o in g if len((s, p, o)) == 3]
        return triples
    except Exception as e:
        print(f"加载 {ttl_path} 时出错: {str(e)}")
        return []


def build_graph_from_triples(triples):
    src, dst, edge_types = [], [], []
    node_map = {}
    edge_type_map = {}
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
    if len(src) == 0:
        return dgl.graph(([0], [0]))
    g = dgl.graph((src, dst))
    g.edata['type'] = torch.tensor(edge_types, dtype=torch.long)
    return g


def preprocess_sem_image(image_path, pixel_size=0.1):
    """
    预处理SEM图像，提取孔隙特征
    pixel_size: 像素尺寸（μm/像素），根据标尺计算
    """
    # 加载灰度图像用于二值化
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("无法加载SEM图像")

    # 中值滤波去噪
    denoised = cv2.medianBlur(image, 5)
    # 阈值分割
    threshold = filters.threshold_otsu(denoised)
    binary_image = (denoised > threshold).astype(np.uint8)

    # 将灰度图像转换为伪RGB图像（复制通道）
    rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # 调整图像大小并转换为张量
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(rgb_image), binary_image


# 数据验证模块
def validate_data_structure(data_root):
    required_dirs = ["stl", "json", "ttl", "sem"]
    missing_dirs = [d for d in required_dirs if not os.path.exists(os.path.join(data_root, d))]
    if missing_dirs:
        raise FileNotFoundError(f"缺失必要目录: {missing_dirs}")
    json_dir = os.path.join(data_root, "json")
    for fname in os.listdir(json_dir):
        if fname.endswith('.json'):
            path = os.path.join(json_dir, fname)
            try:
                with open(path) as f:
                    data = json.load(f)
                if 'projects' not in data:
                    print(f"警告: {fname} 缺少projects字段")
                elif 'project_information' not in data['projects'][0]:
                    print(f"警告: {fname} 缺少project_information字段")
            except Exception as e:
                print(f"解析 {fname} 失败: {str(e)}")


# 数据集类
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
        for fname in os.listdir(stl_dir):
            if fname.endswith('.stl'):
                base = os.path.splitext(fname)[0]
                stl_path = os.path.join(stl_dir, fname)
                json_path = os.path.join(json_dir, f"{base}.json")
                ttl_path = os.path.join(ttl_dir, f"{base}.ttl")
                sem_path = os.path.join(sem_dir, f"{base}.tif")
                if all(os.path.exists(p) for p in [json_path, ttl_path, sem_path]):
                    samples.append((stl_path, json_path, ttl_path, sem_path))
        return samples

    def _generate_labels(self):
        labels = {}
        for stl_path, json_path, _, _ in self.samples:
            base = os.path.splitext(os.path.basename(stl_path))[0]
            try:
                metadata = load_json_metadata(json_path)
                density = metadata[0]
                label = 0 if density < 2000 else 1 if density < 5000 else 2
                labels[base] = label
            except Exception as e:
                print(f"生成 {base} 标签时出错: {str(e)}")
                labels[base] = -1
        return labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        stl_path, json_path, ttl_path, sem_path = self.samples[idx]
        base = os.path.splitext(os.path.basename(stl_path))[0]
        triples = load_ttl_knowledge(ttl_path)
        graph = build_graph_from_triples(triples)
        sem_image, binary_sem = preprocess_sem_image(sem_path)
        return {
            'voxels': torch.from_numpy(stl_to_voxel(stl_path, self.voxel_res)).unsqueeze(0),
            'metadata': torch.tensor(load_json_metadata(json_path)),
            'graph': graph,
            'sem_image': sem_image,
            'label': torch.tensor(self.labels[base]),
            'binary_sem': binary_sem  # 添加二值化SEM图像
        }


# 模型定义
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
        self.voxel_generator = nn.Sequential(
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 64 * 64 * 64), nn.Sigmoid()
        )

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
        graph.ndata['h'] = self.gnn(graph, graph.ndata['h'])
        num_nodes = graph.ndata['h'].size(0)
        graph.ndata['h'] = graph.ndata['h'].view(num_nodes, -1)
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


# 自定义collate函数
def custom_collate(batch):
    voxels = torch.stack([item['voxels'] for item in batch])
    metadata = torch.stack([item['metadata'] for item in batch])
    graphs = [item['graph'] for item in batch]
    sem_images = torch.stack([item['sem_image'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    binary_sems = [item['binary_sem'] for item in batch]  # 添加二值化SEM图像
    assert len(graphs) == len(batch), f"Graph count {len(graphs)} mismatch with batch size {len(batch)}"
    graphs_with_self_loops = [dgl.add_self_loop(g) for g in graphs]
    batched_graph = dgl.batch(graphs_with_self_loops)
    return {
        'voxels': voxels,
        'metadata': metadata,
        'graph': batched_graph,
        'sem_image': sem_images,
        'label': labels,
        'binary_sem': binary_sems
    }


# 训练函数
def train_model():
    data_root = "/pytest/training/PythonProject3/training"
    try:
        validate_data_structure(data_root)
    except FileNotFoundError as e:
        print(f"数据验证失败: {e}")
        return
    dataset = MultiModal3DDataset(data_root=data_root, voxel_res=64)
    valid_samples = [i for i in range(len(dataset)) if
                     dataset.labels[os.path.splitext(os.path.basename(dataset.samples[i][0]))[0]] != -1]
    dataset = Subset(dataset, valid_samples)
    train_idx, test_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=custom_collate)
    test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=custom_collate)
    model = MultiModal3DModel()
    criterion_class = nn.CrossEntropyLoss()
    criterion_voxel = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    output_dir = os.path.join(data_root, "output")
    os.makedirs(output_dir, exist_ok=True)
    for epoch in range(10):
        model.train()
        total_loss = 0
        for batch in enumerate(train_dataloader):
            batch = batch[1]
            optimizer.zero_grad()
            class_output, voxel_output = model(batch['voxels'], batch['metadata'], batch['graph'], batch['sem_image'])
            loss_class = criterion_class(class_output, batch['label'])
            loss_voxel = criterion_voxel(voxel_output, batch['voxels'].squeeze())
            loss = loss_class + 0.5 * loss_voxel
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}, Train Loss: {avg_loss:.4f}")
        model.eval()
        total_test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_dataloader:
                class_output, voxel_output = model(batch['voxels'], batch['metadata'], batch['graph'],
                                                   batch['sem_image'])
                loss_class = criterion_class(class_output, batch['label'])
                loss_voxel = criterion_voxel(voxel_output, batch['voxels'].squeeze())
                test_loss = loss_class + 0.5 * loss_voxel
                total_test_loss += test_loss.item()
                _, predicted = torch.max(class_output.data, 1)
                total += batch['label'].size(0)
                correct += (predicted == batch['label']).sum().item()
            avg_test_loss = total_test_loss / len(test_dataloader)
            test_accuracy = 100 * correct / total
            print(f"Epoch {epoch + 1}, Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
        model_path = os.path.join(output_dir, 'model_cpu.pth')
        torch.save(model.state_dict(), model_path)
        print(f"模型已保存到: {model_path}")


# 后处理函数：调整孔隙率和孔径分布
def adjust_voxel_data(voxel_data, target_porosity=0.5467, target_pore_size_range=(5.26, 320), voxel_size=0.1,
                      max_iterations=100):
    """
    调整体素数据以匹配目标孔隙率和孔径分布
    target_porosity: 目标孔隙率（文章中为54.67%）
    target_pore_size_range: 目标孔径范围（μm）
    voxel_size: 体素尺寸（μm/体素）
    max_iterations: 最大迭代次数
    """
    # 计算当前孔隙率
    current_porosity = np.mean(voxel_data < 0.5)
    print(f"Initial Porosity: {current_porosity:.4f}")

    # 初始二值化
    adjusted_voxel = (voxel_data < 0.5).astype(np.uint8)
    best_voxel = adjusted_voxel.copy()
    best_porosity = current_porosity
    threshold = 0.5

    # 调整孔隙率
    for iteration in range(max_iterations):
        # 更温和的阈值调整
        if current_porosity < target_porosity:
            threshold -= 0.01  # 减小阈值以增加孔隙
        else:
            threshold += 0.01  # 增加阈值以减少孔隙
        adjusted_voxel = (voxel_data < threshold).astype(np.uint8)
        current_porosity = np.mean(adjusted_voxel)
        print(f"Iteration {iteration + 1}, Threshold: {threshold:.4f}, Current Porosity: {current_porosity:.4f}")

        # 记录最佳孔隙率
        if abs(current_porosity - target_porosity) < abs(best_porosity - target_porosity):
            best_voxel = adjusted_voxel.copy()
            best_porosity = current_porosity

        # 如果孔隙率接近目标（误差小于0.01），提前停止
        if abs(current_porosity - target_porosity) < 0.01:
            break

        # 防止孔隙完全消失
        if current_porosity < 0.01:
            print("警告：孔隙率过低，恢复到最佳状态")
            adjusted_voxel = best_voxel.copy()
            current_porosity = best_porosity
            break

    # 调整孔径分布
    distances = ndi.distance_transform_edt(adjusted_voxel)
    pore_diameters = distances[adjusted_voxel == 1] * 2 * voxel_size
    min_diameter, max_diameter = target_pore_size_range

    # 调试：打印孔径分布
    if len(pore_diameters) > 0:
        print(f"Pore Diameter Range: {np.min(pore_diameters):.2f} ~ {np.max(pore_diameters):.2f} μm")
    else:
        print("警告：无孔隙，无法计算孔径分布")

    # 过滤过小和过大的孔隙
    mask = (distances * 2 * voxel_size >= min_diameter) & (distances * 2 * voxel_size <= max_diameter)
    adjusted_voxel = adjusted_voxel * mask

    # 防止孔隙完全消失
    current_porosity = np.mean(adjusted_voxel)
    if current_porosity < 0.01:
        print("警告：孔径过滤后孔隙率过低，恢复到最佳状态")
        adjusted_voxel = best_voxel.copy()
        current_porosity = best_porosity

    # 微调孔隙率
    for _ in range(max_iterations):
        if abs(current_porosity - target_porosity) < 0.01:
            break
        if current_porosity < target_porosity:
            adjusted_voxel = ndi.binary_dilation(adjusted_voxel, iterations=1)
        else:
            adjusted_voxel = ndi.binary_erosion(adjusted_voxel, iterations=1)
        current_porosity = np.mean(adjusted_voxel)
        print(f"Micro-Adjust Iteration, Current Porosity: {current_porosity:.4f}")

    final_porosity = np.mean(adjusted_voxel)
    if final_porosity < 0.01:
        print("警告：调整后的体素网格孔隙率过低，可能无法生成有效模型")
    print(f"Final Adjusted Porosity: {final_porosity:.4f}")
    return adjusted_voxel


# 计算曲折度
def calculate_tortuosity(voxel_data):
    skeleton = skeletonize(voxel_data)
    z_steps = voxel_data.shape[2]
    path_lengths = []
    for x in range(voxel_data.shape[0]):
        for y in range(voxel_data.shape[1]):
            path = skeleton[x, y, :]
            if np.sum(path) > 0:
                actual_length = np.sum(path)
                straight_length = z_steps
                tortuosity = actual_length / straight_length
                path_lengths.append(tortuosity)
    return np.mean(path_lengths) if path_lengths else 1.0


# 生成函数
def generate_3d_model(sem_image_path, model_path, output_dir, voxel_size=0.1):
    """根据SEM图像生成材料与孔隙结构的3D图像，并增加材料与支架合并文件"""
    device = torch.device('cpu')
    model = MultiModal3DModel().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    except FileNotFoundError:
        print(f"错误：未找到模型文件 {model_path}")
        return
    except Exception as e:
        print(f"加载模型时出现其他错误: {e}")
        return
    model.eval()

    dummy_graph = dgl.graph(([0], [0]))
    with torch.no_grad():
        sem_image, binary_sem = preprocess_sem_image(sem_image_path, voxel_size)
        sem_image = sem_image.unsqueeze(0).to(device)
        _, voxel_output = model(
            torch.zeros(1, 1, 64, 64, 64).to(device),
            torch.zeros(1, 3).to(device),
            dummy_graph,
            sem_image
        )

    # 输出路径处理
    base = os.path.splitext(os.path.basename(sem_image_path))[0]
    output_dir = os.path.join(output_dir, base)
    os.makedirs(output_dir, exist_ok=True)

    try:
        voxel_data = voxel_output.cpu().numpy().squeeze()

        # 后处理：调整孔隙率和孔径分布
        adjusted_voxel = adjust_voxel_data(voxel_data, target_porosity=0.5467, target_pore_size_range=(5.26, 320),
                                           voxel_size=voxel_size)

        # 检查调整后的体素网格
        if np.mean(adjusted_voxel) < 0.01:
            raise ValueError("调整后的体素网格孔隙率过低，无法继续生成")

        # 计算曲折度
        tortuosity = calculate_tortuosity(adjusted_voxel)
        print(f"Tortuosity: {tortuosity:.2f} (Target: 2.07)")

        # 1. 材料部分（红色）
        vertices_mat, triangles_mat = mcubes.marching_cubes(1 - adjusted_voxel, 0.5)  # 材料部分为非孔隙
        if len(vertices_mat) > 0 and len(triangles_mat) > 0:
            mesh_mat = trimesh.Trimesh(vertices=vertices_mat, faces=triangles_mat)
            mesh_mat.visual.vertex_colors = [255, 0, 0, 255]  # 红色
            output_path_mat = os.path.join(output_dir, f"{base}_material.ply")
            mesh_mat.export(output_path_mat, file_type='ply')
            print(f"成功生成材料部分: {output_path_mat}")
        else:
            print("警告：材料部分为空，未生成材料网格")

        # 2. 孔隙处理
        pores = adjusted_voxel.astype(np.uint8)
        labeled_pores, num_features = ndi.label(pores)

        # 识别边界连通孔隙
        boundary_pores = np.zeros_like(labeled_pores)
        for i in range(labeled_pores.shape[0]):
            boundary_pores[i, :, :] = labeled_pores[i, :, :]
            boundary_pores[-i - 1, :, :] = labeled_pores[-i - 1, :, :]
        for j in range(labeled_pores.shape[1]):
            boundary_pores[:, j, :] = labeled_pores[:, j, :]
            boundary_pores[:, -j - 1, :] = labeled_pores[:, -j - 1, :]
        for k in range(labeled_pores.shape[2]):
            boundary_pores[:, :, k] = labeled_pores[:, :, k]
            boundary_pores[:, :, -k - 1] = labeled_pores[:, :, -k - 1]
        connected_pores_labels = np.unique(boundary_pores[boundary_pores > 0])
        closed_pores_labels = [l for l in range(1, num_features + 1) if l not in connected_pores_labels]

        # 计算连通孔占比
        connected_pores = np.isin(labeled_pores, connected_pores_labels).astype(np.uint8)
        connected_porosity = np.sum(connected_pores) / np.sum(pores) if np.sum(pores) > 0 else 0
        print(f"Connected Pores Ratio: {connected_porosity:.4f} (Target: >0.99)")

        # 2.1 封闭孔（蓝色）
        closed_pores = np.isin(labeled_pores, closed_pores_labels).astype(np.uint8)
        vertices_closed, triangles_closed = mcubes.marching_cubes(closed_pores, 0.5)
        if len(vertices_closed) > 0 and len(triangles_closed) > 0:
            mesh_closed = trimesh.Trimesh(vertices=vertices_closed, faces=triangles_closed)
            mesh_closed.visual.vertex_colors = [0, 0, 255, 255]  # 蓝色
            output_path_closed = os.path.join(output_dir, f"{base}_closed_pores.ply")
            mesh_closed.export(output_path_closed, file_type='ply')
            print(f"成功生成封闭孔部分: {output_path_closed}")
        else:
            print("警告：封闭孔部分为空，未生成封闭孔网格")

        # 2.2 连通孔骨骼网络（黄色，作为支架部分）
        skeleton = skeletonize(connected_pores).astype(np.uint8)

        def extract_skeleton_lines(skeleton):
            lines = []
            for i in range(1, skeleton.shape[0] - 1):
                for j in range(1, skeleton.shape[1] - 1):
                    for k in range(1, skeleton.shape[2] - 1):
                        if skeleton[i, j, k]:
                            neighbors = [(i + 1, j, k), (i - 1, j, k), (i, j + 1, k), (i, j - 1, k), (i, j, k + 1),
                                         (i, j, k - 1)]
                            for ni, nj, nk in neighbors:
                                if skeleton[ni, nj, nk]:
                                    lines.append([(i, j, k), (ni, nj, nk)])
            return lines

        skeleton_lines = extract_skeleton_lines(skeleton)
        if skeleton_lines:
            line_segments = [line for line in skeleton_lines]
            vertices_skeleton = np.array(line_segments).reshape(-1, 3)
            edges_skeleton = np.arange(len(line_segments) * 2).reshape(-1, 2)
            mesh_skeleton = trimesh.Trimesh(vertices=vertices_skeleton, edges=edges_skeleton, process=False)
            mesh_skeleton.visual.vertex_colors = [255, 255, 0, 255]  # 黄色
            output_path_skeleton = os.path.join(output_dir, f"{base}_skeleton.ply")
            mesh_skeleton.export(output_path_skeleton, file_type='ply')
            print(f"成功生成连通孔骨骼网络: {output_path_skeleton}")
        else:
            print("警告：连通孔骨骼网络为空，未生成骨骼网络")

        # 3. 材料与支架合并文件
        if len(vertices_mat) > 0 and len(triangles_mat) > 0 and skeleton_lines:
            scene = trimesh.Scene()
            scene.add_geometry(mesh_mat)
            scene.add_geometry(mesh_skeleton)
            output_path_combined = os.path.join(output_dir, f"{base}_material_pores_combined.ply")
            with open(output_path_combined, 'wb') as f:
                scene.export(file_obj=f, file_type='ply')
            print(f"成功生成材料与孔隙结合文件: {output_path_combined}")
        else:
            print("警告：无法生成材料与孔隙结合文件")

        # 4. 可视化孔径分布
        distances = ndi.distance_transform_edt(adjusted_voxel)
        pore_diameters = distances[adjusted_voxel == 1] * 2 * voxel_size
        if len(pore_diameters) > 0:
            hist, bins = np.histogram(pore_diameters, bins=50, range=(0, 320))
            plt.figure(figsize=(8, 6))
            plt.bar(bins[:-1], hist, width=np.diff(bins), edgecolor='black')
            plt.title("Pore Size Distribution")
            plt.xlabel("Pore Diameter (μm)")
            plt.ylabel("Frequency")
            plt.savefig(os.path.join(output_dir, f"{base}_pore_size_distribution.png"), dpi=300)
            plt.close()
        else:
            print("警告：无孔隙，无法生成孔径分布图")

    except Exception as e:
        print(f"生成错误: {str(e)}")


# 主流程
if __name__ == "__main__":
    # 训练模型
    train_model()

    # 生成3D模型
    sem_image_path = "/pytest/training/PythonProject3/training/测试样本/测试/3-7实木板-02.tif"
    model_path = "/pytest/training/PythonProject3/training/output/model_cpu.pth"
    output_dir = "/pytest/training/PythonProject3/training/output"
    voxel_size = 0.1  # 根据标尺计算（20 μm / 200 像素 = 0.1 μm/像素）
    generate_3d_model(sem_image_path, model_path, output_dir, voxel_size)