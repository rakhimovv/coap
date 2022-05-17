# adapted from https://github.com/neuralbodies/leap/blob/f2475588ddbc673365b429b21e4ba8f88bfd357c/leap/leap_body_model.py#L96

import os.path as osp
import pickle

import numpy as np
import pytorch3d
import pytorch3d.ops
import pytorch3d.structures
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh

from .modules import ResnetPointnet, QueryEncoder, Decoder

try:
    import torchgeometry
    from skimage import measure
except Exception:
    pass

from .tools.libmise import MISE


class COAPBodyModel(nn.Module):
    def __init__(self, bm_path=None, num_betas=16, dtype=torch.float32, init_modules=True):
        """ Interface for COAP body model.

        Args:
            bm_path (str): Path to a SMPL-compatible model file.
            num_betas (int): Number of shape coefficients for SMPL.
            dtype (torch.dtype): Datatype of pytorch tensors
            device (str): Which device to use.
        """
        super(COAPBodyModel, self).__init__()

        self.num_betas = num_betas
        self.dtype = dtype
        self.num_parts = 22

        # modules
        if init_modules:
            encoder_out_dim, query_dim = 128, 128
            self.encoder = ResnetPointnet(4, 128, encoder_out_dim)
            self.query_encoder = QueryEncoder(in_dim=encoder_out_dim+self.num_parts+1+3, query_dim=query_dim)
            self.decoder = Decoder(query_dim)

        # load SMPL-based model
        smpl_dict, self.model_type = self.load_smpl_model(bm_path)

        # parse loaded model dictionary
        self.num_betas = smpl_dict['shapedirs'].shape[-1] if self.num_betas < 1 else self.num_betas
        weights = smpl_dict['weights'][np.newaxis]  # 1, n, k
        kintree_table = smpl_dict['kintree_table'].astype(np.int32)  # 2, k
        v_template = smpl_dict['v_template'][np.newaxis]  # 1, n, 3
        joint_regressor = smpl_dict['J_regressor']  # k, n
        pose_dirs = smpl_dict['posedirs'].reshape([smpl_dict['posedirs'].shape[0] * 3, -1]).T  # (k-1)*3*3, n*3
        shape_dirs = smpl_dict['shapedirs'][:, :, :self.num_betas]  # n, 3, num_betas

        self.v_template = torch.tensor(v_template, dtype=dtype)
        self.shape_dirs = torch.tensor(shape_dirs, dtype=dtype)
        self.pose_dirs = torch.tensor(pose_dirs, dtype=dtype)
        self.joint_regressor = torch.tensor(joint_regressor, dtype=dtype)
        self.kintree_table = torch.tensor(kintree_table, dtype=torch.int32)
        self.weights = torch.tensor(weights, dtype=dtype)

        # define part info
        self.part_verts_idxs = None
        self.part_faces = None
        self.smpl_dict = smpl_dict
        can_part_meshes = self.build_part_meshes(smpl_dict['weights'], smpl_dict['f'].astype(np.int32),
                                                 self.v_template[0])

        if self.model_type == 'smplx':
            begin_shape_id = 300 if smpl_dict['shapedirs'].shape[-1] > 300 else 10
            num_expressions = 10
            expr_dirs = smpl_dict['shapedirs'][:, :, begin_shape_id:(begin_shape_id + num_expressions)]
            expr_dirs = torch.tensor(expr_dirs, dtype=dtype)
            self.shape_dirs = torch.cat([self.shape_dirs, expr_dirs])

        # init controllable parameters
        self.betas = None
        self.root_loc = None
        self.root_orient = None
        self.pose_body = None
        self.pose_hand = None
        self.pose_jaw = None
        self.pose_eye = None

        # intermediate representations
        self.pose_rot_mat = None
        self.posed_vert = None
        self.can_vert = None
        self.rel_joints = None
        self.fwd_transformation = None
        self.local_shape_codes = None
        self.backward_transforms = None

    def build_part_meshes(self, smpl_weights, smpl_faces, mesh_verts):
        """
        Build canonical body part meshes.
        
        Args:
            smpl_weights (torch.tensort): (skinning weights of shape (n, k)
            smpl_faces (torch.tensort): mesh vertices of shape (m, 3)
            mesh_verts (torch.tensort): mesh vertices of shape (n, 3)
        """

        # common preprocessing
        n, _ = smpl_weights.shape
        k = self.num_parts
        
        if self.part_verts_idxs is None or self.part_faces is None:
            num_faces = smpl_faces.shape[0]
            _n, _num_faces = np.arange(n), np.arange(num_faces)
            part_verts_idxs = [_n[smpl_weights[:, i] > 0.01] for i in range(k)]
            part_faces_idxs = [
                _num_faces[np.in1d(smpl_faces.reshape(-1), part_verts_idxs[i]).reshape(num_faces, 3).any(axis=1)] for i
                in
                range(k)]
            del part_verts_idxs
            self.part_verts_idxs = [torch.from_numpy(np.sort(np.unique(smpl_faces[part_faces_idxs[i]]))).long() for i in
                                    range(k)]
            self.part_faces = [[] for i in range(k)]

            vert_idx_old2new = [{} for i in range(k)]
            for i in range(k):
                for new_idx, old_idx in enumerate(self.part_verts_idxs[i].tolist()):
                    vert_idx_old2new[i][old_idx] = new_idx
                part_faces_i_old = smpl_faces[part_faces_idxs[i]]  # (f, 3)
                for j in part_faces_i_old.reshape(-1).tolist():
                    self.part_faces[i].append(vert_idx_old2new[i][j])
                self.part_faces[i] = torch.from_numpy(
                    np.array(self.part_faces[i]).reshape(part_faces_i_old.shape[0], 3)).long()

        # build part meshes for particular pose
        verts = []
        faces = []
        for i in range(k):
            verts.append(mesh_verts[self.part_verts_idxs[i], :])
            faces.append(self.part_faces[i])

        return pytorch3d.structures.Meshes(verts, faces)

    def get_part_neigbours(self):
        parents = self.kintree_table[0].tolist()
        k = len(parents)
        children = {i: [] for i in range(k)}
        for i in range(k):
            if i > 0:
                children[parents[i]].append(i)
        nearest_parts = []
        for i in range(k):
            nearest_parts.append(children[i])
            if i > 0:
                nearest_parts[-1].append(parents[i])
        # filter
        filtered_nearest_parts = []
        for i in range(self.num_parts):
            filtered_nearest_parts.append([])
            for p in nearest_parts[i]:
                if p < self.num_parts:
                    filtered_nearest_parts[-1].append(p)
        return filtered_nearest_parts

    def set_parameters(self,
                       batch_size: int,
                       betas=None,
                       pose_body=None,
                       pose_hand=None,
                       pose_jaw=None,
                       pose_eye=None,
                       expression=None,
                       device: torch.device = torch.device('cpu')):
        """ Set controllable parameters.

        Args:
            betas (torch.tensor): SMPL shape coefficients (B x betas len).
            pose_body (torch.tensor): Body pose parameters (B x body joints * 3).
            pose_hand (torch.tensor): Hand pose parameters (B x hand joints * 3).
            pose_jaw (torch.tensor): Jaw pose parameters (compatible with SMPL+X) (B x jaw joints * 3).
            pose_eye (torch.tensor): Eye pose parameters (compatible with SMPL+X) (B x eye joints * 3).
            expression (torch.tensor): Expression coefficients (compatible with SMPL+X) (B x expr len).
        """
        if betas is None:
            betas = torch.tensor(np.zeros((batch_size, self.num_betas)), dtype=self.dtype, device=device)
        else:
            betas = betas.view(batch_size, self.num_betas)

        if pose_body is None and self.model_type in ['smpl', 'smplh', 'smplx']:
            pose_body = torch.tensor(np.zeros((batch_size, 63)), dtype=self.dtype, device=device)
        else:
            pose_body = pose_body.view(batch_size, 63)

        # pose_hand
        if pose_hand is None:
            if self.model_type in ['smpl']:
                pose_hand = torch.tensor(np.zeros((batch_size, 1 * 3 * 2)), dtype=self.dtype, device=device)
            elif self.model_type in ['smplh', 'smplx']:
                pose_hand = torch.tensor(np.zeros((batch_size, 15 * 3 * 2)), dtype=self.dtype, device=device)
            elif self.model_type in ['mano']:
                pose_hand = torch.tensor(np.zeros((batch_size, 15 * 3)), dtype=self.dtype, device=device)
        else:
            pose_hand = pose_hand.view(batch_size, -1)

        # face poses
        if self.model_type == 'smplx':
            if pose_jaw is None:
                pose_jaw = torch.tensor(np.zeros((batch_size, 1 * 3)), dtype=self.dtype, device=device)
            else:
                pose_jaw = pose_jaw.view(batch_size, 1 * 3)

            if pose_eye is None:
                pose_eye = torch.tensor(np.zeros((batch_size, 2 * 3)), dtype=self.dtype, device=device)
            else:
                pose_eye = pose_eye.view(batch_size, 2 * 3)

            if expression is None:
                expression = torch.tensor(np.zeros((batch_size, self.num_expressions)), dtype=self.dtype, device=device)
            else:
                expression = expression.view(batch_size, self.num_expressions)

            betas = torch.cat([betas, expression], dim=-1)

        self.root_loc = torch.tensor(np.zeros((batch_size, 1 * 3)), dtype=self.dtype, device=device)
        self.root_orient = torch.tensor(np.zeros((batch_size, 1 * 3)), dtype=self.dtype, device=device)

        self.betas = betas
        self.pose_body = pose_body
        self.pose_hand = pose_hand
        self.pose_jaw = pose_jaw
        self.pose_eye = pose_eye

    def _get_full_pose(self):
        """ Concatenates joints.

        Returns:
            full_pose (torch.tensor): Full pose (B, num_joints*3)
        """
        full_pose = [self.root_orient]
        if self.model_type in ['smplh', 'smpl']:
            full_pose.extend([self.pose_body, self.pose_hand])
        elif self.model_type == 'smplx':
            full_pose.extend([self.pose_body, self.pose_jaw, self.pose_eye, self.pose_hand])
        elif self.model_type in ['mano']:
            full_pose.extend([self.pose_hand])
        else:
            raise Exception('Unsupported model type.')

        full_pose = torch.cat(full_pose, dim=1)
        return full_pose

    def forward(self, points, surface_points, backward_transform):
        """ Checks whether given query points are located inside of a human body.

        Args:
            points (torch.tensor): Query points (B x T x 3)
            surface_points (torch.tensor): Query points (B x K x V x 3)

        Returns:
            occupancy values (torch.tensor): (B x T)
        """
        self.encode(surface_points, backward_transform)
        occupancy = self._query_occupancy(points, backward_transform)
        return occupancy

    @torch.no_grad()
    def sample_surface_points(self):
        neigbours = self.get_part_neigbours()
        num_samples = 1000
        main_num_samples = int(0.5 * num_samples)
        b = len(self.part_meshes)
        k = len(self.part_meshes[0])
        points = []
        for j in range(b):
            pc = []
            for i in range(k):
                part_pc = [pytorch3d.ops.sample_points_from_meshes(self.part_meshes[j][i], main_num_samples)]
                num_neigbours = len(neigbours[i])
                neigb_num_samples = [(num_samples - main_num_samples) // num_neigbours for _ in
                                     range(num_neigbours)]
                neigb_num_samples[-1] += (num_samples - main_num_samples) % num_neigbours
                for ti, t in enumerate(neigbours[i]):
                    part_pc.append(pytorch3d.ops.sample_points_from_meshes(self.part_meshes[j][t],
                                                                           num_samples=neigb_num_samples[ti]))
                part_pc = torch.cat(part_pc, dim=1)  # 1, num_samples, 3
                pc.append(part_pc)
            pc = torch.cat(pc, dim=0)  # k, num_samples, 3
            points.append(pc)
        points = torch.stack(points, dim=0)  # b, k, num_samples, 3
        return points

    def encode(self, points, backward_transform):
        # points: b, k, num_samples, 3

        with torch.no_grad():
            b, k, ns, _3 = points.shape

            # 1. canonicalize
            joint_idxs = torch.arange(k, device=points.device, dtype=torch.long)[None, :, None].expand(b, k, ns)
            joint_idxs = joint_idxs.contiguous().view(b, k * ns)
            points = points.view(b, k * ns, 3)
            can_points = self.canonicalize_to_joints(points, joint_idxs, backward_transform).view(b, k, ns, 3)

            # 2. add geometric prior
            central_parts = can_points[:, :, :int(0.5 * ns), :]
            bb_max = central_parts.max(dim=2).values  # b, k, 3
            bb_min = central_parts.min(dim=2).values  # b, k, 3
            total_size = (bb_max - bb_min).max(dim=2).values  # b, k
            self.bb_loc = (bb_min + bb_max) / 2  # b, k, 3
            self.bb_scale = (total_size / 2) * 1.15  # b, k

        # 3. compute local shape codes
        in_bbox = (can_points - self.bb_loc[:, :, None, :]).abs().max(dim=3).values < self.bb_scale[:, :, None]
        in_bbox = in_bbox.float()  # b, k, ns
        encoder_input = torch.cat([can_points.view(b * k, ns, 3), in_bbox.view(b*k, ns, 1)], dim=-1)

        self.local_shape_codes = self.encoder.forward(encoder_input).view(b, k, -1)

        one_hot = torch.eye(k, device=can_points.device, dtype=can_points.dtype)[None].expand(b, k, k)
        self.local_shape_codes = torch.cat([self.local_shape_codes, one_hot], dim=2)

    def _query_occupancy(self, points, backward_transform, canonical_points=False):
        with torch.no_grad():
            can_points = self.canonicalize(points, backward_transform)  # b, t, k, 3
            z = self.local_shape_codes  # b, k, z_dim+k
            z = z[:, None, :, :].expand(-1, can_points.shape[1], -1, -1)  # b, t, k, z_dim+k
            in_bbox = (can_points - self.bb_loc[:, None, :, :]).abs().max(dim=3).values < self.bb_scale[:, None, :]
            in_bbox = in_bbox.float()[:, :, :, None]  # b, t, k, 1
            x = torch.cat([z, in_bbox, can_points], dim=-1)  # b, t, k, (z_dim+k)+1+3

        query = self.query_encoder.forward(x)  # b, t, k, query_dim
        occupancy = self.decoder.forward(torch.cat([query, can_points], dim=-1))  # b, t, k, 1

        occupancy = torch.sigmoid(occupancy) * in_bbox
        occupancy = torch.max(occupancy, dim=2).values  # b, t, 1
        occupancy = occupancy[:, :, 0]

        return occupancy

    def forward_parametric_model(self):
        B = self.pose_body.shape[0]

        # Compute identity-dependent correctives
        identity_offsets = torch.einsum('bl,ndl->bnd', self.betas, self.shape_dirs)  # b, n, 3 (l=|beta|, d=3)

        # pose to rot matrices
        full_pose = self._get_full_pose()
        full_pose = full_pose.view(B, -1, 3)  # b, k, 3
        self.pose_rot_mat = torchgeometry.angle_axis_to_rotation_matrix(full_pose.view(-1, 3))[:, :3, :3]  # b*k, 3, 3
        self.pose_rot_mat = self.pose_rot_mat.view(B, -1, 3, 3)  # b, k, 3, 3

        # Compute pose-dependent correctives
        eye = torch.eye(3, dtype=self.dtype, device=self.pose_rot_mat.device)
        _pose_feature = self.pose_rot_mat[:, 1:, :, :] - eye  # b, k-1, 3, 3  # fixme why "-eye" ? and not elf.pose_rot_mat[:, [0], :, :]
        pose_offsets = torch.matmul(
            _pose_feature.view(B, -1),  # b, (k-1)*3*3
            self.pose_dirs  # (k-1)*3*3, n*3
        ).view(B, -1, 3)  # b, n, 3

        self.can_vert = self.v_template + identity_offsets + pose_offsets  # b, n, 3

        # Regress joint locations
        self.can_joint_loc = torch.einsum('bnd,kn->bkd',
                                          self.v_template + identity_offsets,
                                          self.joint_regressor)  # b, k, 3

        # Skinning
        self.fwd_transformation, self.rel_joints, self.backward_transforms = self.batch_rigid_transform(self.pose_rot_mat, self.can_joint_loc)
        self.posed_vert = self.lbs_skinning(self.fwd_transformation, self.can_vert)  # b, n, 3

        # Build part meshes
        self.part_meshes = []
        for i in range(B):
            self.part_meshes.append(
                self.build_part_meshes(self.smpl_dict['weights'], self.smpl_dict['f'], self.posed_vert[i]))

    def batch_rigid_transform(self, rot_mats, joints):
        """ Rigid transformations over joints

        Args:
            rot_mats (torch.tensor): Rotation matrices (BxKx3x3).
            joints (torch.tensor): Joint locations (BxKx3).

        Returns:
            rel_transforms (torch.tensor): Relative wrt root joint rigid transformations (BxKx4x4).
            posed_joints (torch.tensor): The locations of the joints after applying transformations (BxKx3).
        """
        B, K = rot_mats.shape[0], joints.shape[1]

        parents = self.kintree_table[0].long()  # (k)

        joints = torch.unsqueeze(joints, dim=-1)  # (b, k, 3, 1)

        rel_joints = joints.clone()  # (b, k, 3, 1)
        rel_joints[:, 1:] -= joints[:, parents[1:]]    # (b, k, 3, 1)

        transforms_mat = torch.cat([
            F.pad(rot_mats.reshape(-1, 3, 3), [0, 0, 0, 1]),  # pad(b*k, 3, 3) -> (b*k, 4, 3)
            F.pad(rel_joints.reshape(-1, 3, 1), [0, 0, 0, 1], value=1)  # pad(b*k, 3, 1) -> (b*k, 4, 1)
        ], dim=2).reshape(-1, joints.shape[1], 4, 4)  # (b, k, 4, 4)

        transform_chain = [transforms_mat[:, 0]]
        for i in range(1, parents.shape[0]):
            curr_res = torch.matmul(transform_chain[parents[i]], transforms_mat[:, i])
            transform_chain.append(curr_res)

        transforms = torch.stack(transform_chain, dim=1)  # (b, k, 4, 4)
        backward_transforms = torch.inverse(transforms)

        joints_hom = torch.cat([
            joints,  # (b, k, 3, 1)
            torch.zeros([B, K, 1, 1], dtype=self.dtype, device=joints.device)
        ], dim=2)  # (b, k, 4, 1)
        init_bone = F.pad(
            torch.matmul(transforms, joints_hom),  # (b, k, 4, 1)
            [3, 0, 0, 0, 0, 0, 0, 0]
        )  # (b, k, 4, 4)
        rel_transforms = transforms - init_bone  # B_k, (b, k, 4, 4)

        return rel_transforms, rel_joints, backward_transforms

    @staticmethod
    def load_smpl_model(bm_path):
        assert osp.exists(bm_path), f'File does not exist: {bm_path}'

        # load smpl parameters
        ext = osp.splitext(bm_path)[-1]
        if ext == '.npz':
            smpl_model = np.load(bm_path, allow_pickle=True)
        elif ext == 'pkl':
            with open(bm_path, 'rb') as smpl_file:
                smpl_model = pickle.load(smpl_file, encoding='latin1')
        else:
            raise ValueError(f'Invalid file type: {ext}')
        
        num_joints = smpl_model['posedirs'].shape[2] // 3
        model_type = {69: 'smpl', 153: 'smplh', 162: 'smplx', 45: 'mano'}[num_joints]

        return smpl_model, model_type

    @staticmethod
    def get_num_joints(bm_path):
        model_type = COAPBodyModel.load_smpl_model(bm_path)[1]

        num_joints = {
            'smpl': 24,
            'smplh': 52,
            'smplx': 55,
            'mano': 16,
        }[model_type]

        return model_type, num_joints

    @staticmethod
    def get_parent_mapping(model_type):
        smplh_mappings = [
            -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 22, 23, 20, 25,
            26, 20, 28, 29, 20, 31, 32, 20, 34, 35, 21, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50
        ]

        if model_type == 'smpl':
            smpl_mappings = smplh_mappings[:22] + [smplh_mappings[25]] + [smplh_mappings[40]]
            return smpl_mappings
        elif model_type == 'smplh':
            return smplh_mappings
        else:
            raise NotImplementedError

    def lbs_skinning(self, fwd_transformation, can_vert):
        """ Conversion of canonical vertices to posed vertices via linear blend skinning.

        Args:
            fwd_transformation (torch.tensor): Forward rigid transformation tensor (B x K x 4 x 4).
            can_vert (torch.tensor): Canonical vertices (B x N x 3).

        Returns:
            posed_vert (torch.tensor): Posed vertices (B x N x 3).
        """
        B = fwd_transformation.shape[0]

        _fwd_lbs_trans = torch.matmul(
            self.weights,  # b, n, k
            fwd_transformation.view(B, -1, 16)  # b, k, 4*4
        ).view(B, -1, 4, 4)  # b, n, 4, 4

        _vert_hom = torch.cat([
            can_vert,
            torch.ones([B, can_vert.shape[1], 1], dtype=self.dtype, device=can_vert.device)
        ], dim=2)  # b, n, 4
        posed_vert = torch.matmul(_fwd_lbs_trans, torch.unsqueeze(_vert_hom, dim=-1))[:, :, :3, 0]
        return posed_vert

    @torch.no_grad()
    def canonicalize_to_joints(self, points, joint_idxs, backward_transform):
        """ Project points to a canonical space via the corresponding bone transformation

        Args:
            points (torch.tensor): input points in posed space (B x T x 3).
            joint_idxs (torch.tensor): corresponding joint indexes (B x T) in range [0, K-1]
            backward_transform (torch.tensor): (B x K x 4 x 4).

        Returns:
            can_points (torch.tensor): canonicalized input points (B x T x 3).
        """
        joint_idxs = joint_idxs[:, :, None, None].expand(-1, -1, 4, 4)  # b, t, 4, 4
        point_backward_transform = torch.gather(backward_transform, dim=1, index=joint_idxs)  # b, t, 4, 4

        points_hom = torch.cat([
            points,
            torch.ones((1, 1, 1), dtype=points.dtype, device=points.device).expand(points.shape[0], points.shape[1], 1)
        ], dim=2)  # b, t, 4
        points_hom = points_hom[:, :, :, None]  # b, t, 4, 1

        can_points = torch.matmul(point_backward_transform, points_hom)  # b, t, 4, 1
        can_points = can_points[:, :, :3, 0]  # b, t, 3

        return can_points

    @torch.no_grad()
    def canonicalize(self, points, backward_transform):
        """ Project points to a canonical space via the corresponding bone transformation

        Args:
            points (torch.tensor): input points in posed space (B x T x 3).
            backward_transform (torch.tensor): (B x K x 4 x 4).

        Returns:
            can_points (torch.tensor): canonicalized input points (B x T x K x 3).
        """
        b, t, _3 = points.shape
        b, k, _4, _4 = backward_transform.shape

        backward_transform = backward_transform[:, None, :, :, :]  # b, 1, k, 4, 4
        backward_transform = backward_transform.expand(b, t, k, 4, 4)

        points_hom = torch.cat([
            points,
            torch.ones((1, 1, 1), dtype=points.dtype, device=points.device).expand(points.shape[0], points.shape[1], 1)
        ], dim=2)  # b, t, 4
        points_hom = points_hom[:, :, None, :, None]  # b, t, 1, 4, 1
        points_hom = points_hom.expand(b, t, k, 4, 1)

        can_points = torch.matmul(backward_transform, points_hom)  # b, t, k, 4, 1
        can_points = can_points[:, :, :, :3, 0]  # b, t, k, 3

        return can_points

    @torch.no_grad()
    def _extract_mesh(self, vertices, resolution0, upsampling_steps, canonical_points=False):
        """ Runs marching cubes to extract mesh for the occupancy representation. """
        device = vertices.device

        # compute scale and loc
        bb_min = np.min(vertices, axis=0)
        bb_max = np.max(vertices, axis=0)
        loc = np.array([
            (bb_min[0] + bb_max[0]) / 2,
            (bb_min[1] + bb_max[1]) / 2,
            (bb_min[2] + bb_max[2]) / 2
        ])
        scale = (bb_max - bb_min).max()

        scale = torch.FloatTensor([scale]).to(device=device)
        loc = torch.from_numpy(loc).to(device=device)

        # create MISE
        threshold = 0.5
        padding = 0.1
        box_size = 1 + padding
        mesh_extractor = MISE(resolution0, upsampling_steps, threshold)

        # sample initial points
        points = mesh_extractor.query()
        while points.shape[0] != 0:
            sampled_points = torch.FloatTensor(points).to(device=device)  # Query points
            sampled_points = sampled_points / mesh_extractor.resolution  # Normalize to bounding box
            sampled_points = box_size * (sampled_points - 0.5)
            sampled_points *= scale
            sampled_points += loc

            # check occupancy for sampled points
            p_split = torch.split(sampled_points, 50000)  # to prevent OOM
            occ_hats = []
            for pi in p_split:
                pi = pi.unsqueeze(0).to(device=device)
                occ_hats.append(self._query_occupancy(pi, canonical_points).cpu().squeeze(0))
            values = torch.cat(occ_hats, dim=0).numpy().astype(np.float64)

            # sample points again
            mesh_extractor.update(points, values)
            points = mesh_extractor.query()

        occ_hat = mesh_extractor.to_dense()
        # Some short hands
        n_x, n_y, n_z = occ_hat.shape
        box_size = 1 + padding
        # Make sure that mesh is watertight
        occ_hat_padded = np.pad(occ_hat, 1, 'constant', constant_values=-1e6)
        vertices, faces, _, _ = measure.marching_cubes(occ_hat_padded, level=threshold)

        vertices -= 1  # Undo padding
        # Normalize to bounding box
        vertices /= np.array([n_x - 1, n_y - 1, n_z - 1])
        vertices = box_size * (vertices - 0.5)
        vertices = vertices * scale.item()
        vertices = vertices + loc.view(1, 3).detach().cpu().numpy()

        # Create mesh
        mesh = trimesh.Trimesh(vertices, faces, process=False)
        return mesh

    @torch.no_grad()
    def extract_canonical_mesh(self, resolution0=32, upsampling_steps=3):
        self.model.eval()
        self.forward_parametric_model()

        mesh = self._extract_mesh(self.can_vert.squeeze(0).detach().cpu().numpy(),
                                  resolution0,
                                  upsampling_steps,
                                  canonical_points=True)
        return mesh

    @torch.no_grad()
    def extract_posed_mesh(self, resolution0=32, upsampling_steps=3):
        self.model.eval()
        self.forward_parametric_model()

        mesh = self._extract_mesh(self.posed_vert.squeeze(0).detach().cpu().numpy(),
                                  resolution0,
                                  upsampling_steps,
                                  canonical_points=False)
        return mesh
