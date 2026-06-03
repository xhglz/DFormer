import torch
import torch.nn as nn
import torch.nn.functional as F


class RGBDDistiller(nn.Module):
    def __init__(
        self,
        student,
        teacher,
        criterion,
        background,
        kd_weight=0.5,
        feat_weight=0.1,
        temperature=4.0,
        feat_indices=(2, 3),
        student_feat_channels=None,
        teacher_feat_channels=None,
    ):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.criterion = criterion
        self.background = background
        self.kd_weight = kd_weight
        self.feat_weight = feat_weight
        self.temperature = temperature
        self.feat_indices = tuple(feat_indices)
        self.adapters = nn.ModuleDict()

        if student_feat_channels is not None and teacher_feat_channels is not None:
            for index in self.feat_indices:
                s_c = int(self._get_channel(student_feat_channels, index))
                t_c = int(self._get_channel(teacher_feat_channels, index))
                if s_c != t_c:
                    self.adapters[str(index)] = nn.Conv2d(s_c, t_c, kernel_size=1, bias=False)

        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    def state_dict(self, *args, **kwargs):
        state = {}
        student_state = self.student.state_dict(*args, **kwargs)
        for k, v in student_state.items():
            state["student." + k] = v
        adapters_state = self.adapters.state_dict(*args, **kwargs)
        for k, v in adapters_state.items():
            state["adapters." + k] = v
        return state

    def load_state_dict(self, state_dict, strict=True):
        if not state_dict:
            return self.student.load_state_dict(state_dict, strict=strict)

        has_prefix = any(k.startswith("student.") or k.startswith("adapters.") for k in state_dict.keys())
        if not has_prefix:
            return self.student.load_state_dict(state_dict, strict=strict)

        student_state = {}
        adapters_state = {}
        for k, v in state_dict.items():
            if k.startswith("student."):
                student_state[k[len("student.") :]] = v
            elif k.startswith("adapters."):
                adapters_state[k[len("adapters.") :]] = v
        student_msg = self.student.load_state_dict(student_state, strict=False)
        adapters_msg = self.adapters.load_state_dict(adapters_state, strict=False)
        return student_msg, adapters_msg

    def _forward_features_and_logits(self, model, rgb, modal_x):
        feats = model.backbone(rgb, modal_x)
        if len(feats) == 2:
            feats = feats[0]
        logits = model.decode_head.forward(feats)
        # logits = F.interpolate(logits, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
        # return feats, logits
        selected_feats = tuple(feats[index] for index in self.feat_indices)
        return selected_feats, logits

    def _get_channel(self, channel_spec, index):
        if isinstance(channel_spec, (list, tuple)):
            return channel_spec[index]
        if isinstance(channel_spec, dict):
            if index in channel_spec:
                return channel_spec[index]
            return channel_spec[str(index)]
        if hasattr(channel_spec, str(index)):
            return getattr(channel_spec, str(index))
        return channel_spec[index]

    def _seg_loss(self, logits, label):
        valid = label.long() != self.background
        return self.criterion(logits, label.long())[valid].mean()

    def _kd_loss(self, student_logits, teacher_logits):
        temperature = self.temperature
        # resize student logits to match teacher's spatial size
        if student_logits.shape[-2:] != teacher_logits.shape[-2:]:
            student_logits = F.interpolate(
                student_logits, size=teacher_logits.shape[-2:], mode="bilinear", align_corners=False
            )
        return (
            F.kl_div(
                F.log_softmax(student_logits / temperature, dim=1),
                F.softmax(teacher_logits / temperature, dim=1),
                reduction="mean",
            )
            * (temperature**2)
        )

    def _feature_loss(self, student_feats, teacher_feats):
        feat_loss = student_feats[0].new_zeros(())
        for feat_offset, feat_index in enumerate(self.feat_indices):
            s = student_feats[feat_offset]
            t = teacher_feats[feat_offset]
            if s.shape[-2:] != t.shape[-2:]:
                s = F.interpolate(s, size=t.shape[-2:], mode="bilinear", align_corners=False)
            adapter = self.adapters[str(feat_index)] if str(feat_index) in self.adapters else None
            if adapter is not None:
                s = adapter(s)
            feat_loss = feat_loss + F.mse_loss(s, t)
        return feat_loss / max(len(self.feat_indices), 1)

    def forward(self, rgb, modal_x=None, label=None):
        if label is None:
            return self.student(rgb, modal_x)

        with torch.no_grad():
            teacher_feats, teacher_logits = self._forward_features_and_logits(self.teacher, rgb, modal_x)
        student_feats, student_logits = self._forward_features_and_logits(self.student, rgb, modal_x)

        seg_logits = F.interpolate(student_logits, size=label.shape[-2:], mode="bilinear", align_corners=False)

        seg_loss = self._seg_loss(seg_logits, label)
        kd_loss = self._kd_loss(student_logits, teacher_logits)
        feat_loss = self._feature_loss(student_feats, teacher_feats)
        total_loss = seg_loss + self.kd_weight * kd_loss + self.feat_weight * feat_loss

        return {
            "loss": total_loss,
            "seg_loss": seg_loss.detach(),
            "kd_loss": kd_loss.detach(),
            "feat_loss": feat_loss.detach(),
        }
