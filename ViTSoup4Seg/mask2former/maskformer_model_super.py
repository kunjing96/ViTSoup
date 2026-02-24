from detectron2.modeling import META_ARCH_REGISTRY

from .maskformer_model import MaskFormer


@META_ARCH_REGISTRY.register()
class SuperMaskFormer(MaskFormer):

    def set_sample_config(self, config: dict):
        if hasattr(self.backbone, "set_sample_config"):
            self.backbone.set_sample_config(config.BACKBONE)
        if hasattr(self.sem_seg_head, "set_sample_config"):
            self.sem_seg_head.set_sample_config(config)

    def params(self):
        params = 0
        params += self.backbone.params()
        params += self.sem_seg_head.params()
        return params

    def flops(self, H, W):
        flops = 0
        flops += self.backbone.flops(H, W)
        flops += self.sem_seg_head.flops(H, W, self.backbone.output_shape())
        return flops
