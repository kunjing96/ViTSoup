from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from .mask_former_head import MaskFormerHead


@SEM_SEG_HEADS_REGISTRY.register()
class SuperMaskFormerHead(MaskFormerHead):

    def set_sample_config(self, config: dict):
        if hasattr(self.pixel_decoder, "set_sample_config"):
            self.pixel_decoder.set_sample_config(config)
        if hasattr(self.pixel_decoder, "set_sample_config"):
            self.predictor.set_sample_config(config)

    def params(self):
        params = 0
        params += self.pixel_decoder.params()
        params += self.predictor.params()
        return params

    def flops(self, H, W, shapes):
        flops = 0
        flops += self.pixel_decoder.flops(H, W, shapes)
        flops += self.predictor.flops(H, W, shapes)
        return flops